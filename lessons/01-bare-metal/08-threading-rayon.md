---
title: "Lesson 8 — From SIMD to Threads (Rayon)"
date: "2026-06-03"
module: "bare-metal"
order: 8
tags: ["threading", "rayon", "work-stealing", "parallelism", "rust", "numa"]
author: "Sudipta Pathak"
prerequisites: ["07-simd-arm-apple"]
---

# Lesson 8 — From SIMD to Threads

## Why this matters

One P-core, fully optimized, is fast. But it's still one core. Modern CPUs have 8 to 64+ cores. If we keep the matmul on one core we leave 80–98% of the chip idle. Threading is how we use the rest — and, done badly, it's also how we lose the speedup we just earned by adding overhead, cache thrash, and false sharing.

The mental model shift: SIMD parallelism is **within** one instruction stream. Thread parallelism is **across** instruction streams. Both can win; they win at different scales, and they compose. Our matmul will end this lesson sitting close to peak per-socket throughput on a modern multi-core CPU.

The threading library we'll use is **Rayon**. It implements work-stealing on top of native threads with a remarkably clean API. It's the de-facto choice for data-parallel CPU work in Rust. For irregular concurrency (network servers, futures), you'd use `tokio` or `async-std` instead — Rayon is for "I have a chunk of computation to spread across cores."

## Concept

### Amdahl's law, briefly

If a fraction `p` of your program is parallelizable and the rest is serial, the maximum speedup with `N` workers is `1 / ((1 - p) + p/N)`. For matmul, `p` can be made very close to 1 — the multiplication itself decomposes cleanly along the `i` and `j` axes. So we should expect near-linear scaling up to memory bandwidth limits.

Why "up to memory bandwidth limits"? Because every core is trying to pull data from the same shared DRAM. If your inner kernel is bandwidth-bound (which a matmul without aggressive packing tends to be), 8 cores all running at full throughput can saturate the memory bus, and the 9th core adds nothing. This is why core counts above a certain point don't help for pure GEMM.

### Work-stealing

Naive parallelism: take the work, divide it into N equal chunks, hand one to each worker. Works fine when chunks finish at the same rate. Falls apart when one chunk is slower (data-dependent cost, scheduler interference) and the other workers sit idle while the slow one drags on.

Work-stealing fixes this. Each worker has a local queue of tasks. When a worker runs out of its own tasks, it steals from another worker's queue. The result: load balances itself. If task A finishes in 1 ms and task B takes 10 ms because of cache pressure, idle workers can subdivide and help finish B.

Rayon implements this on top of native threads via the `rayon-core` crate. You don't think about it. You write `.par_iter().map(...)` and Rayon handles the rest.

### When threading helps and when it doesn't

Threading helps when:
- The work is large enough that thread-startup overhead is amortized. Each thread launch is ~1 µs. Work below ~10 µs per task is a loss.
- The work is parallelizable along an independent axis (matmul: `i` and `j` independent across output rows/cols).
- The cores share enough cache that data doesn't have to be re-loaded per core (or, the per-core working set is small enough to fit in private caches).

Threading hurts when:
- Tasks are tiny — scheduling overhead dwarfs compute.
- Tasks contend on the same data with writes — cache-coherence traffic kills throughput (this is **false sharing**).
- The system is bandwidth-bound — adding cores doesn't add bandwidth, just contention.
- NUMA is involved and the work doesn't respect locality — cross-socket access is 2–5× slower than local.

### NUMA, briefly

A NUMA (Non-Uniform Memory Access) system has multiple memory controllers. Each CPU socket has "near" memory it reads at full speed and "far" memory it reads more slowly via interconnect (UPI/Infinity Fabric). On dual-socket workstations and servers, NUMA is the rule.

For matmul, this matters: if your A and B matrices are allocated on one socket's memory and you spread workers across two sockets, half your workers are paying cross-socket cost on every cache miss. The fix is either to pin each worker to a socket and partition data accordingly, or to *interleave* memory allocation across NUMA nodes so the average cost equalizes.

Apple Silicon is **not NUMA** — it's a single unified memory pool. Most consumer laptops and desktops are single-socket. NUMA matters on workstations and servers, and we'll touch it more in Lesson 11.

### False sharing

If two threads write to *different* variables that happen to live in the *same* cache line, the cache coherency protocol bounces the line back and forth between cores on every write. The threads aren't logically competing — they're competing at the hardware level you can't see from source code.

The fix is to either pad data so different threads' working data live on different cache lines, or to accumulate locally per thread and combine at the end. Both work; the latter is more idiomatic and faster.

A classic example: each thread maintains a counter. If you store the counters as `let counters: Vec<u64> = vec![0; n_threads]`, they sit on the same cache line for the first 8 counters (64 bytes / 8 bytes per u64), and writes serialize. Pad each counter to its own cache line, and throughput scales linearly.

## Code walkthrough

Threading a cache-blocked matmul.

### Partition by output rows

The simplest correct parallelization: each thread is responsible for a contiguous block of output rows. Rows don't share writes (each `C[i, :]` is written by exactly one thread), and the read pattern is the same as serial.

```rust
use rayon::prelude::*;

pub fn matmul_threaded(a: &Mat, b: &Mat, c: &mut MatMut) {
    assert_eq!(a.cols, b.rows);
    assert_eq!(a.rows, c.rows);
    assert_eq!(b.cols, c.cols);

    let m = a.rows;
    let k = a.cols;
    let n = b.cols;

    // Each chunk of c is a horizontal strip of output rows.
    // We process one strip per thread; Rayon's work-stealing distributes them.
    c.data
        .par_chunks_mut(n)            // each chunk = one row of C
        .enumerate()
        .for_each(|(i, c_row)| {
            // For row i, compute c_row = a[i, :] · b
            // Inner loop: cache-blocked along k & j.
            for j_block in (0..n).step_by(64) {
                let j_end = (j_block + 64).min(n);
                for k_block in (0..k).step_by(64) {
                    let k_end = (k_block + 64).min(k);
                    for kk in k_block..k_end {
                        let a_ik = a.data[i * k + kk];
                        for j in j_block..j_end {
                            c_row[j] += a_ik * b.data[kk * n + j];
                        }
                    }
                }
            }
        });
}
```

`par_chunks_mut(n)` chops `c.data` into `n`-element chunks (each chunk is one output row), then runs the inner closure in parallel. Rayon's work-stealing schedules them.

This is simple — perhaps too simple, since one row at a time may not amortize Rayon's overhead. A better version chops C into wider strips:

```rust
const ROWS_PER_TASK: usize = 32;

c.data
    .par_chunks_mut(ROWS_PER_TASK * n)
    .enumerate()
    .for_each(|(strip_idx, strip)| {
        let i_start = strip_idx * ROWS_PER_TASK;
        let n_rows = strip.len() / n;
        for i_local in 0..n_rows {
            let i = i_start + i_local;
            let c_row = &mut strip[i_local * n..(i_local + 1) * n];
            // ... blocked inner kernel as above ...
        }
    });
```

Now each task does ~32 output rows of work — substantially more compute per task, so Rayon's per-task overhead is invisible. Choose `ROWS_PER_TASK` so each task takes 100 µs to a few ms; tasks much shorter than 100 µs are dominated by overhead.

### Calling the SIMD inner kernel

If you've written the NEON or AVX2 kernel from Lessons 6 / 7, you can call it from inside the `for_each` closure. Each thread runs its own micro-kernel; the kernels are independent (no shared writes); scaling should be near-linear up to memory-bandwidth limits.

```rust
c.data
    .par_chunks_mut(ROWS_PER_TASK * n)
    .enumerate()
    .for_each(|(strip_idx, strip)| {
        let i_start = strip_idx * ROWS_PER_TASK;
        // ... outer blocking loops over j and k ...
        // ... call micro_kernel_8x4 (or 8x4_neon) for each inner tile ...
    });
```

Important: the SIMD kernel uses `unsafe` raw pointers. Each task is working on a disjoint strip of `C`, so the raw-pointer aliasing concerns from a single-threaded run don't change — there are no inter-thread races as long as your strips don't overlap. The Rust borrow checker enforces the disjointness for you when you use `par_chunks_mut`.

### Avoiding false sharing in reductions

If your matmul is a *batched matmul* with a reduction across batches (or some statistic you accumulate per-thread), the per-thread accumulators should each live on their own cache line. Rayon's `fold + reduce` pattern handles this:

```rust
let total: f64 = (0..n)
    .into_par_iter()
    .fold(|| 0.0_f64, |acc, i| acc + compute(i))
    .reduce(|| 0.0_f64, |a, b| a + b);
```

`fold` runs per-thread with a thread-local accumulator (initialized by the first closure). `reduce` combines thread-local results. No shared writes during the hot loop; combining is O(n_threads), which is tiny.

## Mental model & pitfalls

The single sentence: **Thread parallelism multiplies your single-core speed by the core count, but only when (a) tasks are big enough to amortize overhead, (b) cores don't write to the same cache lines, and (c) memory bandwidth isn't saturated.**

Pitfalls:

- **Tasks too small.** Rayon's overhead is ~1–5 µs per task. Tasks shorter than 100 µs see more overhead than benefit. Either batch work into bigger units or use a faster scheduler (manual `crossbeam` channels, raw threads with `thread::scope`).
- **False sharing.** Per-thread counters or accumulators that share cache lines. Detect with `perf c2c` (Linux) — it reports false-sharing events directly.
- **Bandwidth bound.** If your inner loop is memory-bound (reads dominate compute), 8 threads contending for memory bus is at best as good as 4. Measure the bandwidth ceiling separately and don't expect to exceed it.
- **Imbalanced work.** Rayon handles imbalance well via work-stealing, but truly pathological tasks (one is 100× slower) will leave workers idle. Try to make tasks roughly equal-sized when you can.
- **NUMA mis-placement on multi-socket.** Workers running on socket 1 accessing memory on socket 0 pay 2–5× the latency. Either bind threads to NUMA nodes, or interleave memory.

## Hands-on (at home)

Extend the project.

1. **Add `rayon` to `Cargo.toml`:**

```toml
[dependencies]
rayon = "1.10"
```

2. **Implement `matmul_threaded`** as in the code walkthrough. Start with the simpler "row at a time" version; once it works, switch to the strip version with `ROWS_PER_TASK`.

3. **Sweep `ROWS_PER_TASK`** from 1 to 256 (powers of two). Plot or print GFLOPs vs ROWS_PER_TASK. You should see a curve: low at very small (overhead-dominated), high in a broad middle (32–128 typically), and *possibly* declining at the high end (load-imbalance becomes a factor).

4. **Vary the number of Rayon threads** to see scaling:

```bash
RAYON_NUM_THREADS=1 cargo run --release
RAYON_NUM_THREADS=2 cargo run --release
RAYON_NUM_THREADS=4 cargo run --release
RAYON_NUM_THREADS=8 cargo run --release
RAYON_NUM_THREADS=$(nproc) cargo run --release
```

Expected: nearly linear scaling up to the P-core count, then plateau (E-cores add less, especially under heat).

5. **Compose with the SIMD kernel.** Wire `micro_kernel_8x4` (or `_neon`) into the per-strip inner loop. Bench the result. Expected on an 8-P-core M-class chip for a 1024-cube:

```
matmul_threaded_neon: ~3.5 ms  600+ GFLOPs
```

On an 8-core x86 desktop with AVX2: ~400–500 GFLOPs aggregate.

6. **Detect false sharing.** Run a quick experiment:

```rust
let counters: Vec<u64> = vec![0; n_threads];
// ... each thread increments counters[thread_id] millions of times ...

// Now pad:
#[repr(align(64))]
struct PaddedU64(u64);
let counters: Vec<PaddedU64> = (0..n_threads).map(|_| PaddedU64(0)).collect();
// ... same work ...
```

The padded version should be much faster when `n_threads > 1`. This is false sharing in action.

7. **(Optional) NUMA experiment.** On a multi-socket Linux box: `numactl --interleave=all cargo run --release` vs default. For matmul, interleaving usually helps a little; cross-NUMA without interleaving usually hurts noticeably.

## Further reading

- *The Art of Multiprocessor Programming* (Herlihy & Shavit) — thorough on the theory of concurrent algorithms, work-stealing especially.
- Rayon's documentation and source — the parallel iterators API is small and worth reading end to end. The implementation of work-stealing in `rayon-core` is dense but instructive.
- "False Sharing" — multiple blog posts and a classic Intel Application Note. Worth seeing the patterns.
- `perf c2c` documentation — Linux's specific tool for detecting cache-coherence pressure between cores.
- Apple's Dispatch (libdispatch) docs — Apple's own work-stealing scheduler, for context on the platform-native option.

Next lesson: `perf`. Now that we have a complex multi-threaded SIMD program, we need professional-grade tooling to see what it's doing. We start with Linux's `perf` — the standard.
