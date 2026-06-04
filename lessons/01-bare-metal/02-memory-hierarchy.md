---
title: "Lesson 2 — The Memory Hierarchy"
date: "2026-06-03"
module: "bare-metal"
order: 2
tags: ["cpu", "cache", "memory", "latency", "prefetch", "rust"]
author: "Sudipta Pathak"
prerequisites: ["01-cpu-mental-model"]
---

# Lesson 2 — The Memory Hierarchy

## Why this matters

Last lesson the CPU was a factory trying to run many operations in parallel. This lesson is about why, most of the time, it isn't running at all — it's waiting for memory. The CPU operates on a clock measured in fractions of nanoseconds. Main memory (DRAM) operates on a clock measured in tens of nanoseconds. That is a factor of ~100. If your code goes to DRAM on every operation, your CPU spends 99% of its time idle.

Caches exist to hide that 100× gap. Their effectiveness — *not* the CPU's instruction throughput — is what determines whether your numeric code runs at 1 GFLOPs or 100. Every fast matmul, every well-tuned convolution, every working FlashAttention implementation is, at its heart, a story about caching the right data at the right time.

You need to know the layers, the sizes, the costs, and the rules of the road. That's this lesson.

## Concept

There is no single "memory." There's a hierarchy of memories, each layer faster and smaller than the one below. Your code reads and writes through this hierarchy whether you ask it to or not.

### The layers

A modern desktop or laptop CPU has roughly:

| Level | Typical size | Latency | Bandwidth (per core) |
| ----- | -----------: | ------: | -------------------: |
| Registers           | ~256 bytes      | <1 ns   | — |
| L1 data cache       | 32–128 KB       | ~1 ns   | hundreds of GB/s     |
| L2 cache            | 256 KB–2 MB     | ~3–5 ns | tens of GB/s         |
| L3 / SLC            | 8–128 MB shared | ~10–25 ns | varies             |
| Main memory (DRAM)  | 16–128 GB       | ~80–150 ns | tens of GB/s system-wide |
| SSD                 | TB+             | ~10–100 µs | GB/s              |

For on-device targets the picture compresses. An iPhone-class M-series chip has tighter L2 and a large unified system-level cache (SLC); a microcontroller may have a few hundred kilobytes of SRAM and no DRAM at all. The hierarchy is universal; the absolute numbers are not.

Internalize one ratio: **L1 hit is ~100× faster than DRAM hit.** Everything else follows.

### Cache lines

The cache doesn't move bytes. It moves **cache lines**, which are 64 bytes on x86 and ARM (128 bytes on Apple Silicon for some accesses). When you read one `int32`, the hardware brings in the surrounding 64 bytes whether you asked for them or not. Subsequent reads to those nearby bytes are free.

This makes **spatial locality** a hardware feature. Code that walks memory in order — `a[0], a[1], a[2], ...` — pays one DRAM trip per 16 ints, because every 16th read pulls a new cache line and the others are L1 hits. Code that walks memory in random order — `a[r0], a[r1], a[r2], ...` — pays a full miss per access. Same number of bytes read; 16× different cost.

### Set associativity (briefly)

Caches aren't fully associative — finding a value would be too slow. Real caches are **N-way set associative**: each address maps to one of M sets, and each set holds N cache lines. When you bring in a line, it evicts whichever line in that set is the least recently used. This means that data spaced exactly some power-of-two apart (a common accident in 2D arrays) will all map to the same set, eviction will be brutal, and the cache will appear to "not work" even though it's large. This is **cache aliasing** and it's the cause of mysterious cliff-edge performance regressions for 1024×1024 matrices vs 1023×1023. You don't need to predict it from first principles; you need to know it exists so you can recognize it when it bites.

### The prefetchers

The CPU has hardware prefetchers that watch your memory access pattern and try to bring data into the cache *before* you ask for it. They're surprisingly capable. They can detect:

- Forward sequential streams (`a[i]` with `i++`).
- Backward sequential streams.
- Strided patterns (`a[i]` with `i += k` for small constant `k`).
- Multiple concurrent streams (so two arrays being walked in parallel both stay in cache).

They can't detect:

- Pointer-chasing through linked structures (the next address depends on data the prefetcher can't see).
- Random or non-stride access patterns.
- Streams that change direction or stride based on data.

This is why a `Vec<f32>` walked in order is many times faster than a linked list, even when both store the same data. The list defeats the prefetcher; the vector loves it.

### Translation Lookaside Buffer (TLB)

Virtual addresses get translated to physical addresses through page tables. The TLB caches those translations. Large working sets (gigabytes) can blow out the TLB and cause an additional miss-on-translation per access. The fix is huge pages (2 MB or 1 GB instead of 4 KB), which reduces TLB pressure by orders of magnitude. You won't need huge pages for most of this module, but you'll meet them again in the GPU chapters and the distributed-training chapters.

## Code walkthrough

The classic experiment that makes the hierarchy visible: stride-N access timing.

```rust
fn stride_sum(xs: &[i64], stride: usize) -> i64 {
    let mut acc: i64 = 0;
    let mut i = 0;
    while i < xs.len() {
        acc = acc.wrapping_add(xs[i]);
        i += stride;
    }
    acc
}
```

Same function, parameterized by stride. We allocate a large array — bigger than L3 so DRAM is in the picture — and run it with strides of 1, 8, 64, 512, 4096, 32768. Then we plot bytes-touched-per-second.

What you'll see, on virtually any CPU:

- **Stride 1**: spatial locality is perfect. Every 8th access (8 bytes × 8 = 64-byte cache line on most CPUs; adjust for the 16-byte i64 stride math) is a miss; the rest are L1 hits. Hardware prefetcher is doing serious work. Bandwidth is at or near the L1/L2 ceiling.
- **Stride 8**: each access lands on a fresh cache line (since 8 × 8 bytes = a full cache line). Every access is a cold miss as far as the prefetcher hasn't run ahead, but the strided prefetcher catches the pattern and keeps up. Slower than stride 1 but still smooth.
- **Stride 512–4096**: each access lands on a different cache line, and the strides are now jumping pages. The prefetcher can still see the pattern but page boundaries make it expensive. TLB misses start mattering.
- **Stride 32768+**: each access is now also a TLB miss. Effective bandwidth collapses. You're measuring DRAM random-access latency, which is about 50–100M accesses/second — roughly 100× slower than the stride-1 case despite reading "the same" data.

This shape — flat for small strides, then a cliff — is in every memory-system textbook for a reason. It's the single best way to internalize "spatial locality is not optional."

### Why row-major matters for matmul

Numerical libraries store matrices in row-major order: row 0 in memory, then row 1, then row 2. The familiar matmul `C[i][j] += A[i][k] * B[k][j]` reads `A` along a row (great — stride 1) and `B` down a column (bad — stride N, one cache line per access for large N).

This is the single biggest reason a naive matmul is slow: the innermost loop's `B` accesses defeat spatial locality. We will fix this several different ways across the next three lessons.

## Mental model & pitfalls

The single sentence: **The CPU is fast. Memory is slow. Caches make memory look fast only when you access it predictably.**

Patterns the cache loves:

- Sequential walks through a contiguous block.
- Strided walks where the stride is small (a few cache lines).
- Repeated reuse of a working set that fits in a particular level.
- Two or three concurrent streams (the prefetcher can track them all).

Patterns the cache hates:

- Pointer-chasing through unrelated heap allocations.
- Random access into something larger than L3.
- Power-of-two strides that hit the same cache set (aliasing).
- Large working sets that thrash a level by exceeding its capacity by a small margin.

The pitfall I see most often: people choose "the elegant data structure" and pay a cache cost they never measure. A hash map for 100 elements is sometimes slower than a sorted `Vec` plus binary search, because the `Vec` is contiguous and prefetcher-friendly. A linked list almost always loses to a contiguous array for ML inner loops. Beauty in the type signature doesn't matter; the hardware never sees it.

The other pitfall: people over-trust the prefetcher. Yes, it's clever. No, it cannot save random access. A "lookup table" of length 100k that you index into pseudo-randomly is a cache thrash machine, even if the lookup itself is one instruction. Either redesign to use sequential access, or accept the cost honestly.

## Hands-on (at home)

Reproduce the stride curve.

```bash
mkdir mem-stride && cd mem-stride
cargo init --bin
```

`src/main.rs`:

```rust
use std::time::Instant;

#[inline(never)]
fn stride_sum(xs: &[i64], stride: usize) -> i64 {
    let mut acc: i64 = 0;
    let mut i = 0;
    while i < xs.len() {
        acc = acc.wrapping_add(xs[i]);
        i += stride;
    }
    acc
}

fn main() {
    // 256MB array — way larger than any L3.
    let n: usize = 32 * 1024 * 1024;
    let xs: Vec<i64> = (0..n as i64).collect();

    let strides = [1usize, 2, 4, 8, 16, 32, 64, 128, 256, 1024, 4096, 16384, 65536];
    for &s in &strides {
        let count = xs.len() / s;
        // Warm cache state for fair comparison
        let _ = stride_sum(&xs, s);

        let t0 = Instant::now();
        let runs = if count > 1_000_000 { 5 } else { 50 };
        let mut sink: i64 = 0;
        for _ in 0..runs {
            sink = sink.wrapping_add(stride_sum(&xs, s));
        }
        let dt = t0.elapsed().as_secs_f64() / runs as f64;

        let bytes_touched = (count * 8) as f64;          // 8 bytes per i64 *accessed*
        let cache_lines = (count as f64).min(bytes_touched / 64.0); // not exact, just illustrative
        let ns_per_access = dt * 1e9 / count as f64;

        println!(
            "stride={:>6}  accesses={:>10}  ns/access={:>6.2}  bytes/s={:>5.2} GB/s  (sink {sink})",
            s, count, ns_per_access, bytes_touched / dt / 1e9
        );
    }
}
```

Run with `cargo run --release`. You should see something like:

```
stride=     1  accesses= 33554432  ns/access=  0.30  bytes/s=26.5 GB/s
stride=     8  accesses=  4194304  ns/access=  1.10  bytes/s= 7.3 GB/s
stride=    64  accesses=   524288  ns/access=  3.20  bytes/s= 2.5 GB/s
stride=   256  accesses=   131072  ns/access=  8.40  bytes/s= 0.95 GB/s
stride=  1024  accesses=    32768  ns/access= 28.00  bytes/s= 0.29 GB/s
stride= 16384  accesses=     2048  ns/access= 88.00  bytes/s= 0.09 GB/s
```

(Numbers will vary — Apple M-series CPUs tend to have lower stride-1 bandwidth-per-core than top-end x86 but flatter degradation; older Intel boxes will have a sharper cliff. The shape is what matters.)

**Interpreting it:**

- Stride 1 should be saturating one core's L1/L2 read bandwidth — many GB/s.
- The dropoff begins around stride 8 because every access is now a fresh cache line.
- Past stride 1024 you're seeing DRAM-dominant latency. `ns/access` should land near your hardware's DRAM random-access cost — usually 30–100 ns.
- Past the TLB-coverage threshold (very large stride) you may see another step-down. On macOS with default 16KB pages this happens later than on Linux with 4KB pages.

**Bonus run.** Add a `total_size` parameter and run stride-1 across arrays sized 16 KB, 256 KB, 4 MB, 64 MB, 256 MB. You'll see plateaus corresponding to L1, L2, L3/SLC, and DRAM. That plot — bandwidth versus working set size — is the canonical "memory mountain" from Bryant & O'Hallaron's textbook.

## Further reading

- *What every programmer should know about memory* (Drepper) — still the single best long-form on this material.
- Anandtech / Chips and Cheese deep dives on M-series and modern x86 microarchitectures — they publish actual measured cache and DRAM latencies that pair well with the mental model here.
- `lstopo` (Linux) / `system_profiler SPHardwareDataType` (macOS) — to see your own chip's cache sizes.
- The `mem-stride` exercise you just ran, but with the addition of a `madvise(HUGE_PAGES)` (Linux) — to see the TLB component isolated.

Next lesson: Rust for systems programming — the language we'll use to write the fast matmul, and why it's the cheapest path to seeing what's actually going on with memory.
