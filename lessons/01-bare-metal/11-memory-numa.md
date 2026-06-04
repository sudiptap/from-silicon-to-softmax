---
title: "Lesson 11 — Memory Profiling and NUMA Awareness"
date: "2026-06-03"
module: "bare-metal"
order: 11
tags: ["memory", "allocator", "numa", "page-faults", "huge-pages", "valgrind", "heaptrack"]
author: "Sudipta Pathak"
prerequisites: ["10-macos-profiling"]
---

# Lesson 11 — Memory Profiling and NUMA Awareness

## Why this matters

We've spent six lessons on the inner kernel. We've barely talked about the *outer* memory layer — the heap allocator, page tables, NUMA boundaries. For most numeric kernels, these are silent. They only show up when they bite, and when they bite they typically take a 2× tax that's invisible in any benchmark you took before the bite started.

Two examples of how they bite. First: a matmul that runs at 50 GFLOPs on small problems suddenly runs at 20 GFLOPs on a problem 20× bigger, despite still fitting in DRAM with room to spare. Diagnosis is often: the working set exceeded the TLB coverage; every access is paying a page-walk. Second: a multi-threaded matmul that scales perfectly to 4 cores stops scaling at 8, hitting 1.2× instead of 2× the throughput. Diagnosis often: NUMA — half the threads are on a different socket from the memory.

Neither of these shows up in `perf stat`'s flagship metrics. Neither will Instruments call out directly. You need to know to look for them.

## Concept

### The heap allocator

When you write `Vec::with_capacity(n)` in Rust, the underlying call goes to the system allocator (glibc's `malloc` on Linux, libmalloc on macOS, jemalloc/mimalloc if you've opted in). The allocator finds a free block of the right size and gives you a pointer. Behind the scenes it manages chunks, slabs, and arenas.

For numeric work, the allocator is usually fine. It becomes a problem when:

- **You allocate in inner loops.** Even allocator-cached allocations are tens of cycles; uncached allocations involve syscalls (page faults). Either case is a disaster inside a hot loop. The rule is: allocate buffers outside the loop, reuse them.
- **Allocation patterns fragment.** Many small allocations of varied sizes, freed in random order, can cause heap fragmentation. For long-running ML processes, this can lead to ever-growing RSS even after frees. Custom allocators (jemalloc, mimalloc) help.
- **Large allocations bypass the cache.** Very large `Vec`s (multi-MB) typically go through `mmap` directly rather than the small-block cache. Allocation is faster but the first touch is more expensive (first-touch fault).

### First-touch and page faults

Here's a subtlety that bites people. When you allocate a 1 GB `Vec<f32>`, the allocator gives you a 1 GB virtual address range. It doesn't actually back any of it with physical memory yet. Each 4 KB page (or 16 KB on macOS) is allocated **on first write** — that's a "minor page fault."

Implication: a benchmark that includes the first iteration of accessing a large array measures the cost of physical allocation, not just the cost of access. Best practice: warm up the array with a memset or a touch loop before timing.

For very large arrays, the cumulative first-touch cost can be hundreds of milliseconds. For NUMA systems, the first-touch policy determines *which NUMA node* gets the physical pages — typically "the node of the thread that first touches the page." This is the foundation of the standard NUMA placement trick.

### Huge pages

Standard pages are 4 KB on Linux, 16 KB on macOS. A 1 GB array is 256K pages (on Linux) — each page needs a TLB entry. The TLB holds maybe 1500 entries. So you can only cover a tiny fraction of a 1 GB array in the TLB at once. The rest is paged in via slow translation walks.

**Huge pages** (2 MB or 1 GB on Linux; 2 MB transparent on some systems) drastically reduce TLB pressure. One 2 MB huge page replaces 512 standard pages with one TLB entry. Coverage jumps from "few MB" to "tens of MB" with the same TLB.

For matmul on large matrices, transparent huge pages (THP) on Linux can give a 10–30% speedup for free if enabled:

```bash
echo always | sudo tee /sys/kernel/mm/transparent_hugepage/enabled
```

For explicitly-controlled huge pages, see `mmap` with `MAP_HUGETLB`, or `madvise(MADV_HUGEPAGE)`. macOS doesn't expose THP at the user level — Apple makes the policy decisions in the kernel.

### NUMA layout

A NUMA system has multiple memory controllers, typically one per CPU socket. Memory attached to socket 0 is "near" for socket-0 cores and "far" for socket-1 cores. Far accesses go over an interconnect (UPI on Intel, Infinity Fabric on AMD) — 2–5× slower than local accesses, with shared bandwidth.

Single-socket workstations, laptops, and Apple Silicon Macs are **not NUMA**. The discussion below is for dual-socket and larger systems — servers, training boxes.

On Linux, `numactl --hardware` shows your NUMA layout:

```
available: 2 nodes (0-1)
node 0 cpus: 0 1 2 3 4 5 6 7
node 0 size: 65536 MB
node 1 cpus: 8 9 10 11 12 13 14 15
node 1 size: 65536 MB
node distances:
node   0   1
  0:  10  21
  1:  21  10
```

The distance numbers are arbitrary units (10 = local, 21 = remote here). Higher = slower.

### NUMA-aware matmul placement

Two strategies:

1. **Interleave**: spread memory across all NUMA nodes round-robin. `numactl --interleave=all ./prog`. Predictable performance regardless of where threads run. Often gives 90% of the optimum with zero code changes.

2. **Bind**: allocate matrix A on node 0, threads working on it run on node 0; same for B/C. Yields peak NUMA-local performance but requires partitioning the algorithm awareness of NUMA topology. `numactl --cpunodebind=0 --membind=0 ./prog` for the simple case.

For most ML work the choice is "interleave or bind." Cross-NUMA blind allocation is the bug to avoid.

### Memory profiling tools

- **`/usr/bin/time -v ./prog`** (Linux): reports max RSS, page faults, context switches. Quick summary.
- **`heaptrack`** (Linux/macOS): records every allocation with stack traces. Excellent for finding "where did all the memory go" in a long-running process.
- **`valgrind --tool=massif`**: profiles heap usage over time, with stack traces. Slower than heaptrack but very thorough.
- **`Instruments → Allocations`** (macOS): GUI heap profiler. Same job as heaptrack with Apple polish.
- **`pmap -x <pid>`** (Linux) / **`vmmap <pid>`** (macOS): shows the process's memory map. Useful for understanding what's mapped where.

## Code walkthrough

Two small examples that show the gotchas.

### Allocation in the inner loop (the bug)

```rust
fn matmul_slow(a: &Mat, b: &Mat, c: &mut MatMut) {
    for i in 0..a.rows {
        // BUG: per-row allocation of a 1MB scratch
        let mut row_scratch: Vec<f32> = vec![0.0; b.cols];
        for kk in 0..a.cols {
            for j in 0..b.cols {
                row_scratch[j] += a.data[i*a.cols + kk] * b.data[kk*b.cols + j];
            }
        }
        for j in 0..b.cols {
            c.data[i*c.cols + j] += row_scratch[j];
        }
    }
}
```

The `vec![0.0; b.cols]` allocates and zero-initializes a buffer per output row. For a 1024-cube, that's 1024 allocations of 4 KB each, plus 1024 zero-fills. Total: tens of milliseconds before any multiply happens. The "right" version hoists the allocation:

```rust
fn matmul_better(a: &Mat, b: &Mat, c: &mut MatMut) {
    let mut row_scratch: Vec<f32> = vec![0.0; b.cols]; // alloc ONCE
    for i in 0..a.rows {
        row_scratch.fill(0.0);                          // reset only
        for kk in 0..a.cols { /* ... */ }
        for j in 0..b.cols {
            c.data[i*c.cols + j] += row_scratch[j];
        }
    }
}
```

One allocation, N resets. The resets touch already-resident pages, so they're cache-fill speed (much faster than fresh allocation).

### Cold-cache first run

```rust
fn main() {
    let mut data = vec![0.0f32; 100_000_000];
    let t0 = Instant::now();
    data.iter_mut().for_each(|x| *x = 1.0);
    println!("first write: {:?}", t0.elapsed());

    let t1 = Instant::now();
    data.iter_mut().for_each(|x| *x = 2.0);
    println!("second write: {:?}", t1.elapsed());
}
```

First write: ~300 ms on a typical Linux box (page-fault on every 4 KB page; ~25K faults).
Second write: ~50 ms (no faults, just memory bandwidth).

The first-time cost is 6× the steady-state cost. If your benchmark only runs once cold, you measure the wrong thing.

### NUMA experiment (on a multi-socket box)

```bash
# Bind: all threads on socket 0, all memory from node 0
numactl --cpunodebind=0 --membind=0 ./target/release/matmul

# Worst case: threads on socket 0, memory on node 1
numactl --cpunodebind=0 --membind=1 ./target/release/matmul

# Interleave: memory spread across both nodes
numactl --interleave=all ./target/release/matmul
```

On a dual-socket workstation for a 4096-cube matmul, expect:

- Local (`bind 0, mem 0`): 100% throughput.
- Cross-NUMA (`bind 0, mem 1`): 40–60% of local. 2× slowdown is typical.
- Interleave: 80–90% of local. Cheap and robust.

## Mental model & pitfalls

Single sentence: **Memory profiling is about the cost of *getting bytes ready*, not the cost of *using them*. Allocator overhead, page faults, TLB misses, and NUMA travel all live in the "getting them ready" bucket — and any of them can be the silent dominant cost.**

Pitfalls:

- **Mistaking a cold-start cost for steady-state.** Always warm up.
- **Allocating inside hot loops.** Allocator calls in inner loops are pure overhead. Hoist.
- **Ignoring TLB for very large working sets.** Multi-GB tensors benefit from huge pages on Linux. macOS handles this implicitly but not perfectly.
- **Default NUMA placement on multi-socket boxes.** Often the worst possible — threads on one socket, memory on another. At minimum, interleave.
- **Trusting `top` / `htop` to show real memory usage.** RSS as displayed is approximate (shared memory accounting, page-table overhead, etc.). For precise numbers, use `pmap -x` or `vmmap`.
- **Allocator choices.** glibc's `malloc` is fine for most things. For ML server processes with heavy concurrent allocation, `jemalloc` or `mimalloc` often beat it by 10–30%. Drop-in replacements; worth A/B testing.

## Hands-on (at home)

1. **Measure cold-vs-warm first-touch cost.** Use the code above. Verify the 5–10× ratio on your system.

2. **Hoist allocations.** Audit your matmul code for any per-iteration allocation. Find the worst one. Move it outside the loop. Measure the wall-clock difference.

3. **Enable transparent huge pages (Linux).**

```bash
cat /sys/kernel/mm/transparent_hugepage/enabled
# If not "always", try:
echo always | sudo tee /sys/kernel/mm/transparent_hugepage/enabled
```

Re-run your large-problem matmul. Look for a 5–25% speedup on problems that previously had high TLB miss counts (visible in `perf stat` as `dTLB-load-misses`).

4. **`/usr/bin/time -v` your matmul.** Note the "Maximum resident set size," "Major (requiring I/O) page faults," and "Minor (reclaiming a frame) page faults." Minor faults should be roughly equal to "matrix bytes / page size" for the first run, lower thereafter.

5. **`heaptrack` or Instruments Allocations.** Run your matmul. Verify the allocation count is small (you should see roughly: one for each matrix buffer + a few for setup). If you see thousands, you have allocation in inner loops to find.

6. **(Optional, requires multi-socket box) NUMA experiment.** Run with `numactl --interleave=all` vs `numactl --cpunodebind=0 --membind=0` vs no `numactl`. Compare GFLOPs.

7. **(Optional) Try jemalloc.** Add to `Cargo.toml`:

```toml
[dependencies]
tikv-jemallocator = "0.5"
```

```rust
#[global_allocator]
static GLOBAL: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;
```

Recompile and re-bench. For matmul specifically the difference is usually small (we don't allocate much in inner loops); for any workload with significant allocation, jemalloc usually wins by 10–30%.

## Further reading

- "What Every Programmer Should Know About Memory" (Drepper) — chapter on NUMA is still excellent.
- Linux kernel docs on transparent huge pages (`Documentation/admin-guide/mm/transhuge.rst`).
- Brendan Gregg's "USE Method" — a structured way to investigate utilization, saturation, and errors at every system level. Applies to memory just as well as CPU.
- `heaptrack` and `mass` (Massif) manuals — both have good examples.
- Apple's memory management documentation — surprisingly readable for a vendor doc.

Next lesson: the module wrap. We put the pieces together, look at where we landed compared to peak, and set up the entry point to the GPU module.
