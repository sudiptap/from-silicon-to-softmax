---
title: "Lesson 9 — Linux `perf` for ML Systems"
date: "2026-06-03"
module: "bare-metal"
order: 9
tags: ["perf", "profiling", "linux", "hardware-counters", "ipc"]
author: "Sudipta Pathak"
prerequisites: ["08-threading-rayon"]
---

# Lesson 9 — Linux `perf` for ML Systems

## Why this matters

You have a fast matmul. You think it's fast. The wall clock says it's fast. But "fast" is a comparison — fast compared to what? Compared to the theoretical peak of your CPU, your matmul might be running at 30% of capacity and you'd never know.

`perf` is the standard Linux tool for reading the CPU's *own* opinion of what your code is doing. The CPU has hardware counters — small numeric registers that the silicon updates as it executes — for cycles, instructions, cache misses, branch mispredictions, FMA throughput, and dozens more. `perf` exposes these counters as a profiler. Five minutes with `perf stat` will tell you what your benchmark plot wouldn't tell you in an hour: where the cycles actually went.

For ML systems work this skill is non-negotiable. Every story about a kernel being slow has a "the L1 miss rate was 40%" or "the branch predictor was wrong on 8% of branches" footnote that pins down the cause. You need to be able to read those footnotes from your own programs.

This lesson is Linux-only. macOS profiling — Instruments, `samply`, `dtrace` — is the next lesson. On Apple Silicon you'll use Instruments; on a Linux GPU server you'll use perf. Both reflect the same model.

## Concept

### What `perf` does

`perf` is a kernel-supported sampler and counter-reader. It can:

- **Count events** over the runtime of a program (`perf stat`). One number per event for the whole run.
- **Sample events** at a configurable rate and attribute samples to the instruction (and source line) that triggered them (`perf record` + `perf report`). This is the "where did my cycles go" answer.
- **Trace events** with timestamps for a deeper drill-down (`perf script`).

It hooks into the kernel's performance subsystem, which talks to the CPU's Performance Monitoring Unit (PMU). The PMU is where hardware counters live. A typical Intel CPU has ~4 programmable counters per core plus 3 fixed-function (cycles, instructions, ref-cycles). AMD's are similar. The number of *event types* available is in the hundreds.

### The vocabulary

Events `perf` cares most about for ML systems:

| Event | What it tells you |
| ----- | ----------------- |
| `cycles` | Cycles spent — the denominator for everything. |
| `instructions` | Retired instructions. `instructions / cycles` is **IPC** (instructions per cycle); peak is 4–8 on modern cores. |
| `cache-misses` | LLC (last-level cache) misses — accesses that went to DRAM. |
| `L1-dcache-load-misses` | L1 data cache load misses — accesses that went to L2 or further. |
| `LLC-load-misses` | Loads that missed all caches and went to DRAM. |
| `branch-misses` | Mispredicted branches. |
| `branches` | Retired branches. `branch-misses / branches` is the misprediction rate. |
| `cpu-cycles:p` (precise) | Higher-precision sampling, useful for `perf record`. |

The two metrics that should be on your bumper sticker for matmul:

- **IPC**: Are you keeping the CPU's execution units busy? Peak is ~4 for serial code, higher with FMAs that count as 2 ops.
- **LLC miss rate** (LLC misses / instructions, roughly): Are you bottlenecked on memory?

Low IPC + high LLC miss rate = memory bound. Fix with blocking, prefetching, layout changes.
Low IPC + low LLC miss rate = bound on something else — dependency chains (Lesson 1), branch mispredictions, instruction decode, register pressure.
High IPC + low LLC miss rate = compute bound. Faster code requires algorithm changes or SIMD/FMA tuning.

### Sampling, briefly

`perf record` samples the CPU at some configurable rate (default 4 kHz). On each sample, it records the current instruction pointer (and optionally the stack). After many samples, the histogram of "where the IP was" approximates "where the time was spent." `perf report` displays this histogram.

If you can attach symbols to addresses (which Rust release builds with `debug = true` allow), you can navigate to source-line granularity. If you can record call stacks (with `-g`), you can see callee-vs-caller attribution.

## Code walkthrough

This lesson is mostly tool usage rather than code, but a small example program illustrates what the counters look like in practice.

### Stat: a one-shot summary

```bash
perf stat -e cycles,instructions,cache-misses,branch-misses,L1-dcache-load-misses,LLC-load-misses \
    ./target/release/matmul-from-scratch
```

Output looks like:

```
 Performance counter stats for './target/release/matmul-from-scratch':

         8,123,456,789      cycles
        21,456,789,012      instructions              #    2.64  insn per cycle
            14,567,890      cache-misses
            56,789,012      branch-misses
         1,234,567,890      L1-dcache-load-misses
            12,345,678      LLC-load-misses

           2.534567890 seconds time elapsed
```

Reading this:

- **IPC = 2.64.** Decent but not great — the FMA-heavy kernel ought to be closer to 3.5+ if it's well-tuned. Suggests room for improvement, possibly in register allocation or dependency chains.
- **Cache misses (LLC-load-misses) = 12M** over 21 billion instructions = ~0.06% miss rate. Excellent — the blocking is working.
- **Branch misses = 57M** out of presumably many billions of branches — well under 1% mispredict rate. Healthy.

What the same `perf stat` looks like on the **naive** matmul, for comparison:

```
            0.31  insn per cycle
       18,234,567      LLC-load-misses     <-- much higher
```

IPC of 0.31 means the CPU is idle 70%+ of the time. LLC-miss is 1.5× higher. That's the "memory-bound" signature.

### Record: where did the cycles go?

```bash
perf record -F 4000 -g ./target/release/matmul-from-scratch
perf report --stdio
```

`-F 4000` is the sampling frequency (4000 Hz default). `-g` records call stacks.

Output is a hierarchical view: function → caller → callsite. For our matmul, you should see ~95%+ of cycles inside `micro_kernel_8x4` or `matmul_blocked` — the inner kernel. If you see significant time elsewhere (e.g., 20% in `Vec::push` or memory allocation), that's a smell to chase.

### Annotated source: line-by-line attribution

```bash
perf annotate -d ./target/release/matmul-from-scratch
```

This shows your source code (or assembly, depending on options) interleaved with the per-line percentage of samples. The single hottest line in a fast matmul should be the FMA instruction inside the inner kernel.

### Top-down analysis (advanced)

For real CPU bottleneck diagnosis, the modern tool is `perf stat --topdown` (Intel) or the equivalent on AMD. This categorizes every cycle into one of four buckets: retiring (productive), bad speculation (mispredict cost), front-end bound (decode/fetch), back-end bound (waiting for data or execution units). The goal: maximize "retiring" — which for matmul might be 60–80% on a well-tuned implementation.

```bash
perf stat --topdown ./your-program
```

You'll see lines like `Frontend Bound: 5%`, `Bad Speculation: 1%`, `Backend Bound: 30%`, `Retiring: 64%`. If backend bound is high, dig into the memory hierarchy. If frontend bound is high, your instruction footprint may be too large for the I-cache (rare for numeric kernels). If bad speculation is high, branchy code is hurting you.

## Mental model & pitfalls

The single sentence: **`perf` tells you what the CPU is actually doing; benchmarks tell you what your code is making it do. You need both.**

Pitfalls:

- **Sampling skid.** On non-precise events, the IP attributed to a sample can be a few instructions off. This makes per-line attribution noisy. Use `:p` or `:pp` suffixes on events (e.g., `cycles:pp`) for precise sampling where supported.
- **Counter multiplexing.** If you ask for more events than the CPU has counters for, `perf` multiplexes (samples each event a fraction of the time). The numbers it reports are then extrapolations. For sensitive metrics, limit to a handful of events per run.
- **Background noise.** Other processes affect counters. For sensitive measurements, use `perf stat -a` (system-wide) carefully, or pin to a core with `taskset` and run with reduced noise.
- **Comparing across CPUs.** Cache-miss counts mean different things on different microarchitectures. The conclusions transfer (high miss = memory-bound), the absolute numbers don't.
- **JITed code (not your problem here, but worth knowing).** `perf` can't symbolize JIT'd code by default; you need a perf map file. Rust release binaries are static, so we're fine.
- **`perf_event_paranoid` on locked-down systems.** Cloud VMs often have `kernel.perf_event_paranoid=2` or 3, which restricts `perf`. Set to 1 (`sudo sysctl -w kernel.perf_event_paranoid=1`) on your own machine. Server admins may have to elevate.

## Hands-on (at home)

Requires Linux. If you're on macOS, skip to Lesson 10 (which covers the Mac side) and come back to this one when you're on a Linux box (cloud GPU instance, dual-boot, container with privileges).

1. **Install `perf`** if you don't have it:

```bash
sudo apt install linux-tools-common linux-tools-$(uname -r)   # Debian/Ubuntu
sudo dnf install perf                                          # Fedora
```

2. **Stat the naive vs blocked vs threaded matmul.** Run each variant under `perf stat` and record IPC, LLC-miss rate, branch-miss rate.

3. **Predict, then measure.**

   - Predict: naive should have low IPC, high LLC-miss. Blocked should improve LLC-miss dramatically. Threaded should keep IPC high but possibly raise LLC-miss as bandwidth contention enters.
   - Run and compare.

4. **`perf record -g` the blocked variant.** Open `perf report`. Confirm 95%+ of samples are in the matmul kernel; chase anything that isn't.

5. **`perf annotate` the inner kernel.** Find the hottest source line. It should be inside the FMA loop, ideally the FMA instruction itself. If it's a load, you're memory-bound and have more to optimize. If it's a store, the store buffer is the bottleneck — consider whether you can restructure to write less.

6. **Top-down analysis.** Run `perf stat --topdown` on the threaded variant. What's the largest bucket?

7. **(Optional) Investigate `perf c2c`.** This is the tool for finding false-sharing across threads. Run it on the threaded matmul and confirm the cache-coherence event rate is low. If you wrote false-sharing-prone counters in Lesson 8's experiment, `perf c2c` will surface them clearly.

8. **(Optional) `perf list`.** This dumps every event your CPU supports. There are hundreds. The interesting ones for ML systems work include `mem_load_uops_retired.l1_hit` (Intel), `armv8_pmuv3_0/l1d_cache_refill/` (some ARM), and many memory-system specific events. Most matmul tuning is done with a small subset; for deep work, the documentation in your CPU's *Optimization Reference Manual* tells you which events matter.

## Further reading

- Brendan Gregg's `perf` page (brendangregg.com/perf.html) — the most useful single resource. Examples, common idioms, pitfalls.
- *Systems Performance* (Brendan Gregg) — the book version, covers perf, eBPF, the whole Linux observability story.
- Intel Optimization Reference Manual — what counters mean and how to interpret them on Intel CPUs.
- AMD Software Optimization Guide — the equivalent for Zen architectures.
- `perfetto` (perfetto.dev) — Google's modern alternative for tracing, more visual; works on Android/Linux/Chrome.

Next lesson: macOS profiling. Same job, different tools. Instruments, `samply`, and `dtrace` — what to use when on Apple Silicon.
