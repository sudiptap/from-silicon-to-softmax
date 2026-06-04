---
title: "Lesson 10 — macOS Profiling (Instruments, samply, hyperfine)"
date: "2026-06-03"
module: "bare-metal"
order: 10
tags: ["profiling", "macos", "instruments", "samply", "hyperfine", "dtrace", "apple-silicon"]
author: "Sudipta Pathak"
prerequisites: ["09-linux-perf"]
---

# Lesson 10 — macOS Profiling

## Why this matters

macOS — and especially Apple Silicon — is increasingly the development environment for on-device AI work. You write the kernel, you tune it, you ship it to phones built on the same instruction set and a related memory architecture. Skipping the Mac-side profiling story means working blind on your own laptop.

The Mac tooling is *different* from `perf`, not worse. Instruments (Apple's flagship profiler) reads from the same kind of hardware counters and shows them through a GUI. `samply` is a `perf record`-style command-line sampler that opens in a browser. `dtrace` is the lower-level tracing primitive Apple inherited from Solaris and still ships. `hyperfine` is a benchmarking microharness. Together they cover the same ground as `perf` plus a couple of Apple-specific tools (Xcode Metal debugger, the Energy / Power profile in Instruments).

This lesson is short on theory — the mental model from Lesson 9 transfers directly — and long on tool-specific recipes.

## Concept

### What Instruments is

Instruments is Apple's GUI profiler. It's bundled with Xcode (free from the App Store) and supports profiling templates:

- **Time Profiler** — function-level sampling, equivalent to `perf record -g`.
- **CPU Counters** — read hardware counters (cycles, IPC, L1/L2 cache misses on supported Macs). The macOS analogue of `perf stat`.
- **Allocations** — heap allocations over time, with stack traces.
- **System Trace** — kernel-level events, syscalls, scheduler decisions.
- **Metal System Trace** — GPU command buffer timing (we'll use this in Modules 2 and 4).
- **Energy** — power-consumption profile; matters for on-device work.

To use Instruments on Apple Silicon profitably, your binary needs debug symbols (Cargo's `debug = true` in the release profile, which we've already set) and ideally no thinning (Rust release binaries are fine).

### `samply` — the better command-line sampler

Instruments is good but heavy. `samply` is a Rust-based sampler that runs the same kind of profiling, produces a `.json` profile, and opens it in the Firefox Profiler web UI for analysis. It works on macOS and Linux. Many engineers prefer it for quick iteration.

```bash
cargo install samply
samply record ./target/release/matmul-from-scratch
# Opens profiler.firefox.com with your data loaded
```

The profile view shows: a flame graph by default, plus the inverted call tree and timeline. It supports source-line annotation if your binary has debug info.

### `hyperfine` — proper benchmark harness

`hyperfine` runs your program multiple times, computes statistics on the wall clock, and gives you mean / stddev / min / max / outliers. It handles warm-up, runs-to-convergence, and comparison between two binaries cleanly.

```bash
brew install hyperfine
hyperfine --warmup 3 \
    --command-name naive   './target/release/matmul-naive' \
    --command-name blocked './target/release/matmul-blocked'
```

Output is a clean table. Useful for sanity-checking the numbers your own `Instant::now()` loop produces, and indispensable for comparing two binaries.

### `dtrace` and friends

`dtrace` is the kernel-side tracing primitive. macOS ships with a useful set of probes — process events, syscalls, IO. Not commonly needed for matmul tuning, but invaluable for "where is my program spending time outside its hot loop" questions:

```bash
sudo dtrace -n 'syscall:::entry { @[execname] = count(); }'
```

This counts syscalls system-wide. There are many recipes; for our purposes, knowing it exists is enough.

### Hardware counters on Apple Silicon

Apple Silicon does expose hardware counters, but only through Instruments (not via a public command-line API the way `perf` is on Linux). The CPU Counters template in Instruments shows:

- Cycles, instructions, IPC.
- L1d / L2 / SLC misses.
- Branch mispredicts.
- Various Apple-specific events (NEON throughput, FMA throughput, FP ops/cycle).

For maximum visibility, you may need to enable counter access in a dev-build of macOS or run with elevated privileges. Most counters are available to user code under reasonable conditions.

## Code walkthrough

This lesson is tool recipes, not code. Three quick recipes you'll use repeatedly.

### Recipe 1: "Where does my matmul actually spend cycles?"

```bash
cargo build --release
samply record ./target/release/matmul-from-scratch
```

Open the profile, look at the flame graph. The widest bar in the matmul portion should be `micro_kernel_8x4_neon` or whichever inner function you have. If it's something else (Vec allocation, memcpy, packing functions taking too long), investigate.

### Recipe 2: "Am I memory-bound or compute-bound?"

Open **Xcode** → File → Open → select your binary → Product → Profile. This loads Instruments. Choose **CPU Counters** template. Set the events:

- Cycles
- Instructions
- L1d cache misses
- L2 cache misses
- (Or click "All Counters" to enable everything; expensive but thorough.)

Run. Instruments computes IPC and miss rates. Same interpretation as `perf stat`:

- IPC < 1: memory or dependency bound.
- IPC > 3 and miss rate < 1%: compute bound; you're near peak.
- IPC ~ 2 with high miss rate: memory bound, room to improve via blocking.

### Recipe 3: "Compare A vs B robustly"

```bash
hyperfine --warmup 5 --runs 20 \
    --command-name 'naive'   './target/release/matmul-naive' \
    --command-name 'neon'    './target/release/matmul-neon' \
    --command-name 'accel'   './target/release/matmul-accel'
```

`hyperfine` outputs:

```
Benchmark 1: naive
  Time (mean ± σ):     85.2 ms ±  1.1 ms
  Range (min … max):   83.5 ms … 88.1 ms

Benchmark 2: neon
  Time (mean ± σ):      2.9 ms ±  0.05 ms
  Range (min … max):    2.85 ms … 3.10 ms

Benchmark 3: accel
  Time (mean ± σ):      1.4 ms ±  0.03 ms
  Range (min … max):    1.36 ms … 1.48 ms

Summary
  'accel' ran 2.07× faster than 'neon'
          ran 60.8× faster than 'naive'
```

This is the cleanest possible way to present "we made it 60× faster."

### Recipe 4: "Is the OS putting me on E-cores?"

Apple Silicon has performance (P) and efficiency (E) cores. The scheduler will sometimes move workloads to E-cores under thermal pressure or QoS hints. For reproducible benchmarks:

```bash
# Request user-interactive QoS, which keeps you on P-cores
taskpolicy -t 0 ./target/release/matmul-from-scratch    # -t 0 = USER_INITIATED
```

Or, in code, you can use `pthread_set_qos_class_self_np` from `libc` to bump your thread's QoS. The cleanest cross-platform pattern is to set the *priority* of the main thread high before starting compute.

To verify which cores are being used:

```bash
# In a separate terminal while benchmark runs
sudo powermetrics --samplers cpu_power -i 100 -n 5
```

Shows per-core utilization in real time. P-cores will be at 80%+ while running matmul; E-cores should be quiet. If you see E-core utilization climbing, your QoS or thermal state is wrong.

## Mental model & pitfalls

Single sentence: **Same mental model as Linux perf; different tool names, GUI in places, but the questions you ask the CPU haven't changed.**

Mac-specific pitfalls:

- **Energy modes affecting measurements.** Macs throttle aggressively under "Low Power Mode" or unplugged on battery. Always benchmark plugged in, cooled, and with energy-saver disabled.
- **Thermal throttling.** Sustained matmul on a fanless MacBook Air will hit thermal limits within 30–60 seconds. P-core frequency drops, benchmarks shift. Either limit run duration or use a Mac with active cooling.
- **QoS bumps.** As above. Default QoS for a CLI binary is reasonable but not guaranteed to stay on P-cores.
- **E-core measurements.** If your workload runs on E-cores, IPC and memory characteristics differ from P-cores. They're not "small P-cores" — they're a different microarchitecture (in-order in places). Bench what you intend to run on.
- **Counter access restrictions.** Some counters require Instruments + Xcode to expose. A pure command-line workflow on macOS has weaker observability than on Linux. Consider running serious perf work on a Linux box and using your Mac for development.

## Hands-on (at home)

1. **Install the tools.**

```bash
xcode-select --install   # Instruments (if Xcode not present)
brew install hyperfine
cargo install samply
```

2. **`samply record`** the threaded NEON matmul. Look at the flame graph. Confirm the inner kernel is the top function. Identify any unexpected callers.

3. **`hyperfine`** comparing your fastest hand-written variant against `matmul_accelerate` from Lesson 7. Note the gap — Accelerate uses AMX, you don't.

4. **Open Instruments → CPU Counters** on the threaded matmul. Record IPC and L1d miss rate. Compare to your Lesson 9 Linux numbers (if you have them). Generally Apple Silicon will report higher IPC and lower miss rate than x86 of similar generation — partly the wider register file, partly the bigger L1d.

5. **Run `sudo powermetrics --samplers cpu_power`** during a multi-second matmul loop. Watch the per-core CPU utilization and frequency. Verify the scheduler is keeping work on P-cores. If not, experiment with `taskpolicy -t 0` and re-check.

6. **(Optional) Profile a memory-bound version vs a compute-bound version.** Run the *unblocked* `matmul_ijk` and the *blocked SIMD* `matmul_neon` under CPU Counters. Compare IPC and L1d-miss directly. The shift should be dramatic and educational — the gap between "memory-bound code" and "compute-bound code" is exactly what the metrics show.

7. **(Optional) Profile energy.** Open Instruments → Energy template. Run your matmul for 30 seconds. Note the joules per second (watts). Apple Silicon's matmul is unusually efficient (~10–20 watts at peak); old x86 servers can be 100+. This matters for on-device work and gets explored further in Module 6.

## Further reading

- *Instruments User Guide* (developer.apple.com/documentation/instruments) — Apple's docs.
- `samply` README — extensive, with examples.
- *Mac OS X Internals* (Singh) — old but still the best book on the kernel layer; relevant for `dtrace` and process-side internals.
- "Apple Silicon CPU Optimization Guide" (developer.apple.com) — the closest thing to an Intel Optimization Manual for Apple chips.
- Brendan Gregg's *BPF Performance Tools* — covers macOS `dtrace` recipes in a chapter, parallel to the Linux eBPF coverage.

Next lesson: memory profiling and NUMA. We zoom in on the allocator, page faults, and the multi-socket story we glanced at last lesson.
