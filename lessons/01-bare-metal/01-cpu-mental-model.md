---
title: "Lesson 1 — The CPU Mental Model"
date: "2026-06-03"
module: "bare-metal"
order: 1
tags: ["cpu", "pipeline", "ilp", "branch-prediction", "ooo", "rust"]
author: "Sudipta Pathak"
prerequisites: []
---

# Lesson 1 — The CPU Mental Model

## Why this matters

When you write `c += a * b` in Python, you have no model of what the CPU actually does. The interpreter dispatches to a C function which calls a hardware multiply, but that hardware multiply is not a single act — it is the visible step at the end of a long, mostly-invisible production line. The CPU pretends to execute one instruction at a time. It does not. Modern CPUs do five to ten things at once, speculate about what's coming, reorder instructions to fill bubbles, and predict your branches before you finish writing the `if`.

Once you internalize this, two things become possible. First, you can write code that the CPU is happy to make fast for you. Second, when the CPU isn't happy, you can read the symptoms — a low instructions-per-cycle number, a branch misprediction spike — and know what the code is doing wrong. Both skills travel with you into GPUs, accelerators, and on-device runtimes. The terms change; the questions don't.

## Concept

A modern CPU is a small factory. The "instruction" you wrote is not the unit of work — the unit of work is the **micro-op**, the lowest-level operation the execution units actually run. Your instruction gets decoded into one or more micro-ops, which the factory then schedules across its parallel functional units.

Four mechanisms make this fast. Each is worth knowing by name because every performance story you'll ever read names them.

### 1. Pipelining

Every instruction goes through stages: fetch, decode, execute, memory access, write-back. A non-pipelined CPU would do these sequentially per instruction. A pipelined CPU runs them concurrently across many instructions — like an assembly line where stage 1 is starting instruction N+4 while stage 5 is finishing instruction N. The depth of the pipeline (12–20 stages on modern x86, ~13 on Apple's Firestorm) is the parallelism you get for free.

The cost of this freedom: when something goes wrong — a branch misprediction, a cache miss — the pipeline has to flush in-flight work. The deeper the pipeline, the more cycles the flush costs.

### 2. Instruction-Level Parallelism (ILP) and Superscalar Execution

A modern CPU has many execution units in parallel. An Apple M-class P-core can issue 8 instructions per cycle. An x86 Zen 5 core can issue 8 too. Most code never gets near that ceiling because the next instruction depends on the previous one. The CPU can only run instructions in parallel when their inputs are ready and their outputs don't collide.

This is why "the same code, slightly differently written" can be 2× faster. You haven't reduced the *number* of instructions — you've reduced the *dependencies* between them, and the CPU now schedules them in parallel.

### 3. Out-of-Order (OoO) Execution

If instruction N+1 is waiting for a slow memory load, but instruction N+5 has all its inputs ready, the CPU will run N+5 first. It maintains a window — the "reorder buffer," typically 300–500 micro-ops on modern CPUs — and shuffles ready instructions ahead of stuck ones, then "retires" them in original order to keep the program's logical sequence consistent.

OoO is why a small piece of unrelated work between two slow loads can keep the CPU busy instead of stalling. It's also why "this code looks dumb but runs fast" — the CPU is reordering it into something smarter.

### 4. Branch Prediction

Every `if`, every loop iteration, every function pointer is a branch. The CPU doesn't wait to find out which way the branch goes — that would cost ~15 cycles of empty pipeline. It guesses. Modern branch predictors are extraordinarily good — 95–99% accuracy on most code — using huge tables of branch history. When they're right, the branch was effectively free. When they're wrong, the pipeline flushes the speculatively-executed work and starts over: 15–20 cycles wasted.

Patterns that the predictor learns: "this branch always goes the same way," "this branch alternates," "this branch correlates with the previous branch." Patterns that defeat it: branches whose direction depends on random data the predictor has never seen.

## Code walkthrough

Let's see ILP and branch prediction in action. We'll write two functions that compute the same sum, with one tiny change, and see how the CPU treats them differently.

```rust
// Version A: sum of an array, straightforward
pub fn sum_a(xs: &[i32]) -> i64 {
    let mut total: i64 = 0;
    for &x in xs {
        total += x as i64;
    }
    total
}

// Version B: four independent partial sums, then combined
pub fn sum_b(xs: &[i32]) -> i64 {
    let mut s0: i64 = 0;
    let mut s1: i64 = 0;
    let mut s2: i64 = 0;
    let mut s3: i64 = 0;
    let mut i = 0;
    while i + 4 <= xs.len() {
        s0 += xs[i + 0] as i64;
        s1 += xs[i + 1] as i64;
        s2 += xs[i + 2] as i64;
        s3 += xs[i + 3] as i64;
        i += 4;
    }
    let mut tail: i64 = 0;
    while i < xs.len() {
        tail += xs[i] as i64;
        i += 1;
    }
    s0 + s1 + s2 + s3 + tail
}
```

These compute the same thing. The CPU loves Version B and finds Version A constraining. Here's why.

Version A has one accumulator, `total`. Every iteration's add depends on the previous iteration's result (`total = total + x`). That dependency is a chain — the CPU's reorder buffer can't get ahead. Even though the CPU has multiple integer adders, only one of them is doing work, because the next add is always waiting for the previous one to finish.

Version B has four independent accumulators. There's no dependency between `s0`, `s1`, `s2`, and `s3` — the four adds can issue in parallel on four different execution units in the same cycle. We've cut the dependency chain from "N adds in series" to "N/4 adds in series, four-wide." On a CPU that can issue 4+ integer ops per cycle, this is close to a 4× speedup, with no algorithm change.

This trick — **multiple accumulators** to break the dependency chain — is one of the most reliably useful micro-optimization patterns you'll see, and it shows up everywhere from matmul kernels to reduction sums.

### Branch prediction in numbers

Now consider a branchy sum:

```rust
pub fn sum_positive(xs: &[i32]) -> i64 {
    let mut total: i64 = 0;
    for &x in xs {
        if x > 0 {            // <-- this branch
            total += x as i64;
        }
    }
    total
}
```

Run this on **sorted** data (all negatives first, then all positives): the branch predictor sees a long run of "false," then a long run of "true." Two predictable phases. Mispredictions ≈ 0. Throughput is excellent.

Run the same function on the **same data shuffled randomly**: the branch direction is unpredictable. Mispredictions ≈ 50%. Each misprediction is ~15 cycles of pipeline flush. The function is now several times slower despite doing the same arithmetic.

This is not a hypothetical — it's the famous Stack Overflow result that's been reproduced on every CPU since Sandy Bridge. Branchless variants (using arithmetic instead of `if`) eliminate the predictor entirely and run at consistent speed on any input.

## Mental model & pitfalls

A useful single sentence to keep in your head: **The CPU is trying to run as much of your code in parallel as the data dependencies allow.** Everything in this lesson is about why that ceiling is high or low for a given piece of code.

Common pitfalls:

- **Optimizing instruction count when the real cost is the dependency chain.** Cutting instructions but leaving a serial chain barely helps. Adding instructions that break the chain can be faster.
- **Assuming "fewer branches = faster."** A perfectly-predicted branch is essentially free. A misprediction-prone branch is the expensive thing. The shape of your data matters more than the count.
- **Believing the source order matters.** The compiler reorders, then the CPU reorders again. Source-level "tightness" is rarely visible by the time the silicon sees the code.
- **Ignoring the compiler.** Rust with `--release` will do many of these tricks automatically — unrolling, SIMDifying, hoisting. The mental model lets you predict when the compiler can do it and when you have to nudge it.

## Hands-on (at home)

This lesson's exercise is small, designed to make the ILP point land in your terminal in under five minutes.

**Setup.** You need a recent Rust toolchain. Install with [rustup](https://rustup.rs) if needed.

```bash
mkdir cpu-model && cd cpu-model
cargo init --bin
```

Replace `src/main.rs` with:

```rust
use std::time::Instant;

fn sum_a(xs: &[i64]) -> i64 {
    let mut total: i64 = 0;
    for &x in xs {
        total = total.wrapping_add(x);
    }
    total
}

fn sum_b(xs: &[i64]) -> i64 {
    let mut s0: i64 = 0;
    let mut s1: i64 = 0;
    let mut s2: i64 = 0;
    let mut s3: i64 = 0;
    let mut i = 0;
    while i + 4 <= xs.len() {
        s0 = s0.wrapping_add(xs[i + 0]);
        s1 = s1.wrapping_add(xs[i + 1]);
        s2 = s2.wrapping_add(xs[i + 2]);
        s3 = s3.wrapping_add(xs[i + 3]);
        i += 4;
    }
    let mut tail = 0i64;
    while i < xs.len() {
        tail = tail.wrapping_add(xs[i]);
        i += 1;
    }
    s0.wrapping_add(s1).wrapping_add(s2).wrapping_add(s3).wrapping_add(tail)
}

fn bench<F: Fn(&[i64]) -> i64>(name: &str, f: F, xs: &[i64]) {
    // Warm up so the result is not dominated by first-touch.
    let _ = f(xs);
    let t0 = Instant::now();
    let mut acc: i64 = 0;
    for _ in 0..50 {
        acc = acc.wrapping_add(f(xs));
    }
    let dt = t0.elapsed().as_secs_f64() / 50.0;
    let gbs = (xs.len() as f64 * 8.0) / dt / 1e9;
    println!("{name:>10}: {:>7.2} ms   {gbs:>5.2} GB/s   (sink: {acc})", dt * 1e3);
}

fn main() {
    let n = 64 * 1024 * 1024; // 64M elements => 512MB
    let xs: Vec<i64> = (0..n as i64).collect();
    bench("sum_a", sum_a, &xs);
    bench("sum_b", sum_b, &xs);
}
```

**Run it.** First in debug mode (slow on purpose), then in release:

```bash
cargo run                # slow, just check it compiles
cargo run --release      # the one that matters
```

**What you should see.** On most CPUs (Apple Silicon, Zen, Skylake+), `sum_b` is roughly **2.5× to 4×** faster than `sum_a`. Both functions read the same bytes, do the same number of adds, and produce the same number. The only difference is the dependency chain in the inner loop.

**If `sum_a` is the same speed as `sum_b`:** your compiler has auto-vectorized the simple version and is already breaking the chain for you. Try with `RUSTFLAGS="-C opt-level=2 -C no-vectorize-loops"` (some toolchains) or shrink the integer type to `i64` of struct-with-side-effects to defeat the auto-SIMDifier. The point of the exercise is to see the *mechanism*; on a fully-optimizing compiler the mechanism may already be in play under the hood.

**Optional second run.** Add a `sum_positive` variant that runs over sorted vs shuffled data and watch the branch-predictor effect. With shuffled data you should see roughly a 3–5× slowdown for the same arithmetic.

## Further reading

- *Computer Organization and Design* (Patterson & Hennessy) — chapters on pipelining and ILP. The standard reference, worth owning.
- Agner Fog's microarchitecture documents (agner.org/optimize) — the bible for x86 microarchitecture details across vendors and generations.
- Apple Silicon CPU Optimization Guide (developer.apple.com) — Apple's own document on M-series microarchitecture, less verbose than Agner but precise where it matters.
- *What every programmer should know about memory* (Ulrich Drepper) — older but still the best single read on the CPU/memory side of the story.

Next lesson: the memory hierarchy. Now that you've seen the CPU is trying to do many things at once, we look at why so much of its time is actually spent waiting — and what the cache layers do about it.
