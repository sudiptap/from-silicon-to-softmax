---
title: "Module 1 — The Low-Level Foundation"
date: "2026-06-03"
module: "bare-metal"
order: 0
tags: ["bare-metal", "cpu", "rust", "simd", "perf", "overview"]
author: "Sudipta Pathak"
prerequisites: []
---

# The Low-Level Foundation

## Why this module exists

Python is a fine cockpit. It is not a fine engine room.

Most ML engineers work in Python and reach for a GPU the moment they want speed. That habit hides the entire CPU side of the computation — and in on-device AI, where there is no NVLink and no datacenter cooling, the CPU side is often the difference between a feature that ships and one that doesn't. Even on a phone, the bottleneck of a quantized LLM is usually not the matmul kernel; it's the memory traffic feeding it. You cannot reason about that without a mental model of how the CPU and its caches actually work.

This module builds that mental model. By the end you should be able to take a simple piece of numeric code (matrix multiply is our running example), look at it, predict roughly how fast it will run, run it, see why your prediction was wrong, and fix it. Twice or three times.

We do this in Rust. Not because Rust is mandatory for ML systems — it isn't — but because Rust forces you to see allocations, lifetimes, and data layout. Python lets you ignore all of those. C lets you ignore them differently (by being wrong about them silently). Rust is the cheapest language to learn the *systems* part in, even if your day job stays in Python.

## How this fits

This is module 1 of the depth track. It deliberately doesn't talk about GPUs, models, or attention — those come later. It earns the right to talk about them by making sure you can already explain why a naive `for i { for j { for k { c[i][j] += a[i][k] * b[k][j] } } }` matmul runs at 0.3 GFLOPs when your hardware will happily do 100.

## The roadmap

Twelve lessons. The arc is: build a mental model, then beat the naive code with that model.

### Mental models

1. **The CPU mental model** — pipelines, ILP, branch prediction, out-of-order execution. The thing the Python `for` loop is hiding.
2. **Memory hierarchy** — L1/L2/L3, cache lines, prefetchers; the latency numbers every engineer should have on a sticker.
3. **Rust for systems programming** — ownership and borrowing through a systems lens; why ML engineers benefit from one project in this language.

### Building a fast matmul

4. **Naive matrix multiply** — the triple-loop version. Measure it. Understand why it's slow before you change it.
5. **Cache-blocked matmul** — tiling, loop reordering, why a 64×64 tile is magic; the first big speedup.
6. **SIMD on x86** — SSE, AVX2, AVX-512 intrinsics; vectorizing the inner kernel.
7. **SIMD on ARM / Apple Silicon** — NEON intrinsics; the M-series story, where it matches x86 and where it differs.
8. **From SIMD to threads** — Rayon, work stealing, NUMA-aware partitioning; when threading helps and when it just heats the chip.

### Measuring like a professional

9. **Linux `perf`** — cache miss counters, IPC, branch mispredictions; reading the truth instead of guessing.
10. **macOS profiling** — Instruments, `samply`, `hyperfine`, `dtrace`; how to do the same work on the platform you actually have.
11. **Memory profiling and NUMA awareness** — heap allocation cost, page faults, `numactl` (Linux) / mach VM (macOS); the parts that surprise everyone.

### Wrap

12. **From 1 GFLOPs to 50+ GFLOPs** — the journey assembled end to end; what we measured, what we learned, where the remaining gap to peak is.

## How to work through it

Each lesson stands alone for reading. Code lives inline so you can follow on a phone. The "Hands-on (at home)" section at the bottom of each lesson tells you what to run when you're at your machine — commands, expected output, what to look at when things differ.

There's a single running project: a matmul that gets faster every lesson. You'll start at hundreds of milliseconds and end at single-digit milliseconds for the same problem, with every step traceable to a specific hardware reason.
