---
title: "Module 2 — GPU & Parallelism"
date: "2026-06-03"
module: "gpu-computing"
order: 0
tags: ["gpu", "cuda", "triton", "metal", "mps", "parallelism", "overview"]
author: "Sudipta Pathak"
prerequisites: ["bare-metal"]
---

# GPU & Parallelism

## Why this module exists

CPUs hit a wall at ~100 GFLOPs/core in FP32. Modern GPUs hit 50–100+ TFLOPs/chip. The factor of 1000× between them is the entire reason ML systems exist as a distinct discipline. Every important model — every LLM, every diffusion model, every modern vision system — is shaped by what fits in GPU memory and how fast the GPU's tensor cores can chew through it.

For on-device AI the same is true at smaller scale. Apple Silicon's GPU is an order of magnitude faster than its CPU for matmul-heavy workloads. The Adreno or Mali GPU in your phone is similarly a multiplier over its ARM cores. Even a Raspberry Pi 5 has a VideoCore GPU that beats its CPU for inference. Knowing how to write GPU code is no longer the data-center engineer's problem; it's the on-device engineer's problem too.

This module rebuilds the matmul story on GPUs. The arc parallels Module 1 — naive kernel → blocked kernel → SIMD-equivalent (tensor cores) — except every multiplier in the chain is larger because the hardware is more aggressive. We cover the two dominant programming surfaces: **CUDA** for NVIDIA (still the workhorse for training) and **Metal** for Apple Silicon (the workhorse for on-device). We also cover **Triton**, the Python DSL that has become the common language for fast GPU kernels in the LLM era.

## How this fits

Module 2 of the depth track. Module 1 gave you the mental model of caches, SIMD, and threading on CPUs. Module 2 transfers that intuition to GPUs — where the cache hierarchy looks different (shared memory ≈ programmer-managed L1), the SIMD width is enormous (32-thread warps, hundreds-wide on Apple Silicon), and threading is the basic unit rather than an optimization (10K+ simultaneous threads per kernel launch).

The output of this module: the ability to write GPU kernels by hand in CUDA / Metal / Triton, predict their performance from a structural read, and identify the bottleneck in someone else's kernel from the access pattern alone. This skill is the foundation for Module 4 (MLX & Apple Silicon Internals) and Module 7 (Inference from Scratch).

## The roadmap

Thirteen lessons.

### The mental model

1. **The GPU mental model** — what's different from a CPU; how parallelism, memory, and control flow look from the inside; the SIMT execution model.
2. **NVIDIA GPU architecture** — SMs, warps, occupancy, the memory hierarchy (HBM, L2, shared memory, registers); what Ampere, Hopper, and Blackwell changed.
3. **Apple Silicon GPU architecture** — how it differs from NVIDIA: unified memory, threadgroup memory vs shared memory naming, SIMD groups, the lack of a tensor core, the AMX wrinkle.

### CUDA from scratch

4. **CUDA basics: your first kernel** — the launch syntax, thread/block/grid, the host/device split, error handling, and a naive matmul kernel.
5. **CUDA memory hierarchy** — global, shared, L1, registers; coalesced access; bank conflicts; the hierarchy of "where is my data" decisions.

### The cross-vendor frontier

6. **Triton: same GPU, 10× less code** — block-level programming, when Triton wins and when it doesn't, the Triton matmul.

### Metal for Apple Silicon

7. **Metal Shading Language** — what Metal is, threadgroup memory, SIMD-group ops, the same matmul kernel in MSL.

### The patterns that matter

8. **Kernel fusion** — why fewer GPU trips means faster code; vertical vs horizontal fusion; the case study of fused attention.
9. **The reduction problem** — adding a million numbers fast on a GPU; the warp-level shuffle reductions every GPU programmer ends up using.

### The big one

10. **FlashAttention demystified** — it's about memory, not math. The online softmax trick, the tiling, the back-of-envelope math for why it works.
11. **FlashAttention on Apple Silicon** — porting the same idea to MLX / MPSGraph / MSL; what changes and what stays the same.

### Measuring and wrap

12. **Profiling GPU kernels** — Nsight on NVIDIA, Xcode Metal debugger on Apple; the metrics that tell you whether you're memory-bound, compute-bound, or occupancy-bound.
13. **Module wrap: when to leave the compiler alone** — when vendor BLAS / cuBLAS / Metal Performance Shaders is the right answer; when hand-writing pays off; how this module's skills compose into Module 4 and Module 7.

---

## What this module deliberately won't cover

- **Multi-GPU.** Module 8 (Distributed Systems) covers NCCL, NVLink, sharding. Module 2 is one-GPU.
- **Production training systems.** PyTorch's autograd, mixed-precision training, gradient scaling — those belong to a different curriculum.
- **AMD / ROCm.** The CUDA chapters apply with minimal changes; we'd be repeating ourselves to write a separate ROCm track. Where the differences matter, we mention them.
- **Older GPU features.** Volta and earlier are skipped; we focus on Ampere+ for NVIDIA and M-series for Apple.

## How to work through it

Same pattern as Module 1. Every lesson is fully readable as prose. Hands-on sections are clearly marked at the end of each lesson. To do the hands-on you need either:

- An NVIDIA GPU (consumer 3080+ or any datacenter card works; cloud rentals on Lambda, RunPod, Modal, and others are inexpensive for an evening).
- An Apple Silicon Mac (M1 onward).
- Both is best, but the prose stands without the hands-on if you're between machines.

The running project from Module 1 carries forward: we add `matmul_cuda`, `matmul_triton`, `matmul_metal` variants alongside the existing `matmul_naive` / `matmul_blocked` / `matmul_neon` and watch the speedup land.
