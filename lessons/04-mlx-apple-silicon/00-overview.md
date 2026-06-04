---
title: "Module 4 — MLX & Apple Silicon Internals"
date: "2026-06-04"
module: "mlx-apple-silicon"
order: 0
tags: ["apple-silicon", "mlx", "metal", "mps", "unified-memory", "ane", "on-device", "overview"]
author: "Sudipta Pathak"
prerequisites: ["bare-metal", "gpu-computing", "ml-internals"]
---

# MLX & Apple Silicon Internals

## Why this module exists

Most ML systems courses treat "GPU" as synonymous with "NVIDIA GPU with discrete VRAM connected via PCIe." That model is wrong for Apple Silicon, wrong for most mobile SoCs, and increasingly wrong for the segment of AI that runs closest to users.

Apple Silicon doesn't have a separate GPU memory pool. Its GPU isn't programmed with CUDA. It has a Neural Engine that PyTorch can't directly use. Its memory hierarchy looks more like a console than a workstation. The matrix accelerators are an entirely separate execution surface — not in the GPU, not in the CPU — that most ML libraries can't reach. None of this makes Apple Silicon worse; it makes it different, with a different set of optimal patterns and a different framework ecosystem.

This module covers the SoC architecture, the four programming surfaces (Metal, MPS, MLX, Core ML), and the practical question every Apple-Silicon ML engineer has to answer: *which framework do I reach for, when, and why.* By the end, you should be able to look at a workload and predict which surface will dominate (and which will silently fall back to CPU), write custom Metal kernels integrated with MLX when the framework primitives aren't enough, and benchmark cleanly across the available paths.

## How this fits

Module 4 of the depth track. Modules 1–3 built the foundation — CPU primitives, GPU kernels, model compression. This module takes that foundation and applies it to the specific substrate that's quietly become the dominant on-device AI platform: the M-series Mac (and, by extension, the A-series iPhone/iPad, which shares much of the architecture). The quantized model from Module 3 lands here, on a chip whose memory model removes the host-device transfer cost but introduces bandwidth contention that has its own optimization story.

The output of this module: the ability to deploy a quantized LLM (from Module 3) on an Apple Silicon device at competitive token rates, with a clear understanding of why one framework was the right choice over the alternatives, and with the capacity to write custom kernels for the workloads where the off-the-shelf primitives don't fit.

This module is the foundation for Module 5 (Mobile & Edge Runtimes — the broader edge story including Android NPUs and ExecuTorch), Module 6 (On-Device LLM Inference — the production-grade LLM serving on consumer devices), and Module 7 (Inference from Scratch — where the production-grade pipeline composes all of the above).

## The roadmap

Twelve lessons.

### The hardware

1. **The M-series SoC at a glance** — chip layout, P-cores, E-cores, GPU cluster, ANE, AMX matrix coprocessor, Secure Enclave, ProRes engines. What is actually inside the package and which parts are reachable from ML code.
2. **Unified memory architecture** — why there's no `cudaMemcpy`, what zero-copy actually means in hardware terms, the bandwidth/latency profile, and the cost model (especially: when "free" host-device transfers aren't actually free).
3. **The memory hierarchy on Apple Silicon** — L1/L2 on P-cores vs E-cores vs GPU, the system level cache (SLC), bandwidth contention when CPU and GPU run concurrently, how to reason about real-world performance.

### The programming surfaces

4. **Metal & Metal Performance Shaders** — what Metal is at the API level, what MPS provides as high-level operations, MPSGraph as Apple's autodiff layer, and where each fits in a stack.
5. **PyTorch's MPS backend** — what's implemented, what's not, the operator coverage problem, when it silently falls back to CPU, and the practical implications for someone porting a CUDA-trained model.
6. **MLX intro** — Apple's MLX framework: lazy evaluation, unified-memory-native, NumPy-like API, automatic differentiation, when MLX wins over PyTorch MPS and when it doesn't.
7. **MLX internals** — arrays, the computational graph, the streams model, materialization (`mx.eval`), `mx.compile`, and the async dispatch that makes lazy evaluation pay off.

### Writing fast code

8. **Custom Metal kernels from MLX** — when MLX's built-in primitives aren't enough; writing a custom MSL kernel and calling it from MLX; threadgroup memory and SIMD-group ops in the integrated workflow.
9. **KV cache strategies on unified memory** — why unified memory changes the KV cache design space, the streaming patterns that work on M-series, and the in-place update strategies that avoid the wasteful copies common in CUDA-first runtimes.
10. **Quantization on Apple Silicon** — running the Module 3 recipes for real: MLX's native quantized formats, `mlx-lm`, GGUF via llama.cpp, what each runtime supports, and which delivers the best tokens/sec at INT4.

### The neural accelerator

11. **The Apple Neural Engine** — what the ANE actually is, what it can run (and what it can't), the Core ML bridge, the latency-vs-flexibility tradeoff, and why most LLM inference still runs on the GPU even though the ANE exists.

### Wrap

12. **Picking your tool: MLX vs PyTorch MPS vs Core ML vs llama.cpp** — a decision tree backed by benchmarks on a Llama-class model, when each framework is the right answer, and the handoff to Module 5 (the broader edge story).

---

## What this module deliberately won't cover

- **iOS/macOS app integration.** Wrapping a Core ML model in a SwiftUI app, the App Store deployment story, model signing — important but app-development territory, not ML systems.
- **General Metal graphics programming.** Render pipelines, rasterization, shader artistry — Metal is a graphics API too; we use only the compute subset.
- **MPS-specific media operations.** Image processing, video encoding via MPS — adjacent but not ML.
- **Older Apple GPU generations** (pre-M1). The architecture diverged enough at M1 that pre-M1 Macs run a different story; we focus on M1 onward, with notes where M2/M3/M4 differ materially.
- **Distributed inference across multiple Macs.** A real frontier in 2025–2026 (e.g., `exo` and similar projects); we touch it lightly in the wrap lesson but the full story belongs to Module 8 (Distributed Systems).

## How to work through it

Every lesson is fully readable as prose on a phone. The hands-on sections assume an Apple Silicon Mac — M1 onward — running macOS 14+ (Sonoma) or 15+ (Sequoia). The MLX hands-on requires `pip install mlx mlx-lm`; the llama.cpp hands-on requires building llama.cpp from source (5 minutes); the Core ML / ANE hands-on requires Xcode and the Core ML tools (a larger install, ~10 GB).

If you don't have an Apple Silicon Mac available, the reading still stands — the cross-framework decision-tree and the mental models port to similar discussions on Android NPUs (Module 5) — but the numerical results in the hands-on sections won't be reproducible.

The running project from Modules 1–3 carries forward: the matmul kernels gain MLX variants; the quantized model from Module 3's capstone runs through MLX, PyTorch MPS, and llama.cpp on the same hardware so the tradeoffs are visible. Module 4's capstone (Lesson 12) is the canonical "which-runtime" benchmark table that informs every deployment decision on Apple Silicon for the rest of the curriculum.

A note on tempo: this module sits between the algorithm-heavy Module 3 and the broader-ecosystem Module 5. The hardware lessons (1–3) and the framework lessons (4–7) are dense; the writing-fast-code lessons (8–10) are practical; the ANE lesson (11) is short because the ANE is, frankly, smaller in practical impact than its marketing suggests; the wrap (12) collects everything into the deployment workflow.
