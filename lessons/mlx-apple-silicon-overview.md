---
title: "MLX & Apple Silicon Internals: Module Overview"
date: "2026-06-03"
excerpt: "Apple Silicon as a first-class ML target. Unified memory, the M-series SoC layout, Metal & MPS, MLX framework internals, the Apple Neural Engine, and the practical tradeoffs between MLX, PyTorch MPS, and Core ML for on-device workloads."
module: "mlx-apple-silicon"
order: 0
tags: ["apple-silicon", "mlx", "metal", "mps", "unified-memory", "ane", "on-device", "overview"]
author: "Sudipta Pathak"
prerequisites: ["bare-metal", "gpu-computing", "ml-internals"]
---

# MLX & Apple Silicon Internals

## Why this module exists

Most ML systems courses treat "GPU" as synonymous with "NVIDIA GPU with discrete VRAM connected via PCIe." That model is wrong for Apple Silicon, wrong for most mobile SoCs, and increasingly wrong for the segment of AI that runs closest to users.

Apple Silicon doesn't have a separate GPU memory pool. Its GPU isn't programmed with CUDA. It has a Neural Engine that PyTorch can't directly use. Its memory hierarchy looks more like a console than a workstation. None of this makes it worse — it makes it different, with a different set of optimal patterns.

This module covers the SoC architecture, the programming surfaces (Metal, MPS, MLX, Core ML), and the practical question every Apple Silicon ML engineer has to answer: *which framework do I reach for, when, and why.*

## How this fits

This is module 4 of the depth track, sitting after the foundation (Rust, GPU & parallelism, ML internals & quantization) and before the wider edge runtimes (module 5) and on-device LLM serving (module 6). The output of this module is the ability to make informed framework choices and write fast code on M-series hardware.

## The roadmap

12 tutorials. Topics get linked here as they ship.

### The hardware

1. **The M-series SoC at a glance** — chip layout, P-cores, E-cores, GPU cluster, ANE, Secure Enclave, ProRes engines; what's actually inside the package
2. **Unified memory architecture** — why there's no `cudaMemcpy`, what zero-copy actually means, the bandwidth/latency picture, the cost model
3. **The memory hierarchy on Apple Silicon** — caches across P/E/GPU, system level cache (SLC), bandwidth vs latency, how to reason about it

### The programming surfaces

4. **Metal & Metal Performance Shaders** — what Metal is, what MPS does, where each fits, MPSGraph as the autodiff layer
5. **PyTorch's MPS backend** — what's implemented, what's slow, the operator coverage problem, fallbacks to CPU
6. **MLX intro** — what makes MLX different (lazy evaluation, unified memory native, NumPy-like API), when to use it
7. **MLX internals** — arrays, graphs, the streams model, materialization, compilation

### Writing fast code

8. **Custom Metal kernels** — when MLX/MPS isn't enough; threadgroup memory, SIMD groups, the Metal Shading Language tour
9. **KV cache strategies on unified memory** — why the unified-memory model changes KV cache design; the streaming patterns that work
10. **Quantization on Apple Silicon** — gguf, mlx-format, mlx-lm; precision recipes for M-series; using INT4/INT8 with MLX

### The neural accelerator

11. **The Apple Neural Engine** — what the ANE actually is, what it can run, the Core ML bridge, the latency-vs-flexibility tradeoff
12. **Picking your tool** — MLX vs PyTorch MPS vs Core ML vs llama.cpp; a decision tree backed by real benchmarks

---

## What I'm filling in over time

This is the topic scaffold. Tutorials ship as I write them — each one is a self-contained read with code examples, mental models, and a "hands-on at home" section for what to run on your M-series machine. Open an issue or reach out if there's an ordering or topic you'd like prioritized.
