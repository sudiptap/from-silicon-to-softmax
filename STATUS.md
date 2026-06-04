# Generation Status

Tracks how much of the curriculum has been written. Each module has an **overview** (the lesson scaffold — titles + descriptions) and the **lesson content** itself (the actual readable material with code).

Updated: 2026-06-03

## Summary

| #  | Module | Lessons planned | Overview | Lessons drafted |
| -- | ------ | --------------- | -------- | --------------- |
| 1  | The Low-Level Foundation     | ~12 | ✅ | 0 / 12 |
| 2  | GPU & Parallelism            | ~13 | ⬜ | 0 / 13 |
| 3  | ML Internals & Optimization  | ~11 | ⬜ | 0 / 11 |
| 4  | MLX & Apple Silicon Internals | 12 | ✅ | 0 / 12 |
| 5  | Mobile & Edge Runtimes        | 10 | ✅ | 0 / 10 |
| 6  | On-Device LLM Inference       | 12 | ✅ | 0 / 12 |
| 7  | Inference from Scratch        | 54 | ✅ | 0 / 54 |
| 8  | Distributed Systems          | ~11 | ⬜ | 0 / 11 |
| 9  | Cluster Orchestration         | 15 | ✅ | 0 / 15 |
| 10 | ML Platform Engineering       | 14 | ✅ | 0 / 14 |
| 11 | Agents from Scratch           | 24 | ✅ | 0 / 24 |
|    | **Total**                     | **~188** |  | **0 / 188** |

Legend: ✅ done · 🟡 in progress · ⬜ not started

## Per-module status

Each module's overview is the canonical source for its lesson list. As lessons get drafted, they appear as separate files under `lessons/` and the count above ticks up.

### 1. The Low-Level Foundation
Overview pending — to be written alongside Lesson 1. Planned lesson list:

1. The CPU mental model: pipelines, ILP, branch prediction
2. Memory hierarchy: caches, prefetching, the latency numbers every engineer should know
3. Rust for systems programming: why ML systems care
4. Matrix multiply, the naive version: what makes it slow
5. Cache-blocked matmul: tiling and loop reordering
6. SIMD on x86: SSE/AVX intrinsics and autovectorization
7. SIMD on ARM / Apple Silicon: NEON intrinsics
8. From SIMD to threads: rayon and work stealing
9. Linux `perf` for ML systems
10. macOS profiling: Instruments, samply, hyperfine
11. Memory profiling and NUMA awareness
12. Module wrap: from 1 GFLOPs to 50+ GFLOPs

### 2. GPU & Parallelism
Overview pending. Planned lessons:

1. The GPU mental model
2. NVIDIA GPU architecture: SMs, warps, occupancy
3. Apple Silicon GPU architecture: how it differs
4. CUDA basics: your first kernel
5. CUDA memory hierarchy: global, shared, registers
6. Triton: same GPU, 10x less code
7. Metal Shading Language for Apple Silicon
8. Kernel fusion: fewer GPU trips, faster code
9. The reduction problem: adding a million numbers fast
10. FlashAttention demystified: it's about memory, not math
11. FlashAttention on Apple Silicon (MLX, MPSGraph)
12. Profiling GPU kernels: Nsight, Xcode Metal debugger
13. Module wrap: when to leave the compiler alone

### 3. ML Internals & Optimization
Overview pending. Planned lessons:

1. The arithmetic of neural nets: where the FLOPs actually go
2. FP16 vs BF16 vs FP8: the precision landscape
3. INT8 quantization: math and recipes
4. INT4 quantization: per-channel, per-group, the formats
5. Calibration data and quantization-aware training basics
6. AWQ: activation-aware weight quantization
7. GPTQ: error-correcting quantization
8. Pruning: structured vs unstructured, lottery tickets
9. Distillation: from logits to step-by-step
10. The Rust GPU frontier: cubecl, rust-gpu, where things stand
11. Module wrap: picking your compression budget

### 4. MLX & Apple Silicon Internals
See [overview](lessons/mlx-apple-silicon-overview.md).

### 5. Mobile & Edge Runtimes
See [overview](lessons/mobile-edge-runtimes-overview.md).

### 6. On-Device LLM Inference
See [overview](lessons/on-device-llm-inference-overview.md).

### 7. Inference from Scratch
See [overview](lessons/inference-from-scratch-overview.md). 54 lessons across 9 parts (attention, positional encodings, KV cache, sampling, MoE, quantization, serving, long context, block-level choices).

### 8. Distributed Systems
Overview pending. Planned lessons:

1. The network layer that lets ML scale: RDMA, InfiniBand, RoCE
2. NVLink, NVSwitch, and the bandwidth wall
3. NCCL: collectives and topology awareness
4. AllReduce algorithms: ring, tree, double binary tree
5. Distributed Data Parallel (DDP) from first principles
6. ZeRO and FSDP: sharding optimizer state
7. Tensor parallelism: Megatron-style sharding
8. Pipeline parallelism: GPipe, PipeDream, 1F1B
9. The 3D parallelism grid
10. Sequence parallelism and Ring Attention
11. Module wrap: building a 3D-parallel training run

### 9. Cluster Orchestration
See [overview](lessons/cluster-orchestration-overview.md).

### 10. ML Platform Engineering
See [overview](lessons/ml-platform-engineering-overview.md).

### 11. Agents from Scratch
See [overview](lessons/agents-from-scratch-overview.md).

## Generation cadence

Lessons are generated in batches across sessions, in order. Each generated lesson is committed and pushed so it's readable on phone/browser immediately. Hands-on / runnable sections are fenced clearly so you know what to defer to laptop time.

## Currently being written

**Module 1 — The Low-Level Foundation** (lessons 1–12).
