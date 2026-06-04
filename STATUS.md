# Generation Status

Tracks how much of the curriculum has been written. Each module has an **overview** (the lesson scaffold — titles + descriptions) and the **lesson content** itself (the actual readable material with code).

Updated: 2026-06-03

## Summary

| #  | Module | Lessons planned | Overview | Lessons drafted |
| -- | ------ | --------------- | -------- | --------------- |
| 1  | The Low-Level Foundation     | 12 | ✅ | 12 / 12 |
| 2  | GPU & Parallelism            | 13 | ✅ | 4 / 13 |
| 3  | ML Internals & Optimization  | ~11 | ⬜ | 0 / 11 |
| 4  | MLX & Apple Silicon Internals | 12 | ✅ | 0 / 12 |
| 5  | Mobile & Edge Runtimes        | 10 | ✅ | 0 / 10 |
| 6  | On-Device LLM Inference       | 12 | ✅ | 0 / 12 |
| 7  | Inference from Scratch        | 54 | ✅ | 0 / 54 |
| 8  | Distributed Systems          | ~11 | ⬜ | 0 / 11 |
| 9  | Cluster Orchestration         | 15 | ✅ | 0 / 15 |
| 10 | ML Platform Engineering       | 14 | ✅ | 0 / 14 |
| 11 | Agents from Scratch           | 24 | ✅ | 0 / 24 |
|    | **Total**                     | **188** |  | **16 / 188** |

Legend: ✅ done · 🟡 in progress · ⬜ not started

## Per-module status

Each module's overview is the canonical source for its lesson list. As lessons get drafted, they appear as separate files under `lessons/` and the count above ticks up.

### 1. The Low-Level Foundation
See [overview](lessons/01-bare-metal/00-overview.md). All 12 lessons drafted:

1. [The CPU Mental Model](lessons/01-bare-metal/01-cpu-mental-model.md)
2. [The Memory Hierarchy](lessons/01-bare-metal/02-memory-hierarchy.md)
3. [Rust for Systems Programming](lessons/01-bare-metal/03-rust-for-systems.md)
4. [Naive Matrix Multiply](lessons/01-bare-metal/04-naive-matmul.md)
5. [Cache-Blocked Matrix Multiply](lessons/01-bare-metal/05-cache-blocked-matmul.md)
6. [SIMD on x86 (SSE, AVX2, AVX-512)](lessons/01-bare-metal/06-simd-x86.md)
7. [SIMD on ARM and Apple Silicon (NEON, AMX)](lessons/01-bare-metal/07-simd-arm-apple.md)
8. [From SIMD to Threads (Rayon)](lessons/01-bare-metal/08-threading-rayon.md)
9. [Linux `perf` for ML Systems](lessons/01-bare-metal/09-linux-perf.md)
10. [macOS Profiling](lessons/01-bare-metal/10-macos-profiling.md)
11. [Memory Profiling and NUMA Awareness](lessons/01-bare-metal/11-memory-numa.md)
12. [Module Wrap: From 1 GFLOPs to 50+ GFLOPs](lessons/01-bare-metal/12-module-wrap.md)

### 2. GPU & Parallelism
See [overview](lessons/02-gpu-parallelism/00-overview.md). 4 of 13 lessons drafted:

1. [The GPU Mental Model](lessons/02-gpu-parallelism/01-gpu-mental-model.md)
2. [NVIDIA GPU Architecture](lessons/02-gpu-parallelism/02-nvidia-architecture.md)
3. [Apple Silicon GPU Architecture](lessons/02-gpu-parallelism/03-apple-silicon-gpu-architecture.md)
4. [CUDA Basics: Your First Kernel](lessons/02-gpu-parallelism/04-cuda-basics.md)
5. CUDA Memory Hierarchy *(next session)*
6. Triton: same GPU, 10× less code
7. Metal Shading Language for Apple Silicon
8. Kernel fusion
9. The reduction problem
10. FlashAttention demystified
11. FlashAttention on Apple Silicon
12. Profiling GPU kernels
13. Module wrap

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

**Module 1 complete (12/12).** Module 2 partial (4/13). Next session resumes at Module 2, Lesson 5 (CUDA Memory Hierarchy).
