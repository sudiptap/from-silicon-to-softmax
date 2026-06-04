# Generation Status

Tracks how much of the curriculum has been written. Each module has an **overview** (the lesson scaffold — titles + descriptions) and the **lesson content** itself (the actual readable material with code).

Updated: 2026-06-04

## Summary

| #  | Module | Lessons planned | Overview | Lessons drafted |
| -- | ------ | --------------- | -------- | --------------- |
| 1  | The Low-Level Foundation     | 12 | ✅ | 12 / 12 |
| 2  | GPU & Parallelism            | 13 | ✅ | 13 / 13 |
| 3  | ML Internals & Optimization  | 11 | ✅ | 11 / 11 |
| 4  | MLX & Apple Silicon Internals | 12 | ✅ | 12 / 12 |
| 5  | Mobile & Edge Runtimes        | 10 | ✅ | 10 / 10 |
| 6  | On-Device LLM Inference       | 12 | ✅ | 12 / 12 |
| 7  | Inference from Scratch        | 54 | ✅ | 54 / 54 |
| 8  | Distributed Systems          | 11 | ✅ | 11 / 11 |
| 9  | Cluster Orchestration         | 15 | ✅ | 15 / 15 |
| 10 | ML Platform Engineering       | 14 | ✅ | 0 / 14 |
| 11 | Agents from Scratch           | 24 | ✅ | 0 / 24 |
|    | **Total**                     | **188** |  | **150 / 188** |

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
See [overview](lessons/02-gpu-parallelism/00-overview.md). All 13 lessons drafted:

1. [The GPU Mental Model](lessons/02-gpu-parallelism/01-gpu-mental-model.md)
2. [NVIDIA GPU Architecture](lessons/02-gpu-parallelism/02-nvidia-architecture.md)
3. [Apple Silicon GPU Architecture](lessons/02-gpu-parallelism/03-apple-silicon-gpu-architecture.md)
4. [CUDA Basics: Your First Kernel](lessons/02-gpu-parallelism/04-cuda-basics.md)
5. [CUDA Memory Hierarchy](lessons/02-gpu-parallelism/05-cuda-memory-hierarchy.md)
6. [Triton: Same GPU, 10× Less Code](lessons/02-gpu-parallelism/06-triton.md)
7. [Metal Shading Language](lessons/02-gpu-parallelism/07-metal-shading-language.md)
8. [Kernel Fusion](lessons/02-gpu-parallelism/08-kernel-fusion.md)
9. [The Reduction Problem](lessons/02-gpu-parallelism/09-reduction-problem.md)
10. [FlashAttention Demystified](lessons/02-gpu-parallelism/10-flashattention-demystified.md)
11. [FlashAttention on Apple Silicon](lessons/02-gpu-parallelism/11-flashattention-apple.md)
12. [Profiling GPU Kernels](lessons/02-gpu-parallelism/12-profiling-gpu-kernels.md)
13. [Module Wrap: When to Leave the Compiler Alone](lessons/02-gpu-parallelism/13-module-wrap.md)

### 3. ML Internals & Optimization
See [overview](lessons/03-ml-internals/00-overview.md). All 11 lessons drafted:

1. [The Arithmetic of Neural Nets](lessons/03-ml-internals/01-flop-arithmetic.md)
2. [FP16 vs BF16 vs FP8: The Precision Landscape](lessons/03-ml-internals/02-precision-landscape.md)
3. [INT8 Quantization: Math and Recipes](lessons/03-ml-internals/03-int8-quantization.md)
4. [INT4 Quantization: Per-Channel, Per-Group, and the Formats](lessons/03-ml-internals/04-int4-quantization.md)
5. [Calibration Data and Quantization-Aware Training Basics](lessons/03-ml-internals/05-calibration-qat.md)
6. [AWQ: Activation-Aware Weight Quantization](lessons/03-ml-internals/06-awq.md)
7. [GPTQ: Error-Correcting Quantization](lessons/03-ml-internals/07-gptq.md)
8. [Pruning: Structured vs Unstructured, Lottery Tickets](lessons/03-ml-internals/08-pruning.md)
9. [Distillation: From Logits to Step-by-Step](lessons/03-ml-internals/09-distillation.md)
10. [The Rust GPU Frontier: cubecl, rust-gpu, Burn](lessons/03-ml-internals/10-rust-gpu-frontier.md)
11. [Module Wrap: Picking Your Compression Budget](lessons/03-ml-internals/11-module-wrap.md)

### 4. MLX & Apple Silicon Internals
See [overview](lessons/04-mlx-apple-silicon/00-overview.md). All 12 lessons drafted:

1. [The M-Series SoC at a Glance](lessons/04-mlx-apple-silicon/01-m-series-soc.md)
2. [Unified Memory Architecture](lessons/04-mlx-apple-silicon/02-unified-memory.md)
3. [The Memory Hierarchy on Apple Silicon](lessons/04-mlx-apple-silicon/03-memory-hierarchy.md)
4. [Metal & Metal Performance Shaders](lessons/04-mlx-apple-silicon/04-metal-mps.md)
5. [PyTorch's MPS Backend](lessons/04-mlx-apple-silicon/05-pytorch-mps.md)
6. [MLX Intro](lessons/04-mlx-apple-silicon/06-mlx-intro.md)
7. [MLX Internals](lessons/04-mlx-apple-silicon/07-mlx-internals.md)
8. [Custom Metal Kernels from MLX](lessons/04-mlx-apple-silicon/08-custom-metal-kernels.md)
9. [KV Cache Strategies on Unified Memory](lessons/04-mlx-apple-silicon/09-kv-cache-unified-memory.md)
10. [Quantization on Apple Silicon](lessons/04-mlx-apple-silicon/10-quantization-apple-silicon.md)
11. [The Apple Neural Engine](lessons/04-mlx-apple-silicon/11-apple-neural-engine.md)
12. [Picking Your Tool](lessons/04-mlx-apple-silicon/12-picking-your-tool.md)

### 5. Mobile & Edge Runtimes
See [overview](lessons/05-mobile-edge-runtimes/00-overview.md). All 10 lessons drafted:

1. [The Edge Runtime Map](lessons/05-mobile-edge-runtimes/01-edge-runtime-map.md)
2. [Model Formats](lessons/05-mobile-edge-runtimes/02-model-formats.md)
3. [Core ML Deep Dive](lessons/05-mobile-edge-runtimes/03-coreml-deep-dive.md)
4. [ONNX Runtime on iOS/macOS](lessons/05-mobile-edge-runtimes/04-ort-apple.md)
5. [ExecuTorch](lessons/05-mobile-edge-runtimes/05-executorch.md)
6. [LiteRT (Formerly TFLite)](lessons/05-mobile-edge-runtimes/06-litert.md)
7. [ONNX Runtime Mobile](lessons/05-mobile-edge-runtimes/07-ort-mobile.md)
8. [Qualcomm Hexagon NPU + QNN SDK](lessons/05-mobile-edge-runtimes/08-qualcomm-hexagon-qnn.md)
9. [llama.cpp / ggml](lessons/05-mobile-edge-runtimes/09-llama-cpp-ggml.md)
10. [The Runtime Decision Tree](lessons/05-mobile-edge-runtimes/10-runtime-decision-tree.md)

### 6. On-Device LLM Inference
See [overview](lessons/06-on-device-llm-inference/00-overview.md). All 12 lessons drafted:

1. [The On-Device LLM Stack](lessons/06-on-device-llm-inference/01-on-device-stack.md)
2. [Picking the Model](lessons/06-on-device-llm-inference/02-picking-the-model.md)
3. [Aggressive Quantization Recipes](lessons/06-on-device-llm-inference/03-aggressive-quantization-recipes.md)
4. [Mixed Precision and Per-Channel Quantization](lessons/06-on-device-llm-inference/04-mixed-precision.md)
5. [Pruning and Distillation as a Complement](lessons/06-on-device-llm-inference/05-pruning-distillation-complement.md)
6. [KV Cache for Tiny Memory Budgets](lessons/06-on-device-llm-inference/06-kv-cache-tiny-budget.md)
7. [Memory-Mapped Weights and Weight Streaming](lessons/06-on-device-llm-inference/07-mmap-weight-streaming.md)
8. [Streaming Generation Patterns](lessons/06-on-device-llm-inference/08-streaming-generation.md)
9. [Speculative Decoding On-Device](lessons/06-on-device-llm-inference/09-speculative-decoding.md)
10. [LoRA Hot-Swap](lessons/06-on-device-llm-inference/10-lora-hotswap.md)
11. [On-Device Multimodal](lessons/06-on-device-llm-inference/11-on-device-multimodal.md)
12. [Real-Time Interactive Use Cases (Module Wrap)](lessons/06-on-device-llm-inference/12-real-time-interactive-wrap.md)

### 7. Inference from Scratch
See [overview](lessons/07-inference-from-scratch/00-overview.md). All 54 lessons drafted across 9 parts.

Part 1 — Attention family: lessons [1](lessons/07-inference-from-scratch/01-self-attention-first-principles.md)–[9](lessons/07-inference-from-scratch/09-linear-attention.md).
Part 2 — Positional encodings: lessons [10](lessons/07-inference-from-scratch/10-why-positions-matter.md)–[15](lessons/07-inference-from-scratch/15-rope-scaling.md).
Part 3 — KV cache & memory: lessons [16](lessons/07-inference-from-scratch/16-why-kv-cache-exists.md)–[21](lessons/07-inference-from-scratch/21-streaming-llm-attention-sinks.md).
Part 4 — Sampling & decoding: lessons [22](lessons/07-inference-from-scratch/22-sampling-methods.md)–[28](lessons/07-inference-from-scratch/28-multi-token-prediction.md).
Part 5 — Mixture of Experts: lessons [29](lessons/07-inference-from-scratch/29-moe-from-scratch.md)–[34](lessons/07-inference-from-scratch/34-expert-parallelism.md).
Part 6 — Quantization for inference: lessons [35](lessons/07-inference-from-scratch/35-int8-int4-basics.md)–[40](lessons/07-inference-from-scratch/40-bitnet.md).
Part 7 — Serving systems: lessons [41](lessons/07-inference-from-scratch/41-continuous-batching.md)–[46](lessons/07-inference-from-scratch/46-tensor-parallelism-inference.md).
Part 8 — Long context & test-time compute: lessons [47](lessons/07-inference-from-scratch/47-ring-attention.md)–[50](lessons/07-inference-from-scratch/50-reasoning-models.md).
Part 9 — Block-level + wrap: lessons [51](lessons/07-inference-from-scratch/51-rmsnorm-vs-layernorm.md)–[54](lessons/07-inference-from-scratch/54-tokenization-and-module-wrap.md).

### 8. Distributed Systems
See [overview](lessons/08-distributed-systems/00-overview.md). All 11 lessons drafted:

1. [RDMA, InfiniBand, RoCE](lessons/08-distributed-systems/01-rdma-infiniband-roce.md)
2. [NVLink, NVSwitch, and the Bandwidth Wall](lessons/08-distributed-systems/02-nvlink-nvswitch.md)
3. [NCCL: Collectives and Topology Awareness](lessons/08-distributed-systems/03-nccl.md)
4. [AllReduce Algorithms: Ring, Tree, Double Binary Tree](lessons/08-distributed-systems/04-allreduce-algorithms.md)
5. [Distributed Data Parallel (DDP) from First Principles](lessons/08-distributed-systems/05-ddp.md)
6. [ZeRO and FSDP: Sharding Optimizer State](lessons/08-distributed-systems/06-zero-fsdp.md)
7. [Tensor Parallelism: Megatron-Style Sharding](lessons/08-distributed-systems/07-tensor-parallelism.md)
8. [Pipeline Parallelism: GPipe, PipeDream, 1F1B](lessons/08-distributed-systems/08-pipeline-parallelism.md)
9. [The 3D Parallelism Grid](lessons/08-distributed-systems/09-3d-parallelism.md)
10. [Sequence Parallelism and Ring Attention](lessons/08-distributed-systems/10-sequence-parallelism-ring.md)
11. [Module Wrap: Building a 3D-Parallel Training Run](lessons/08-distributed-systems/11-module-wrap.md)

### 9. Cluster Orchestration
See [overview](lessons/09-cluster-orchestration/00-overview.md). All 15 lessons drafted:

1. [Kubernetes for ML](lessons/09-cluster-orchestration/01-kubernetes-for-ml.md)
2. [NVIDIA GPU Operator + Device Plugin](lessons/09-cluster-orchestration/02-nvidia-gpu-operator.md)
3. [The Cluster-Side View of a Training Job](lessons/09-cluster-orchestration/03-training-job-anatomy.md)
4. [Kueue](lessons/09-cluster-orchestration/04-kueue.md)
5. [Volcano](lessons/09-cluster-orchestration/05-volcano.md)
6. [MPI Operator](lessons/09-cluster-orchestration/06-mpi-operator.md)
7. [Training Operator (Kubeflow)](lessons/09-cluster-orchestration/07-training-operator.md)
8. [Slurm](lessons/09-cluster-orchestration/08-slurm.md)
9. [Slurm vs Kubernetes](lessons/09-cluster-orchestration/09-slurm-vs-kubernetes.md)
10. [KubeRay](lessons/09-cluster-orchestration/10-kuberay.md)
11. [Topology-Aware Scheduling](lessons/09-cluster-orchestration/11-topology-aware-scheduling.md)
12. [Multi-Tenancy](lessons/09-cluster-orchestration/12-multi-tenancy.md)
13. [Spot / Preemptible Scheduling](lessons/09-cluster-orchestration/13-spot-preemptible.md)
14. [Storage for Clusters](lessons/09-cluster-orchestration/14-storage-for-clusters.md)
15. [Cluster Networking + Module Wrap](lessons/09-cluster-orchestration/15-networking-and-wrap.md)

### 10. ML Platform Engineering
See [overview](lessons/ml-platform-engineering-overview.md).

### 11. Agents from Scratch
See [overview](lessons/agents-from-scratch-overview.md).

## Generation cadence

Lessons are generated in batches across sessions, in order. Each generated lesson is committed and pushed so it's readable on phone/browser immediately. Hands-on / runnable sections are fenced clearly so you know what to defer to laptop time.

## Currently being written

**Modules 1–9 complete (150/188 = 80%).** Next session: Module 10 — ML Platform Engineering (14 lessons).
