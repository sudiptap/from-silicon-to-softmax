# From Silicon to Softmax

A structured learning path from bare-metal systems programming to on-device LLM inference and distributed training clusters — the full stack of modern AI infrastructure, built from the silicon up.

Eleven modules, organized as a T-shaped portfolio. The depth track (1–7) walks from CPU caches and SIMD up through Apple Silicon, MLX, mobile runtimes, and on-device LLM serving. The crossbar (8–11) covers the data-center side — distributed training, cluster orchestration, ML platforms, and agents — for when the work scales beyond a single device.

The curriculum has a deliberate on-device lean. Apple Silicon, MLX, Core ML, ExecuTorch, llama.cpp, and Qualcomm Hexagon get first-class treatment alongside CUDA, NCCL, and Kubernetes.

## Modules

### Depth track — on-device foundations

| # | Module | Focus |
| - | ------ | ----- |
| 1 | [The Low-Level Foundation](lessons/intro-to-ml-systems.md) | Systems programming in Rust, CPU architecture, SIMD, memory hierarchy, Linux & macOS performance profiling. |
| 2 | GPU & Parallelism | CUDA programming, Triton kernels, Metal & MPS, parallel algorithms, kernel fusion, FlashAttention. |
| 3 | ML Internals & Optimization | Quantization (FP16/BF16/INT8/INT4), inference optimization, the math behind precision tradeoffs. |
| 4 | [MLX & Apple Silicon Internals](lessons/mlx-apple-silicon-overview.md) | Unified memory, M-series SoC layout, Metal, MLX framework internals, ANE, MLX vs PyTorch MPS. |
| 5 | [Mobile & Edge Runtimes](lessons/mobile-edge-runtimes-overview.md) | Core ML, ONNX Runtime mobile, ExecuTorch, LiteRT (TFLite), llama.cpp/ggml, Qualcomm QNN/Hexagon. |
| 6 | [On-Device LLM Inference](lessons/on-device-llm-inference-overview.md) | Aggressive quantization, KV cache for tiny budgets, speculative decoding, LoRA hot-swap, the SLM landscape, on-device multimodal. |
| 7 | [Inference from Scratch](lessons/inference-from-scratch-overview.md) | Attention variants (GQA, MLA), positional encodings, KV cache, Mixture of Experts, multi-token prediction, serving internals. |

### Crossbar — when work scales beyond a single device

| # | Module | Focus |
| - | ------ | ----- |
| 8 | Distributed Systems | RDMA, InfiniBand, NCCL, distributed training with DDP/FSDP, the 3D parallelism grid. |
| 9 | [Cluster Orchestration](lessons/cluster-orchestration-overview.md) | Kubernetes for ML, Slurm, Volcano/Kueue, MPI Operator, KubeRay, topology-aware scheduling, multi-tenancy. |
| 10 | [ML Platform Engineering](lessons/ml-platform-engineering-overview.md) | Experiment tracking, model registry, training observability, workflow orchestration, CI/CD for models, cost attribution. |
| 11 | [Agents from Scratch](lessons/agents-from-scratch-overview.md) | The agent loop, tool use, memory, RAG, planning, context engineering, multi-agent systems, distributed agent infrastructure. |

## Layout

```
lessons/      Module overviews and per-lesson writeups
legacy/       Older planning docs kept for reference
PROGRESS.md   Personal day-by-day learning tracker
STATUS.md     Per-module / per-lesson generation status
```

## How each lesson is structured

Every lesson is written to be fully readable on a phone or browser tab during the day, with hands-on code clearly fenced off so you know what to come back to at your machine:

```
# Lesson N: Title
## Why this matters
## Concept
## Code walkthrough         ← shown inline, fully readable
## Mental model & pitfalls
## Hands-on (at home)       ← deferred to laptop time
## Further reading
```

## Status

See [STATUS.md](STATUS.md) for module/lesson generation progress.

## License

Content is licensed under [CC-BY-4.0](LICENSE). You're free to share and adapt with attribution.
