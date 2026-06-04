# From Silicon to Softmax

A structured learning path from bare-metal systems programming to distributed GPU clusters — the full stack of AI infrastructure, built from the silicon up.

Eight modules, organized as a T-shaped portfolio: deep technical stem (silicon → kernels → attention internals) and a wider crossbar covering the orchestration and platform layers that turn one good training run into a reliable model factory.

## Modules

| # | Module | Focus |
| - | ------ | ----- |
| 1 | [The Low-Level Foundation](lessons/intro-to-ml-systems.md) | Systems programming in Rust, CPU architecture, SIMD, memory hierarchy, Linux performance profiling. |
| 2 | GPU & Parallelism | CUDA programming, Triton kernels, parallel algorithms, kernel fusion, FlashAttention. |
| 3 | Distributed Systems | RDMA, InfiniBand, NCCL, distributed training with DDP/FSDP, the 3D parallelism grid. |
| 4 | ML Internals & Optimization | Quantization, inference optimization, and the Rust GPU frontier. |
| 5 | [Cluster Orchestration](lessons/cluster-orchestration-overview.md) | Kubernetes for ML, Slurm, Volcano/Kueue, MPI Operator, KubeRay, topology-aware scheduling, multi-tenancy. |
| 6 | [ML Platform Engineering](lessons/ml-platform-engineering-overview.md) | Experiment tracking, model registry, training observability, workflow orchestration, CI/CD for models, cost attribution. |
| 7 | [Inference from Scratch](lessons/inference-from-scratch-overview.md) | Attention variants (GQA, MLA), positional encodings, KV cache, Mixture of Experts, multi-token prediction, serving internals. |
| 8 | [Agents from Scratch](lessons/agents-from-scratch-overview.md) | The agent loop, tool use, memory, RAG, planning, context engineering, multi-agent systems, distributed agent infrastructure. |

Modules 1–4 form the deep systems stem; 5–6 are the orchestration crossbar; 7–8 are tutorial series that apply the foundation to modern LLM workloads.

## Layout

```
lessons/   Module overviews and (in time) per-lesson writeups
legacy/    Older planning docs kept for reference
PROGRESS.md  Personal day-by-day learning tracker
```

## Status

Curriculum scaffolding is complete. Lesson content is being written and worked through.

## License

MIT — see [LICENSE](LICENSE).
