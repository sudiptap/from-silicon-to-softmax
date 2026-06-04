---
title: "From Silicon to Softmax: Course Overview"
date: "2025-09-15"
excerpt: "A structured learning path from bare-metal systems programming to distributed GPU clusters — the full stack of AI infrastructure, from the silicon up."
module: "bare-metal"
order: 0
tags: ["systems-engineering", "curriculum", "overview"]
author: "Sudipta Pathak"
prerequisites: []
---

# From Silicon to Softmax

There's a quiet divide forming in the AI engineering world.

On one side, you have engineers who can call APIs, chain prompts, and build apps on top of models. That's valuable work. But it's also increasingly commoditized — every framework, every wrapper, every "AI startup" is fighting over the same thin layer of the stack.

On the other side, you have the engineers who make the infrastructure actually work. The ones who understand why a training run is leaving 40% of GPU compute on the table. The ones who can look at an NCCL profile and know immediately that the all-reduce is bottlenecked by PCIe bandwidth, not the network. The ones who can write a custom CUDA kernel that cuts inference latency by 3x because they understand memory access patterns at the hardware level.

This second group is small, the skillset is rare and hard-won, and they are increasingly the bottleneck in scaling AI infrastructure.

## The Core Thesis

To work at the level where you're making a $100 billion data center actually perform, you need to move away from "apps" and toward **systems, hardware, and math**. You're becoming a digital mechanic — someone who understands every layer from the silicon up.

This curriculum is organized in four phases, each building on the last. You cannot skip phases. Phase 2 is meaningless without Phase 1. Phase 4 is impossible without Phase 2.

---

## Phase 1: The Low-Level Foundation

**You cannot optimize what you don't understand.**

Most AI engineers live in Python-land, where memory is managed for you, allocations are invisible, and the CPU is an abstraction. To do systems work, you need to break through that abstraction layer and understand what's actually happening on the hardware.

- **Rust for systems programming** — Manual memory management with safety guarantees. Ownership, lifetimes, and zero-cost abstractions.
- **CPU architecture** — How L1/L2/L3 caches work. Why accessing memory sequentially is 10-100x faster than random access. SIMD for vectorized computation.
- **Linux internals** — How the kernel schedules threads, manages memory pages, and handles I/O. `perf`, `numactl`, and why NUMA topology matters.

### Lessons

1. **Building a Fast Matrix Multiply in Rust: From Naive to SIMD** — A progression from a triple-loop matmul to a SIMD-optimized version with 50-100x speedup.
2. **What `perf` Revealed About My "Optimized" Code** — Profiling with Linux `perf`. Real cache miss rates, IPC numbers, and the story of how I thought my code was fast until the hardware told me otherwise.

---

## Phase 2: GPU & Parallelism

**Modern AI doesn't run on CPUs — it runs on thousands of parallel GPU cores.**

The mental model shift from CPU to GPU is significant. CPUs are optimized for single-thread latency. GPUs are optimized for throughput — thousands of simple cores, massive memory bandwidth, and a programming model where you think in terms of 10,000+ concurrent threads.

- **CUDA programming** — Write kernels that run directly on NVIDIA GPUs. Master the memory hierarchy: registers, shared memory, global memory. Learn to avoid bank conflicts and ensure coalesced memory access.
- **Triton** — OpenAI's language for writing GPU kernels in Python. Block-level thinking instead of thread-level. Gets you 90-95% of hand-tuned CUDA with 10x less code.
- **Kernel fusion & FlashAttention** — Combining multiple operations into single GPU kernels. The single most important optimization in modern LLMs.

### Lessons

3. **The GPU Mental Model: How Parallel Hardware Actually Thinks** — Warps, occupancy, memory coalescing, and why GPU programming is nothing like CPU programming.
4. **Writing CUDA Kernels That Don't Suck** — ReLU, Softmax, and matrix multiply in CUDA. From naive to shared memory tiling.
5. **From CUDA to Triton: Same GPU, 10x Less Code** — Reimplementing the same kernels in Triton. Side-by-side code and performance comparison.
6. **Kernel Fusion: Why Fewer GPU Trips Means Faster Code** — Fusing matmul + activation into a single kernel.
7. **FlashAttention Demystified: It's About Memory, Not Math** — Implementing FlashAttention from scratch in Triton. Online softmax, tiling, and IO-awareness.

---

## Phase 3: Distributed Systems

**Meta isn't running one GPU. They're running clusters of 100,000+.**

Once you can write efficient code for a single GPU, the next challenge is making thousands of GPUs work together. This is where networking, collective communication, and distributed training frameworks become critical.

- **GPU networking** — RDMA, InfiniBand, GPUDirect — protocols that let machines read each other's GPU memory without involving the CPU. NCCL primitives: AllReduce, AllGather, ReduceScatter.
- **Distributed training** — Data Parallelism, Tensor Parallelism, Pipeline Parallelism. ZeRO optimization stages. The 3D parallelism grid.
- **Orchestration** — Kubernetes for GPU workloads. Topology-aware placement and fault tolerance at scale.

### Lessons

8. **RDMA, InfiniBand, and Why TCP Can't Scale AI Training** — The networking stack behind 100K GPU clusters. Real NCCL benchmark numbers.
9. **Benchmarking GPU Communication: AllReduce, NVLink, and the Bandwidth Wall** — Running NCCL tests, interpreting bus bandwidth, and diagnosing bottlenecks.
10. **Distributed Training from First Principles: DDP to FSDP** — Training the same model with DDP and FSDP. Memory, throughput, and communication overhead.
11. **The 3D Parallelism Grid: Tensor, Pipeline, and Data** — How Megatron-LM and DeepSpeed combine parallelism strategies at scale.

---

## Phase 4: ML Internals & Optimization

**You must understand the mechanics of the models you're optimizing.**

This phase isn't about inventing new architectures. It's about understanding the existing ones deeply enough to optimize them at the infrastructure level.

- **Quantization** — Converting 32-bit weights to 8-bit or 4-bit with minimal quality loss. GPTQ, AWQ, GGUF K-quants.
- **Inference optimization** — Continuous batching, speculative decoding, PagedAttention.
- **The Rust GPU frontier** — `wgpu`, `burn`, and `candle`. Where Rust fits in the future of AI infrastructure.

### Lessons

12. **The Math Behind Quantization: FP32 to INT4** — How quantization works at the bit level. Group quantization and why K-quants allocate precision non-uniformly.
13. **Quantizing Llama-3: The Precision-Performance Tradeoff** — Full walkthrough with llama.cpp. Perplexity vs. speed vs. VRAM at every quant level.
14. **Building GPU Compute in Rust: The Frontier** — Exploring the Rust GPU ecosystem and where it fits in AI infrastructure.

---

## The Meta-Skill: Reading Code

The repos that taught me the most:

| Repo | What It Teaches |
|------|----------------|
| [karpathy/llm.c](https://github.com/karpathy/llm.c) | GPT-2 training in C/CUDA. The single best CUDA learning resource. |
| [NVIDIA/cutlass](https://github.com/NVIDIA/cutlass) | How NVIDIA thinks about matrix multiplication at the hardware level. |
| [triton-lang/triton](https://github.com/triton-lang/triton) | The compiler that makes GPU programming accessible. |
| [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention) | The most impactful optimization in modern LLMs. |
| [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) | Quantization, inference optimization, runs on everything. |
| [NVIDIA/Megatron-LM](https://github.com/NVIDIA/Megatron-LM) | How to split a model across 1,000 machines. |
| [vllm-project/vllm](https://github.com/vllm-project/vllm) | Production inference: PagedAttention, continuous batching. |

---

## Who This Is For

This curriculum is for engineers who are already comfortable building software and want to go deeper:

- Proficiency in at least one systems language (C++, Rust, or C)
- Basic understanding of linear algebra (matrix multiplication, dot products)
- Familiarity with PyTorch or similar frameworks
- Access to a GPU (cloud instances like RunPod work fine)

## What's Next

Each lesson comes with full code, benchmarks, and the mistakes I made along the way. Start with Phase 1. Write the code. Run the benchmarks. Read the repos. The rest follows.
