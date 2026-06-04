---
title: "Module 8 — Distributed Systems"
date: "2026-06-04"
module: "distributed-systems"
order: 0
tags: ["distributed", "nccl", "rdma", "infiniband", "tensor-parallel", "pipeline-parallel", "fsdp", "overview"]
author: "Sudipta Pathak"
prerequisites: ["bare-metal", "gpu-computing", "inference-from-scratch"]
---

# Distributed Systems

## Why this module exists

A single GPU caps at ~80 GB of memory and ~3 TB/s of bandwidth. The frontier models — Llama 3.1 405B, DeepSeek-V3 671B, GPT-4o-class — have hundreds of billions to a trillion parameters. The math is unambiguous: they don't fit on one GPU; they don't even fit on one server. Frontier-scale training and inference happens across hundreds or thousands of GPUs connected by specialized networking.

The networking and the parallelism strategies that make this work are a discipline of their own. Module 7 used distributed primitives (TP, EP, Ring Attention) without deriving them; this module derives them. By the end you should understand what `all_reduce` actually does on InfiniBand, why DDP works, why ZeRO replaced naive data parallelism for training, and how the 3D parallelism grid composes tensor + pipeline + data parallelism for trillion-parameter training.

## How this fits

Module 8 of the depth track. Modules 1-7 covered the single-system stack — kernels, GPU architecture, compression, runtimes, inference engines. Module 8 is the multi-system layer: the network primitives, the collective communication, the parallelism strategies. It's the substrate for Module 9 (Cluster Orchestration) and Module 10 (ML Platform Engineering).

The output of this module: the ability to reason about distributed training and serving configurations from first principles. Given a model size and a hardware budget, you should be able to design the parallelism (TP/PP/DP), estimate the communication cost, identify the bottleneck, and pick the right collective.

## The roadmap

Eleven lessons.

### Networking foundations

1. **The network layer that lets ML scale: RDMA, InfiniBand, RoCE** — what each is, the latency / bandwidth profile, why RDMA is the unifying abstraction.
2. **NVLink, NVSwitch, and the bandwidth wall** — intra-node GPU-to-GPU networking, how it differs from inter-node, the physical bandwidth ceiling.

### Collective communication

3. **NCCL: collectives and topology awareness** — NVIDIA's collective communication library; topology-aware algorithm selection; what the API does and what it expects.
4. **AllReduce algorithms: ring, tree, double binary tree** — the three families of all-reduce implementations and when each wins.

### Parallelism strategies

5. **Distributed Data Parallel (DDP) from first principles** — the simplest parallelism: replicate the model, split the batch, all-reduce gradients. The baseline.
6. **ZeRO and FSDP: sharding optimizer state** — DeepSpeed's ZeRO and PyTorch's FSDP shard parameters/gradients/optimizer state to make DDP memory-feasible at large model sizes.
7. **Tensor parallelism: Megatron-style sharding** — splitting a single matmul across GPUs (recap and extension of Module 7 Lesson 46).
8. **Pipeline parallelism: GPipe, PipeDream, 1F1B** — splitting layers across GPUs with pipelined micro-batches; the bubble problem and its mitigations.
9. **The 3D parallelism grid** — TP × PP × DP composition; the topology you arrange GPUs in for the largest training runs.

### Sequence parallelism (revisit)

10. **Sequence parallelism and Ring Attention** — the fourth parallelism dimension (after TP, PP, DP); Module 7 Lesson 47 was the inference view; this is the training and combined view.

### Wrap

11. **Module wrap: building a 3D-parallel training run** — a worked example assembling everything; the decision tree for parallelism choice; handoff to Module 9.

---

## What this module deliberately won't cover

- **Specific cluster vendor architectures** in depth (Slurm, Kubernetes specifics). Module 9 covers cluster orchestration.
- **Application-layer ML platform** (experiment tracking, model registry). Module 10.
- **Reinforcement learning training** distributed setups specifically. Adjacent and important; not the focus here.
- **Heterogeneous-GPU training** (mixed A100 + H100 in one job). Real but a specialized topic.
- **Federated learning** (across user devices). Different problem; different communication patterns.

## How to work through it

Every lesson is fully readable as prose. The hands-on sections require multi-GPU access for most parts; cloud rentals (Lambda, RunPod, Modal, Coreweave) are the practical path. For the AllReduce and DDP lessons, 2 GPUs suffice. For the 3D parallelism lesson, ideally 8-16 GPUs.

If you only have a single GPU, the prose still stands. The mental models are framework-agnostic; the algorithms are derived from first principles. The hands-on for the single-GPU path uses `torch.distributed` simulation (multi-process on a single GPU).

A note on tempo: this module is dense and engineering-heavy. Each lesson has a clear "this is what happens at the wire level" component and a "this is how PyTorch / DeepSpeed exposes it" component. Read both; the abstractions hide the wire-level details, but the wire level is what determines performance.

The capstone: by the end of the module, you should be able to estimate the time-to-train of a 70B model given a specific GPU count and interconnect, and identify where the bottleneck is. The last lesson walks through this exercise.
