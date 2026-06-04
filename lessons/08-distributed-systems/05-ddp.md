---
title: "Lesson 5 — Distributed Data Parallel (DDP) from First Principles"
date: "2026-06-04"
module: "distributed-systems"
order: 5
tags: ["ddp", "data-parallel", "gradient-allreduce", "pytorch"]
author: "Sudipta Pathak"
prerequisites: ["04-allreduce-algorithms"]
---

# Lesson 5 — Distributed Data Parallel (DDP) from First Principles

## Why this lesson exists

DDP (Distributed Data Parallel) is the simplest distributed parallelism strategy: replicate the model on every GPU; split the batch across GPUs; each GPU computes gradients on its mini-batch; all-reduce the gradients; every GPU updates its replica with the averaged gradients.

The result: every GPU has identical model state at all times. The aggregate batch size scales linearly with GPU count. The optimizer step is also identical across GPUs (same gradients → same updates).

DDP is the baseline. It works for any model that fits on a single GPU. For larger models, DDP combines with other parallelisms (TP, PP, ZeRO). Even at frontier scale, DDP is part of the stack.

This lesson covers DDP from the algorithmic level, the PyTorch implementation, and the standard performance optimizations.

The lesson is reading. The Hands-on writes a DDP training loop with the PyTorch primitives.

## The algorithm

For `N` GPUs and a global batch size `B`:

1. **Setup**: every GPU has an identical copy of the model. Broadcast initial weights from rank 0 if needed.
2. **Data sharding**: the data loader gives each GPU `B/N` samples per step.
3. **Forward**: each GPU runs its forward pass on its local batch. Produces local activations and loss.
4. **Backward**: each GPU computes local gradients.
5. **AllReduce gradients**: across GPUs, average the gradients. Every GPU ends with the global average gradient.
6. **Optimizer step**: every GPU applies the same averaged gradient to its replica. They stay in sync.
7. **Repeat**.

The AllReduce is the only communication; everything else is local.

For correctness: the global average gradient is mathematically equivalent to a gradient computed on the full batch. DDP at `N` GPUs with batch `B/N` per GPU is equivalent (up to small numerical differences in summation order) to single-GPU training at batch `B`.

## The PyTorch implementation

The `torch.nn.parallel.DistributedDataParallel` wrapper:

```python
model = MyModel().to(device)
ddp_model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device])
```

The wrapper:
- Registers backward hooks on each parameter.
- When a parameter's gradient is computed, triggers an async AllReduce of that gradient.
- The AllReduce overlaps with the rest of the backward pass.
- By the end of backward, all gradients are reduced.

The overlap is critical: without it, communication time adds directly to step time. With overlap, communication hides behind compute.

## Gradient bucketing

A naïve implementation would AllReduce each parameter's gradient as soon as it's computed. For a model with thousands of parameters, that's thousands of small AllReduces — terrible for performance (latency-dominated).

The optimization: *bucket* gradients. Group small gradients together; AllReduce a bucket of ~25 MB at a time.

The bucketing happens in reverse parameter order (last layer's gradients are computed first in backward). PyTorch's DDP wrapper handles this automatically.

The bucket size is a hyperparameter (default 25 MB in PyTorch). Larger buckets → fewer AllReduces → less latency overhead. Smaller buckets → earlier reduction start → more overlap with backward. The default is reasonable for most workloads.

## Synchronous vs asynchronous SGD

DDP is *synchronous* SGD: every GPU waits for the AllReduce to complete before the optimizer step. This guarantees all GPUs stay in sync.

Asynchronous SGD (used in some early frameworks): each GPU updates independently with its local gradients; gradients eventually propagate to others. Faster per-step but harder to converge because gradients are stale.

For LLMs, synchronous SGD is the universal choice. Convergence is more reliable; the gradient-average property is mathematically clean.

## What DDP costs

Per step, DDP communicates: `2 × model_size` bytes via AllReduce (the `2×` is from ring AllReduce's 2× bandwidth factor).

For Llama 70B in BF16 (140 GB):
- AllReduce per step: 280 GB.
- On NVLink (900 GB/s) within a node: ~0.3 s.
- On RoCE (100 GB/s) across nodes: ~3 s.

For a 1-node DDP run: ~0.3s per step is OK if step time is much larger. For multi-node DDP at 70B: not great; the all-reduce dominates step time.

This is why pure DDP doesn't scale well past a single node for very large models. Beyond a single node, you typically combine DDP with TP or ZeRO (Lesson 6) to reduce the communication burden.

## The "effective batch size" question

DDP scales the effective batch size linearly with GPU count. At 8 GPUs with per-GPU batch 32, the effective batch is 256.

This is usually a *good* thing — larger batches train more stably. But:
- The learning rate schedule needs adjustment (the "linear scaling rule" — LR scales linearly with batch size, up to a saturation point).
- At very large effective batches (10K+), convergence may slow even with proper LR scaling (the "large-batch generalization gap").
- The data loader needs to keep up — at 8 GPUs, you need 8× the data throughput.

Most production training uses gradient accumulation + DDP together: per-GPU batch × accumulation steps × world_size = total effective batch. This decouples per-step memory from effective batch size.

## DDP variations: HSDP and others

**Hybrid Sharded Data Parallel (HSDP)**: a recent variant (PyTorch 2.x) that does ZeRO-style sharding within a "small DP" group and full replication across "large DP" groups. The intra-group sharding reduces memory; the inter-group replication keeps cross-node communication manageable.

For models that almost fit on a node, HSDP is the sweet spot. For models way too large, full FSDP across all nodes is needed.

## What you should believe after this lesson

Three sentences:

**1. DDP replicates the model on every GPU, splits the batch, and AllReduces gradients each step.** The math is equivalent to single-GPU training with the larger effective batch; convergence is well-understood.

**2. PyTorch's DDP overlaps AllReduce with the backward pass via parameter-grouped buckets** (default 25 MB). The overlap hides most of the communication time behind compute; without it, AllReduce would dominate.

**3. Pure DDP doesn't scale well past a single node for very large models** because the per-step AllReduce of the full model weights becomes prohibitive over cross-node IB/RoCE. ZeRO/FSDP (Lesson 6) and combination with TP (Lesson 7) address this.

## Hands-on (at home)

A minimal DDP training loop (requires 2+ GPUs or multi-process on one GPU).

```python
# ddp_demo.py
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import os

def main():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    
    # Simple model.
    model = nn.Sequential(
        nn.Linear(1024, 1024), nn.ReLU(),
        nn.Linear(1024, 10)
    ).to(device)
    model = DDP(model, device_ids=[rank])
    
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    for step in range(5):
        # Each rank uses its own per-rank batch.
        x = torch.randn(32, 1024, device=device)
        y = torch.randint(0, 10, (32,), device=device)
        
        out = model(x)
        loss = criterion(out, y)
        
        optimizer.zero_grad()
        loss.backward()  # Triggers async AllReduce of gradients.
        optimizer.step()
        
        if rank == 0:
            print(f"step {step}: loss = {loss.item():.4f}")
    
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
```

Run with `torchrun --nproc_per_node=2 ddp_demo.py`. You'll see the same loss values across ranks (because gradients are averaged, and weights stay in sync).

For a real measurement of DDP overhead, set `NCCL_DEBUG=INFO` and profile a training step. The PyTorch profiler shows the AllReduce time vs forward/backward time.

## Further reading

- PyTorch DDP documentation.
- "PyTorch Distributed: Experiences on Accelerating Data Parallel Training" (Li et al, 2020).
- "Linear Scaling Rule" (Goyal et al, 2017) — for LR schedules under DDP scaling.
- HSDP documentation in PyTorch 2.x.

Next lesson: **ZeRO and FSDP — sharding optimizer state.** Pure DDP requires the model + optimizer state to fit on every GPU. ZeRO and FSDP shard parameters / gradients / optimizer state across GPUs to make DDP work at much larger model sizes.
