---
title: "Lesson 6 — ZeRO and FSDP: Sharding Optimizer State"
date: "2026-06-04"
module: "distributed-systems"
order: 6
tags: ["zero", "fsdp", "sharded-data-parallel", "optimizer-state", "memory"]
author: "Sudipta Pathak"
prerequisites: ["05-ddp"]
---

# Lesson 6 — ZeRO and FSDP: Sharding Optimizer State

## Why this lesson exists

Pure DDP replicates everything on every GPU: parameters, gradients, optimizer state. For Adam-style optimizers, the total memory per GPU is roughly:
- Parameters (FP16/BF16): 2 bytes per param.
- Gradients (FP16/BF16): 2 bytes per param.
- Optimizer state (Adam: momentum + variance + master weights in FP32): 12 bytes per param.

Total: 16 bytes per param. For a 7B model: 112 GB per GPU. Doesn't fit on an A100 (80 GB) or H100 (80 GB).

ZeRO (Zero Redundancy Optimizer, DeepSpeed 2019) and its PyTorch equivalent FSDP (Fully Sharded Data Parallel, 2022) shard parameters, gradients, and optimizer state across the data-parallel group. Each GPU holds a fraction, gathers what it needs for compute, frees it after.

This makes DDP-style training viable at much larger model sizes — currently up to ~70B in pure FSDP, more with mixed strategies.

The lesson is reading. The Hands-on enables FSDP for a small model.

## ZeRO levels

DeepSpeed's ZeRO has three "stages":

**ZeRO-1**: shard *only* the optimizer state. Parameters and gradients are still replicated.
- Saves the big optimizer-state portion (12 bytes per param for Adam).
- All-reduce of gradients is unchanged.
- Optimizer step: each GPU updates its shard; then all-gather updated parameters.
- Memory per GPU: `4 + 12/N` bytes per param (params + grads + sharded optimizer state).

**ZeRO-2**: shard optimizer state *and* gradients.
- Each GPU holds only `1/N` of the gradients.
- Reduce-scatter (not all-reduce) is used in backward: each GPU receives the reduced gradient for its shard.
- Optimizer step: each GPU updates its shard's parameters.
- Memory per GPU: `2 + 2/N + 12/N` bytes per param.

**ZeRO-3** (the most aggressive): shard optimizer state, gradients, *and* parameters.
- Each GPU holds only `1/N` of the parameters.
- For forward/backward, parameters are *gathered* on-demand (all-gather), used, then re-sharded.
- Memory per GPU: `2/N + 2/N + 12/N = 16/N` bytes per param.

ZeRO-3 (also called "Fully Sharded") is the most memory-efficient; it requires more communication (parameter all-gather per layer).

## FSDP (PyTorch's equivalent)

FSDP is roughly ZeRO-3 baked into PyTorch's native distributed API:

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
model = MyModel().to(device)
model = FSDP(model)
```

Behavior:
- Parameters are sharded across the DP group at initialization.
- For each forward call to a layer, parameters are all-gathered, used, then re-sharded.
- For backward, gradients are reduce-scattered.
- Optimizer state lives on the shard.

The all-gather + reduce-scatter pattern: per layer, each GPU does (1/N × all_gather_size) of work for the gather, plus (1/N × reduce_scatter_size) for the scatter.

FSDP became the PyTorch default for large-model training in 2023; pure DDP is for small models that fit.

## The communication cost

Per training step under ZeRO-3 / FSDP:
- All-gather of parameters per layer (forward): `param_size` total per GPU per layer.
- All-gather of parameters per layer (backward, for some configurations): another `param_size`.
- Reduce-scatter of gradients per layer (backward): `param_size`.

For Llama 70B at 80 layers: each step does ~80 × 3 × `params_per_layer / N` GB of communication.

For 8 GPUs and BF16 weights (140 GB total): per-step communication ~420 GB / 8 = ~52 GB per GPU per step.

At NVLink-4 (900 GB/s): ~60 ms purely for FSDP communication. Acceptable on top of the compute time of a 70B forward pass.

The communication is also overlappable with compute: while one layer's parameters are being gathered for backward, the next layer's compute can proceed. PyTorch's FSDP does this overlap automatically.

## The Adam mixed-precision detail

A typical Adam training stores:
- FP32 master weights: 4 bytes per param.
- FP32 momentum: 4 bytes per param.
- FP32 variance: 4 bytes per param.
- BF16/FP16 parameters: 2 bytes per param (the live copy used in forward/backward).
- BF16/FP16 gradients: 2 bytes per param.

Total: 16 bytes per param.

Under FSDP/ZeRO-3, the 12 bytes of FP32 optimizer state are sharded → 12/N per GPU. The 4 bytes of BF16 params + grads are also sharded (under ZeRO-3) → 4/N per GPU.

For Llama 70B on 8 GPUs: 16/8 = 2 bytes per param per GPU = 140 GB. Wait — that's still big. The trick: 8-GPU is one node; ZeRO-3 across many nodes is what really brings it down. ZeRO-3 across 64 GPUs (8 nodes): 140 GB × 64 = 9 TB... 9 TB / 64 = 140 GB per GPU still doesn't seem right.

Let me recalculate: 70B × 16 bytes / 64 GPUs = 17.5 GB per GPU. *That's* fittable. The 1/N scaling is the central win.

## When ZeRO/FSDP is the right choice

The clear case: training a model that doesn't fit on a single GPU with all of its optimizer state. For LLMs above ~3B parameters at full-precision Adam, you need ZeRO/FSDP or another sharding scheme.

Combinations:
- **FSDP only**: works up to ~70B. The communication cost grows; throughput suffers vs. smaller models.
- **FSDP + TP**: TP within a node (8 GPUs), FSDP across nodes. The TP keeps within-node communication bounded; FSDP handles cross-node memory.
- **FSDP + PP**: pipeline parallelism for very deep models combined with FSDP for memory.

For a 400B+ model: 3D parallelism (Lesson 9) combines TP + PP + DP/FSDP.

## What you should believe after this lesson

Three sentences:

**1. ZeRO and FSDP shard parameters, gradients, and optimizer state across the DP group** — each GPU holds `1/N` of the model's training state. Total memory per GPU drops from `16 × params` to `16 × params / N` bytes (for Adam mixed-precision).

**2. The cost is extra communication**: per-layer all-gather of parameters in forward, reduce-scatter of gradients in backward. PyTorch's FSDP overlaps this with compute; the per-step cost is modest at moderate N but grows with N.

**3. FSDP is the modern PyTorch default for large-model training**; pure DDP is for small models. FSDP combines with TP and PP for the largest training runs (Lessons 7-9).

## Hands-on (at home)

Enable FSDP for a small model.

```python
# fsdp_demo.py
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
import functools

def main():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    
    # Toy model.
    model = nn.Sequential(*[nn.Linear(1024, 1024) for _ in range(10)]).to(device)
    
    # Wrap with FSDP. Auto-wrap policy splits at parameter-count boundaries.
    wrap_policy = functools.partial(size_based_auto_wrap_policy, min_num_params=100000)
    model = FSDP(model, auto_wrap_policy=wrap_policy)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    for step in range(5):
        x = torch.randn(32, 1024, device=device)
        y = torch.randn(32, 1024, device=device)
        
        out = model(x)
        loss = (out - y).pow(2).mean()
        
        optimizer.zero_grad()
        loss.backward()  # Triggers reduce-scatter of gradients.
        optimizer.step()
        
        if rank == 0:
            mem = torch.cuda.memory_allocated() / 1e9
            print(f"step {step}: loss = {loss.item():.4f}, mem = {mem:.2f} GB")
    
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
```

Run with `torchrun --nproc_per_node=2 fsdp_demo.py`. Compare GPU memory usage to a DDP version of the same model — FSDP should use roughly half the memory at 2 GPUs.

## Further reading

- "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models" (Rajbhandari et al, 2019).
- PyTorch FSDP documentation.
- "PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel" (Zhao et al, 2023).
- DeepSpeed documentation on ZeRO.

Next lesson: **Tensor parallelism — Megatron-style sharding.** Module 7 Lesson 46 covered TP for inference; this lesson extends to training (the column/row partition pattern, the all-reduce structure, and Megatron's TP design).
