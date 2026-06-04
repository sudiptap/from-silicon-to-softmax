---
title: "Lesson 7 — Tensor Parallelism: Megatron-Style Sharding"
date: "2026-06-04"
module: "distributed-systems"
order: 7
tags: ["tensor-parallelism", "megatron", "column-parallel", "row-parallel", "training"]
author: "Sudipta Pathak"
prerequisites: ["06-zero-fsdp"]
---

# Lesson 7 — Tensor Parallelism: Megatron-Style Sharding

## Why this lesson exists

Module 7 Lesson 46 covered TP from the inference perspective. This lesson extends to training and goes deeper on the canonical Megatron design.

TP shards individual matmuls across GPUs. Where FSDP shards parameters by *moment in time* (load when needed, free after), TP shards them by *spatial partition* (each GPU owns its slice all the time). The matmul is split such that partial results combine via all-reduce or all-gather.

For training, TP combines naturally with DP and PP to form the 3D parallelism grid (Lesson 9).

This lesson covers Megatron's exact TP design: which matmuls are column-parallel, which are row-parallel, how the all-reduces fall out, and the activation memory implications.

The lesson is reading. The Hands-on simulates TP on a single machine.

## The Megatron TP pattern

For a standard transformer block (attention + FFN):

**Attention**:
- Q, K, V projections: column-parallel. Each GPU computes a subset of attention heads.
- Output projection: row-parallel. Each GPU computes a partial result; an AllReduce combines.

**FFN (MLP)**:
- Up projection (gate + up for SwiGLU): column-parallel. Each GPU computes a subset of intermediate channels.
- Down projection: row-parallel. Each GPU computes a partial result; an AllReduce combines.

The pattern: column-parallel feeds row-parallel; the AllReduce happens at the boundary between the row-parallel output and the next column-parallel input.

For each transformer block, two AllReduces per forward pass (one for attention output, one for FFN output), and two for backward (the gradient AllReduces). Total: 4 AllReduces per layer per step.

## Column-parallel vs row-parallel

For a matmul `Y = X @ W^T` with `W` shape `[out, in]`:

**Column-parallel** (split W along `out`):
- Each GPU holds `W[i × out/N : (i+1) × out/N]` — a slice of output rows.
- Each GPU computes `Y_i = X @ W_i^T` — a slice of output columns.
- No AllReduce needed; the partial outputs are concatenated along the output dimension.
- But: the next operation needs the full Y as input. If the next op is also column-parallel along a different dimension, you may need AllGather here.

**Row-parallel** (split W along `in`):
- Each GPU holds `W[:, j × in/N : (j+1) × in/N]` — a slice of input columns.
- Each GPU computes `Y_j = X_j @ W_j^T` (where X is also split along the input dim) — a partial output.
- The partial outputs are summed across GPUs via AllReduce.

The Megatron pattern: column-parallel followed by row-parallel. The column-parallel's output (which is naturally split) becomes the row-parallel's input (which expects a split). The AllReduce happens only at the row-parallel output, not in between.

## Activation memory

A key benefit: TP also shards *activations* during the parallel regions. Each GPU's intermediate activations are 1/N the full size.

For a 70B model with TP=8: per-GPU activation memory drops by 8× during the FFN's intermediate (which has 4× expansion → can be 4× the per-token bytes of the residual stream).

This makes long-context training feasible — activations would otherwise dominate memory.

## The all-reduce cost (training)

For TP across 8 GPUs in a node (NVLink-4 at 900 GB/s):
- Per-layer AllReduce: ~hidden_size × seq_len × batch × 2 bytes.
- For Llama 70B (hidden=8192), seq=4096, batch=4: 8192 × 4096 × 4 × 2 = 268 MB per AllReduce.
- Time at 900 GB/s with 2× ring factor: 268 × 2 / 900,000 = 0.6 ms.
- Per layer: 4 AllReduces (2 forward, 2 backward gradient). 2.4 ms.
- 80 layers: 192 ms per step purely from TP communication.

For a step time of ~500 ms (typical), TP communication is ~40% of step time. Significant but not dominant.

Cross-node TP (over RoCE at 100 GB/s): 9× slower communication. TP communication would dominate; not viable.

## Sequence parallelism (preview)

Within TP, the *non-parallel* regions (layer norms, residuals, dropout) are replicated on every GPU. For long sequences, this replicated activation memory dominates.

Sequence parallelism splits these along the sequence dimension: each GPU holds 1/N of the sequence's residual stream. Reduce communication is needed at the parallel-to-non-parallel transitions.

Sequence parallelism + TP is sometimes called "tensor + sequence parallelism" (TSP). It's Megatron's recommendation for large models. Lesson 10 covers sequence parallelism in more depth.

## Combining TP with FSDP

A common pattern at scale:
- TP=8 within a node (the model's matmuls are sharded; weights are replicated within the TP group but split across the NVLink GPUs).
- FSDP across nodes (the weight replicas across TP groups are themselves sharded).

This combines TP's per-step communication efficiency (NVLink) with FSDP's memory scaling (across nodes).

For Llama 70B on 8 nodes × 8 GPUs (64 total):
- TP=8 within each node: 70B weights split across 8 GPUs → ~8.75B per GPU.
- FSDP across 8 nodes: each TP group of 8 holds 1/8 of the optimizer state.
- Total per-GPU memory: 8.75B × 2 (BF16 params) + 8.75B × 12/8 (sharded Adam state) ≈ 30 GB. Fits comfortably.

## Training-time TP differences from inference

Module 7 Lesson 46 covered inference TP. Training adds:
- **Backward pass** with gradient AllReduces.
- **Optimizer step** that operates on the sharded parameters (each GPU updates its slice).
- **Activation memory** is much larger during training (backward needs them); the per-step memory profile is different.
- **Larger batch sizes** typical in training mean smaller communication overhead relative to compute.

The Megatron framework (NVIDIA) is the canonical TP-for-training reference. PyTorch's `torch.distributed.tensor.parallel` is a more recent native API.

## What you should believe after this lesson

Three sentences:

**1. Megatron-style TP pairs column-parallel and row-parallel matmuls**: column-parallel splits the output dim (no AllReduce needed); row-parallel splits the input dim and AllReduces the partial outputs. The pattern fits transformer attention and FFN cleanly with 2 AllReduces per layer per forward.

**2. TP shards activations as well as parameters during the parallel regions**, making long-context training memory-feasible. Sequence parallelism extends this to the non-parallel regions (layer norms, residuals) for further memory savings.

**3. TP within a node + FSDP across nodes is the standard combination** for large-model training. TP exploits NVLink for low-latency intra-node communication; FSDP exploits IB/RoCE for cross-node memory sharding.

## Hands-on (at home)

A simulated 2-way TP for an FFN.

```python
# tp_simulation.py
import torch
import torch.nn.functional as F

# Standard FFN.
D = 1024
dff = 4 * D
W_up = torch.randn(dff, D)
W_down = torch.randn(D, dff)
x = torch.randn(8, D)

# Standard forward.
y_std = F.gelu(x @ W_up.t()) @ W_down.t()
print(f"Standard FFN output shape: {y_std.shape}")

# Simulate TP=2.
# Up: column-parallel — split W_up along output dim (dff).
W_up_0 = W_up[:dff//2]   # GPU 0's slice
W_up_1 = W_up[dff//2:]   # GPU 1's slice

# Each GPU computes its slice of the up projection.
up_0 = F.gelu(x @ W_up_0.t())  # [8, dff/2]
up_1 = F.gelu(x @ W_up_1.t())  # [8, dff/2]

# Down: row-parallel — split W_down along input dim (dff).
W_down_0 = W_down[:, :dff//2]
W_down_1 = W_down[:, dff//2:]

# Each GPU computes a partial down output.
down_0 = up_0 @ W_down_0.t()  # [8, D]
down_1 = up_1 @ W_down_1.t()  # [8, D]

# AllReduce: sum the partial outputs.
y_tp = down_0 + down_1
print(f"TP FFN output shape: {y_tp.shape}")
print(f"Match: {torch.allclose(y_std, y_tp, atol=1e-5)}")
```

For real TP training, see Megatron-LM's reference code or PyTorch's `torch.distributed.tensor.parallel.ColwiseParallel` and `RowwiseParallel`.

## Further reading

- "Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism" (Shoeybi et al, 2019).
- "Reducing Activation Recomputation in Large Transformer Models" (Korthikanti et al, 2022) — introduces sequence parallelism with TP.
- Module 7 Lesson 46 — TP for inference.
- PyTorch tensor parallelism documentation.

Next lesson: **Pipeline parallelism — GPipe, PipeDream, 1F1B.** The other dimension of model parallelism: split layers across GPUs rather than splitting matmuls. We cover the pipelining schedules and the "bubble" problem.
