---
title: "Lesson 46 — Tensor Parallelism for Inference"
date: "2026-06-04"
module: "inference-from-scratch"
order: 46
tags: ["tensor-parallelism", "megatron", "inference", "all-reduce", "nvlink"]
author: "Sudipta Pathak"
prerequisites: ["45-cuda-graphs"]
---

# Lesson 46 — Tensor Parallelism for Inference

## Why this lesson exists

Tensor parallelism (TP) — splitting a single matmul across multiple GPUs — was developed by Megatron-LM (Shoeybi et al, 2019) for *training* very large models. The training-side story is well-established.

TP for *inference* has a different cost profile, and the right design is not the same as training. Inference has stricter latency requirements, smaller batch sizes (so communication is a larger fraction of work), and different memory-vs-compute tradeoffs.

This lesson covers TP specifically for inference: how it differs from training TP, the communication patterns, and when TP is the right answer (vs sticking with a smaller model on a single GPU).

The lesson is reading. The Hands-on profiles the all-reduce communication cost in a TP setup.

## What TP does

The transformer block has matmuls. TP splits each matmul across GPUs:

- For QKV projections and the FFN's first projection (`gate`/`up`): split along the *output* dimension. Each GPU computes a different slice of the output.
- For the attention output projection and the FFN's `down` projection: split along the *input* dimension. Each GPU computes a partial result; all results are summed via *all-reduce*.

The pattern: split per-column then split per-row, with all-reduce between layers. This is the Megatron-LM TP pattern.

For an 8-way TP setup:
- Each GPU holds 1/8 of each layer's parameters.
- Per-layer compute is parallelized 8-way.
- Per-layer communication: one all-reduce after attention output, one after FFN down.

Effective per-token throughput: roughly 8× the single-GPU rate (limited by communication overhead).

## The inference-specific tradeoffs

TP for inference has different constraints than TP for training:

**1. Batch sizes are smaller.** Training batches are large (1024+); the communication cost per token is amortized. Inference batches are smaller (1-64); communication is a larger fraction.

**2. Latency matters.** Training cares about throughput per dollar; inference cares about both throughput and latency. TP adds latency (all-reduces add 0.1-1 ms per layer); for latency-critical applications, this matters.

**3. Memory budgets differ.** Training needs gradients and optimizer state; inference doesn't. For the same model, inference TP can support smaller TP sizes because the memory pressure is lower.

The inference-specific recipe:
- Use TP only when the model doesn't fit on a single GPU at the required precision.
- Prefer NVLink-connected GPUs (within one node) — Infiniband across nodes is too slow for TP all-reduce.
- TP=2 or TP=4 within a node is common; TP=8 is the upper limit for most production inference.

## The all-reduce cost

For TP=8 on a Hopper-class node:
- All-reduce within a node over NVLink: ~10-50 GB/s effective.
- Bytes per all-reduce per layer: `2 × hidden_dim × batch_tokens × bytes_per_elem`. For hidden=8192, batch=32 tokens, FP16: 4 MB.
- Per-layer all-reduce latency: ~100 μs at moderate batch sizes; less at higher batch.

For 80 layers (Llama 3.1 70B), total all-reduce overhead per step: ~8 ms. For a 50 ms decode step, that's ~16%. Real but not dominant.

For NVLink-less setups (cross-node Infiniband), all-reduce is 10-100× slower; TP becomes impractical.

## TP vs PP vs DP for inference

The other parallelism dimensions:

**Pipeline parallelism (PP)**: split *layers* across GPUs. Each GPU runs a chunk of layers; results pipeline between them. For inference, PP has a pipelining-bubble problem at small batches — most stages are idle most of the time. PP is rarely used for inference; it's mostly a training technique.

**Data parallelism (DP)**: replicate the whole model on each GPU; serve different requests on each. For inference, DP is the throughput multiplier — N GPUs serve N× the requests per second. Used in combination with TP.

**Expert parallelism (EP, Lesson 34)**: only for MoE; shards experts.

A typical large-model inference deployment combines:
- TP within a node (e.g., TP=8 on an 8×H100 node).
- DP across nodes (replicate the TP setup, serve more requests).
- EP for MoE (across nodes if necessary).

## When TP is the right answer

The clear cases:
- The model doesn't fit on a single GPU at acceptable precision. A 70B model in FP16 (140 GB) doesn't fit on a single 80 GB H100; TP=2 (70 GB per GPU) does.
- You want low latency for a single request and the model is large enough that compute parallelism via TP helps.
- You have an NVLink-connected multi-GPU node.

When TP isn't right:
- The model fits on one GPU with room to spare. Adding TP just adds communication overhead.
- You're optimizing for total throughput across many requests; DP scales more efficiently than TP.
- You don't have NVLink. Cross-node TP is too slow.

For a 7B model on H100: don't bother with TP; DP across multiple H100s serves more requests.
For a 70B model: TP=2 within a node is the default.
For a 405B model (Llama 3.1 405B): TP=8 within a node; possibly cross-node sharding.

## What you should believe after this lesson

Three sentences:

**1. Tensor parallelism splits each layer's matmuls across GPUs** with all-reduce communication between layers. TP for inference has smaller batches and stricter latency than TP for training; the communication overhead is a larger fraction of total work.

**2. TP within a node (over NVLink) is fine** — all-reduce latency is ~100 μs per layer at moderate batches. Cross-node TP (over Infiniband) is impractical because the all-reduce becomes too slow.

**3. The 2026 rule of thumb**: use TP only when the model doesn't fit on a single GPU; combine with DP for throughput across multiple TP-setups. PP is mostly for training, not inference. EP only for MoE.

## Hands-on (at home)

Multi-GPU TP requires multiple GPUs. The conceptual breakdown without multi-GPU:

```python
# tp_sketch.py
import torch

# Simulate 4-way TP on a single GPU by partitioning matrices.
D = 1024
H = 4  # TP world size

# Full-size weight matrix W: [D, D]
W = torch.randn(D, D)

# Partition for column-parallel (split along output dim).
W_chunks_cols = W.chunk(H, dim=1)  # each shard: [D, D/H]

x = torch.randn(8, D)
# Each "rank" computes its chunk.
local_outs = [x @ Wc for Wc in W_chunks_cols]  # each: [8, D/H]
# Concatenate along columns.
local_full = torch.cat(local_outs, dim=1)
# Equivalent to full matmul.
full = x @ W
print(f"Match (column-parallel): {torch.allclose(local_full, full)}")

# Row-parallel (split along input dim).
W_chunks_rows = W.chunk(H, dim=0)  # each shard: [D/H, D]
x_chunks = x.chunk(H, dim=1)  # each: [8, D/H]
local_outs = [xc @ Wc for xc, Wc in zip(x_chunks, W_chunks_rows)]  # each [8, D]
# All-reduce: sum across ranks.
local_full = sum(local_outs)
print(f"Match (row-parallel): {torch.allclose(local_full, x @ W)}")
```

This shows the math. Real multi-GPU TP uses `torch.distributed` collectives (`all_reduce`) over NCCL.

For real multi-GPU TP benchmarks, use vLLM with `--tensor-parallel-size 2/4/8` on a multi-GPU machine and measure throughput at various batch sizes.

## Further reading

- "Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism" (Shoeybi et al, 2019) — original TP paper.
- "Efficiently Scaling Transformer Inference" (Pope et al, 2022) — discusses inference TP at scale.
- vLLM tensor parallelism documentation.
- TensorRT-LLM tensor parallelism configuration.

End of Part 7. Next: Part 8 begins with **Ring Attention** — sharding the sequence dimension across GPUs. The frontier of long-context inference at scale.
