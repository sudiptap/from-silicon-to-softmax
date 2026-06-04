---
title: "Lesson 47 — Ring Attention"
date: "2026-06-04"
module: "inference-from-scratch"
order: 47
tags: ["ring-attention", "sequence-parallelism", "long-context", "distributed"]
author: "Sudipta Pathak"
prerequisites: ["46-tensor-parallelism-inference"]
---

# Lesson 47 — Ring Attention

## Why this lesson exists

For very long contexts (1M+ tokens), even FlashAttention runs into memory limits — the KV cache alone might exceed a single GPU's memory. Ring Attention (Liu et al, 2023) shards the *sequence dimension* across multiple GPUs: each GPU holds a slice of the K and V; queries traverse the ring; partial attention results are combined.

The result: attention over arbitrarily long contexts becomes feasible by distributing across multiple GPUs. The communication cost is moderate; the algorithm preserves the exact attention computation.

This lesson covers Ring Attention as the canonical sequence parallelism technique for long context.

The lesson is reading. The Hands-on sketches the ring algorithm.

## The setup

For a sequence of N tokens distributed across W workers (GPUs):
- Each worker holds N/W tokens' worth of K, V (and Q for those positions).
- During attention, each query needs to attend to all K positions — but most K positions are on other workers.

The naive approach: gather all K and V to every worker. Communication cost: O(N × W); for large N this is prohibitive.

Ring Attention's approach: query the local K, V first; rotate K, V around the ring of workers; each worker progressively sees all K, V slices and accumulates partial attention.

## The algorithm

Setup: W workers in a ring (worker 0 → 1 → 2 → ... → W-1 → 0).

For each "round" `r` from 0 to W-1:

1. Each worker has Q (its local queries) and K, V (a particular slice).
2. Each worker computes partial attention: `score = Q × K^T`, partial output += softmax-style accumulation.
3. Each worker sends its current K, V slice to the next worker and receives a new slice from the previous worker.

After W rounds, every worker has attended to every K, V slice in the sequence. The accumulated partial outputs become the final attention output.

The clever part: the online-softmax pattern (from FlashAttention; Lesson 8) makes the partial accumulation numerically equivalent to standard attention. The result is exact.

## The communication cost

Per round: each worker sends one K, V slice (size N/W tokens × kv_per_token bytes).

Total communication per worker: W × (N/W × kv_per_token bytes) = N × kv_per_token bytes.

So total communication is proportional to N (the sequence length), independent of W. This is the key property — adding more workers doesn't increase total communication; it just splits it across more rounds.

For 1M tokens at FP16 with 32 KV heads × 128 d_head: ~1 GB per worker per inference. At NVLink speeds (200 GB/s), ~5 ms total — acceptable.

## When Ring Attention matters

The clear case: contexts where the KV cache exceeds a single GPU's memory:
- 1M+ token contexts on H100 (80 GB) for 7B-class models.
- 32K+ token contexts on smaller GPUs.
- Multi-million-token contexts (the cutting edge in 2026).

For shorter contexts that fit on a single GPU, FlashAttention alone is sufficient; Ring Attention's communication overhead isn't worth it.

## Production deployments

Ring Attention is research-grade in early 2024 but has matured rapidly:
- vLLM has experimental Ring Attention support.
- Google's Gemini 1.5 Pro (1M+ context) almost certainly uses some form of sequence parallelism (Google hasn't publicly specified).
- Anthropic's Claude 3 (200K context) likely uses something similar.

For self-hosted long-context inference, Ring Attention is the path to support contexts that don't fit on one GPU.

## Combining with other parallelisms

Ring Attention is sequence-parallel; it composes with:
- **Tensor parallelism (TP)**: split the model's parameters across GPUs as well as the sequence. TP within a node + sequence parallel across nodes.
- **Pipeline parallelism**: works orthogonally.
- **Data parallelism**: each replica handles a different request; sequence-parallel within each replica.

The 1M-token deployment recipe might be: TP=8 within a node + Ring Attention across 4 nodes = 32 GPUs effective.

## What you should believe after this lesson

Three sentences:

**1. Ring Attention shards the sequence dimension across GPUs**, with K and V slices rotating around a ring; each worker progressively attends to all positions. Total communication is O(N), independent of the number of workers.

**2. The online-softmax accumulation pattern (from FlashAttention) makes the partial result mathematically equivalent to standard attention**: exact, not approximate.

**3. Ring Attention enables very long contexts (1M+ tokens)** that don't fit on a single GPU. For shorter contexts, FlashAttention alone suffices; the communication overhead isn't worth it.

## Hands-on (at home)

A conceptual sketch of the ring rotation (single-machine simulation).

```python
# ring_attention_sketch.py
import torch
import torch.nn.functional as F
import math

def ring_attention_simulation(Q_full, K_full, V_full, world_size=4):
    N, d = K_full.shape
    chunk = N // world_size
    
    # Partition into per-worker slices.
    Q_workers = list(Q_full.chunk(world_size))
    K_workers = list(K_full.chunk(world_size))
    V_workers = list(V_full.chunk(world_size))
    
    # Each worker's output: accumulator.
    outputs = [torch.zeros_like(Q_workers[w]) for w in range(world_size)]
    # Online-softmax state (one per worker).
    max_per_q = [torch.full((Q_workers[w].shape[0], 1), float('-inf')) for w in range(world_size)]
    sum_per_q = [torch.zeros(Q_workers[w].shape[0], 1) for w in range(world_size)]
    
    for round in range(world_size):
        for w in range(world_size):
            # Worker w currently holds K_workers[(w+round) % world_size], V_workers[(w+round) % world_size].
            k_idx = (w + round) % world_size
            K_local = K_workers[k_idx]
            V_local = V_workers[k_idx]
            Q_local = Q_workers[w]
            
            scores = (Q_local @ K_local.T) / math.sqrt(d)  # [n_q_w, n_k_local]
            # Online-softmax accumulation.
            local_max = scores.max(dim=-1, keepdim=True).values
            new_max = torch.maximum(max_per_q[w], local_max)
            # Rescale old state.
            scale_factor = torch.exp(max_per_q[w] - new_max)
            outputs[w] *= scale_factor
            sum_per_q[w] *= scale_factor
            # Add this round's contribution.
            exp_scores = torch.exp(scores - new_max)
            sum_per_q[w] += exp_scores.sum(dim=-1, keepdim=True)
            outputs[w] += exp_scores @ V_local
            max_per_q[w] = new_max
    
    # Normalize.
    outputs = [outputs[w] / sum_per_q[w] for w in range(world_size)]
    # Concatenate.
    return torch.cat(outputs, dim=0)

# Verify against standard attention.
torch.manual_seed(0)
N, d = 64, 32
Q = torch.randn(N, d)
K = torch.randn(N, d)
V = torch.randn(N, d)
out_ring = ring_attention_simulation(Q, K, V, world_size=4)
scores = (Q @ K.T) / math.sqrt(d)
weights = F.softmax(scores, dim=-1)
out_standard = weights @ V
print(f"Max diff: {(out_ring - out_standard).abs().max().item():.6f}")
```

The outputs should match within floating-point rounding. The simulation captures the algorithm; real Ring Attention uses NCCL for the K, V rotation.

For multi-GPU benchmarks: requires multi-GPU access; the open-source `ring_flash_attention` package implements it production-grade.

## Further reading

- "Ring Attention with Blockwise Transformers for Near-Infinite Context" (Liu et al, 2023).
- "Striped Attention" — a related variant.
- vLLM long-context documentation.
- DeepSpeed-Ulysses — alternative sequence parallelism.

Next lesson: **Tree Attention for branched generation.** A different angle: when generating multiple candidate sequences (best-of-N, beam search, speculative tree), share the common prefix's attention computation.
