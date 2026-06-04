---
title: "Lesson 10 — Sequence Parallelism and Ring Attention"
date: "2026-06-04"
module: "distributed-systems"
order: 10
tags: ["sequence-parallelism", "ring-attention", "long-context", "context-parallel"]
author: "Sudipta Pathak"
prerequisites: ["09-3d-parallelism"]
---

# Lesson 10 — Sequence Parallelism and Ring Attention

## Why this lesson exists

The 3D parallelism grid (Lesson 9) handles model parameters via TP + PP + DP. It doesn't handle the *sequence dimension* — the long contexts that modern LLM training and inference need (32K → 1M+ tokens).

Sequence parallelism (Korthikanti et al, 2022; "Reducing Activation Recomputation in Large Transformer Models") was the first step: shard the non-parallel regions of TP (layer norms, residuals, dropout) along the sequence dimension. Modest memory savings.

Ring Attention (Liu et al, 2023; Module 7 Lesson 47) is more aggressive: shard the K, V along the sequence dimension during attention itself; rotate around a ring of GPUs; combine partial attention with the online-softmax trick.

Together, sequence parallelism + Ring Attention enable training and inference at 1M+ token contexts that no single-GPU memory could fit.

This lesson is the training-side view (Module 7 Lesson 47 covered inference) and the combined picture with TP/PP/DP.

The lesson is reading. The Hands-on sketches sequence-parallel layer norm.

## Sequence parallelism (Korthikanti)

Recall from Lesson 7: TP shards the matmuls but the *non-parallel* regions (layer norms, residuals, dropout) are *replicated* on every TP rank. For long sequences, the replicated activation memory dominates.

Sequence parallelism shards these regions along the sequence dimension:
- Each TP rank holds `seq_len / TP` rows of the residual stream during the non-parallel regions.
- For the parallel regions (attention, FFN matmuls), the TP rank holds the full sequence but a partition of the hidden dimension.
- Transitions between parallel and non-parallel regions require AllGather (parallel → non-parallel) and ReduceScatter (non-parallel → parallel).

The total bytes communicated are the same as standard TP (the AllReduce in TP becomes AllGather + ReduceScatter in TP+SP, which has the same bandwidth cost). But the activation memory drops significantly.

For Llama 70B at seq=8K, TP=8: activation memory savings ~5-10× from SP.

In practice, SP is enabled together with TP in modern Megatron and DeepSpeed configurations. It's a small extra setting; the memory savings make long-context training feasible.

## Ring Attention as sequence parallelism

Ring Attention takes the sequence sharding into attention itself:
- Each GPU holds `seq_len / W` Q, K, V (for W workers).
- During attention, Q is fixed; K, V slices rotate around the ring; partial attention is accumulated.
- Online-softmax makes the math exact.

The communication: each worker sends one K, V slice per round; W rounds total; total bytes per worker = sequence length. Total system-wide bandwidth: `O(N)`, independent of W.

Ring Attention scales to contexts that don't fit even with TP+SP within a node. For 1M-token contexts, Ring Attention spans multiple nodes; the K, V rotation goes over IB.

Module 7 Lesson 47 covered the inference algorithm; Lesson 47's same approach applies to training (with backward also rotating gradients around the ring).

## Combined parallelism: TP + SP + PP + DP + Ring

The full-spectrum parallelism for a 1M-context model:
- TP within a node (NVLink, intra-node compute parallel).
- SP within a node (along TP, shards non-parallel activations).
- Ring Attention across nodes (shards Q, K, V along sequence dimension).
- PP across pipeline groups (deep model).
- DP across full configurations (throughput multiplier).

That's 5 parallelism dimensions. The combination is what enables Llama 3.1's 128K context or Gemini's 1M+ context.

For most deployments, you don't need all 5. The common 2026 configurations:
- 70B at 8K context: TP+SP+DP.
- 70B at 128K context: TP+SP+Ring+DP.
- 405B at 8K context: TP+SP+PP+DP.
- 405B at 128K context: TP+SP+Ring+PP+DP.

## Context parallelism (a newer name)

Different frameworks have slightly different names for sequence-related parallelism:
- **Sequence parallelism** (Korthikanti): the original SP for the non-parallel regions of TP.
- **Context parallelism**: a more general term covering Ring Attention and other sequence-sharding approaches.
- **Ulysses** (DeepSpeed): an alternative sequence parallel that uses all-to-all instead of ring rotation.

The taxonomy isn't fully standardized. The underlying idea is the same: shard along sequence dimension; deal with the attention pattern via collectives.

## Training-time consideration: memory and stability

Long-context training has a quirk: many tasks don't actually benefit from very long context. Most pretraining data is far shorter than 128K tokens. Models trained at long context don't always use the context well.

The 2026 practical recipe:
- Pretrain at moderate context (8K-32K) without sequence parallelism — much cheaper.
- Continue training at long context (128K-1M) with sequence parallelism — necessary for the position embeddings to learn the long range.
- The long-context phase is much shorter than the main pretraining (~1-5% of total tokens).

This phased approach lets you train at long context only where needed, keeping the bulk of compute in the cheap regime.

## What you should believe after this lesson

Three sentences:

**1. Sequence parallelism extends TP** by sharding the non-parallel regions (layer norms, residuals, dropout) along the sequence dimension. Modest memory savings; small extra communication cost. Enabled with TP by default in modern Megatron / DeepSpeed.

**2. Ring Attention shards Q, K, V along the sequence dimension during attention itself**, rotating K, V around a ring with online-softmax accumulation. Enables contexts (1M+) that no single GPU could fit; can span nodes for very long contexts.

**3. The full-spectrum parallelism (TP + SP + Ring + PP + DP)** is what enables training the largest long-context models. The 2026 production recipe phases the parallelism: short-context pretraining is cheap; long-context continuation pulls in the heavy parallelism only briefly.

## Hands-on (at home)

Sketch a sequence-parallel layer norm.

```python
# sp_layernorm_sketch.py
import torch
import torch.nn as nn

class SequenceParallelLayerNorm(nn.Module):
    """Layer norm where the input is sharded along the sequence dimension."""
    def __init__(self, d_model, tp_world_size, eps=1e-5):
        super().__init__()
        self.tp_world_size = tp_world_size
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        # x: [B, N/W, D] — sequence sharded across W workers.
        # LayerNorm is independent per-token, so each worker can compute locally.
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        normalized = (x - mean) / torch.sqrt(var + self.eps)
        return normalized * self.gamma + self.beta

# In a real training run, the layer norm comes between TP's parallel regions.
# Before the layer norm: AllGather across TP workers to get the full sequence.
# After: ReduceScatter to shard back along the sequence.
# These collectives are the cost of SP; the layer-norm compute itself is local.

# For demo, just verify the layer norm produces the right shape.
torch.manual_seed(0)
W = 4
B, N, D = 1, 64, 128
x = torch.randn(B, N // W, D)
ln = SequenceParallelLayerNorm(D, W)
out = ln(x)
print(f"input shape: {x.shape}")
print(f"output shape: {out.shape}")  # same as input
```

For real sequence parallelism, see Megatron-LM's `core/tensor_parallel/layers.py` for the SP-aware layer implementations.

## Further reading

- "Reducing Activation Recomputation in Large Transformer Models" (Korthikanti et al, 2022) — sequence parallelism paper.
- "Ring Attention with Blockwise Transformers for Near-Infinite Context" (Liu et al, 2023).
- "Ulysses" (DeepSpeed) — alternative sequence parallel.
- Module 7 Lesson 47 — Ring Attention for inference.

Next lesson: **Module wrap — building a 3D-parallel training run.** A worked example. We assemble everything: a model size, a cluster, a parallelism configuration, a step-time estimate, and the decision rationale. The capstone of Module 8.
