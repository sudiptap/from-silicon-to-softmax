---
title: "Lesson 2 — Multi-Head Attention (MHA)"
date: "2026-06-04"
module: "inference-from-scratch"
order: 2
tags: ["attention", "mha", "multi-head", "heads"]
author: "Sudipta Pathak"
prerequisites: ["01-self-attention-first-principles"]
---

# Lesson 2 — Multi-Head Attention (MHA)

## Why this lesson exists

Self-attention with one set of `(W_Q, W_K, W_V)` projections is rank-limited: it can only learn one "kind" of similarity at a time. Different relationships between tokens — syntactic vs semantic, local vs long-distance, agreement vs reference — would have to share one projection space, competing for representational capacity.

Multi-head attention (MHA) addresses this by splitting the `d`-dimensional vector space into `h` smaller subspaces, each with its own projections. Each "head" computes attention independently over a subspace of dimension `d_head = d / h`, and the outputs are concatenated and projected back. The model can learn `h` different attention patterns simultaneously.

This lesson is MHA from the ground up: how the split works, what each head can specialize in, the parameter accounting, and the implementation. MHA is the default attention variant from "Attention Is All You Need" through GPT-3-era models.

The lesson is reading. The Hands-on implements MHA from scratch and inspects what different heads attend to.

## The shape transformation

Start from the inputs of Lesson 1: `Q`, `K`, `V` each of shape `[B, N, D]` where `D = d_model`.

For MHA with `h` heads, each head has dimension `d_head = D / h`. The transformation:

1. Project to Q, K, V (same as Lesson 1).
2. Reshape: `[B, N, D]` → `[B, N, h, d_head]` → `[B, h, N, d_head]`.
3. Run scaled dot-product attention *per head, in parallel*: each head sees `Q[b, head]`, `K[b, head]`, `V[b, head]` of shape `[N, d_head]`.
4. Concatenate heads: `[B, h, N, d_head]` → `[B, N, h, d_head]` → `[B, N, D]`.
5. Final output projection: `[B, N, D]` → `[B, N, D]` via `W_O`.

The output projection `W_O` mixes information across heads. It's an underrated piece — without it, the heads' outputs would be in separate subspaces and the model couldn't combine them.

## Parameter count

For a transformer block with `d_model = D`, `h` heads, `d_head = D/h`:

- `W_Q`, `W_K`, `W_V` each: `D × D` parameters. Total: `3D²`.
- `W_O`: `D × D` parameters.
- Total attention parameters: `4D²`.

Note the parameter count is independent of `h`. The choice of `h` doesn't change the parameter budget; it just slices the same projection differently. Common choices: `h = 8` for 512-dim models (Vaswani 2017), `h = 12` for GPT-2 small (768 dim), `h = 32` for Llama 2 7B (4096 dim).

The convention `d_head = D / h` is universal but not required. Some recent architectures (Llama 3 with GQA, MLA) break it.

## What each head learns

A standard observation when probing trained transformers: different heads specialize in different patterns. Common findings:

- **Positional heads**: attend to nearby tokens (the previous token, the next token, fixed offsets).
- **Syntactic heads**: attend to grammatically related tokens (subject of the current verb, the noun a pronoun refers to).
- **Semantic heads**: attend to semantically related tokens (a word's antonym, a topic-related word).
- **Copy heads**: attend strongly to similar / identical tokens earlier in the sequence (used for repeated names, citation patterns).
- **Induction heads**: a specific pattern where one head attends back to a previous occurrence and copies what comes next (the basis of in-context learning).

Not every head is interpretable; many do something the probing techniques can't characterize. But the diversity is real and is the empirical justification for multi-head over single-head.

## Why split rather than wider single-head?

A natural question: instead of `h` heads of dimension `D/h`, why not one head of dimension `D`?

The answer is mostly empirical: the multi-head version trains better. The single-head version has the same parameter count and the same expressivity in theory (since the heads' outputs are linearly recombined), but in practice the head split provides an inductive bias that helps learning. Different heads can develop specializations more easily when they're forced to live in separate subspaces.

A secondary answer: the multi-head version is more parallelism-friendly. Each head's computation is independent, so they can run concurrently on hardware that supports tensor parallelism. Llama 3 70B running with 8-way tensor parallel splits the heads across 8 GPUs.

## Causal masking with multi-head

Same as Lesson 1 — the causal mask is applied per head. The implementation handles all heads in one go via broadcasting:

```python
# scores: [B, h, N, N]
mask = torch.tril(torch.ones(N, N)) == 0  # [N, N]
scores = scores.masked_fill(mask, float('-inf'))  # broadcasts over B and h
```

Every head has the same mask; no head can see the future.

## Complexity and memory

MHA's per-token compute and memory are dominated by:

- **Compute**: `O(N² D)` for attention (the `Q @ K^T` and `weights @ V`).
- **Memory**: `O(N² h)` for the attention matrix (one `[N, N]` matrix per head). At long context, this dominates.

For Llama 2 7B (`D=4096`, `h=32`, `N=4096`): 32 heads × 4096² × 4 bytes (FP32) = 2 GB just for the attention matrices at the FP32 path. At FP16, 1 GB. This is the cost FlashAttention (Lesson 8) eliminates by never materializing the full `[N, N]` matrices.

## Implementation

Two common conventions for MHA implementation:

**Style 1: separate `nn.Linear` for Q, K, V** (clearer):

```python
self.W_q = nn.Linear(D, D)
self.W_k = nn.Linear(D, D)
self.W_v = nn.Linear(D, D)
```

**Style 2: fused QKV projection** (faster, what most production code does):

```python
self.W_qkv = nn.Linear(D, 3 * D)  # one matmul producing Q | K | V
```

The fused version saves two matmul launches and lets the GPU do the projection more efficiently. It's invisible to the math; just a CUDA optimization.

## Heads and the KV cache

For inference, the KV cache stores the keys and values per head per token. The cache size scales with `h × d_head × n_tokens = D × n_tokens`. So the KV cache per token is `D` per layer for K + `D` per layer for V = `2D` per layer.

For Llama 2 7B (32 layers, D=4096) at FP16: `2 × 32 × 4096 × 2 = 524288` bytes ≈ 512 KB per token. At 4K context, ~2 GB.

This `2D` per-token cost is what MQA (Lesson 3) and GQA (Lesson 4) target. They reduce the number of K and V "heads" while keeping the number of Q heads constant, shrinking the KV cache without changing the model's Q-head capacity.

## What you should believe after this lesson

Three sentences:

**1. Multi-head attention splits the projection space into `h` subspaces of dimension `d_head = D/h`** so the model can learn multiple attention patterns in parallel. The parameter count is `4D²` regardless of `h`; the empirical benefit is in learnability, not capacity.

**2. Different heads learn different patterns** — positional, syntactic, semantic, induction. Not all are interpretable, but the diversity is real and empirically justifies the multi-head split.

**3. The `O(N² h)` attention-matrix memory is the long-context bottleneck**, and the `2D × n_layers × n_tokens` KV cache cost per token is what MQA and GQA reduce. Every attention variant in the next several lessons trades something for one of these costs.

## Hands-on (at home)

Implement MHA from scratch and inspect what the heads do.

```python
# mha.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, causal=True):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.causal = causal
        # Fused QKV projection.
        self.W_qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        B, N, D = x.shape
        qkv = self.W_qkv(x)  # [B, N, 3*D]
        q, k, v = qkv.chunk(3, dim=-1)  # each [B, N, D]
        # Reshape to [B, h, N, d_head].
        q = q.view(B, N, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, N, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, N, self.n_heads, self.d_head).transpose(1, 2)
        # Scaled dot-product attention per head.
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)  # [B, h, N, N]
        if self.causal:
            mask = torch.tril(torch.ones(N, N, device=x.device)) == 0
            scores = scores.masked_fill(mask, float('-inf'))
        weights = F.softmax(scores, dim=-1)
        out = weights @ v  # [B, h, N, d_head]
        # Concatenate heads.
        out = out.transpose(1, 2).contiguous().view(B, N, D)
        # Output projection.
        return self.W_o(out), weights  # also return weights for inspection

# Test.
torch.manual_seed(0)
mha = MultiHeadAttention(d_model=64, n_heads=4, causal=True)
x = torch.randn(1, 8, 64)
y, attn_weights = mha(x)
print(f"output shape: {y.shape}")
print(f"attention weights shape: {attn_weights.shape}")  # [B, h, N, N]

# Inspect what each head attends to.
print("Attention weights per head (row = query, col = key, last row only):")
for h in range(4):
    row = attn_weights[0, h, -1, :]  # last query position
    print(f"  head {h}: {row.detach().numpy()}")
```

Run with a small model and you'll see each head's last-row attention distribution. Some heads will be uniform (haven't specialized — common for random initialization); after training, different heads develop visible patterns.

For interpretability on a *trained* model, use a model like GPT-2:

```python
from transformers import GPT2Model, GPT2Tokenizer
tok = GPT2Tokenizer.from_pretrained("gpt2")
m = GPT2Model.from_pretrained("gpt2", output_attentions=True)
inputs = tok("The cat sat on the mat", return_tensors="pt")
out = m(**inputs)
# out.attentions is a tuple of [batch, n_heads, n_tokens, n_tokens] per layer.
# Visualize one layer's attention pattern with matplotlib for any head.
```

## Further reading

- "Attention Is All You Need" Section 3.2.2 — the original MHA description.
- "A Mathematical Framework for Transformer Circuits" (Anthropic, 2021) — for the induction-heads characterization and other head specializations.
- "BertViz" (visualization tool) — for interactive exploration of trained model attention patterns.
- "Attention is Not Only a Weight: Analyzing Transformers with Vector Norms" (Kobayashi et al, 2020) — for why raw attention weights aren't the full story.

Next lesson: **Multi-Query Attention (MQA).** The first variant that breaks the "every head has its own K, V" convention — MQA shares one K and one V across all heads, drastically shrinking the KV cache at the cost of some model capacity. The first stop on the road to GQA and MLA.
