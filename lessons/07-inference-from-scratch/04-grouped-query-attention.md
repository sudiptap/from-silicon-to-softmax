---
title: "Lesson 4 — Grouped-Query Attention (GQA)"
date: "2026-06-04"
module: "inference-from-scratch"
order: 4
tags: ["attention", "gqa", "kv-cache", "llama", "groups"]
author: "Sudipta Pathak"
prerequisites: ["03-multi-query-attention"]
---

# Lesson 4 — Grouped-Query Attention (GQA)

## Why this lesson exists

MHA is the full-capacity baseline; MQA is the bandwidth-optimized variant that costs quality. GQA (Ainslie et al, 2023) lives between them: instead of one K/V (MQA) or one K/V per head (MHA), GQA uses `g` K/V groups, where `g` divides `n_heads`. Each group of `n_heads / g` query heads shares one K/V pair.

This single parameter (`g`) lets you tune the MHA↔MQA spectrum continuously. At `g = n_heads`, GQA reduces to MHA. At `g = 1`, GQA reduces to MQA. Llama 2 (70B) introduced GQA-8; Llama 3 uses GQA with various group counts; Mistral 7B uses GQA-8.

The empirical sweet spot: a small number of groups (typically 4 or 8) recovers most of MHA's quality at a fraction of the KV cache cost. GQA is the default attention variant for production LLMs in 2026.

This lesson is the mechanics, the implementation, and the practical group-count choice.

The lesson is reading. The Hands-on implements GQA and compares to MHA and MQA on KV cache size and decode speed.

## The mechanics

Setup: `n_heads` query heads, `g` KV groups (must divide `n_heads`).

- Each group has its own K and V projection (produces a `d_head`-dim vector each).
- Each query head belongs to a group; the head's query attends against its group's K and V.
- Equivalent: copy each K/V `n_heads / g` times to broadcast back to per-head shape, then do MHA-style attention.

The shape transformation:

1. Project Q: `[B, N, D]` → `[B, n_heads, N, d_head]`.
2. Project K, V: `[B, N, D]` → `[B, g, N, d_head]` (just `g` heads, not `n_heads`).
3. Repeat K, V along the head dimension: `[B, g, N, d_head]` → `[B, n_heads, N, d_head]` (each K/V replicated `n_heads / g` times).
4. Standard MHA-style scaled dot product.

The repeat step is conceptual — production implementations don't actually copy data; they use broadcasting or the attention kernel handles the indexing.

## Parameter and KV cache accounting

For `n_heads = h`, `g` groups, `d_head = D/h`:

- `W_Q`: `D × D` (full).
- `W_K`: `D × g × d_head = D × (g D / h)` parameters.
- `W_V`: `D × g × d_head` parameters.
- `W_O`: `D × D`.

Total: `2 D² + 2 D × (g D / h) = D² × (2 + 2g/h)`.

For MHA (`g = h`): `D² × 4` — confirming MHA.
For MQA (`g = 1`): `D² × (2 + 2/h)` — approximately `2 D²`, confirming MQA.
For GQA-8 on a model with 32 heads: `D² × (2 + 16/32) = 2.5 D²` — meaningful savings over MHA's 4D² without going all the way to MQA's 2D².

KV cache per token per layer:

```
2 × g × d_head × bytes = 2 × g × (D/h) × bytes
```

MHA (g=h): `2D × bytes`.
MQA (g=1): `2(D/h) × bytes`.
GQA-8 with h=32: `2 × (D/4) × bytes` — 4× smaller than MHA, 8× larger than MQA.

For Llama 2 70B (D=8192, h=64, g=8 → groups of 8 Q heads per KV pair):
- MHA equivalent KV: `2 × 8192 × 2 = 32 KB` per token per layer.
- GQA-8 actual: `2 × 8 × (8192/64) × 2 = 4 KB` per token per layer. 8× smaller.
- MQA equivalent: `2 × (8192/64) × 2 = 256 bytes` per token per layer.

Across 80 layers and a 4K context:
- MHA: 80 × 32 KB × 4096 = 10 GB.
- GQA-8: 80 × 4 KB × 4096 = 1.25 GB.
- MQA: 80 × 256 × 4096 = 80 MB.

GQA's 1.25 GB is much more manageable than MHA's 10 GB while preserving most of the head specialization.

## Why GQA tends to win

The empirical observation: as you go from MHA to GQA-8 to GQA-2 to MQA:
- KV cache shrinks monotonically.
- Inference throughput improves monotonically.
- Model quality degrades monotonically.

The shape of the quality curve isn't linear. MHA → GQA-8 (4× KV reduction) loses ~0.05 perplexity. GQA-8 → GQA-2 (4× more reduction) loses ~0.15 perplexity. GQA-2 → MQA (2× more reduction) loses another ~0.5 perplexity. The marginal quality cost grows as you compress more aggressively.

This is why the sweet spot is GQA with 4-8 groups: you get most of MQA's benefit (close to 8x KV reduction) without paying MQA's quality cost. The quality vs cost tradeoff is most favorable in this middle range.

Why does the quality degradation accelerate near MQA? Hypothesis: a small number of KV "channels" (groups) is enough to capture the dominant attention patterns. Going below ~4 starts to compress the model's ability to differentiate between truly distinct attention modes.

## How to pick the group count

For new architectures, the rule of thumb is:
- 4 or 8 groups is the sweet spot for most models.
- Larger groups (`g = 16` or more) for very large models where capacity matters and the KV saving is less proportionally important.
- Smaller groups (`g = 2`) for very inference-heavy deployments.

Llama 3 8B uses GQA-8 (32 heads, 8 KV groups). Llama 3 70B uses GQA-8. Mistral 7B uses GQA-8. Qwen 2.5 uses GQA-8 / GQA-2 depending on the size variant. The convergence on `g=8` is not coincidence; it's the empirical sweet spot multiple teams found independently.

## GQA from an MHA checkpoint

Like MQA, you can convert an MHA-trained model to GQA. The recipe:

1. For each group, *mean-pool* the original `n_heads / g` heads' K and V projections.
2. The output is a `g`-head K/V model.
3. Brief fine-tune (typically 5-10% of the original training cost) to recover quality.

The Llama 2 7B → Llama 2 7B-GQA conversion follows this recipe; the resulting model has comparable downstream task quality to the original with 8× smaller KV cache.

This conversion path is useful when you have an MHA model and want to deploy it more efficiently without retraining from scratch.

## A subtle implementation point

GQA requires the attention kernel to handle the "broadcast K and V across head groups" operation. Native PyTorch's `torch.nn.functional.scaled_dot_product_attention` supports this via the `enable_gqa=True` flag (PyTorch 2.3+).

For older PyTorch or custom kernels, you implement the broadcast explicitly:

```python
# k: [B, g, N, d_head] → [B, h, N, d_head]
n_per_group = n_heads // g
k = k.repeat_interleave(n_per_group, dim=1)
v = v.repeat_interleave(n_per_group, dim=1)
# Then standard MHA-style attention.
```

`repeat_interleave` is conceptually a copy but PyTorch implements it efficiently. For production inference with FlashAttention, the FA kernel handles GQA natively without the explicit repeat — it indexes into the K/V groups directly during the streaming attention computation.

## What you should believe after this lesson

Three sentences:

**1. GQA is the MHA↔MQA spectrum**, with `g` group parameters — `g = n_heads` is MHA, `g = 1` is MQA, intermediate `g` (typically 4 or 8) is the sweet spot. Llama 2/3, Mistral, Qwen, and most modern LLMs use GQA with `g = 8`.

**2. The KV cache scales linearly with `g`**, so GQA-8 is 4× smaller than MHA for a 32-head model. The quality cost is mild at `g = 8` and grows rapidly as you go below ~4. Empirically `g = 8` is the empirically discovered sweet spot.

**3. You can convert MHA → GQA via mean-pooling K/V across heads + brief fine-tune** — useful for retrofitting existing checkpoints into more inference-friendly architectures.

## Hands-on (at home)

Implement GQA and compare to MHA and MQA.

```python
# gqa.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model, n_heads, n_kv_groups, causal=True):
        super().__init__()
        assert d_model % n_heads == 0
        assert n_heads % n_kv_groups == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_groups = n_kv_groups
        self.d_head = d_model // n_heads
        self.heads_per_group = n_heads // n_kv_groups
        self.causal = causal
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        # K and V project to (n_kv_groups * d_head) instead of d_model.
        self.W_k = nn.Linear(d_model, n_kv_groups * self.d_head, bias=False)
        self.W_v = nn.Linear(d_model, n_kv_groups * self.d_head, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        B, N, D = x.shape
        q = self.W_q(x).view(B, N, self.n_heads, self.d_head).transpose(1, 2)
        k = self.W_k(x).view(B, N, self.n_kv_groups, self.d_head).transpose(1, 2)
        v = self.W_v(x).view(B, N, self.n_kv_groups, self.d_head).transpose(1, 2)
        # Broadcast K, V across heads_per_group.
        k = k.repeat_interleave(self.heads_per_group, dim=1)
        v = v.repeat_interleave(self.heads_per_group, dim=1)
        # Standard MHA-style attention.
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        if self.causal:
            mask = torch.tril(torch.ones(N, N, device=x.device)) == 0
            scores = scores.masked_fill(mask, float('-inf'))
        weights = F.softmax(scores, dim=-1)
        out = weights @ v
        out = out.transpose(1, 2).contiguous().view(B, N, D)
        return self.W_o(out)

# Compare across configurations.
D, h = 512, 8
for g in [8, 4, 2, 1]:
    m = GroupedQueryAttention(D, h, g)
    params = sum(p.numel() for p in m.parameters())
    kv_per_token = 2 * g * (D // h) * 2  # FP16
    name = {8: "MHA equivalent", 4: "GQA-4", 2: "GQA-2", 1: "MQA equivalent"}[g]
    print(f"{name:20s} (g={g}): {params:>8,} params, {kv_per_token:5d} bytes/token KV")
```

You'll see the KV-per-token scales linearly with `g`; parameter count drops as `g` decreases.

For perplexity comparison, train all configurations to convergence on the same dataset (use a small char-level Shakespeare for speed) — you'll observe the quality vs KV tradeoff directly.

## Further reading

- "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints" (Ainslie et al, 2023) — the foundational paper.
- "Llama 2: Open Foundation and Fine-Tuned Chat Models" (Touvron et al, 2023) — Llama 2 70B adopts GQA; the report discusses the choice.
- "Mistral 7B" (Jiang et al, 2023) — Mistral also uses GQA-8.
- "Llama 3 Herd of Models" (Meta, 2024) — for the larger-scale GQA story.

Next lesson: **Multi-Head Latent Attention (MLA).** DeepSeek's variant that takes the KV-compression idea further by projecting K and V into a low-rank latent space, achieving compression beyond what GQA can. We look at the math, the inference-time gymnastics, and the place MLA holds in the 2026 attention landscape.
