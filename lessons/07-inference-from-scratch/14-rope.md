---
title: "Lesson 14 — RoPE: Rotary Position Embeddings"
date: "2026-06-04"
module: "inference-from-scratch"
order: 14
tags: ["positional-encoding", "rope", "rotary", "relative-position", "llama"]
author: "Sudipta Pathak"
prerequisites: ["13-alibi"]
---

# Lesson 14 — RoPE: Rotary Position Embeddings

## Why this lesson exists

RoPE (Rotary Position Embeddings, Su et al, 2021, popularized by Llama) is the positional encoding scheme that dominates production LLMs in 2026. It's used by Llama 2/3, Mistral, Qwen, Gemma, Phi, DeepSeek, and almost every other open-weights LLM.

The mechanism: rotate the query and key vectors by a position-dependent angle before computing the attention dot product. The rotation has the magical property that the dot product `Q_i · K_j` *automatically* depends only on the relative position `(i - j)`, not on the absolute positions of either.

This lesson is the derivation, the implementation, and what makes RoPE the right combination of mathematical properties for the LLM context.

The lesson is reading. The Hands-on implements RoPE from scratch.

## The key insight

Suppose we want the dot product `Q_i · K_j` to encode the relative position `(i - j)`. We could just add a relative-position bias to the dot product (T5 bias). RoPE achieves something stronger: the *dot product itself* is a function of the relative position.

The trick: rotate `Q_i` by angle `θ × i` and `K_j` by angle `θ × j`. The rotated dot product satisfies:

```
rotate(Q, θ × i) · rotate(K, θ × j) = (Q · K) × cos(θ × (i - j)) + (some sign-flipped term) × sin(θ × (i - j))
```

The right-hand side depends only on `(i - j)`, not on `i` or `j` individually. The absolute position information is removed by the rotation; relative position is what survives.

This is the elegant property RoPE exploits: a 2D rotation by `θ × (j - i)` is what you get from rotating one vector by `θ × i` and another by `θ × j` and dotting them.

## The 2D case (the building block)

For 2-dimensional vectors `Q` and `K`, rotation by angle `α` is:

```
R(α) = [cos α   -sin α]
       [sin α    cos α]
```

Rotated dot product:

```
(R(α_q) Q) · (R(α_k) K) = Q · R(α_k - α_q) K = Q · R(α_k - α_q) K
```

The product of the two rotation matrices reduces to the rotation by their difference: `R(α_k - α_q)`. So the rotated dot product depends only on `α_k - α_q`.

If we set `α_i = θ × i` for position `i`, then `α_k - α_q = θ × (j - i)` — exactly the relative position.

## Extending to higher dimensions

For a `d`-dim vector, split into `d/2` pairs of consecutive elements. Each pair gets its own rotation angle, depending on both the position and the pair's index.

Specifically, for the `i`-th token and the `k`-th pair (with `k = 0, 1, ..., d/2 - 1`), rotate by:

```
θ_k = base^(-2k/d)
α = i × θ_k
```

where `base` is typically 10000 (matching the sinusoidal encoding's base).

Different pairs rotate at different rates:
- Pair 0 (`θ_0 = 1`): rotates by 1 radian per position. Fast rotation — captures local position.
- Pair `d/2 - 1` (`θ_{d/2-1} = base^(-2(d/2-1)/d) ≈ 1/base`): rotates very slowly. Captures global position.

This is the same exponential frequency spacing as the original sinusoidal encoding, repurposed for rotations.

## The implementation

```python
def apply_rope(q, k, position_ids, base=10000):
    # q, k: [B, h, N, d_head]
    # position_ids: [N] tensor of token positions [0, 1, 2, ..., N-1]
    d = q.shape[-1]
    # Compute frequencies for each pair.
    inv_freq = 1.0 / (base ** (torch.arange(0, d, 2).float() / d))  # [d/2]
    # Outer product: [N, d/2] of position × inv_freq.
    freqs = torch.einsum('n,k->nk', position_ids.float(), inv_freq)  # [N, d/2]
    # cos and sin: [N, d/2].
    cos = freqs.cos()
    sin = freqs.sin()
    # Replicate to [N, d]: [cos0, cos0, cos1, cos1, ..., cos_{d/2}, cos_{d/2}].
    # (This is the convention; some implementations interleave differently.)
    cos = cos.repeat_interleave(2, dim=-1)
    sin = sin.repeat_interleave(2, dim=-1)
    # Apply rotation: for each pair (q_2k, q_2k+1):
    # q_2k' = q_2k * cos - q_2k+1 * sin
    # q_2k+1' = q_2k * sin + q_2k+1 * cos
    def rotate_half(x):
        x1, x2 = x[..., 0::2], x[..., 1::2]
        return torch.stack([-x2, x1], dim=-1).flatten(-2)
    q_rotated = q * cos + rotate_half(q) * sin
    k_rotated = k * cos + rotate_half(k) * sin
    return q_rotated, k_rotated
```

In practice, RoPE is applied per attention head (the rotations operate on the head's dimension, not the full model dimension). The cos and sin tables are precomputed once (they only depend on the position) and reused across all layers.

## Why RoPE became the default

Several properties that combined to win:

**1. Encodes relative position in the dot product itself.** The score `Q_i · K_j` directly carries position information; the model uses position information naturally through the attention computation, not as an external bias.

**2. No learnable parameters.** Like ALiBi, no parameter cost.

**3. Extrapolation: not great by default, but excellent with scaling.** Vanilla RoPE doesn't extrapolate beyond its training length well (the rotations at long positions look "novel" to the model). But with the scaling tricks (Lesson 15 — Linear, NTK-aware, YaRN, LongRoPE), RoPE can extrapolate 4-8× cleanly, better than ALiBi.

**4. Composes well with KV cache.** Once K is rotated, it's stored in the cache with its rotation applied. Future Q's at any position can compute the right rotated dot product against cached K. No per-step extra work.

**5. Numerically clean.** The rotation is a unitary operation — it doesn't change the norm of Q or K. The attention scores stay in a well-behaved range.

**6. Empirically better than alternatives at scale.** When models scaled to 7B, 13B, 70B+ parameters, RoPE consistently outperformed competing positional encodings.

## A subtle implementation detail

Two interleaving conventions for RoPE:
- **Pair-interleaved** (`[a, b, a, b, ...]`): pairs are at positions `(2k, 2k+1)`. Used in Meta's reference Llama implementation.
- **Half-interleaved** (`[a, a, ..., b, b, ...]`): first half is one part of each pair, second half is the other. Used in some HuggingFace implementations.

The two are mathematically equivalent (just a permutation of the dimensions). When loading weights between implementations with different conventions, you need to reorder the K projection's output dimensions. This is the source of many "I converted a Llama checkpoint and it gives garbage" issues.

## RoPE vs ALiBi, side by side

| Property | RoPE | ALiBi |
| -------- | ---- | ----- |
| Learnable params | 0 | 0 |
| Mechanism | Rotation of Q, K | Bias on attention score |
| Position info location | In dot product | Added to score post-dot |
| Extrapolation (vanilla) | Modest | Good |
| Extrapolation (with scaling) | Excellent (YaRN, etc.) | Limited |
| KV cache integration | Clean (K rotated once at insert) | Clean (bias depends only on i, j) |
| Mathematical elegance | High | High |
| 2026 adoption | Dominant | Niche |

Both are elegant. RoPE's combination of (better scaling potential + better quality at scale) carried the day.

## What you should believe after this lesson

Three sentences:

**1. RoPE rotates Q and K by position-dependent angles**, with the consequence that the dot product `Q_i · K_j` automatically depends only on the relative position `(i - j)`. The relative-position information is built into the attention computation, not added as an external bias.

**2. RoPE has zero learnable parameters** and clean KV cache integration (K is rotated once on insert, reused indefinitely). With long-context scaling tricks (Lesson 15) it extrapolates 4-8× the training length.

**3. RoPE is the 2026 default** — Llama, Mistral, Qwen, Gemma, Phi, DeepSeek all use it. The combination of mathematical elegance + empirical performance at scale + ecosystem maturity made it the standard despite ALiBi's earlier promise.

## Hands-on (at home)

Implement RoPE from scratch.

```python
# rope.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def precompute_freqs_cis(d, max_len, base=10000):
    """Precompute the cos and sin tables for RoPE."""
    inv_freq = 1.0 / (base ** (torch.arange(0, d, 2).float() / d))  # [d/2]
    pos = torch.arange(max_len).float()  # [max_len]
    freqs = torch.outer(pos, inv_freq)  # [max_len, d/2]
    return freqs.cos(), freqs.sin()  # each [max_len, d/2]

def apply_rope(x, cos, sin):
    """Apply RoPE to x. x: [..., N, d]; cos, sin: [N, d/2]."""
    # Split into pairs: x_even = x[..., 0::2], x_odd = x[..., 1::2].
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    # Rotate: new_even = even*cos - odd*sin; new_odd = even*sin + odd*cos.
    new_even = x_even * cos - x_odd * sin
    new_odd = x_even * sin + x_odd * cos
    # Interleave back.
    out = torch.stack([new_even, new_odd], dim=-1).flatten(-2)
    return out

class RoPEAttention(nn.Module):
    def __init__(self, d_model, n_heads, max_len=2048):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.W_qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        cos, sin = precompute_freqs_cis(self.d_head, max_len)
        self.register_buffer('cos', cos)
        self.register_buffer('sin', sin)

    def forward(self, x):
        B, N, D = x.shape
        q, k, v = self.W_qkv(x).chunk(3, dim=-1)
        q = q.view(B, N, self.n_heads, self.d_head).transpose(1, 2)  # [B, h, N, d_head]
        k = k.view(B, N, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, N, self.n_heads, self.d_head).transpose(1, 2)
        # Apply RoPE to q and k.
        cos = self.cos[:N].view(1, 1, N, -1)
        sin = self.sin[:N].view(1, 1, N, -1)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        # Standard attention with causal mask.
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        mask = torch.tril(torch.ones(N, N, device=x.device)) == 0
        scores = scores.masked_fill(mask, float('-inf'))
        weights = F.softmax(scores, dim=-1)
        out = (weights @ v).transpose(1, 2).contiguous().view(B, N, D)
        return self.W_o(out)

# Test the relative-position property.
torch.manual_seed(0)
attn = RoPEAttention(d_model=64, n_heads=4)

# Two sequences with the same relative position structure should have similar attention patterns.
# This is hard to verify directly because content also influences attention.
# Easier verification: the same Q,K dotted at different absolute positions but
# same relative position give the same score.
x = torch.randn(1, 8, 64)
y1 = attn(x)
# Shift the sequence (pad with random tokens at start; attention scores should shift consistently).
x_shifted = torch.cat([torch.randn(1, 4, 64), x], dim=1)
y2 = attn(x_shifted)
print(f"Output shape: {y1.shape}")
print(f"Shifted shape: {y2.shape}")
# Note: outputs won't match exactly because the new prefix tokens influence attention,
# but the RoPE rotation is correctly applied per position.
```

For testing extrapolation specifically: train a small transformer with RoPE on context 512, then run inference at context 1024, 2048, ... and watch perplexity. With vanilla RoPE it degrades; with the YaRN-scaled version (Lesson 15) the degradation is much milder.

## Further reading

- "RoFormer: Enhanced Transformer with Rotary Position Embedding" (Su et al, 2021) — the original RoPE paper with the full derivation.
- "Llama 2: Open Foundation and Fine-Tuned Chat Models" (Touvron et al, 2023) — production-scale RoPE deployment.
- "A Mathematical Framework for Rotary Position Embeddings" (various 2023-2024 follow-ups) — for deeper theoretical analysis.
- Eleuther AI's GPT-NeoX implementation (early open-source large-scale RoPE).

Next lesson: **Long-context RoPE scaling.** Vanilla RoPE doesn't extrapolate; the scaling tricks (Linear, NTK-aware, YaRN, LongRoPE) make it extrapolate well. We cover what each scaling does, what they trade off, and how Llama 3.1 / Mistral / Qwen handle long context.
