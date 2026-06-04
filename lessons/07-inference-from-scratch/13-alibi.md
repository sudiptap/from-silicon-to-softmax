---
title: "Lesson 13 — ALiBi: Attention with Linear Biases"
date: "2026-06-04"
module: "inference-from-scratch"
order: 13
tags: ["positional-encoding", "alibi", "extrapolation", "linear-bias"]
author: "Sudipta Pathak"
prerequisites: ["12-relative-position"]
---

# Lesson 13 — ALiBi: Attention with Linear Biases

## Why this lesson exists

ALiBi (Press et al, 2022) is a deliciously simple positional encoding scheme. Replace the entire learned-bias-per-bucket machinery of T5 with: add `-m × (i - j)` to attention scores, where `m` is a fixed per-head slope. No learnable parameters at all. The attention naturally biases toward attending to recent tokens; the bias falls off linearly with distance.

The surprising result: ALiBi extrapolates dramatically better than every preceding scheme. A model trained on 1K context can run reasonably at 16K context with ALiBi; the absolute and T5-style alternatives degrade much sooner.

This lesson is what ALiBi is, why it works, and the question that eventually moved most production LLMs to RoPE despite ALiBi's elegance.

The lesson is reading. The Hands-on implements ALiBi attention and tests extrapolation behavior.

## The mechanism

The attention score with ALiBi:

```
score_ij = (Q_i K_j^T) / sqrt(d) + m × (j - i)
```

where:
- `(j - i)` is the relative position offset (negative for causal attention, since `j ≤ i`).
- `m` is a per-head fixed slope (no learning).
- The combined effect: a more-recent token (small `|j - i|`) gets less of a bias subtraction; a far-back token gets a large negative bias that competes with the content-similarity score.

For causal attention, `j ≤ i`, so `(j - i) ≤ 0`. The bias is non-positive: `0` at the diagonal (`j = i`), more negative further back. Softmax weights tokens close to `i` more than tokens far from `i`, with the magnitude controlled by `m`.

Different heads have different slopes. ALiBi assigns slopes geometrically:

```
m_h = 2^(-8h/n_heads)   # for head h ∈ {1, ..., n_heads}
```

For 8 heads: `m = 1/2, 1/4, 1/8, 1/16, 1/32, 1/64, 1/128, 1/256`. The "low-m" heads can see far back (mild bias); the "high-m" heads focus on local context (strong bias).

That's it. No learned parameters. Just a fixed per-head slope and a bias that's a linear function of the position offset.

## Why it works

Two intuitions:

**1. Recency bias is sensible.** Natural language has the property that recent tokens are usually more relevant than distant ones. ALiBi bakes this prior directly into the attention mechanism. The model doesn't need to learn it from data; it gets it for free.

**2. Multiple heads with different slopes cover the range.** One head with a steep slope captures very-local syntax. Another head with a gentle slope captures longer-range references. Together they span the distance spectrum.

The extrapolation property comes from the linearity: at any distance `(j - i)`, the bias is well-defined (just multiply by `m`). The model wasn't trained at distance 8192, but distance 8192's bias is just `m × 8192` — a deterministic value the same as the in-training distances scaled up. The model's other parameters were trained against a *consistent* recency bias; extending the bias linearly extends the same prior.

Compare to absolute encodings: the encoding pattern at position 8192 looks fundamentally different from the patterns the model saw during training (different sinusoidal phase, or no learned vector at all). Compare to T5 bias: the bucketization means distances at 8192 fall into a bucket that's far from any bucket the model saw clearly.

ALiBi's extrapolation simply works because the bias mechanism is naturally length-invariant.

## Empirical results

The ALiBi paper's key result: a model trained on 1024-token context handles 16384-token inference with only modest perplexity degradation. Absolute encodings and T5 bias fail much earlier.

Production adoption:
- **MPT** (MosaicML, 2023): 7B and 30B models used ALiBi as their positional encoding.
- **BLOOM** (BigScience, 2022): 176B used ALiBi.
- Several smaller research models.

Then the trend shifted. Most flagship 2024-2026 LLMs adopted RoPE instead. Why?

## Why RoPE won over ALiBi

ALiBi's elegance is undeniable, but RoPE has properties that, on net, the field preferred:

**1. RoPE preserves token-level content distinctions across positions.** ALiBi's bias just makes distant tokens less attended-to, but doesn't differentiate them by *which* distant position. RoPE encodes the distance into the actual dot-product computation in a way that preserves more positional information.

**2. RoPE handles bidirectional patterns better.** ALiBi's monotonic recency bias is well-suited for causal attention but awkward for tasks where future or non-local information matters (encoder-only, cross-attention).

**3. RoPE composes cleanly with KV cache.** The position-dependent rotation is a property of K, so cached K can be rotated once and reused. ALiBi's bias depends on the relative position of Q and K, requiring per-step bias computation. Tiny difference; mattered for some implementations.

**4. RoPE has better long-context scaling tricks.** Linear, NTK-aware, YaRN — these scaling methods (Lesson 15) let RoPE extrapolate 4-8× the training length. ALiBi's natural extrapolation is good but caps out earlier.

**5. Ecosystem inertia.** Once Llama 2 used RoPE, the world rebuilt around it. Tooling for RoPE matured faster.

ALiBi remains a great pedagogical example of "simpler often works" and is still used in some new architectures (notably some Chinese open-weights families). For new general-purpose LLMs in 2026, RoPE is the default.

## When to consider ALiBi

A few scenarios where ALiBi is still the right choice:

- **You want a simple-to-implement, no-learned-parameters positional encoding** for an experiment or research model.
- **You're working with a fixed architecture that already uses ALiBi** and want to avoid changing it.
- **Extrapolation matters more than absolute quality** — ALiBi's "no scaling tricks needed" property is a real practical win in some deployment scenarios.

For new flagship LLMs: use RoPE.

## What you should believe after this lesson

Three sentences:

**1. ALiBi adds a linear bias to attention scores**: `score += m_head × (j - i)` with `m_head` a fixed per-head slope from a geometric progression. Zero learnable parameters; dramatically better extrapolation than absolute or T5-style encodings.

**2. ALiBi works because recency bias is a sensible prior** and because the linear bias is length-invariant — distances beyond training are just extrapolated by the same linear function. The model's other parameters were trained against a consistent recency bias.

**3. RoPE eventually won** despite ALiBi's elegance, mainly because RoPE preserves more position information in the attention scores, composes cleanly with KV cache, and has well-established long-context scaling methods (Lesson 15). ALiBi persists in some models but is no longer the default for new flagships.

## Hands-on (at home)

Implement ALiBi attention.

```python
# alibi.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def get_alibi_slopes(n_heads):
    """ALiBi's geometric slope schedule for n_heads heads."""
    def get_slopes_power_of_2(n):
        start = 2**(-2**-(math.log2(n)-3))
        ratio = start
        return [start * ratio**i for i in range(n)]
    if math.log2(n_heads).is_integer():
        return get_slopes_power_of_2(n_heads)
    else:
        # For non-power-of-2 heads, interpolate.
        closest_power = 2**math.floor(math.log2(n_heads))
        slopes = get_slopes_power_of_2(closest_power)
        extra = get_slopes_power_of_2(2*closest_power)[0::2][:n_heads-closest_power]
        return slopes + extra

class ALiBiAttention(nn.Module):
    def __init__(self, d_model, n_heads, causal=True):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.W_qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        self.register_buffer('slopes', torch.tensor(get_alibi_slopes(n_heads)))
        self.causal = causal

    def alibi_bias(self, N, device):
        # [n_heads, N, N] = slope * (j - i) for each head.
        i = torch.arange(N, device=device).unsqueeze(1).float()
        j = torch.arange(N, device=device).unsqueeze(0).float()
        relative = j - i  # [N, N]; negative below diagonal
        # Per-head: -slope * |relative| (for causal: j <= i, so |relative| = i - j).
        # The original ALiBi formula: bias = -slope * (i - j) for j <= i.
        return -self.slopes.view(-1, 1, 1) * (i - j).unsqueeze(0)  # [n_heads, N, N]

    def forward(self, x):
        B, N, D = x.shape
        q, k, v = self.W_qkv(x).chunk(3, dim=-1)
        q = q.view(B, N, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, N, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, N, self.n_heads, self.d_head).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)  # [B, h, N, N]
        # Add ALiBi bias.
        bias = self.alibi_bias(N, x.device).unsqueeze(0)  # [1, h, N, N]
        scores = scores + bias
        if self.causal:
            mask = torch.tril(torch.ones(N, N, device=x.device)) == 0
            scores = scores.masked_fill(mask, float('-inf'))
        weights = F.softmax(scores, dim=-1)
        out = (weights @ v).transpose(1, 2).contiguous().view(B, N, D)
        return self.W_o(out)

# Inspect slopes.
slopes = get_alibi_slopes(8)
print(f"ALiBi slopes for 8 heads: {slopes}")
# The smallest slope means the head can attend far back; the largest is local.

# Run it.
torch.manual_seed(0)
m = ALiBiAttention(d_model=64, n_heads=8)
x = torch.randn(1, 32, 64)
y = m(x)
print(f"output shape: {y.shape}")

# Visualize the bias for one head.
bias = m.alibi_bias(16, 'cpu')
print(f"ALiBi bias for head 0 (slope {slopes[0]:.4f}):")
print(bias[0].numpy())
# Values are 0 on diagonal, increasingly negative below.
```

For extrapolation testing on real models: ALiBi-trained models (MPT-7B, MPT-30B available on HF) can be loaded and tested at context lengths well above their nominal training context. Compare perplexity at 1K, 4K, 16K and observe the gentler degradation than absolute-encoded models.

## Further reading

- "Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation" (Press et al, 2022) — the original ALiBi paper.
- "MPT-7B: A Commercially Usable LLM" (MosaicML, 2023) — production deployment of ALiBi at moderate scale.
- "BLOOM: A 176B-Parameter Open-Access Multilingual Language Model" (BigScience, 2022) — large-scale ALiBi.
- The RoPE paper (next lesson) — discusses ALiBi in comparison.

Next lesson: **RoPE — Rotary Position Embeddings.** The positional encoding scheme that became the 2024-2026 default. We derive RoPE from scratch: how rotating Q and K by position-dependent angles encodes relative position in the dot-product score, why it has cleaner mathematical properties than ALiBi, and the implementation details.
