---
title: "Lesson 12 — Relative Position Encodings"
date: "2026-06-04"
module: "inference-from-scratch"
order: 12
tags: ["positional-encoding", "relative", "shaw", "t5-bias", "extrapolation"]
author: "Sudipta Pathak"
prerequisites: ["11-absolute-encodings"]
---

# Lesson 12 — Relative Position Encodings

## Why this lesson exists

Absolute encodings give each position a unique fingerprint. Relative encodings (Shaw et al, 2018; T5, 2019) take a different angle: encode the *distance* between two tokens, not their absolute positions. Instead of "I am at position 7," the encoding says "you are 3 positions before me." When you change context length, the relative-distance vocabulary doesn't change — distances 0, 1, 2, ... -1, -2, ... still mean the same thing.

This shift is the key step toward extrapolation: a model trained on distances up to ±N can in principle handle longer sequences, as long as the relative-distance space is well-defined for the longer distances too.

This lesson is the two foundational relative-encoding approaches: Shaw et al's original formulation (positional information added to keys and values) and T5's simplified bias (positional information added directly to attention scores). The simpler T5 approach is the one that mostly took over.

The lesson is reading. The Hands-on implements T5-style relative bias attention.

## Shaw et al's relative encoding

The original (2018): inject relative-position vectors into both the keys and values of attention. The attention score becomes:

```
score_ij = Q_i (K_j + a_{i-j})^T / sqrt(d)
```

where `a_{i-j}` is a learnable vector that depends on the relative distance `i - j`.

Similarly for the value computation:

```
output_i = sum_j weight_ij × (V_j + b_{i-j})
```

The vectors `a_k` and `b_k` for each distance `k = -(N-1), ..., N-1` are learned parameters. To keep the parameter count bounded, distances beyond a clip range `[-K, +K]` get clipped to `±K`.

The mechanism is principled — relative positions are first-class members of the attention computation — but the implementation is messy. You need to maintain per-distance learnable vectors and handle the clipping. It also adds parameters proportional to the clip range.

## T5's relative attention bias

T5 (Raffel et al, 2019) simplified to a key insight: the position information doesn't need to be a full vector added to K and V. It can be a *scalar bias* added directly to the attention score:

```
score_ij = (Q_i K_j^T) / sqrt(d) + b_{i-j}
```

`b_{i-j}` is a scalar per relative distance, per head. The total parameter count: `num_heads × num_distances`.

T5 bucketizes the distances: distances 0-7 each get their own bucket, then 8-15 share a bucket, 16-31 share another, and so on (logarithmic spacing). This caps the number of distinct biases per head at ~32, regardless of sequence length.

Implementation:

```python
def t5_bias(N, n_heads, bucket_count=32):
    # Compute distance buckets for each pair of positions.
    i = torch.arange(N).unsqueeze(1)
    j = torch.arange(N).unsqueeze(0)
    distance = i - j  # [N, N]
    buckets = compute_t5_bucket(distance, bucket_count)
    # bias_table: [n_heads, bucket_count] learnable parameters
    bias = bias_table[:, buckets]  # [n_heads, N, N]
    return bias

def compute_t5_bucket(distance, n_buckets):
    # First 16 buckets are exact (distances 0-15); next 16 are log-spaced up to ~256.
    ...  # see T5 reference implementation
```

The attention computation:

```python
scores = (Q @ K^T) / sqrt(d) + t5_bias  # add bias before softmax
weights = softmax(scores, dim=-1)
out = weights @ V
```

Properties:
- **Few parameters**: ~32 biases per head, vs Shaw's `2 × clip_range × d` parameters.
- **Position information added to scores, not vectors**: simpler implementation, slightly less expressive.
- **Extrapolation is reasonable**: the bucketization means new distances at long context fall into existing buckets (the log-spaced "far-away" buckets), so the model has some learned response.

T5 bias extrapolates better than absolute encodings but worse than ALiBi (Lesson 13) or RoPE (Lesson 14).

## The extrapolation question

A model trained with T5 bias on context length N can handle longer contexts in two ways:
1. **In-distribution**: distances within the trained range work directly.
2. **Out-of-distribution**: distances larger than seen during training fall into the long-distance bucket; the bias for that bucket was learned but on shorter distances. Quality degrades but doesn't collapse.

In practice, T5-style models extrapolate maybe 1.5-2× their trained context with reasonable quality. Not as good as ALiBi (which extrapolates much further) or RoPE with scaling (which can extrapolate 4-8×), but better than absolute encodings.

## When relative encodings won, when they lost

The win: T5 and BART used relative encodings successfully. The simplicity (T5 bias) and the extrapolation improvement made them attractive over the absolute encodings of BERT.

The eventual loss: when ALiBi appeared (Lesson 13) and especially when RoPE (Lesson 14) emerged with even cleaner mathematical properties and better extrapolation, T5-style relative encodings stopped being the default.

In 2026, T5 bias is not in any flagship LLM. Llama 3, Mistral, Qwen, Gemma, Phi, DeepSeek — all use RoPE.

T5 bias does persist in:
- T5 derivatives (FLAN-T5, T5x).
- Some encoder-decoder models for translation.
- Some research papers.

## What relative encodings taught us

The conceptual shift from absolute to relative was important, even though the specific implementations weren't the long-term winners:

- Position is fundamentally a *pairwise* concept. The dot-product between two tokens cares about their relative position more than their absolute positions.
- Encoding position via an additive bias on the score (T5) is simpler than via additions to K, V (Shaw).
- A bounded set of position "buckets" gives reasonable extrapolation without unbounded parameters.

All of these insights carried into RoPE.

## What you should believe after this lesson

Three sentences:

**1. Relative encodings encode the *distance* between two tokens** rather than absolute positions. This shifts the extrapolation problem from "what does position 8192 look like" to "what does distance 4096 look like" — distances are bounded by the trained range but generalize better than absolute positions.

**2. T5's bias** is the simplified form that took over from Shaw's original: a per-head per-distance-bucket scalar added directly to attention scores. ~32 biases per head; trivial parameter overhead; extrapolation 1.5-2× the trained context.

**3. Relative encodings were a stepping stone**, not the destination. The insights (position as a pairwise concept; bounded parameter count; in-attention-score injection) carried into ALiBi and RoPE, which dominate in 2026.

## Hands-on (at home)

Implement T5-style relative bias attention.

```python
# t5_relative_bias.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class T5RelativeAttention(nn.Module):
    def __init__(self, d_model, n_heads, max_distance=128, n_buckets=32):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.W_qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        # Bias table: [n_heads, n_buckets] learnable parameters.
        self.relative_bias = nn.Embedding(n_buckets, n_heads)
        self.max_distance = max_distance
        self.n_buckets = n_buckets

    def compute_buckets(self, N, device):
        i = torch.arange(N, device=device).unsqueeze(1)
        j = torch.arange(N, device=device).unsqueeze(0)
        relative_position = i - j  # [N, N]
        # T5-style bucket assignment: first n_buckets/2 buckets for small distances,
        # remaining for log-spaced large distances. Causal mask: only positive distances.
        ret = 0
        n = -relative_position  # for causal we look at j → i with j ≤ i
        n = n.clamp(min=0)
        max_exact = self.n_buckets // 2
        is_small = n < max_exact
        val_if_large = max_exact + (
            torch.log(n.float() / max_exact + 1e-6) / math.log(self.max_distance / max_exact)
            * (self.n_buckets - max_exact)
        ).long()
        val_if_large = val_if_large.clamp(max=self.n_buckets - 1)
        bucket = torch.where(is_small, n, val_if_large)
        return bucket  # [N, N], values in [0, n_buckets)

    def forward(self, x):
        B, N, D = x.shape
        q, k, v = self.W_qkv(x).chunk(3, dim=-1)
        q = q.view(B, N, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, N, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, N, self.n_heads, self.d_head).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)  # [B, h, N, N]
        # Add T5 bias.
        buckets = self.compute_buckets(N, x.device)
        bias = self.relative_bias(buckets).permute(2, 0, 1)  # [n_heads, N, N]
        scores = scores + bias.unsqueeze(0)  # [B, h, N, N]
        # Causal mask.
        mask = torch.tril(torch.ones(N, N, device=x.device)) == 0
        scores = scores.masked_fill(mask, float('-inf'))
        weights = F.softmax(scores, dim=-1)
        out = (weights @ v).transpose(1, 2).contiguous().view(B, N, D)
        return self.W_o(out)

# Test it runs.
torch.manual_seed(0)
m = T5RelativeAttention(d_model=64, n_heads=4)
x = torch.randn(1, 16, 64)
y = m(x)
print(f"output shape: {y.shape}")

# Inspect the bucket assignment.
buckets = m.compute_buckets(16, 'cpu')
print("Distance buckets (i=row, j=col):")
print(buckets.numpy())
# You should see small buckets near the diagonal (small distances), larger buckets
# off-diagonal (large distances).
```

The bucket-based bias gives the model a coarse-grained position signal — within 16 positions it's nearly token-level; beyond that the buckets are logarithmic. This is what enables the modest extrapolation.

## Further reading

- "Self-Attention with Relative Position Representations" (Shaw et al, 2018) — the original relative-encoding paper.
- "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer" (Raffel et al, 2019) — T5; introduces the relative attention bias.
- "Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation" (Press et al, 2022) — the ALiBi paper, our next lesson; compares against T5 bias and Shaw et al.

Next lesson: **ALiBi — Attention with Linear Biases.** A radically simpler relative-encoding scheme: just add a linear (in distance) bias to attention scores, with a fixed per-head slope. Zero learnable parameters, dramatically better extrapolation than T5 bias. The lesson covers what ALiBi is, why it works, and why it nonetheless lost to RoPE.
