---
title: "Lesson 9 — Linear and Sub-Quadratic Attention"
date: "2026-06-04"
module: "inference-from-scratch"
order: 9
tags: ["attention", "linear-attention", "performer", "linformer", "mamba", "rwkv"]
author: "Sudipta Pathak"
prerequisites: ["08-flashattention"]
---

# Lesson 9 — Linear and Sub-Quadratic Attention

## Why this lesson exists

FlashAttention attacks `O(N²)` by changing the implementation. Linear and sub-quadratic attention attack it by changing the algorithm: replace the softmax-attention mechanism entirely with something that has lower asymptotic complexity. Performer uses random features to approximate the softmax kernel; Linformer projects K and V to a fixed low-rank subspace; the more recent state-space models (Mamba, RWKV) build attention-free architectures with constant-per-step compute.

For inference, these are interesting because they offer fundamentally different cost curves. For training, the established practice still uses softmax attention. Whether the linear variants ever replace softmax attention at scale is one of the open questions in 2026.

This lesson is a tour: what the major linear variants do, what they trade away, and the place they hold in the 2026 architecture landscape.

The lesson is reading. The Hands-on implements a basic linear-attention forward pass.

## The math that's possible

Standard softmax attention:

```
output = softmax(Q K^T / sqrt(d)) V
```

The `softmax(Q K^T)` is `O(N²)`. The fundamental observation: if we could replace `softmax(Q K^T)` with `φ(Q) ψ(K)^T` for some feature maps `φ` and `ψ`, then we could rearrange:

```
output = (φ(Q) ψ(K)^T) V = φ(Q) (ψ(K)^T V)
```

`ψ(K)^T V` is `O(N × d × d)` — linear in N. Multiplying by `φ(Q)` is another `O(N × d × d)`. Total: `O(N × d²)`. Linear in N.

The trick is finding `φ` and `ψ` such that `φ(Q) ψ(K)^T` approximates `softmax(Q K^T / sqrt(d))` — or such that the result is good enough that the model trains and inferences competitively.

## Performer (random features)

Performer (Choromanski et al, 2020) uses random Fourier features to approximate the softmax kernel:

```
softmax(Q K^T) ≈ φ(Q) ψ(K)^T
```

where `φ` and `ψ` are random projections that statistically approximate the exponential kernel. The Performer paper proves bounds on the approximation quality.

In practice, Performer's approximation is good enough for some tasks but causes meaningful quality loss on language modeling. It's faster than softmax attention at long N but rarely matches the quality.

## Linformer (low-rank K, V projection)

Linformer (Wang et al, 2020) takes a different angle: assume the attention matrix is low-rank, and project K and V down to a fixed sequence length `k << N`:

```
K' = K @ E  # [k, d], where E is [N, k]
V' = V @ F  # [k, d]
output = softmax(Q K'^T / sqrt(d)) @ V'
```

The attention is now `O(N × k)` instead of `O(N²)`. For `k = 256` and `N = 2048`, an 8× reduction.

The catch: the assumption that attention is low-rank is dubious. It holds for some tasks (mostly classification) and breaks for others (generation, where information from arbitrary past positions matters).

Linformer also has issues with autoregressive decoding — the K, V projection isn't easily extensible to streaming sequences.

## RetNet and the gated linear attention family

RetNet (Sun et al, 2023) and its descendants (GLA, Gated Linear Attention) use an explicit kernel decomposition with gating:

```
S_t = γ S_{t-1} + φ(K_t) V_t^T   # state update
y_t = ψ(Q_t) S_t                 # output
```

Where `γ` is a decay factor. The state `S_t` accumulates information from all previous tokens; the gating controls how fast old information decays.

Properties:
- Constant memory per token during inference (the state `S_t` is fixed size).
- Linear training cost via a parallel form (equivalent matrix computation that recovers the recurrent computation).
- Quality somewhat competitive with softmax attention; not yet matching the very best transformers at large scale.

The "parallel + recurrent" duality is the key engineering observation: during training, run as parallel matmul; during decoding, run as recurrent state update. The same model, two computation modes.

## Mamba (State Space Models)

Mamba (Gu & Dao, 2023) is the most prominent recent attention-free architecture. It uses a *selective state-space model* (SSM): a recurrent computation parameterized to selectively retain or forget information based on input.

Mathematically:

```
h_t = A_t h_{t-1} + B_t x_t   # hidden state update
y_t = C_t h_t                  # output
```

The matrices `A_t`, `B_t`, `C_t` are input-dependent (the "selective" part) and the state `h_t` is much smaller than the full token history. Linear in sequence length.

Mamba's claims:
- Comparable quality to transformers at small-to-mid scale (up to ~7B params).
- Faster inference: constant per-token cost (the state is fixed size).
- Linear training cost via a parallel algorithm based on prefix-sum on the matrices.

The reality in 2026:
- Mamba models exist (Mamba-2, Codestral Mamba, etc.) and run.
- Quality is competitive but hasn't decisively beaten softmax attention at scale.
- Inference throughput is genuinely better for long contexts where attention's KV-cache cost would dominate.
- Adoption is growing but most production LLMs are still attention-based.

## Hybrid architectures

A pattern that's emerging: hybrid models that interleave attention and state-space layers. Examples: Jamba (AI21 Labs, 2024), Zamba (Zyphra, 2024). The idea: get attention's quality on some layers and SSM's efficiency on others.

These often outperform pure attention on long-context throughput while keeping competitive quality. Whether the pattern dominates remains to be seen.

## Why softmax attention still wins for now

The empirical situation in mid-2026:

1. **Quality at scale**: pure softmax attention models (Llama 3.x, GPT-4-class) still represent the quality frontier at the largest scales. Linear-attention models haven't crossed the same scale at comparable quality.

2. **Ecosystem**: every tool, runtime, and downstream application assumes softmax attention. Linear variants force you to maintain a separate stack.

3. **Long-context tricks**: FlashAttention + sliding window + GQA + MLA collectively bring softmax attention's effective cost down enough that linear attention's theoretical advantage shrinks. For 32K-128K context, FlashAttention is fast enough.

4. **In-context learning**: there's some evidence that softmax attention's exact key-matching is what enables strong in-context learning, and linear variants lose some of this capability.

The case for linear attention strengthens at:
- **Very long context** (1M+ tokens) where even FlashAttention struggles.
- **Resource-constrained inference** where the linear-time cost matters more than the quality difference.
- **Specific tasks** where attention's specific behavior isn't load-bearing.

## What you should believe after this lesson

Three sentences:

**1. Linear attention replaces `softmax(QK^T)V` with `φ(Q) (ψ(K)^T V)`** — exploiting matrix associativity to drop complexity to `O(N)`. Performer uses random features; Linformer uses low-rank projection; RetNet and GLA use gated recurrent forms; Mamba builds an SSM-based attention-free architecture.

**2. The 2026 picture is "competitive but not dominant"**: linear/sub-quadratic models exist, work, and run faster at long context, but pure softmax-attention still represents the quality frontier at scale. Hybrid architectures (Jamba, Zamba) attempt to combine strengths.

**3. FlashAttention + sliding window + GQA + MLA collectively neutralize most of softmax attention's long-context disadvantage**, which is why production LLMs stayed softmax-based. The linear-attention frontier matters most for very-long-context (1M+) and resource-constrained edge inference.

## Hands-on (at home)

A basic linear-attention forward pass demonstrating the math rearrangement.

```python
# linear_attention.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class LinearAttention(nn.Module):
    """Performer-style linear attention with elu+1 feature map (simpler than random features)."""
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def feature_map(self, x):
        return F.elu(x) + 1  # ensures positivity; approximates softmax-like behavior

    def forward(self, x):
        B, N, D = x.shape
        q = self.W_q(x).view(B, N, self.n_heads, self.d_head).transpose(1, 2)  # [B, h, N, d]
        k = self.W_k(x).view(B, N, self.n_heads, self.d_head).transpose(1, 2)
        v = self.W_v(x).view(B, N, self.n_heads, self.d_head).transpose(1, 2)
        # Apply feature map.
        q_phi = self.feature_map(q)
        k_phi = self.feature_map(k)
        # The associativity trick: compute k_phi^T @ v first (small [d, d]), then q_phi @ that.
        kv = torch.einsum('bhnd,bhne->bhde', k_phi, v)  # [B, h, d, d]
        out = torch.einsum('bhnd,bhde->bhne', q_phi, kv)  # [B, h, N, d]
        # Normalization: each query position's denominator is sum of attention weights.
        z = torch.einsum('bhnd,bhd->bhn', q_phi, k_phi.sum(dim=2))  # [B, h, N]
        out = out / z.unsqueeze(-1).clamp(min=1e-6)
        out = out.transpose(1, 2).contiguous().view(B, N, D)
        return self.W_o(out)

# Compare scaling.
import time
D, h = 128, 4
lin = LinearAttention(D, h)

for N in [512, 2048, 8192]:
    x = torch.randn(1, N, D)
    # Warm.
    for _ in range(3): lin(x)
    t0 = time.time()
    for _ in range(5): lin(x)
    dt = (time.time() - t0) / 5
    print(f"N={N:5d}  linear attn: {dt*1000:.2f} ms")
```

You should see roughly linear scaling in `N` — doubling `N` roughly doubles the time. Compare to FlashAttention's near-linear behavior with much higher absolute throughput; for short-to-medium `N`, the constant factors of FlashAttention make it faster than this naive linear attention.

For the Mamba forward pass, the `mamba-ssm` package provides a reference implementation; the per-token cost is genuinely constant once the state is initialized.

## Further reading

- "Performer: Rethinking Attention with Performers" (Choromanski et al, 2020).
- "Linformer: Self-Attention with Linear Complexity" (Wang et al, 2020).
- "Retentive Network: A Successor to Transformer" (Sun et al, 2023) — RetNet.
- "Mamba: Linear-Time Sequence Modeling with Selective State Spaces" (Gu & Dao, 2023).
- "Jamba: A Hybrid Transformer-Mamba Language Model" (Lieber et al, 2024).
- "Linear Attention Is (Maybe) All You Need (to Understand Transformer Optimization)" — theoretical analysis of what linear vs softmax attention loses.

End of Part 1. Next: Part 2 begins with **Why positions matter** — the permutation-invariance problem that motivates positional encodings. We've covered attention's full family; now we turn to the auxiliary mechanism that gives transformers any notion of order at all.
