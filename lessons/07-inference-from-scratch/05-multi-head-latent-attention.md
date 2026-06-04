---
title: "Lesson 5 — Multi-Head Latent Attention (MLA)"
date: "2026-06-04"
module: "inference-from-scratch"
order: 5
tags: ["attention", "mla", "deepseek", "kv-cache", "low-rank", "compression"]
author: "Sudipta Pathak"
prerequisites: ["04-grouped-query-attention"]
---

# Lesson 5 — Multi-Head Latent Attention (MLA)

## Why this lesson exists

GQA shrinks the KV cache by reducing the *number* of K/V heads. MLA (Multi-Head Latent Attention, introduced by DeepSeek in DeepSeek-V2, 2024) takes a different angle: instead of fewer K/V heads, it compresses each K/V into a *low-rank latent space*. The KV cache stores the small latent vector (typically 512 dim or so) instead of the per-head K and V vectors (`h × d_head = D` dim).

The advertised compression: ~10× smaller KV cache than MHA at comparable quality. DeepSeek-V2 (236B parameters, MoE) used MLA to fit a competitive long-context model under deployment constraints; DeepSeek-V3 (671B) extended the same approach.

MLA is more involved than GQA — the inference-time math has gymnastics that make the implementation tricky. This lesson is the construction, the math that makes it work, and the practical deployment story.

The lesson is reading. The Hands-on implements MLA's KV-compression path.

## The core idea

Stash both K and V information into a single small latent vector per token, rather than storing the full per-head K and V tensors.

Define a "latent" projection that maps each input token's hidden state to a compressed vector `c_kv` of dimension `d_c` (typically much smaller than `D`):

```
c_kv = x @ W_DKV   # [B, N, d_c]
```

This is the KV cache. Per token per layer: `d_c × bytes_per_element`. For `d_c = 512` at FP16: 1024 bytes per token per layer. Compare to MHA's `2D × bytes = 16 KB` for a 4096-dim model — 16× smaller.

At inference time, K and V are reconstructed from `c_kv` via per-head decompression matrices:

```
K_per_head = c_kv @ W_UK   # [B, N, h, d_head]
V_per_head = c_kv @ W_UV   # [B, N, h, d_head]
```

`W_UK` and `W_UV` are small per-head projection matrices (`d_c × d_head`).

## The clever inference-time trick

The naive implementation: store `c_kv` in the cache; at every attention step, decompress to per-head K and V, then run standard attention. Saves cache memory but adds decompression compute per step.

The trick: you can *fold the decompression matrices into the Q and O projections* so the inference never explicitly decompresses. Specifically:

Original attention: `output[head] = softmax(Q[head] @ K[head]^T) @ V[head]` followed by `output @ W_O`.

Substitute the MLA decompressions:
- `K[head] = c_kv @ W_UK[head]`
- `V[head] = c_kv @ W_UV[head]`

Now:
- `Q[head] @ K[head]^T = Q[head] @ W_UK[head]^T @ c_kv^T = (Q[head] @ W_UK[head]^T) @ c_kv^T`.

We can absorb `W_UK[head]` into `W_Q[head]`: store an "effective Q" that's the original Q post-multiplied by `W_UK^T`. Then the attention computation operates directly on the small `c_kv`, never reconstructing the per-head K.

Similarly for V: absorb `W_UV` into the output projection `W_O`.

Net result: during inference, the KV cache stores only `c_kv` (small); the Q and O projections are slightly modified at model load time (to absorb the decompression matrices); the attention kernel works on a small per-token vector.

This is the "absorbed" form of MLA. The training-time form computes K and V explicitly; the inference-time form uses the absorbed projections.

## The rotary position embedding complication

MLA has a wrinkle that GQA doesn't: it interacts awkwardly with RoPE (rotary position embeddings, Lesson 14). RoPE applies a position-dependent rotation to Q and K *per head*. If you absorb `W_UK` into `W_Q`, the rotation can't be applied to K because K no longer exists as a separate tensor.

DeepSeek's fix: split each head's K into two parts. One part comes from the latent projection (no RoPE applied). The other part is a separate "decoupled" small K (with RoPE applied). The attention computes scores against both parts and adds them.

```
K_no_rope = c_kv @ W_UK   # latent decompression, no position info
K_rope = (x @ W_KR) with RoPE applied   # small, position-aware
```

The RoPE part has a much smaller dimension (typically `d_rope = 64` or so). The cache stores both `c_kv` and the RoPE-applied K, but the RoPE part is small.

This complication is the main reason MLA is harder to implement than GQA. The conceptual model is clean; the RoPE integration adds friction.

## Parameter and KV cache accounting

For `d_model = D`, `n_heads = h`, `d_head = D/h`, `d_c` (latent dim, ~512), `d_rope` (RoPE-K dim, ~64):

- `W_DKV` (down-projection to latent): `D × d_c`.
- `W_UK`, `W_UV` (up-projections): `d_c × (h × d_head) = d_c × D`.
- `W_KR` (RoPE K projection): `D × d_rope` (shared across heads or per-head; smaller dim).
- `W_Q`: `D × D`.
- `W_O`: `D × D`.

Total: roughly `2D² + 2 × d_c × D + D × d_rope ≈ 2D² + 2 × d_c × D`.

For `D = 5120` (DeepSeek-V2), `d_c = 512`: `2 × 5120² + 2 × 512 × 5120 ≈ 57M` parameters per attention block. About 30% more than MHA's `4 × 5120² = 105M`. (Wait — that's less than MHA. The down-projection + small up-projections sum to less than the full Q, K, V, O.) So MLA actually has *fewer* attention parameters than MHA while having a much smaller KV cache. The win is twofold.

KV cache per token per layer: `(d_c + d_rope) × bytes`. For DeepSeek-V2 (`d_c=512`, `d_rope=64`): `576 × 2 = 1152` bytes. For MHA at D=5120: `2 × 5120 × 2 = 20480` bytes. ~18× smaller.

## What MLA loses

The quality cost relative to MHA: small (~0.05-0.1 perplexity in published evals). The DeepSeek team has demonstrated MLA's quality competitive with GQA at much smaller cache, which makes it attractive for long-context deployment.

Where MLA loses:
- **Implementation complexity.** The RoPE integration is non-trivial; integrating with existing kernel libraries (FlashAttention) requires modifications.
- **Less ecosystem support.** GQA is supported natively in every major runtime; MLA is supported in fewer (vLLM, sglang, DeepSeek's own implementations). llama.cpp added MLA support in 2024 with effort.
- **Less battle-tested.** MLA emerged in 2024; GQA has been deployed at scale since 2023.

For new deployments in 2026, MLA is the most aggressive practical KV-compression choice. For new model architectures specifically designed for long context, MLA is increasingly the default.

## When MLA is the right answer

The cases where MLA's complexity is worth it:

- **Long-context inference at scale.** When you're serving 32K+ context windows and the KV cache dominates memory, MLA's 16-18× compression is transformative.
- **Multi-tenant serving.** More cache headroom means more concurrent sessions.
- **MoE models.** DeepSeek-V2 / V3 combine MLA with MoE; the combination produces models with high parameter count but moderate active parameters per token *and* moderate KV cache.

When GQA is fine:
- Most standard deployments where 4-8× KV reduction is enough.
- When you want ecosystem support today, not 2025+.
- When deployment simplicity matters.

The 2026 picture: GQA-8 is the safe default; MLA is the choice for long-context-heavy or extremely cost-sensitive deployments.

## Implementation sketch

A simplified MLA in pseudo-code (ignoring the RoPE complication for brevity):

```python
class MLA(nn.Module):
    def __init__(self, d_model, n_heads, d_c):
        super().__init__()
        self.W_q = nn.Linear(d_model, d_model)
        self.W_dkv = nn.Linear(d_model, d_c)  # down-projection
        self.W_uk = nn.Linear(d_c, d_model)   # up-projection for K
        self.W_uv = nn.Linear(d_c, d_model)   # up-projection for V
        self.W_o = nn.Linear(d_model, d_model)
        self.n_heads, self.d_head = n_heads, d_model // n_heads

    def forward(self, x, kv_cache=None):
        B, N, D = x.shape
        q = self.W_q(x).view(B, N, self.n_heads, self.d_head).transpose(1, 2)
        # Cache the latent, not the per-head K and V.
        c_kv = self.W_dkv(x)  # [B, N, d_c]
        if kv_cache is not None:
            c_kv = torch.cat([kv_cache, c_kv], dim=1)
            kv_cache_new = c_kv
        # Decompress (or "absorb" the up-projections for production).
        k = self.W_uk(c_kv).view(B, -1, self.n_heads, self.d_head).transpose(1, 2)
        v = self.W_uv(c_kv).view(B, -1, self.n_heads, self.d_head).transpose(1, 2)
        # Standard attention.
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        # Causal mask, softmax, weighted sum, output projection ...
        out = ...
        return out, kv_cache_new
```

Production implementations bake the up-projections into Q and O matrices for inference-time efficiency. The `c_kv` cache is the small thing that gets stored.

## What you should believe after this lesson

Three sentences:

**1. MLA compresses K and V into a low-rank latent vector** (typically `d_c = 512`) stored in the cache, shrinking the KV cache by ~16-18× compared to MHA at comparable model quality. The decompression matrices can be absorbed into Q and O at inference time, so attention runs directly on the small latent.

**2. The RoPE integration is the implementation wrinkle** — RoPE per-head rotation doesn't naturally compose with the latent decompression, so MLA splits K into a no-RoPE latent-decompressed part and a small RoPE-applied direct part. Both contribute to attention scores.

**3. MLA is the most aggressive practical KV compression in 2026**; reach for it when long-context inference dominates and the implementation complexity is worth it. GQA-8 remains the default for typical deployments.

## Hands-on (at home)

A simplified MLA without RoPE.

```python
# mla_simple.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MLASimplified(nn.Module):
    def __init__(self, d_model, n_heads, d_c, causal=True):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.causal = causal
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_dkv = nn.Linear(d_model, d_c, bias=False)
        self.W_uk = nn.Linear(d_c, d_model, bias=False)
        self.W_uv = nn.Linear(d_c, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        self.d_c = d_c

    def forward(self, x):
        B, N, D = x.shape
        q = self.W_q(x).view(B, N, self.n_heads, self.d_head).transpose(1, 2)
        c_kv = self.W_dkv(x)  # [B, N, d_c]
        k = self.W_uk(c_kv).view(B, N, self.n_heads, self.d_head).transpose(1, 2)
        v = self.W_uv(c_kv).view(B, N, self.n_heads, self.d_head).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        if self.causal:
            mask = torch.tril(torch.ones(N, N, device=x.device)) == 0
            scores = scores.masked_fill(mask, float('-inf'))
        weights = F.softmax(scores, dim=-1)
        out = (weights @ v).transpose(1, 2).contiguous().view(B, N, D)
        return self.W_o(out)

# Compare KV-cache size assumption: store c_kv (size d_c per token) vs full K,V.
D, h, d_c = 1024, 16, 256
mla = MLASimplified(D, h, d_c)
print(f"MLA cache per token: {d_c * 2} bytes (FP16)")  # store c_kv
print(f"MHA cache per token: {2 * D * 2} bytes (FP16)")
print(f"Reduction: {2 * D / d_c:.1f}x")
```

For a real MLA implementation with RoPE, the DeepSeek-V2 reference implementation on GitHub is the canonical example.

## Further reading

- "DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model" (DeepSeek, 2024) — the paper that introduced MLA.
- "DeepSeek-V3 Technical Report" (DeepSeek, 2024) — the larger-scale follow-up.
- "DeepSeek MoE" (DeepSeek, 2024) — the MoE side of the architecture (Module 7 Lesson 32).
- vLLM and SGLang MLA implementations — the production reference code for serving MLA models.

Next lesson: **Sliding Window Attention.** A different angle on `O(N²)` attention: instead of compressing K/V, restrict attention to a window of recent tokens. Mistral, Longformer, and StreamingLLM are the canonical examples. Combines with attention sinks to give effectively unbounded context.
