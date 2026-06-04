---
title: "Lesson 16 — Why the KV Cache Exists"
date: "2026-06-04"
module: "inference-from-scratch"
order: 16
tags: ["kv-cache", "prefill", "decode", "autoregressive"]
author: "Sudipta Pathak"
prerequisites: ["15-rope-scaling"]
---

# Lesson 16 — Why the KV Cache Exists

## Why this lesson exists

LLM inference splits into two distinct phases — prefill (processing the prompt) and decode (generating tokens one at a time). They have radically different computational profiles, and the KV cache exists specifically to handle the asymmetry.

Without a KV cache, each new generated token would require recomputing attention against the entire prior sequence. With one, the K and V for prior tokens are stored once; only the new token's Q runs through attention. The complexity drops from `O(n²)` per token to `O(n)`.

This lesson is the *why* — not the implementation details (those are Lessons 17-21), but the structural reason the KV cache is the central data structure of inference. After this lesson, the engineering complexity that surrounds the KV cache (paged attention, prefix caching, KV quantization, attention sinks) makes immediate sense as solutions to the management problem.

The lesson is reading. The Hands-on times the same forward pass with and without a KV cache.

## The two phases

**Prefill**: process the user's prompt. Input: a sequence of `N_prompt` tokens. The model runs one forward pass over all of them at once, computing Q, K, V for each, computing attention, FFN, etc. Output: the model's hidden states for each prompt token (used only to initialize the KV cache) and the logits for the *next* token after the prompt.

Cost: `O(N_prompt² × D)` for attention, `O(N_prompt × D²)` for FFN. Dominated by the FFN at moderate `N_prompt`, by attention at long `N_prompt`.

**Decode**: generate output tokens one at a time. At each step:
1. Take the most recent token's embedding.
2. Compute Q, K, V *for just that one token*.
3. Attend the new Q against the cached K, V from all prior tokens (plus the new K, V).
4. Run FFN.
5. Sample the next token.
6. Append the new K, V to the cache.
7. Repeat.

Cost per token: `O(N_current × D)` for attention (linear in current context length), `O(D²)` for FFN.

The asymmetry: prefill is one big batch of `N_prompt` tokens; decode is many small batches of 1 token.

## What the KV cache eliminates

Without a cache, the decode step would recompute K and V for every prior token at every step. The first generated token: compute K, V for `N_prompt + 1` tokens. The second: compute for `N_prompt + 2` tokens. The 100th: compute for `N_prompt + 100` tokens.

Total work for generating 100 tokens: roughly proportional to `sum_{i=1}^{100} (N_prompt + i) ≈ 100 × (N_prompt + 50)`. Quadratic blowup in generation length, on top of the `N_prompt²` prefill.

With a cache: K, V for prior tokens are computed once and stored. Each new decode step only computes K, V for one new token. Total work: linear in generation length.

For `N_prompt = 1000` and 100 tokens of output:
- Without cache: ~100K K/V computations. With everything else (attention, FFN), 100× the per-token cost.
- With cache: 100 K/V computations. Decode is fast.

The KV cache makes interactive LLM inference feasible.

## Why prefill is compute-bound, decode is bandwidth-bound

The asymmetry continues into the hardware-bound regime:

**Prefill** processes `N_prompt` tokens through every layer's matmul. The matmul shapes are `[N_prompt, D] @ [D, D]` — fat matmuls that hit the GPU's tensor cores at peak throughput. Compute-bound.

**Decode** processes 1 token through every layer's matmul. The matmul shapes are `[1, D] @ [D, D]` — thin matmuls that are dominated by the *load* of the `[D, D]` weight matrix, not by the compute of one row times that matrix. Bandwidth-bound. (Module 3 Lesson 1 established this.)

The implication: techniques that help prefill (better compute utilization, parallelism) don't help decode. Techniques that help decode (smaller weights via quantization, smaller KV cache via GQA/MQA) don't speed up prefill proportionally.

This split shapes every aspect of inference engineering. Continuous batching (Lesson 41) addresses it by overlapping prefill of new requests with decode of in-progress ones. Prefill/decode disaggregation (Lesson 43) takes it further — different hardware for each phase.

## Time-to-first-token (TTFT) vs decode rate

The two phases produce two user-visible latencies:

- **TTFT**: prefill time + first decode token. This is "how fast does the model start responding."
- **Decode rate (tok/s)**: how fast subsequent tokens come out.

For a typical 7B model on a 4090 with a 500-token prompt and 100-token output:
- Prefill: ~50 ms (10K tok/s effective).
- First decode token: ~25 ms.
- TTFT ≈ 75 ms.
- Subsequent 99 decode tokens at ~40 tok/s = 2.5 seconds total decode.

The user perceives:
- Quick initial response (TTFT).
- Smooth streaming at 40 tok/s.

Different applications care about different things: chat cares about both; code completion cares mostly about TTFT (the response is short); summarization cares mostly about decode rate (the output is long).

## What the KV cache holds

For each transformer layer, per token, the cache holds:
- K (the key tensor): shape `[n_kv_heads, d_head]`.
- V (the value tensor): shape `[n_kv_heads, d_head]`.

Total per-token cost: `2 × n_kv_heads × d_head × bytes`. This is the budget we tracked in Modules 3, 4, 6, and earlier in this module.

At Llama 3.2 3B (28 layers, n_kv_heads=8, d_head=128, FP16):
- Per token per layer: `2 × 8 × 128 × 2 = 4096 bytes`.
- Per token across all layers: `28 × 4096 ≈ 115 KB`.
- For 2K context: ~230 MB.
- For 32K context: ~3.7 GB.

The KV cache eclipses the weight memory at long context. This is why KV-cache management is so much of inference engineering.

## What the KV cache doesn't hold

Notably *not* in the cache:
- The Q tensor. Q is computed for the new token each decode step; it's not reused across steps.
- The attention weights. Computed and discarded each step.
- The FFN intermediates. Same — computed and discarded.
- The hidden states. Each layer's output becomes the next layer's input, then is discarded.

The cache is just K and V. Everything else is recomputed each step.

The reason K and V are special: in attention, K and V are computed from past tokens and *reused* against future Q's. Q comes from the new token; it doesn't get reused.

## Decode is structurally weird

The decode-step matmul (`[1, D] @ [D, D]`) has arithmetic intensity ~1 FLOP per byte loaded. For a 4090 with peak ~1 FLOP per byte (effective bandwidth-bound ceiling), this is the limit.

This is the central oddity of LLM inference: the hardware is built for big matmul throughput; decode forces the hardware into bandwidth-bound mode where most of the FLOPs go unused. Decode at batch 1 effectively wastes most of the GPU's compute capacity — you're just moving weights through.

The fixes:
- **Quantization** (Modules 3, 6, and Part 6 of this module): smaller weights → less to load.
- **GQA / MQA / MLA** (Lessons 3-5): smaller KV cache → less to load.
- **Batching** (Lesson 41): multiple decodes in parallel → amortize weight loads.
- **Speculative decoding** (Lessons 25-27): more tokens per main-pass → fewer loads per token.

Every optimization in the rest of this module is, at its core, about closing the gap between decode's bandwidth-bound performance and the chip's compute peak.

## What you should believe after this lesson

Three sentences:

**1. LLM inference splits into prefill (compute-bound, processes the prompt) and decode (bandwidth-bound, generates one token at a time)** — the two phases have radically different cost profiles, and most inference engineering is about handling each appropriately.

**2. The KV cache eliminates the quadratic blowup of re-computing K and V for prior tokens every decode step**, dropping decode complexity from `O(n²)` per token to `O(n)`. K and V are special because they're reused across steps; Q is recomputed each step.

**3. Decode is bandwidth-bound** — the `[1, D] @ [D, D]` matmul wastes most of the GPU's compute capacity. Every subsequent optimization in this module (quantization, GQA, batching, speculative decoding) is some attempt to close the bandwidth-vs-compute gap.

## Hands-on (at home)

Compare attention with and without a KV cache.

```python
# kv_cache_demo.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time

class CachedAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.W_qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, kv_cache=None):
        # x: [B, N_new, D]; kv_cache: (k_prev, v_prev) each [B, h, N_prev, d_head] or None
        B, N_new, D = x.shape
        q, k_new, v_new = self.W_qkv(x).chunk(3, dim=-1)
        q = q.view(B, N_new, self.n_heads, self.d_head).transpose(1, 2)
        k_new = k_new.view(B, N_new, self.n_heads, self.d_head).transpose(1, 2)
        v_new = v_new.view(B, N_new, self.n_heads, self.d_head).transpose(1, 2)
        if kv_cache is None:
            k, v = k_new, v_new
        else:
            k_prev, v_prev = kv_cache
            k = torch.cat([k_prev, k_new], dim=2)
            v = torch.cat([v_prev, v_new], dim=2)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        # No causal mask needed if decoding one token at a time (q attends to all of k naturally).
        weights = F.softmax(scores, dim=-1)
        out = (weights @ v).transpose(1, 2).contiguous().view(B, N_new, D)
        out = self.W_o(out)
        return out, (k, v)

# Simulate generating 50 tokens after a 500-token prompt.
torch.manual_seed(0)
D, h = 256, 4
attn = CachedAttention(D, h)
prompt = torch.randn(1, 500, D)
total_steps = 50

# WITH cache.
t0 = time.time()
_, kv = attn(prompt)  # prefill
for _ in range(total_steps):
    new_tok = torch.randn(1, 1, D)
    out, kv = attn(new_tok, kv_cache=kv)
t_with_cache = time.time() - t0

# WITHOUT cache: each decode step re-runs over the full sequence.
t0 = time.time()
full_seq = prompt.clone()
for _ in range(total_steps):
    out, _ = attn(full_seq)  # recompute over whole sequence
    new_tok = torch.randn(1, 1, D)
    full_seq = torch.cat([full_seq, new_tok], dim=1)
t_without_cache = time.time() - t0

print(f"with cache:    {t_with_cache*1000:.0f} ms total ({t_with_cache*1000/total_steps:.1f} ms/decode)")
print(f"without cache: {t_without_cache*1000:.0f} ms total ({t_without_cache*1000/total_steps:.1f} ms/decode)")
print(f"speedup: {t_without_cache / t_with_cache:.1f}x")
```

With a cache, decode is fast (the per-step cost is constant after prefill); without, it grows linearly each step. On a real LLM with 32 layers and a 500-token prompt, the speedup is 10-100×.

## Further reading

- "Efficiently Scaling Transformer Inference" (Pope et al, 2022) — the canonical reference for the prefill/decode asymmetry at scale.
- "Fast Transformer Decoding: One Write-Head is All You Need" (Shazeer, 2019) — MQA paper; the motivation section has a clean prefill/decode discussion.
- Module 3 Lesson 1 and Module 6 Lesson 6 — earlier coverage of the budget side of KV cache.

Next lesson: **KV cache memory layout and arithmetic intensity.** We zoom in on the bandwidth math: how the cache is laid out in memory, what arithmetic intensity decode actually has, and the roofline analysis that explains why decode is bandwidth-bound regardless of how powerful the GPU is.
