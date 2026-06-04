---
title: "Lesson 3 — Multi-Query Attention (MQA)"
date: "2026-06-04"
module: "inference-from-scratch"
order: 3
tags: ["attention", "mqa", "kv-cache", "memory-bandwidth"]
author: "Sudipta Pathak"
prerequisites: ["02-multi-head-attention"]
---

# Lesson 3 — Multi-Query Attention (MQA)

## Why this lesson exists

MHA keeps a separate K and V projection per head. At inference, this means the KV cache stores per-token: `n_heads × d_head × 2 (K + V) × bytes_per_element` for every layer. For a typical 7B model with 32 heads × 128 d_head, that's `32 × 128 × 2 × 2 = 16 KB` per token per layer. Across 32 layers and a 4K-token context: 2 GB. The KV cache dominates memory at long context (Module 3 Lesson 1 covered this).

MQA (Shazeer 2019) makes a simple architectural choice: keep the `h` separate Q heads, but use only *one* shared K and one shared V across all heads. The KV cache shrinks by `h×` — for the 7B example above, from 2 GB to 64 MB. The decode bandwidth — which loads the KV cache per token — drops proportionally.

The cost: model capacity. With only one K and V projection (instead of `h`), the model has fewer parameters to learn the key-value structure. Empirically this hurts a bit; in practice the bandwidth win at inference dominates for production deployments.

This lesson is MQA's mechanics and the inference-time math that motivates it.

The lesson is reading. The Hands-on implements MQA and benchmarks the KV-cache size and decode latency vs MHA.

## What changes

The Q projection is unchanged: still `n_heads × d_head` separately for each head.

The K and V projections produce a single `d_head`-dimensional vector each (shared across all heads). Per-token shape:

- Q: `[n_heads, d_head]` — one query per head.
- K: `[d_head]` — one shared key.
- V: `[d_head]` — one shared value.

The attention computation:
- For each head, compute `Q[head] @ K^T / sqrt(d_head)` — but K is the same single vector across heads.
- Softmax, weighted sum with V.

Implementation: `K` and `V` are *broadcast* across the head dimension; the per-head Q dotproducts the same K (and the same V).

## Parameter count

MHA: `4 D²` (Q, K, V, O each `D × D`).

MQA:
- `W_Q`: `D × D` parameters.
- `W_K`: `D × d_head` parameters (produces one `d_head`-dim vector per token).
- `W_V`: `D × d_head` parameters.
- `W_O`: `D × D` parameters.
- Total: `2 D² + 2 D × d_head ≈ 2 D²` (the `D × d_head` terms are small).

Roughly half the attention parameters of MHA. Most of the parameter loss is in K and V; Q and O retain their full size.

## KV cache size

The whole point. MHA cache per token per layer:

```
2 × n_heads × d_head × bytes = 2 × h × (D/h) × bytes = 2 × D × bytes
```

For Llama 2 7B at FP16: `2 × 4096 × 2 = 16384 bytes` ≈ 16 KB per token per layer.

MQA cache per token per layer:

```
2 × d_head × bytes
```

For the same model with d_head=128 at FP16: `2 × 128 × 2 = 512 bytes` per token per layer. **32× smaller** (one head worth instead of all heads).

Across all layers and a 4K context:
- MHA: 32 layers × 16 KB × 4096 tokens = 2 GB.
- MQA: 32 layers × 512 bytes × 4096 tokens = 64 MB.

This shifts the inference bottleneck dramatically. The decode-time bandwidth required to load the KV cache drops 32×, and the memory required to *hold* the cache drops 32×.

## What you lose

Empirically:
- Quality degradation (perplexity) on the order of 0.5–1.5 points at MHA-equivalent training.
- More pronounced quality loss at smaller models / shorter training; smaller at larger / longer training.

The intuition: with one shared K and V, all heads must work with the same key/value space. Heads can no longer specialize in completely different aspects of the input.

The aggregate effect on downstream tasks is often smaller than the perplexity number suggests — quality on standard evals (MMLU, HellaSwag) is comparable for an MQA model trained at scale.

## When MQA wins, when it loses

Wins:
- **Inference-heavy production**. The KV cache is the bottleneck; MQA dramatically improves throughput.
- **Long-context deployments**. KV cache grows linearly with context; MQA's 32× saving compounds.
- **Multi-tenant serving with batching**. More memory free for batching more requests.

Loses:
- **Small models where capacity matters**. The parameter savings hurt more than they help.
- **Models with relatively short context** where the KV cache wasn't the bottleneck anyway.
- **Training-heavy workloads** (training cost is what dominates, not inference; the inference saving is wasted).

Production models that adopted MQA: PaLM, Falcon, several Chinese open-weights families. Notably, *most* modern flagship LLMs use GQA instead (Lesson 4), which is a middle ground that gives most of MQA's benefits without all of MQA's quality loss.

## A note on training MQA from MHA

You can't just take an MHA-trained model and use one of its K, V projections as MQA. The model's K and V across heads aren't equivalent; picking one arbitrarily destroys the model.

There are two viable paths to get an MQA model from an MHA-trained model:
1. **Average the K and V projections** across heads (mean-pool). Lossy but cheap; "MQA-from-MHA via mean."
2. **Fine-tune from the averaged starting point**. Brief continued training to recover quality lost in step 1.

This is the recipe that made MQA practical for some early adopters: train MHA at scale, then convert to MQA for inference deployment. Modern training pipelines bake the choice in from the start.

## Inference math reconsidered

For autoregressive decode, the per-token attention cost is:

- **Loading KV cache**: `n_layers × kv_per_token × bytes` per token.
- **Attention compute**: `n_layers × n_heads × N × d_head` FLOPs per token (linear in current context length `N`).

The bandwidth term (KV loading) dominates when memory bandwidth is the bottleneck (the on-device case from Module 6, the multi-tenant case from server). MQA's `32×` KV reduction directly translates to up to `32×` decode-throughput improvement when the workload is KV-bandwidth-bound. In practice it's less than 32× because (a) the weights also dominate bandwidth, and (b) the attention compute itself has overhead. Realistic decode-throughput improvement: 1.5–3× over MHA.

For prefill (where you process all `N` tokens of the prompt at once), MQA helps less because the workload is compute-bound, not bandwidth-bound. The QK matmul still requires the full `n_heads × N × N` work; MQA just reuses the same K across heads, saving a small amount of bandwidth but not the compute.

## What you should believe after this lesson

Three sentences:

**1. MQA replaces `h` separate K and V projections with one shared K and V**, shrinking the KV cache by `h×` (typically 32× for modern LLMs) and proportionally reducing decode-time memory bandwidth. The Q projections stay separate; the heads still have their own queries.

**2. The quality cost is moderate** — ~0.5–1.5 perplexity in standard settings. The cost is concentrated in capacity loss in the K and V projections; downstream task evals show a smaller gap.

**3. MQA's win is concentrated in inference, especially long-context and multi-tenant** — the KV cache shrinks, decode-time bandwidth drops, you can batch more requests. The next lesson (GQA) is a middle ground that recovers most of the quality lost in MQA while keeping most of the KV benefit.

## Hands-on (at home)

Implement MQA from scratch.

```python
# mqa.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiQueryAttention(nn.Module):
    def __init__(self, d_model, n_heads, causal=True):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.causal = causal
        # Separate projections: Q gets full size, K and V get d_head.
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, self.d_head, bias=False)
        self.W_v = nn.Linear(d_model, self.d_head, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        B, N, D = x.shape
        q = self.W_q(x).view(B, N, self.n_heads, self.d_head).transpose(1, 2)  # [B, h, N, d_head]
        k = self.W_k(x).view(B, N, 1, self.d_head).transpose(1, 2)             # [B, 1, N, d_head]
        v = self.W_v(x).view(B, N, 1, self.d_head).transpose(1, 2)             # [B, 1, N, d_head]
        # Broadcast k and v across the n_heads dimension during the matmul.
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)  # [B, h, N, N]
        if self.causal:
            mask = torch.tril(torch.ones(N, N, device=x.device)) == 0
            scores = scores.masked_fill(mask, float('-inf'))
        weights = F.softmax(scores, dim=-1)
        out = weights @ v  # [B, h, N, d_head] via broadcasting v
        out = out.transpose(1, 2).contiguous().view(B, N, D)
        return self.W_o(out)

# Compare parameter counts.
from torch import nn
D, h = 512, 8
mha = nn.ModuleDict({
    'qkv': nn.Linear(D, 3 * D, bias=False),  # MHA fused QKV
    'o': nn.Linear(D, D, bias=False),
})
mqa = MultiQueryAttention(D, h)
def count(m):
    return sum(p.numel() for p in m.parameters())
print(f"MHA parameters: {count(mha):,}")
print(f"MQA parameters: {count(mqa):,}")
print(f"MQA is {count(mqa) / count(mha) * 100:.1f}% of MHA")

# KV cache size per token.
print(f"MHA KV per token: {2 * D * 2} bytes (FP16)")
print(f"MQA KV per token: {2 * D//h * 2} bytes (FP16)")
print(f"Reduction: {h}x")
```

For a real decode-throughput comparison, implement a generation loop with KV cache and time MHA vs MQA on the same model. On a 7B-class model the MQA version typically decodes 2× faster, all from reduced KV-cache bandwidth.

## Further reading

- "Fast Transformer Decoding: One Write-Head is All You Need" (Shazeer, 2019) — the original MQA paper, two pages, very readable.
- "PaLM: Scaling Language Modeling with Pathways" (Chowdhery et al, 2022) — PaLM is one of the prominent MQA adopters at scale.
- "Falcon-180B" technical report — for another large-scale MQA deployment.
- "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints" (Ainslie et al, 2023) — the next lesson's reference; also discusses MQA-from-MHA conversion.

Next lesson: **Grouped-Query Attention (GQA).** The middle ground between MHA's full per-head K/V and MQA's single shared K/V. Groups of Q heads share a K/V pair; you pick the group count. Llama 2/3, Mistral, and most modern LLMs use this. The lesson covers why GQA is usually the right tradeoff.
