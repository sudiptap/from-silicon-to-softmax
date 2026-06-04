---
title: "Lesson 6 — Sliding Window Attention"
date: "2026-06-04"
module: "inference-from-scratch"
order: 6
tags: ["attention", "sliding-window", "mistral", "longformer", "attention-sinks", "long-context"]
author: "Sudipta Pathak"
prerequisites: ["05-multi-head-latent-attention"]
---

# Lesson 6 — Sliding Window Attention

## Why this lesson exists

GQA and MLA reduce the cost *per token*. Sliding window attention reduces the cost *per attention step* by restricting each query to a fixed-size window of recent tokens — not the full history. The attention complexity drops from `O(N²)` to `O(N × W)` where `W` is the window size. For long sequences (`N >> W`), this is a huge win.

The catch: each token can only see the most recent `W` tokens. Information from outside the window is lost — unless something else carries it forward. The base sliding-window design loses information; combined with attention sinks (Module 6 Lesson 6 introduced this) or with "global tokens" (Longformer's approach), it becomes practical for arbitrarily long contexts.

This lesson covers the basic sliding window, its layered variants, and the modern attention-sink combination that makes it work for production LLMs (Mistral 7B and others).

The lesson is reading. The Hands-on implements sliding-window attention and demonstrates the long-context memory footprint.

## The basic mechanism

For each query token at position `i`, attend only to key positions in `[max(0, i - W + 1), i]`. The window has size `W`. Earlier tokens beyond `i - W + 1` are masked out.

Implementation: the same scaled-dot-product attention, but with a different mask. Instead of just lower-triangular:

```python
# Causal sliding-window mask: True means "masked out".
def sliding_window_mask(N, W):
    i = torch.arange(N).unsqueeze(1)
    j = torch.arange(N).unsqueeze(0)
    # Mask if j > i (causal) OR j < i - W + 1 (outside window).
    return (j > i) | (j < i - W + 1)
```

`scores.masked_fill(mask, -inf)` then softmax does the rest.

The cost:

- The attention matrix has at most `W` non-masked entries per row, so the effective work per row is `O(W)` instead of `O(N)`.
- Total: `O(N × W)` per layer.

For Mistral 7B's `W = 4096` and `N = 32K`: `32K × 4K = 128M` ops instead of `32K × 32K = 1024M`. 8× reduction.

The KV cache implication: once a token is more than `W` positions behind, no future query will attend to it. You can *evict* it from the KV cache. The cache size is bounded by `W` tokens regardless of generation length.

For Mistral 7B with `W=4096`, the KV cache is permanently capped at 4096-token-worth of K and V — regardless of whether the conversation runs 4K or 100K tokens. Memory becomes a constant rather than growing.

## Receptive field across layers

A naive concern: a token at position `i` only sees `W` previous tokens, so it can't access information older than `W`. But in a multi-layer transformer, *each layer's attention contributes to the receptive field*. Layer 1 sees `W` tokens. Layer 2's hidden state at position `i` incorporates information from layer 1's positions `[i-W+1, i]` — each of which in turn incorporated information from layer 0's earlier positions.

After `L` layers of sliding-window attention with window `W`, the effective receptive field is `L × W` tokens. For Mistral 7B (32 layers, W=4096): effective receptive field 131K tokens. Comfortably beyond the practical context length.

This is the conceptual justification for sliding-window: layered attention transitively builds up a wide receptive field even when each individual layer's window is narrow.

The catch: the information "transitively" available through multiple layers is *compressed* through hidden states. A token 100K tokens ago doesn't have its raw representation; it has a heavily-processed hidden-state summary. The fidelity of long-range information is worse than full attention.

## Why the layered receptive field isn't quite enough

In practice, sliding-window attention without modification loses information at long range. The reason: when a token leaves the window, its KV is evicted; future queries can't attend to it directly. The information lives only through the chain of intermediate hidden states.

The empirical result: on benchmarks that test long-range information retention (recalling a fact stated 20K tokens ago, "needle in haystack" tests), pure sliding-window models perform poorly beyond their window size. The transitive receptive field is real but not enough.

This is where attention sinks come in.

## Attention sinks (StreamingLLM)

Xiao et al's StreamingLLM paper (2023) made a clean observation: when you apply sliding-window attention to a *pretrained* (non-sliding-window) model at inference time, the model breaks dramatically. The cause: the first few tokens of the sequence (typically `<BOS>`, system prompt start, etc.) act as "sinks" — most attention heads have learned to dump excess probability mass into these tokens. When the sinks get evicted from the window, the softmax distribution becomes mis-calibrated, and the model collapses.

The fix: pin the first ~4 tokens in the cache. The window slides over the rest, but the sink tokens stay.

```
Cache: [sink_0, sink_1, sink_2, sink_3, recent_n-W+5, ..., recent_n]
```

The attention computation includes the 4 sink tokens plus the most recent `W-4` tokens.

For a pretrained model, this single change (pin the first 4 tokens) is enough to enable unbounded-context inference with sliding-window. The sinks soak up the excess attention; the model behaves as if the full history were available.

For models trained with sliding window from the start (Mistral 7B, Gemma some variants), the model has already learned not to need explicit sinks — the sliding-window training has built the right inductive bias.

## Longformer and global tokens

Longformer (Beltagy et al, 2020) is the canonical pre-LLM sliding-window paper. It introduced the *global tokens* mechanism: certain pre-designated tokens (e.g., the `[CLS]` token in classification, the question tokens in QA) get full attention to all other tokens, and all other tokens get full attention to them.

The architecture:
- Most tokens: sliding-window attention.
- A small number of "global" tokens: full attention.

This combines sliding-window efficiency with the ability to inject a few "hub" tokens that aggregate information from anywhere in the sequence.

Longformer was designed for encoder-only (BERT-style) tasks. The pattern transfers to decoder-only models with some adaptation but isn't standard. StreamingLLM's attention-sink approach is closer to a decoder-only adaptation of the global-tokens idea (sinks act as global tokens that anchor attention).

## Combining with GQA

Mistral 7B combines sliding-window attention with GQA-8. The two compose orthogonally:

- Sliding window reduces KV cache size to a constant `W` instead of growing with sequence length.
- GQA reduces KV per token by 8×.

Combined: Mistral 7B's KV cache is `W × kv_per_token_GQA = 4096 × ~512 bytes per layer = 2 MB per layer`. Across 32 layers, 64 MB total — regardless of context length. By comparison, a non-sliding-window MHA model at 32K context would need >2 GB of KV cache. The combination is the practical enabler of long context on consumer hardware.

## When sliding window wins

The clear cases:

- **Long-context inference where the model was trained for sliding window**. Mistral 7B, Ministral 3B, Gemma some variants.
- **Pretrained models you want to extend to long context via StreamingLLM**. Pin a few sinks; the model can run effectively unbounded contexts (at the cost of long-range recall).
- **Streaming workloads** (real-time chat, audio transcription) where the model processes tokens one at a time and the cache mustn't grow without bound.

When it loses:

- **Tasks requiring high-fidelity recall of distant tokens** (long-document QA where the answer is far from the question, needle-in-haystack). The sliding window loses too much information.
- **Pretrained models *not* trained for sliding-window**. Even with attention sinks, the quality at very long context is worse than full-attention.

## What you should believe after this lesson

Three sentences:

**1. Sliding-window attention restricts each query to a fixed window of recent tokens**, dropping attention cost from `O(N²)` to `O(N × W)` and bounding KV cache memory to `W` tokens regardless of generation length. The trade-off is loss of long-range information.

**2. The multi-layer receptive field stacks** — `L` layers of window `W` give effective receptive field `L × W` tokens, but the long-range information is transitively compressed through hidden states and lower-fidelity than full attention.

**3. Attention sinks (StreamingLLM)** make sliding-window viable for *pretrained* non-sliding-window models by pinning the first ~4 tokens in the cache. Models trained with sliding window from the start (Mistral) don't need explicit sinks; their inductive bias is built-in.

## Hands-on (at home)

Implement sliding-window attention and demonstrate constant memory at long context.

```python
# sliding_window.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def sliding_window_mask(N, W, device):
    i = torch.arange(N, device=device).unsqueeze(1)
    j = torch.arange(N, device=device).unsqueeze(0)
    return (j > i) | (j < i - W + 1)

class SlidingWindowAttention(nn.Module):
    def __init__(self, d_model, n_heads, window_size):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.window_size = window_size
        self.W_qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        B, N, D = x.shape
        q, k, v = self.W_qkv(x).chunk(3, dim=-1)
        q = q.view(B, N, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, N, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, N, self.n_heads, self.d_head).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        mask = sliding_window_mask(N, self.window_size, x.device)
        scores = scores.masked_fill(mask, float('-inf'))
        weights = F.softmax(scores, dim=-1)
        out = (weights @ v).transpose(1, 2).contiguous().view(B, N, D)
        return self.W_o(out)

# Demonstrate that the attention mask narrows attention.
D, h, W = 64, 4, 8
sw = SlidingWindowAttention(D, h, W)
x = torch.randn(1, 32, D)
y = sw(x)
print(f"output shape: {y.shape}")

# Visualize the mask.
mask = sliding_window_mask(16, W, 'cpu').float()
print(f"window mask (1=masked):\n{mask.int().tolist()}")
```

For the constant-memory-at-long-context demonstration, use llama.cpp's `--keep` flag (Module 6 Lesson 6 hands-on) to see the steady-state KV cache size as you generate increasingly long sequences. With sliding-window-trained models (Mistral) you'd see the cache plateau at `W × per_token_cost`.

For the attention-sink experiment, apply sliding window without sinks to a non-sliding-window model (Llama 3.2 1B) — quality should collapse beyond the window. Then add `--keep 4` (the sink count); quality should recover.

## Further reading

- "Longformer: The Long-Document Transformer" (Beltagy et al, 2020) — sliding-window with global tokens.
- "Mistral 7B" (Jiang et al, 2023) — sliding-window attention in a production LLM.
- "Efficient Streaming Language Models with Attention Sinks" (Xiao et al, 2023) — the StreamingLLM paper.
- "Big Bird: Transformers for Longer Sequences" (Zaheer et al, 2020) — sparse attention with random + window + global; a related approach.

Next lesson: **Cross-Attention.** All the variants so far have been self-attention (Q, K, V from the same input). Cross-attention takes Q from one source and K, V from another — the foundation of encoder-decoder transformers (translation, summarization originally) and modern multimodal fusion (vision tokens → text decoder).
