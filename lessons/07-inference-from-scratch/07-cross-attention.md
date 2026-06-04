---
title: "Lesson 7 — Cross-Attention"
date: "2026-06-04"
module: "inference-from-scratch"
order: 7
tags: ["attention", "cross-attention", "encoder-decoder", "multimodal", "fusion"]
author: "Sudipta Pathak"
prerequisites: ["06-sliding-window-attention"]
---

# Lesson 7 — Cross-Attention

## Why this lesson exists

All the attention variants so far have been *self*-attention: Q, K, V all come from the same input. Cross-attention takes Q from one source and K, V from another. This single change enables one of the original Transformer use cases (encoder-decoder for machine translation) and several modern critical patterns: encoder-decoder summarization, vision-language fusion (vision tokens as K/V, text query as Q), and audio-language fusion.

The mechanics are nearly identical to self-attention. The differences are: where the projections' inputs come from, the cache-management implications, and the use cases.

This lesson is the cross-attention construction and the contexts where it dominates over self-attention.

The lesson is reading. The Hands-on implements cross-attention and uses it for a simple "look at this sequence while generating this other sequence" setup.

## The basic mechanism

Self-attention:

```
Q = X @ W_Q
K = X @ W_K
V = X @ W_V
output = softmax(Q @ K^T / sqrt(d)) @ V
```

All projections are from the same `X`.

Cross-attention:

```
Q = X_query @ W_Q          # decoder hidden state
K = X_context @ W_K        # encoder output (or vision tokens, or other source)
V = X_context @ W_V
output = softmax(Q @ K^T / sqrt(d)) @ V
```

`X_query` and `X_context` are typically different sequences with different lengths and different semantics. The attention computation is the same; the shapes don't need to match in sequence length.

Result shape: `[B, N_query, D]` — one output per query position.

## Encoder-decoder transformers

The original 2017 transformer was encoder-decoder. Setup:

- **Encoder**: standard self-attention over the source sequence (e.g., English sentence). Produces encoder outputs of shape `[B, N_src, D]`.
- **Decoder**: alternating *self*-attention over the target tokens (causal) and *cross*-attention against the encoder outputs.

The decoder's cross-attention layer asks: "what in the source should I look at to produce my next token?" During English-to-French translation, when producing the French word for "cat", the cross-attention query attends to the encoder representation of "cat" in the English input.

Encoder-decoder was the standard for translation, summarization, and text-to-text tasks from 2017 through 2021 or so. Then GPT-style *decoder-only* models took over for most language tasks. The decoder-only models drop the encoder entirely and treat everything as one autoregressive sequence: "<source>: Hello → <target>:" produces "Bonjour."

Why the shift? Decoder-only models are simpler (one block type, easier scaling), train better on web-scale data, and benefit from the same architecture for many downstream tasks. Encoder-decoder is still alive in specialized contexts (T5, BART, Whisper) but isn't the default for general-purpose LLMs.

## Cross-attention in multimodal models

The dominant modern use of cross-attention: feeding non-text modalities into a language model.

**Vision-language models** (VLMs from Module 6 Lesson 11): the vision encoder produces a sequence of vision tokens; these are injected into the language decoder. Two common patterns:

1. **Prepend** vision tokens to the text input; use standard self-attention. The text decoder treats the vision tokens as if they were text tokens. Llama 3.2 Vision and many other VLMs use this.

2. **Cross-attention** at every layer: the text decoder's hidden states attend to vision tokens via cross-attention, in addition to text self-attention. The Flamingo (DeepMind, 2022) and Llama 3.2 Vision (in some variants) use this.

The prepend approach is simpler and more common; the cross-attention approach lets vision and text live in slightly separate representations and can scale to longer text contexts without inflating the text sequence.

**Audio-language models** (Whisper): the audio encoder produces a sequence of audio tokens; the text decoder cross-attends to them. Whisper is encoder-decoder explicitly.

## The KV cache implication

Cross-attention has a useful property for cache management: **the K and V from the context are static across decode steps.** Once you've encoded the source (vision or audio or encoder output), K and V don't change as you decode tokens. The KV cache for cross-attention is computed once and reused for the entire generation.

This is different from self-attention's KV cache, which grows as new tokens are appended.

For VLMs using cross-attention: the vision tokens' K/V are computed once when the image is encoded, then reused for every decode step. The vision-token-KV portion of the cache doesn't grow.

For encoder-decoder models: the encoder runs once on the source; its K/V are cached for all decode steps.

## Inference latency profile

Cross-attention's compute cost per decode step is `O(N_query × N_context × D)` for the QK matmul and similar for the attention-output matmul.

For each decode step with `N_query = 1`: `O(N_context × D)` — linear in context length. Comparable to self-attention's per-token cost.

The savings come from amortizing the K/V computation. Self-attention recomputes K/V for the new token at every decode step; cross-attention computes K/V once for the context and reuses indefinitely.

For a Whisper-style audio decoder with 30 seconds of audio (300 audio tokens) generating 100 text tokens: cross-attention K/V is computed once over 300 audio tokens; each decode step's cross-attention does a small matmul against 300 cached tokens. Self-attention happens over a growing text sequence.

## Implementation

```python
class CrossAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x_query, x_context):
        # x_query: [B, N_query, D]
        # x_context: [B, N_context, D]
        B, N_q, D = x_query.shape
        N_c = x_context.shape[1]
        q = self.W_q(x_query).view(B, N_q, self.n_heads, self.d_head).transpose(1, 2)
        k = self.W_k(x_context).view(B, N_c, self.n_heads, self.d_head).transpose(1, 2)
        v = self.W_v(x_context).view(B, N_c, self.n_heads, self.d_head).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        # No causal mask — queries can attend to all context positions.
        weights = F.softmax(scores, dim=-1)
        out = (weights @ v).transpose(1, 2).contiguous().view(B, N_q, D)
        return self.W_o(out)
```

The main differences from self-attention:
- Q comes from `x_query`; K and V come from `x_context`.
- No causal mask (the query can attend to any context position; the causality is between decoder time-steps, not query-to-context).
- Context sequence length is fixed across decode steps (in the autoregressive setting).

## When to use cross-attention

The choice between cross-attention and prepended self-attention for multimodal:

**Cross-attention wins**:
- When the context tokens are static / cacheable across decoded outputs.
- When the context is much longer than the decoder output (audio: 1000+ tokens; vision: 500+).
- When the text and other modality should have somewhat separate representation spaces.

**Prepended self-attention wins**:
- Simpler architecture; works directly with decoder-only LLM substrate.
- Allows the text tokens to also attend to each other through the same attention layer.
- Easier to compose with existing decoder-only training pipelines.

In 2026 the prepend approach is more common for VLMs because it slots into existing decoder-only pipelines without architectural surgery. Cross-attention persists in encoder-decoder models (Whisper, T5) and in some newer VLMs (Idefics-style architectures).

## What you should believe after this lesson

Three sentences:

**1. Cross-attention is self-attention with Q from one source and K, V from another** — typically a decoder querying an encoder's output, or a language decoder querying vision/audio encoded tokens. The mechanism is identical to self-attention; the projections' inputs are what differ.

**2. The cross-attention K/V is static across decoder steps** (the context is fixed once encoded), so it's computed once and reused — a meaningful win over self-attention's growing KV cache for the context portion.

**3. The 2026 multimodal default is "prepend modality tokens to the text input"** (using self-attention) rather than explicit cross-attention, because the prepend approach slots into existing decoder-only architectures cleanly. Cross-attention persists in encoder-decoder models (Whisper, T5) and some newer VLMs.

## Hands-on (at home)

A minimal cross-attention example: a "summarize" task where the decoder cross-attends to a fixed input.

```python
# cross_attention.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class CrossAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x_query, x_context):
        B, N_q, D = x_query.shape
        N_c = x_context.shape[1]
        q = self.W_q(x_query).view(B, N_q, self.n_heads, self.d_head).transpose(1, 2)
        k = self.W_k(x_context).view(B, N_c, self.n_heads, self.d_head).transpose(1, 2)
        v = self.W_v(x_context).view(B, N_c, self.n_heads, self.d_head).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        weights = F.softmax(scores, dim=-1)
        out = (weights @ v).transpose(1, 2).contiguous().view(B, N_q, D)
        return self.W_o(out), weights

# Simulate: source = "the cat sat on the mat" (6 tokens), target = "le chat" (2 tokens).
torch.manual_seed(0)
D, h = 64, 4
xa = CrossAttention(D, h)
src = torch.randn(1, 6, D)  # source sequence (encoder output)
tgt = torch.randn(1, 2, D)  # decoder hidden states
out, weights = xa(tgt, src)
print(f"output shape: {out.shape}")  # [1, 2, 64] — one per target position
print(f"attention weights shape: {weights.shape}")  # [1, 4, 2, 6] — per head, target→source
print(f"first head, first target token, attention over source:")
print(f"  {weights[0, 0, 0, :].detach().numpy()}")  # 6-dim distribution
```

The attention weights row tells you which source positions the first target position attended to. In a trained model, you'd see meaningful alignment patterns (target token attending to corresponding source token).

For a real example: load Whisper and inspect its cross-attention weights between audio frames and text tokens. The patterns make alignment visible — see which audio frames produce which transcribed words.

## Further reading

- "Attention Is All You Need" Section 3.2 — describes both self- and cross-attention in the original encoder-decoder transformer.
- "Whisper" (Radford et al, 2022) — the encoder-decoder architecture with cross-attention from text decoder to audio encoder.
- "Flamingo: a Visual Language Model for Few-Shot Learning" (Alayrac et al, 2022) — the explicit cross-attention VLM architecture.
- "T5: Text-to-Text Transfer Transformer" (Raffel et al, 2020) — a modern encoder-decoder LLM.

Next lesson: **FlashAttention v1 → v2 → v3.** The kernel optimization that made long-context training and inference feasible. We trace the evolution from IO-aware tiling (v1) through better parallelism (v2) to Hopper-specific async paths (v3), and the broader lesson it teaches about kernel design.
