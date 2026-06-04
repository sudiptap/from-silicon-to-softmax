---
title: "Lesson 1 — Self-Attention from First Principles"
date: "2026-06-04"
module: "inference-from-scratch"
order: 1
tags: ["attention", "self-attention", "queries", "keys", "values", "softmax"]
author: "Sudipta Pathak"
prerequisites: ["00-overview"]
---

# Lesson 1 — Self-Attention from First Principles

## Why this lesson exists

Self-attention is the operation. Every transformer is built around it. Every attention variant in this part (MHA, MQA, GQA, MLA, FlashAttention, sliding-window, cross-attention, linear) is a modification of one core formula. Before we vary it, we need to understand what the unmodified version *does* and *why* — not as a formula to memorize but as an operation with a clear purpose.

The plain-language version: self-attention lets each token in a sequence look at every other token and pull information from the ones that matter to it. That's it. The math is the implementation of "look at" (compute a similarity score) and "pull information" (weighted average of values).

The lesson is reading. The Hands-on builds self-attention from scratch in ~30 lines of PyTorch.

## The setup

We have a sequence of `n` tokens, each represented as a vector of dimension `d`. Call this input matrix `X` of shape `[n, d]`. For each token, we want to produce an output vector that incorporates information from all the other tokens — weighted by relevance.

We need three things per token:
- A **query** vector: "what am I looking for?"
- A **key** vector: "what do I represent?"
- A **value** vector: "what information do I carry?"

The query of token A is compared against the keys of every token (including A itself) to get scores; the scores are turned into weights; the weights mix the values; the result is A's output.

The queries, keys, and values are computed from `X` by three linear projections:

```
Q = X @ W_Q   # [n, d]
K = X @ W_K   # [n, d]
V = X @ W_V   # [n, d]
```

where `W_Q`, `W_K`, `W_V` are learned `[d, d]` weight matrices.

## The dot product as similarity

How do you measure "is token A interested in token B"? The dot product.

`Q[A] · K[B]` is large and positive if Q[A] and K[B] point in the same direction (high cosine similarity), zero if they're orthogonal, large and negative if they point in opposite directions.

Doing this for all pairs:

```
scores = Q @ K.T   # [n, n], where scores[i, j] = Q[i] · K[j]
```

`scores[i, j]` tells you how interested token `i` is in token `j`.

## The scale factor

Why divide by `sqrt(d)`? It's not arbitrary. Consider two random vectors `q` and `k` of dimension `d`, each component sampled from a unit-variance distribution. The expected value of `q · k` has variance `d`. So as `d` grows, the dot products grow on a scale of `sqrt(d)`.

If we then feed `q · k` to softmax (next step), large values cause the softmax to saturate — almost all probability mass concentrates on a single token, and gradients vanish through the saturated softmax. The fix: divide by `sqrt(d)` to keep the variance constant regardless of `d`:

```
scaled_scores = (Q @ K.T) / sqrt(d)
```

This is the "scaled" in scaled-dot-product attention.

## The softmax

Scores are real numbers — they could be anything. We need them to be a probability distribution (non-negative, summing to 1) so they can act as weights. Softmax does this:

```
weights = softmax(scaled_scores, axis=-1)   # [n, n]
```

Each row of `weights` is now a probability distribution over the `n` tokens. `weights[i, j]` is how much token `i` attends to token `j`. The softmax is applied row-wise: for each query position `i`, the keys' scores get turned into a distribution.

The softmax does two things:
1. Makes scores non-negative (via `exp`).
2. Normalizes to sum to 1.

Properties:
- A score that's much larger than the others dominates (winner-take-all behavior).
- Scores that are close to each other produce a more uniform distribution (the model is "uncertain" which token to attend to).
- Adding a constant to all scores doesn't change the distribution.

## The weighted sum

Now we use the weights to mix the values:

```
output = weights @ V   # [n, d]
```

`output[i]` is the weighted average of all values, with weights given by `weights[i]`. If row `i` of `weights` puts most of its mass on token `j`, then `output[i]` is mostly `V[j]`.

That's it. Self-attention is:

```
attention(Q, K, V) = softmax(Q @ K.T / sqrt(d)) @ V
```

## What it's actually computing

Conceptually: for each token, look at all tokens (including yourself), figure out which ones you care about (the dot-product scoring), turn those into weights (softmax), and produce your output as a weighted combination of what they say (the values).

Why is this useful? Because the model can learn what to attend to. For "the cat sat on the mat", when processing "sat", the model can learn (via gradient descent on the loss) to attend strongly to "cat" — answering "who sat?" — and weakly to "the" — which isn't informative.

The learned weight matrices `W_Q`, `W_K`, `W_V` shape *what kind of similarity matters*. Different layers learn different similarities; different heads (next lesson) learn different similarities within a layer.

## Causal masking

For language modeling (and any other autoregressive task), each token should only attend to itself and earlier tokens — not to future tokens. The fix: zero out the upper-triangular part of the scores matrix before softmax. Equivalently, set those positions to `-inf` so they go to 0 after softmax.

```
mask = torch.tril(torch.ones(n, n)) == 0   # upper triangle is True (to be masked)
scaled_scores = (Q @ K.T) / sqrt(d)
scaled_scores = scaled_scores.masked_fill(mask, float('-inf'))
weights = softmax(scaled_scores, axis=-1)
output = weights @ V
```

Now `weights[i, j] = 0` for all `j > i`. Each token only attends to itself and previous tokens. This is what every LLM uses.

The non-causal version (no mask) is used in encoder-only models (BERT) where bidirectional attention is appropriate.

## Complexity

Self-attention is `O(n² d)`. The `n²` comes from the `Q @ K.T` (which has shape `[n, n]`) and `weights @ V`. The `d` is the inner dimension.

This `n²` is the headache. For a 2K-token context, `n² = 4M`. For a 32K context, `n² = 1G` (a billion). The matrix `Q @ K.T` is `[n, n]` — at 32K context that's a 1 GB matrix per attention head. This is the cost FlashAttention (Lesson 8) attacks, and the cost sliding-window (Lesson 6) and linear attention (Lesson 9) sidestep.

For inference specifically, the situation is asymmetric (Lesson 16): during prefill (processing the input prompt) you have `n` queries and `n` keys/values, so the cost is `O(n²)`. During decode (generating one token at a time) you have 1 query and `n` cached keys/values, so the cost is `O(n)` per token. The KV cache (Module 3 Lesson 1; Module 4 Lesson 9; Part 3 of this module) is what enables the linear decode cost.

## What you should believe after this lesson

Three sentences:

**1. Self-attention is `softmax(Q K^T / sqrt(d)) V`** — for each token, compute scaled dot-product similarities against all other tokens, turn into weights via softmax, take a weighted average of values. The whole transformer architecture is built around this operation.

**2. The scale factor `1/sqrt(d)` exists to keep the softmax from saturating** as `d` grows; the causal mask makes attention autoregressive (each token sees itself and previous tokens only).

**3. The `O(n²)` cost in `n` is the central headache** — at long context the `n × n` attention matrix dominates memory and compute. Every attention variant in this part either accepts the cost (and optimizes around it: FlashAttention), reduces it via structural choices (sliding window, MoE, GQA), or replaces it with something subquadratic (linear attention).

## Hands-on (at home)

Implement self-attention in ~30 lines.

```python
# self_attention.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SelfAttention(nn.Module):
    def __init__(self, d_model, causal=True):
        super().__init__()
        self.d_model = d_model
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.causal = causal

    def forward(self, x):
        # x: [batch, seq_len, d_model]
        B, N, D = x.shape
        Q = self.W_q(x)  # [B, N, D]
        K = self.W_k(x)  # [B, N, D]
        V = self.W_v(x)  # [B, N, D]
        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(D)  # [B, N, N]
        if self.causal:
            mask = torch.tril(torch.ones(N, N, device=x.device)) == 0
            scores = scores.masked_fill(mask, float('-inf'))
        weights = F.softmax(scores, dim=-1)
        return weights @ V  # [B, N, D]

# Test.
torch.manual_seed(0)
attn = SelfAttention(d_model=64, causal=True)
x = torch.randn(1, 8, 64)
y = attn(x)
print(f"input shape:  {x.shape}")
print(f"output shape: {y.shape}")  # same as input

# Verify causality: changing future tokens shouldn't change earlier outputs.
x2 = x.clone()
x2[0, 5:] = torch.randn(3, 64)  # change last 3 tokens
y2 = attn(x2)
print(f"first 5 outputs unchanged: {torch.allclose(y[0, :5], y2[0, :5])}")
```

The causality check is the most important assertion: if your implementation of causal masking is correct, changing future tokens never changes earlier outputs.

For a richer test, run on a real sentence: embed a tokenized string, run through `SelfAttention`, inspect the attention weights matrix and see which tokens attend to which.

## Further reading

- "Attention Is All You Need" (Vaswani et al, 2017) — the foundational transformer paper. Section 3.2.1 has the canonical scaled-dot-product-attention description.
- "The Annotated Transformer" (Harvard NLP) — line-by-line walkthrough of the paper's code.
- Andrej Karpathy's "Let's build GPT" YouTube video — the same content from scratch in code, with excellent intuition-building.
- 3Blue1Brown's transformer video series — for the visual / geometric intuition of attention.

Next lesson: **Multi-Head Attention.** Self-attention with one set of `(W_Q, W_K, W_V)` is limiting because it can only learn one kind of similarity at a time. MHA splits the dimension into multiple "heads," each with their own projections, so the model can learn multiple kinds of similarity in parallel.
