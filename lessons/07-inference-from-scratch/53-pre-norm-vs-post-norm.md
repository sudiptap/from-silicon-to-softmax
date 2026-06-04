---
title: "Lesson 53 — Pre-Norm vs Post-Norm"
date: "2026-06-04"
module: "inference-from-scratch"
order: 53
tags: ["normalization", "pre-norm", "post-norm", "residual", "training-stability"]
author: "Sudipta Pathak"
prerequisites: ["52-swiglu-vs-gelu"]
---

# Lesson 53 — Pre-Norm vs Post-Norm

## Why this lesson exists

The transformer block has a residual stream: each sublayer's output is added to its input. Where does the LayerNorm/RMSNorm go relative to this residual?

**Post-norm** (original 2017 transformer): `x = LN(x + Sublayer(x))`. Normalize after the residual.

**Pre-norm** (most modern LLMs): `x = x + Sublayer(LN(x))`. Normalize the input to the sublayer; the residual stream itself remains un-normalized.

The choice has training-stability implications: pre-norm allows much deeper transformers to train stably. Almost all modern LLMs use pre-norm.

This lesson explains both placements and the trade-off.

The lesson is reading. The Hands-on visualizes the two arrangements.

## The arrangements

**Post-norm** (original):
```
Sublayer 1 (attention):
  attn_out = Attention(x)
  x = LayerNorm(x + attn_out)
Sublayer 2 (FFN):
  ffn_out = FFN(x)
  x = LayerNorm(x + ffn_out)
```

The residual stream is normalized at the output of every sublayer. After normalization, the magnitude of the residual stream is bounded.

**Pre-norm** (modern):
```
Sublayer 1 (attention):
  attn_out = Attention(LayerNorm(x))
  x = x + attn_out
Sublayer 2 (FFN):
  ffn_out = FFN(LayerNorm(x))
  x = x + ffn_out
```

Each sublayer's *input* is normalized; the residual stream is unbounded. The output is the un-normalized residual stream.

## Why pre-norm matters for stability

The training problem with post-norm: the residual stream's magnitude is bounded (by the LayerNorm); the sublayer outputs have to fit within this bounded range. As depth grows, the sublayers fight for budget in the bounded residual stream; deep post-norm networks become hard to train.

Pre-norm solves this: the residual stream grows freely; the LayerNorm only acts on each sublayer's input. Sublayers don't compete for budget; they all contribute additively. Empirically, pre-norm transformers train stably at hundreds of layers; post-norm typically caps around 20-30 layers without specialized techniques (warmup, careful initialization).

For modern LLMs with 60-120 layers, pre-norm is essential.

## The final norm

Pre-norm has one wrinkle: since the residual stream isn't normalized, the final output (before the LM head) can have large magnitude. Most pre-norm models add a *final* LayerNorm before the LM head:

```
hidden = transformer_block_N(hidden)
hidden = LayerNorm(hidden)  # final norm
logits = lm_head(hidden)
```

Post-norm doesn't need this; the residual is already normalized.

## What does each one trade?

**Pre-norm**:
- ✓ Stable training at large depth.
- ✓ Simpler optimization (no warmup needed for stability).
- ✗ The "representation" at deep layers is less constrained; can drift.
- ✗ Requires a final norm before the LM head.

**Post-norm**:
- ✓ Better representation properties (the residual stream is normalized; representations are more directly comparable).
- ✗ Training stability issues at depth.
- ✗ Needs careful learning-rate warmup.

For LLMs (the focus of this curriculum), pre-norm wins for the training-stability reason. The representation arguments matter less in practice; quality differences in fully-trained models are small.

## Post-norm comebacks

A few attempts to recover post-norm:

**DeepNet** (Wang et al, 2022): a careful scaling of the residual connection that allows post-norm training at depth. Used in some research models.

**Sub-LN** (Ding et al, 2023): a variant that normalizes inside the sublayer (after the linear projection but before the activation). Compromise between pre- and post-norm.

For 2026 production, pre-norm dominates. The variants are research-grade.

## Inference-side consequences

The inference path is symmetric for pre- and post-norm: just compute the forward pass according to the model's definition. The norm placement doesn't change the inference complexity.

The minor practical issue: when loading a model checkpoint, you need to match the norm placement to the model's architecture. Pre-norm and post-norm checkpoints are not interchangeable; the order of operations differs.

## What you should believe after this lesson

Three sentences:

**1. Pre-norm and post-norm differ in where the LayerNorm sits relative to the residual**: pre-norm normalizes the sublayer's input (`x + Sublayer(LN(x))`); post-norm normalizes the sublayer's output (`LN(x + Sublayer(x))`). 

**2. Pre-norm is essential for training stability at large depth** (60-120+ layers in modern LLMs); post-norm caps around 20-30 layers without specialized techniques. Pre-norm is the default for all modern LLMs.

**3. Pre-norm requires a final LayerNorm before the LM head** (because the residual stream isn't bounded); post-norm doesn't. Inference complexity is unchanged; the choice is purely about training stability and representation properties.

## Hands-on (at home)

Build both pre-norm and post-norm transformer blocks.

```python
# pre_post_norm.py
import torch
import torch.nn as nn

class PostNormBlock(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, num_heads=8, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(d, 4*d), nn.GELU(), nn.Linear(4*d, d))
        self.ln1 = nn.LayerNorm(d)
        self.ln2 = nn.LayerNorm(d)
    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        x = self.ln1(x + attn_out)
        ffn_out = self.ffn(x)
        x = self.ln2(x + ffn_out)
        return x

class PreNormBlock(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, num_heads=8, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(d, 4*d), nn.GELU(), nn.Linear(4*d, d))
        self.ln1 = nn.LayerNorm(d)
        self.ln2 = nn.LayerNorm(d)
    def forward(self, x):
        attn_in = self.ln1(x)
        attn_out, _ = self.attn(attn_in, attn_in, attn_in)
        x = x + attn_out
        ffn_in = self.ln2(x)
        ffn_out = self.ffn(ffn_in)
        x = x + ffn_out
        return x

torch.manual_seed(0)
post = PostNormBlock(64)
pre = PreNormBlock(64)
x = torch.randn(1, 8, 64)
print(f"post-norm output norm: {post(x).norm():.3f}")
print(f"pre-norm output norm: {pre(x).norm():.3f}")
# Pre-norm output magnitude can drift; post-norm is bounded.

# Stack 12 blocks of each and observe.
def stack(block_cls, n_layers, d):
    blocks = nn.ModuleList([block_cls(d) for _ in range(n_layers)])
    def forward(x):
        for b in blocks:
            x = b(x)
        return x
    return forward

post_stack = stack(PostNormBlock, 12, 64)
pre_stack = stack(PreNormBlock, 12, 64)
print(f"post-norm 12-layer output norm: {post_stack(x).norm():.3f}")
print(f"pre-norm 12-layer output norm:  {pre_stack(x).norm():.3f}")
# Pre-norm typically grows more.
```

In a real training run, deep pre-norm models train stably; deep post-norm models typically diverge unless you use warmup + careful initialization.

## Further reading

- "On Layer Normalization in the Transformer Architecture" (Xiong et al, 2020) — analysis of pre- vs post-norm.
- "DeepNet: Scaling Transformers to 1,000 Layers" (Wang et al, 2022) — making post-norm work at extreme depth.
- "Sub-LN" and other variants — for the recent norm-placement research.

Next lesson: **Tokenization for inference + module wrap.** We close Module 7 with the tokenizer story (BPE, SentencePiece, tiktoken, byte-level) and the module wrap that ties everything together.
