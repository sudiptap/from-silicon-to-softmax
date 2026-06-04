---
title: "Lesson 26 — Medusa: Multi-Head Speculative"
date: "2026-06-04"
module: "inference-from-scratch"
order: 26
tags: ["medusa", "self-speculation", "multi-head", "speculative-decoding"]
author: "Sudipta Pathak"
prerequisites: ["25-speculative-decoding"]
---

# Lesson 26 — Medusa: Multi-Head Speculative

## Why this lesson exists

Speculative decoding requires a draft model. Maintaining two models has costs: extra memory, extra training (if you want a high-acceptance-rate draft), separate version management. Medusa (Cai et al, 2024) sidesteps this: attach additional "Medusa heads" to the main model that predict the second, third, ..., k-th future tokens directly. No separate draft.

This lesson is the Medusa architecture, the training recipe, and the tradeoffs vs draft-based speculative decoding.

The lesson is reading. The Hands-on sketches Medusa-style multi-head prediction in PyTorch.

## The architecture

Standard transformer decoder: the final hidden state at position `i` is projected to logits via `W_lm_head`, predicting the next token at position `i+1`.

Medusa adds extra heads: `W_medusa_2`, `W_medusa_3`, ..., `W_medusa_k`. Each is a separate small head (typically a small MLP) projecting the same final hidden state to logits, but predicting the token at position `i+j` (for the j-th head).

```
hidden = transformer(x_<=i)
logits_1 = lm_head(hidden)                # predict x_i+1 (standard)
logits_2 = medusa_head_2(hidden)          # predict x_i+2
logits_3 = medusa_head_3(hidden)          # predict x_i+3
...
```

The heads are *added on top of a pretrained model* — the original lm_head and the base transformer are frozen; only the new Medusa heads are trained. Training data: just feed sequences and have head j predict the j-th-ahead token.

## How verification works

At each speculative step:
1. Run one forward pass; get the base model's logits + Medusa heads' logits.
2. Sample from each head: token candidates `t_1` (next), `t_2` (second-ahead), ..., `t_k` (k-th ahead).
3. Build a tree of candidate continuations from these: positions 1 from head 1; positions 2 from head 2; etc.
4. Run another forward pass to *verify* — compute the target's logits for all candidate prefixes in parallel.
5. Accept the longest valid prefix from the candidates.

The acceptance is based on consistency: does the base model agree that head 2's candidate is reasonable as the position-2 token (given position 1 was filled by head 1's candidate)?

The original Medusa paper uses "typical acceptance" — accept a candidate if its probability under the target is above a threshold. This isn't theoretically exact (unlike rejection-sampling speculative decoding) but is computationally simpler.

## Throughput

Empirically, Medusa with `k=4` heads achieves 2-3× throughput improvement on standard LLM benchmarks. Less than rejection-sampling speculative decoding with a small draft model, but with simpler deployment.

The drop in throughput is because:
- Acceptance per Medusa head is lower than acceptance from a coherent draft model (each Medusa head independently predicts; they don't reason about prior tokens).
- The candidate-tree exploration adds overhead.

## Tradeoffs vs draft model

| Aspect | Medusa | Draft model |
| ------ | ------ | ----------- |
| Extra memory | Small (extra heads ~10-50M params) | Large (draft model 0.5-1B params) |
| Training cost | Lightweight (train heads, freeze base) | None if reusing existing models; large if custom |
| Maintenance | Single model | Two models |
| Acceptance rate | 60-70% per head | 75-85% for matched draft |
| Throughput | 2-3× | 3-5× for large main model |

For deployments where you don't have a good draft model (or where you want to ship a single artifact), Medusa is attractive. For maximum throughput, draft-based speculative decoding usually wins.

## EAGLE (a Medusa successor)

EAGLE (Li et al, 2024) is a more sophisticated variant: instead of independent Medusa heads, it adds a small autoregressive sub-network that consumes the base model's hidden states and produces a sequence of speculative tokens. The autoregressive structure means later predictions can condition on earlier ones, giving higher acceptance rates.

Throughput: 3-4× over base, closer to rejection-sampling speculative decoding with a draft model. The lightweight autoregressive component is faster than running a full draft model.

EAGLE-2 (2024 update) further refined this. As of 2026, EAGLE-style self-speculation is competitive with the best draft-based approaches, with the deployment advantage of a single model.

## Production support

Medusa and EAGLE support:
- vLLM has Medusa and EAGLE support.
- TensorRT-LLM has Medusa support.
- Some HuggingFace models ship with Medusa heads pretrained (the "vicuna-medusa" variants).

Adoption isn't universal yet; draft-based speculative decoding remains more common in production deployments.

## What you should believe after this lesson

Three sentences:

**1. Medusa adds extra heads to the main model** to predict the 2nd, 3rd, ..., k-th future tokens directly. No separate draft model; the heads are trained while the base model is frozen.

**2. Throughput is 2-3× via tree-based verification** of candidate continuations — less than draft-based speculative decoding (3-5×) but with simpler deployment (single model).

**3. EAGLE (Medusa's successor) adds a lightweight autoregressive sub-network** that produces speculative tokens; it achieves competitive throughput with draft-based approaches while keeping the single-model deployment advantage. Both Medusa and EAGLE are gaining production adoption.

## Hands-on (at home)

A toy Medusa-style multi-head prediction.

```python
# medusa_heads.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class MedusaModel(nn.Module):
    """A toy 'base + Medusa heads' model."""
    def __init__(self, vocab_size, d_model, n_medusa_heads=4):
        super().__init__()
        # The "base" model: just an embedding + linear (toy).
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.base = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(),
            nn.Linear(d_model, d_model), nn.ReLU(),
        )
        # The standard LM head (next-token).
        self.lm_head = nn.Linear(d_model, vocab_size)
        # Medusa heads: each is a small MLP + projection.
        self.medusa_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model), nn.SiLU(),
                nn.Linear(d_model, vocab_size),
            )
            for _ in range(n_medusa_heads)
        ])

    def forward(self, x):
        h = self.embedding(x)
        h = self.base(h)
        # Predict next k+1 tokens from the same hidden state.
        outputs = [self.lm_head(h)]  # next
        for head in self.medusa_heads:
            outputs.append(head(h))  # 2nd, 3rd, ... ahead
        return outputs  # list of [B, N, V] tensors

# Demonstrate.
torch.manual_seed(0)
vocab, d = 1000, 128
model = MedusaModel(vocab, d, n_medusa_heads=4)
x = torch.randint(0, vocab, (1, 16))
logits_list = model(x)
print(f"Base logits shape: {logits_list[0].shape}")
print(f"Medusa head 0 logits shape: {logits_list[1].shape}")
print(f"Total candidate-positions per step: {len(logits_list)}")

# At inference, you'd sample from each and verify by re-running the base.
```

In a real Medusa deployment, the heads are trained on next-2, next-3, ... token prediction loss with the base model frozen. The verification step uses tree-based candidate exploration.

For the EAGLE-style autoregressive variant, the additional architecture is a small (1-2 layer) transformer that takes the base model's hidden state plus the previously-generated speculative token as input, producing the next speculative token.

## Further reading

- "Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads" (Cai et al, 2024).
- "EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty" (Li et al, 2024).
- "EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees" (Li et al, 2024).
- vLLM Medusa and EAGLE documentation.

Next lesson: **EAGLE & lookahead decoding.** We cover EAGLE in more depth and introduce a related technique — lookahead decoding — that uses the model's own ability to predict short n-gram sequences to bootstrap speculation without any additional training.
