---
title: "Lesson 28 — Multi-Token Prediction (MTP)"
date: "2026-06-04"
module: "inference-from-scratch"
order: 28
tags: ["mtp", "multi-token-prediction", "deepseek", "training", "speculative-decoding"]
author: "Sudipta Pathak"
prerequisites: ["27-eagle-lookahead"]
---

# Lesson 28 — Multi-Token Prediction (MTP)

## Why this lesson exists

Standard next-token prediction trains the model to predict only the *immediately next* token from each hidden state. Multi-Token Prediction (MTP) — used at training time by DeepSeek-V3 (2024) and others — trains the model to predict the next *several* tokens from each hidden state, via auxiliary heads.

The motivation isn't purely inference speed (though it can be used for speculative decoding); the bigger benefit is **training-time signal**: each token position contributes more learning signal, which improves sample efficiency.

This lesson covers MTP as a training technique, how it differs from Medusa (which adds heads to a *pretrained* model post-hoc), and how MTP enables faster inference via speculative decoding using the auxiliary heads.

The lesson is reading. The Hands-on builds an MTP-style training loop.

## The training setup

For a model with `K` MTP heads (DeepSeek-V3 uses K=2 — predict next + next-2), the loss at each token position is:

```
L_total = L_next + sum_{k=2}^{K+1} α_k × L_(k-th-next)
```

Where:
- `L_next` is the standard next-token cross-entropy loss.
- `L_(k-th-next)` is cross-entropy for predicting the token `k` positions ahead.
- `α_k` are weighting factors (typically decreasing for further-ahead heads).

Each MTP head has the same architecture: a small projection block from the shared backbone to vocabulary logits. The heads share most of the model (the transformer layers) and differ only in the final projection.

The total parameter cost: a few percent of the base model size (one extra projection per head).

## Why this helps training

Two reasons:

**1. More learning signal per token.** Each position contributes K+1 loss terms instead of 1. The gradient information per training token is richer; the model converges faster on the same amount of data.

**2. The model learns better representations.** Predicting multiple tokens ahead forces the hidden state to encode more about future context. This regularizes the model toward useful representations.

DeepSeek-V3's experiments showed ~10-15% improvement in convergence speed on the same data and ~0.5-1% absolute accuracy improvement on downstream benchmarks. Real but not earth-shattering; valuable as part of a stack of training techniques.

## MTP at inference

The auxiliary heads can serve dual purpose at inference:

**Standard inference**: ignore the MTP heads; use only the main next-token head. Quality is the trained baseline.

**Speculative inference**: use the MTP heads as a draft. The k-th MTP head predicts the k-th-ahead token; treat these as candidates; verify against the main next-token head's predictions.

The speculative-inference variant is structurally similar to Medusa (Lesson 26), but with one key advantage: the MTP heads were trained alongside the base model, so they're more aligned with the model's representation than post-hoc-trained Medusa heads. Higher acceptance rates.

DeepSeek-V3 uses this dual purpose: train with MTP for the training-signal benefit, use MTP heads at inference for speculation. Reports a 2-3× inference speedup over standard next-token decoding.

## MTP vs Medusa

| Aspect | MTP | Medusa |
| ------ | --- | ------ |
| When trained | During base-model training | After base-model training |
| Base model frozen during head training? | No | Yes |
| Quality gain on base? | Yes (~0.5-1%) | No |
| Inference speedup | 2-3× via speculation | 2-3× via speculation |
| Engineering cost | Must control training | Lightweight (fine-tune the heads) |

The choice depends on whether you're training from scratch (MTP is the right call) or augmenting an existing pretrained model (Medusa is the practical option).

## Looking ahead: MTP and reasoning models

A speculative observation: MTP-trained models seem to do somewhat better at long-form generation. The training-time signal about "what comes a few tokens later" may help the model plan ahead in a way that pure next-token training doesn't.

This is consistent with the broader observation that reasoning models (o1, R1) — which generate very long chains of thought — benefit from any training technique that encourages planning. MTP fits this picture; it's plausible that future reasoning models more aggressively use MTP-like multi-step training.

As of 2026, it's an open question. The DeepSeek-V3 paper notes the benefit but doesn't fully attribute it to MTP.

## What you should believe after this lesson

Three sentences:

**1. MTP trains the model to predict multiple future tokens** via auxiliary heads, with each head adding a loss term to the total training objective. The training-signal benefit is real (~0.5-1% absolute accuracy improvement, ~10-15% faster convergence).

**2. The MTP heads serve dual purpose**: training-time learning signal, inference-time speculative decoding draft. DeepSeek-V3 uses this combination for 2-3× inference speedup with quality matching the base.

**3. MTP differs from Medusa primarily in timing**: MTP is trained alongside the base; Medusa is bolted onto a pretrained base. MTP's tighter integration produces higher acceptance rates at the cost of requiring control over the training process.

## Hands-on (at home)

A minimal MTP-style multi-token loss in PyTorch.

```python
# mtp_training.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class MTPModel(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, n_mtp_heads=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        # Simplified "transformer" backbone.
        self.backbone = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead=8, batch_first=True),
            num_layers=n_layers,
        )
        # The standard next-token head.
        self.lm_head = nn.Linear(d_model, vocab_size)
        # MTP heads: predict 2nd, 3rd, ... ahead.
        self.mtp_heads = nn.ModuleList([
            nn.Linear(d_model, vocab_size) for _ in range(n_mtp_heads)
        ])

    def forward(self, x):
        h = self.embedding(x)
        h = self.backbone(h)
        next_logits = self.lm_head(h)  # [B, N, V]
        mtp_logits = [head(h) for head in self.mtp_heads]  # list of [B, N, V]
        return next_logits, mtp_logits

def mtp_loss(model, x, targets, alphas=[0.5, 0.3]):
    # x: [B, N] input tokens; targets: [B, N] = next-tokens (the standard target).
    next_logits, mtp_logits = model(x)
    # Standard loss: predict next token.
    L_next = F.cross_entropy(next_logits.view(-1, next_logits.size(-1)), targets.view(-1))
    # MTP losses: predict 2nd, 3rd, ... ahead.
    total = L_next
    for k, (head_logits, alpha) in enumerate(zip(mtp_logits, alphas), start=2):
        # Target for head k: targets shifted left by (k-1) positions.
        shifted_targets = targets[:, k-1:]  # truncate first k-1
        head_logits_trunc = head_logits[:, :-(k-1)]  # last k-1 don't have a valid target
        L_k = F.cross_entropy(
            head_logits_trunc.contiguous().view(-1, head_logits_trunc.size(-1)),
            shifted_targets.contiguous().view(-1)
        )
        total = total + alpha * L_k
    return total, L_next, [F.cross_entropy(
        mtp_logits[k][:, :-(k+1)].contiguous().view(-1, mtp_logits[k].size(-1)),
        targets[:, k+1:].contiguous().view(-1)
    ) for k in range(len(mtp_logits))]

# Toy training step.
torch.manual_seed(0)
V, D, L = 100, 64, 2
model = MTPModel(V, D, L, n_mtp_heads=2)
x = torch.randint(0, V, (2, 16))
targets = torch.randint(0, V, (2, 16))
loss, l_next, l_mtps = mtp_loss(model, x, targets)
print(f"Total loss: {loss.item():.3f}")
print(f"Next-token loss: {l_next.item():.3f}")
print(f"MTP head 2 loss: {l_mtps[0].item():.3f}")
print(f"MTP head 3 loss: {l_mtps[1].item():.3f}")
```

In real DeepSeek-V3 training, the MTP heads use the same target shifts but with the model's full architecture (not a toy transformer); the alphas are tuned and the heads are deeper.

## Further reading

- "DeepSeek-V3 Technical Report" (DeepSeek, 2024) — the production MTP application.
- "Better & Faster Large Language Models via Multi-token Prediction" (Gloeckle et al, Meta, 2024) — earlier MTP study.
- Comparisons of MTP-trained vs single-target-trained models across recent literature.

End of Part 4. Next: Part 5 begins with **MoE from scratch** — the architectural shift behind the largest open models. We've covered attention's evolution and decoding mechanics; the next 6 lessons build the MoE machinery from gating to routing to expert parallelism.
