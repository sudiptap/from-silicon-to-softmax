---
title: "Lesson 31 — Switch Transformer (top-1) vs GShard (top-2)"
date: "2026-06-04"
module: "inference-from-scratch"
order: 31
tags: ["moe", "switch-transformer", "gshard", "top-1", "top-2", "routing"]
author: "Sudipta Pathak"
prerequisites: ["30-load-balancing"]
---

# Lesson 31 — Switch Transformer (top-1) vs GShard (top-2)

## Why this lesson exists

The first major production MoE recipes from Google were GShard (2020) and Switch Transformer (2021). They differ on one parameter: **how many experts to route each token to** — `k=2` (GShard) or `k=1` (Switch).

That single parameter shapes everything downstream: compute per token, training dynamics, expert utilization patterns, communication costs in distributed training. The modern MoE recipes (Mixtral with k=2, DeepSeek with various, others) all sit on this spectrum.

This lesson covers the two designs, the practical implications of each choice, and the takeaway that has solidified over the subsequent literature.

The lesson is reading. The Hands-on compares top-1 vs top-2 routing.

## GShard (top-2)

GShard (Lepikhin et al, 2020) is the cleaner of the two designs in retrospect:
- Route each token to *top-2* experts.
- Softmax over the two gate scores; combine experts' outputs as a weighted sum.
- Auxiliary loss for balancing.
- Capacity factor + token dropping.

The compute per token: 2 expert forward passes. The output is a smooth combination of two experts' contributions.

Why top-2 (and not top-1 or top-k for larger k):
- Top-1 has discrete routing decisions; gradients flow only through the gate, not through alternate experts. Training is sometimes unstable.
- Higher k means more compute per token. The compute/capacity tradeoff degrades.
- k=2 is the sweet spot: dense enough to have stable gradients, sparse enough to keep compute manageable.

Production GShard models: Google's various large translation models, some early-2020s research.

## Switch Transformer (top-1)

Switch Transformer (Fedus et al, 2021) argued for top-1:
- Route each token to *one* expert (the highest-scoring).
- Use only that expert's output; no combination.
- Auxiliary loss + tighter capacity factor + harder token dropping.

The compute per token: 1 expert forward pass — same as a dense model with one expert's worth of FFN. The model can have many experts (Switch experimented with up to 2048) at the same compute cost as a dense FFN.

Why top-1:
- **Lowest possible compute** for a given expert count. 2× faster than top-2 at the same expert count.
- **Cleaner sparsity**. Each token corresponds to exactly one expert, simplifying routing.
- **Larger expert counts feasible**. Switch went to 2048 experts; top-2 at that scale would be too much compute.

Trade-offs:
- Less gradient stability (only the chosen expert sees the gradient).
- Quality sometimes worse than top-2 at the same effective dense-FFN-equivalent compute.
- Token dropping is more impactful (a dropped token loses 100% of its FFN contribution, not 50%).

## The empirical convergence

By 2024, the practitioner consensus settled around top-2 for most production MoE:
- Mixtral 8×7B: top-2.
- Mixtral 8×22B: top-2.
- Most open-weights MoE models: top-2.

DeepSeek-V3 is the notable exception: it uses top-8 of 256 experts (much higher than k=2). The "fine-grained MoE" design (Lesson 32) — many small experts with higher k — is a different point in the design space that we cover next.

The practical decision matrix for choosing k:
- **k=1**: only if you want extreme compute-to-capacity ratio and have engineering bandwidth for the training-stability issues.
- **k=2**: the sweet spot for most cases.
- **k=4 or higher**: only with fine-grained MoE (many small experts) where each is too small to carry much alone.

## Token dropping in production

Token dropping is uncomfortable. Conceptually you're throwing away the FFN contribution for some tokens. In practice it works because:
- The router learns to avoid creating drop situations (it spreads tokens).
- The dropped contribution is replaced by the residual connection (the token's input passes through to the next layer unchanged).
- The model's other layers compensate.

Modern MoE recipes often avoid token dropping entirely (capacity factor very high, or no capacity limit) at the cost of memory waste for unbalanced batches.

## Comparison table

| Property | Switch (top-1) | GShard (top-2) |
| -------- | -------------- | -------------- |
| k | 1 | 2 |
| Compute per token | 1× dense FFN | 2× dense FFN |
| Combination | None (single expert) | Softmax-weighted sum |
| Token dropping | Common | Less common |
| Training stability | Harder | Easier |
| Max experts (historically) | 2048 | 256 (Mixtral 8×7B uses 8) |
| Modern adoption | Rare | Standard |

The modern winner: top-2 (GShard-style) for most. Top-1 (Switch-style) for specific scaling experiments.

## What you should believe after this lesson

Three sentences:

**1. The two foundational MoE recipes** differ on `k` — Switch uses top-1, GShard uses top-2. The choice determines compute per token, gradient stability, and effective max expert count.

**2. Top-2 won the modern consensus** — Mixtral, most open-weights MoE use it. The compute-cost-vs-stability sweet spot. Top-1 is reserved for specific compute-efficiency experiments at very high expert counts.

**3. DeepSeek-V3's high k (top-8 of 256)** is a different point in the design space — fine-grained experts (Lesson 32) where each expert is small and you combine more of them. We cover that next.

## Hands-on (at home)

Compare top-1 vs top-2 routing on the same input.

```python
# top1_vs_top2.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class MoEKHead(nn.Module):
    def __init__(self, d_model, d_ff, n_experts, k):
        super().__init__()
        self.k = k
        self.n_experts = n_experts
        self.router = nn.Linear(d_model, n_experts)
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(d_model, d_ff), nn.SiLU(), nn.Linear(d_ff, d_model))
            for _ in range(n_experts)
        ])

    def forward(self, x):
        B, N, D = x.shape
        x_flat = x.view(-1, D)
        gate = self.router(x_flat)
        top_k_logits, top_k_idx = gate.topk(self.k, dim=-1)
        weights = top_k_logits.softmax(dim=-1)
        out = torch.zeros_like(x_flat)
        for e in range(self.n_experts):
            em = (top_k_idx == e)
            tm = em.any(dim=-1)
            if not tm.any(): continue
            ew = (em.float() * weights).sum(dim=-1)[tm]
            eo = self.experts[e](x_flat[tm])
            out[tm] += ew.unsqueeze(-1) * eo
        return out.view(B, N, D)

torch.manual_seed(0)
moe_top1 = MoEKHead(d_model=64, d_ff=128, n_experts=8, k=1)
moe_top2 = MoEKHead(d_model=64, d_ff=128, n_experts=8, k=2)
# Reset router/experts to same values for fair compute-cost comparison.

x = torch.randn(2, 16, 64)
out1 = moe_top1(x)
out2 = moe_top2(x)
print(f"top-1 output shape: {out1.shape}")
print(f"top-2 output shape: {out2.shape}")
print(f"top-1 vs top-2 difference (random init): {(out1 - out2).abs().mean().item():.4f}")
# top-2 has roughly 2x the compute of top-1.

import time
N_iter = 50
t0 = time.time()
for _ in range(N_iter): moe_top1(x)
t_top1 = time.time() - t0
t0 = time.time()
for _ in range(N_iter): moe_top2(x)
t_top2 = time.time() - t0
print(f"top-1: {t_top1/N_iter*1000:.2f} ms; top-2: {t_top2/N_iter*1000:.2f} ms")
print(f"top-2 / top-1 ratio: {t_top2 / t_top1:.2f}")
```

The compute ratio should be roughly 2× (top-2 does 2 expert FFNs per token; top-1 does 1).

## Further reading

- "GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding" (Lepikhin et al, 2020).
- "Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity" (Fedus et al, 2021).
- "Mixtral of Experts" (Jiang et al, 2024) — production top-2 design.
- "ST-MoE: Designing Stable and Transferable Sparse Expert Models" (Zoph et al, 2022) — training-stability tricks for top-2 and top-1.

Next lesson: **DeepSeek MoE — fine-grained experts + shared experts.** A different design philosophy: many small experts (256+ with k=8) plus a few "shared experts" that always activate. Has shown excellent results at the 200B+ parameter scale.
