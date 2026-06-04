---
title: "Lesson 30 — Load Balancing: Auxiliary Loss and Expert Utilization"
date: "2026-06-04"
module: "inference-from-scratch"
order: 30
tags: ["moe", "load-balancing", "auxiliary-loss", "router-collapse", "expert-utilization"]
author: "Sudipta Pathak"
prerequisites: ["29-moe-from-scratch"]
---

# Lesson 30 — Load Balancing: Auxiliary Loss and Expert Utilization

## Why this lesson exists

The naive MoE router has a problem: it tends to collapse. The router learns to send tokens to a few "favorite" experts; other experts receive few or no tokens; the unused experts don't get gradient signal; the router becomes more confident in its favorites; the gap widens. The model converges to using maybe 20-30% of its experts effectively.

This wastes the MoE's whole reason for existence. Load balancing — making the router spread tokens evenly across experts — is essential.

This lesson covers the standard load-balancing techniques: auxiliary loss, capacity factor + token-dropping, and the alternatives that emerged in 2023-2024.

The lesson is reading. The Hands-on adds load-balancing loss to the Lesson 29 MoE.

## Why routers collapse

In the early stages of training:
- The router's outputs are random.
- Some experts happen to get more tokens than others.
- Those experts get more gradient signal; they improve faster.
- The router's gate weights drift to prefer the improved experts.
- The preference loop reinforces.

After a few thousand steps, the model is using 2-3 experts heavily and the rest are dead weight. The total compute is the same (you still ran the routing logic), but the model behaves like a much smaller dense model.

The fix: explicitly penalize unequal expert usage.

## Auxiliary loss

The standard fix from Shazeer 2017 onward: add an auxiliary loss that encourages equal usage.

For each batch, compute:
- `f_i = fraction of tokens routed to expert i` (the "load").
- `P_i = average gating probability for expert i` (the "importance").

The auxiliary loss:

```
L_aux = n_experts × sum_i (f_i × P_i)
```

Why this works: if the router is balanced, `f_i ≈ P_i ≈ 1/n_experts`, and the sum is `n_experts × n_experts × (1/n_experts)² = 1`. If one expert dominates, both `f_i` and `P_i` for it are large, the product is much bigger, and the loss grows. The model learns to keep `f_i` and `P_i` close to `1/n_experts`.

Added to the training loss with a weighting factor:

```
L_total = L_lm + α × L_aux  # typical α: 0.01
```

The weight `α` is small — too high and the model sacrifices quality to balance load; too low and the router collapses anyway.

## Capacity factor and token dropping

A complementary approach used by Switch Transformer (Lesson 31): cap the number of tokens any expert can receive per batch.

The *capacity factor* `c` (typically 1.0-1.5) sets the cap:

```
capacity = ceil(c × tokens_per_batch / n_experts)
```

For c=1.25 and 1024 tokens across 8 experts, each expert can receive at most `ceil(1.25 × 1024 / 8) = 160` tokens.

If the router sends more than `capacity` tokens to expert i, the excess are *dropped* (treated as zero contribution).

This hard cap forces the router to spread; if it tries to send too many to one expert, the extras are wasted, providing a learning signal to route them elsewhere.

Trade-offs:
- Higher capacity (c=1.5): less dropping, less balanced, more memory.
- Lower capacity (c=1.0): more dropping, better balanced, less memory.
- Dropping tokens hurts quality; the model effectively loses that token's contribution from the FFN.

Token-dropping was the standard with Switch Transformer; modern MoE recipes (like DeepSeek-V3's no-aux-loss approach, Lesson 33) avoid it.

## Router z-loss

A subtler issue: the router's logits can grow unbounded, making the softmax saturated and the gradients sparse. Router z-loss adds a regularization on the logit magnitudes:

```
L_z = sum (log_sum_exp(logits))²
```

This keeps the logits bounded. Commonly added with a small weight (~1e-3) alongside the auxiliary loss.

## Diagnostic: monitor expert utilization

The actionable monitoring metric: per-batch, what fraction of tokens went to each expert? Plot it over training; if you see one expert dominating, the load balancing is failing.

Healthy distribution: roughly uniform across experts (within 2× of `1/n_experts`).
Collapse: one or two experts get >50% of tokens.

Most production MoE training pipelines log this. If you see collapse, increase the auxiliary loss weight or reduce the capacity factor.

## What you should believe after this lesson

Three sentences:

**1. Naive MoE routers collapse**: a positive-feedback loop concentrates tokens on a few "favorite" experts, leaving others dead weight. Without load balancing, MoE quickly degrades to a dense model with extra unused parameters.

**2. Auxiliary loss is the standard fix**: `L_aux = n_experts × sum_i (f_i × P_i)`, where `f_i` is the routing fraction and `P_i` is the gate probability per expert. Added to total loss with a small weight (~0.01); balances the expert usage.

**3. Capacity factor + token dropping is a complementary mechanism** that hard-caps how many tokens each expert can receive per batch, with excess dropped. Forces balance via constraints rather than soft penalty; standard in Switch Transformer.

## Hands-on (at home)

Add load-balancing loss to the MoE layer.

```python
# moe_with_aux_loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class MoELayerWithAux(nn.Module):
    def __init__(self, d_model, d_ff, n_experts, k=2):
        super().__init__()
        self.n_experts = n_experts
        self.k = k
        self.router = nn.Linear(d_model, n_experts, bias=False)
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(d_model, d_ff), nn.SiLU(), nn.Linear(d_ff, d_model))
            for _ in range(n_experts)
        ])

    def forward(self, x):
        B, N, D = x.shape
        x_flat = x.view(-1, D)
        gate_logits = self.router(x_flat)  # [B*N, n_experts]
        gate_probs = gate_logits.softmax(dim=-1)
        top_k_logits, top_k_indices = gate_logits.topk(self.k, dim=-1)
        weights = top_k_logits.softmax(dim=-1)

        # === Compute auxiliary loss components ===
        # f_i: fraction of tokens that selected expert i (in their top-k).
        expert_mask = torch.zeros(x_flat.shape[0], self.n_experts, device=x.device)
        expert_mask.scatter_(1, top_k_indices, 1.0)
        f = expert_mask.mean(dim=0)  # [n_experts]
        # P_i: average gate prob for expert i.
        P = gate_probs.mean(dim=0)  # [n_experts]
        # Load-balancing loss.
        aux_loss = self.n_experts * (f * P).sum()

        # === Routing (naive loop) ===
        output = torch.zeros_like(x_flat)
        for expert_idx in range(self.n_experts):
            em = (top_k_indices == expert_idx)
            tm = em.any(dim=-1)
            if not tm.any():
                continue
            ew = (em.float() * weights).sum(dim=-1)[tm]
            eo = self.experts[expert_idx](x_flat[tm])
            output[tm] += ew.unsqueeze(-1) * eo

        return output.view(B, N, D), aux_loss

# Test: compare expert utilization with and without aux loss in a tiny training run.
torch.manual_seed(0)
moe = MoELayerWithAux(d_model=64, d_ff=128, n_experts=8, k=2)
opt = torch.optim.Adam(moe.parameters(), lr=1e-3)

# Random data; loss is a regression to constant target (just to drive learning).
target = torch.randn(4, 8, 64)
for step in range(200):
    x = torch.randn(4, 8, 64)
    out, aux = moe(x)
    main_loss = F.mse_loss(out, target)
    total = main_loss + 0.01 * aux
    opt.zero_grad(); total.backward(); opt.step()
    if step % 50 == 0:
        # Inspect expert utilization.
        with torch.no_grad():
            gate = moe.router(x.view(-1, 64))
            top_k = gate.topk(2, dim=-1).indices
            utilization = torch.bincount(top_k.view(-1), minlength=8).float() / (top_k.numel())
        print(f"step {step}: main={main_loss:.3f}, aux={aux:.3f}, util={utilization.numpy().round(3)}")
```

Without the auxiliary loss, you'd see one or two experts dominate. With it, the utilization stays roughly uniform.

## Further reading

- "Outrageously Large Neural Networks" (Shazeer et al, 2017) — original auxiliary loss formulation.
- "Switch Transformer" (Fedus et al, 2021) — capacity factor; token dropping.
- "ST-MoE" (Zoph et al, 2022) — router z-loss and stability tricks.
- "DeepSeek-V3" (DeepSeek, 2024) — Lesson 33's reference; introduces aux-loss-free balancing.

Next lesson: **Switch Transformer vs GShard.** Two foundational MoE designs that established the modern recipe. Switch uses top-1 routing (one expert per token); GShard uses top-2. We cover both and the tradeoff.
