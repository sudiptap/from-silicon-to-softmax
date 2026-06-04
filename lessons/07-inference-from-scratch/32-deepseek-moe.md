---
title: "Lesson 32 — DeepSeek MoE: Fine-Grained Experts + Shared Experts"
date: "2026-06-04"
module: "inference-from-scratch"
order: 32
tags: ["moe", "deepseek", "fine-grained", "shared-experts", "v3"]
author: "Sudipta Pathak"
prerequisites: ["31-switch-vs-gshard"]
---

# Lesson 32 — DeepSeek MoE: Fine-Grained Experts + Shared Experts

## Why this lesson exists

DeepSeek's MoE architecture (introduced in DeepSeek-V2, refined in DeepSeek-V3) takes a different path from Mixtral's "8 large experts, top-2." Instead: **256 small experts, top-8** ("fine-grained"), plus a handful of **shared experts** that activate on every token.

The design has produced some of the strongest open-weights models in 2024-2026 (DeepSeek-V3 671B total / 37B active). This lesson covers the fine-grained-MoE design philosophy and why the variant has succeeded.

The lesson is reading. The Hands-on builds a small fine-grained MoE with shared experts.

## The motivation

Two observations the DeepSeek team made:

**1. Fewer-larger experts vs more-smaller**: with 8 experts of size D each, you get 8D of FFN parameters. With 64 experts of size D/8 each, you also get 8D — but you can route top-8 of the 64 small experts per token, getting a richer combination.

The hypothesis: combining many small experts gives finer-grained specialization than combining few large experts. Each small expert can specialize in a narrow topic; the combination is the model's per-token "ensemble."

**2. Some computation is needed for every token**: not all useful FFN computation is specialized. A model needs to do some baseline processing on every token (e.g., normalization, general feature extraction) regardless of content. A standard MoE wastes capacity by making every token select all its computation from "specialized" experts; better to have a small dedicated "shared expert" that runs on every token.

These together motivate the DeepSeek design:
- Many small "routed" experts (256 of them in V3).
- A few "shared" experts (1-2) that always activate.
- High k (8 of 256) so the per-token output is a combination of multiple specialists.

## The architecture

For each token:

1. *Shared experts* always activate; their outputs are summed.
2. *Routed experts* are scored by the gate; top-8 of 256 are selected; their outputs are weighted-summed.
3. The total FFN output is the shared output plus the routed output.

```python
# Pseudo-code:
shared_output = sum(shared_expert(x) for shared_expert in shared_experts)
gate_logits = router(x)
top_k_logits, top_k_indices = gate_logits.topk(k=8)
weights = top_k_logits.softmax(dim=-1)
routed_output = weighted_sum(routed_experts[top_k_indices], weights, x)
ffn_out = shared_output + routed_output
```

## Why fine-grained works

The empirical results: at the same total parameter count and same active parameter count, fine-grained MoE beats coarse-grained MoE by ~1-2% on most benchmarks.

The intuition: more diversity in the expert combinations means more "modes" the model can express per token. With 8 large experts × top-2, there are 28 possible expert combinations. With 256 small experts × top-8, there are millions.

Most of those combinations aren't used in practice, but the *expressive capacity* is much higher. The router can carve up the input space more finely.

## Why shared experts help

The shared experts handle the "always needed" computation:
- General language modeling features (next-token statistics).
- Common content patterns.
- Baseline FFN transformation.

Without shared experts, the routed experts have to learn this baseline too — wasting capacity. With dedicated shared experts, the routed experts specialize more.

Empirically: shared experts contribute about ~10-20% of the FFN's expressivity but cost only a fraction of the parameter budget. Good ratio.

## Expert balance with fine-grained MoE

256 experts is a lot. Naive load balancing struggles — auxiliary loss with this many experts is unstable; the gradients per expert are tiny.

DeepSeek-V2 used aux-loss-balancing (Lesson 30 variants); DeepSeek-V3 introduced aux-loss-free balancing (Lesson 33) which is more elegant. Either way, fine-grained MoE training requires more careful balancing infrastructure than coarse-grained.

## Compute scaling

Per-token compute for DeepSeek-V3 (256 routed experts, top-8 + 1 shared expert):
- 8 routed expert FFNs at small size + 1 shared expert FFN at small size.
- Total: ~9 × (D_model × D_ff_per_expert).
- Compare to Mixtral 8×7B (top-2): 2 × (D_model × D_ff_per_expert).

If small experts are 1/4 the size of Mixtral's, DeepSeek's per-token compute is ~9 × 0.25 = 2.25 — comparable to Mixtral's. The capacity (total parameters) is much higher.

## Production considerations

Implementing fine-grained MoE efficiently is harder than coarse-grained:
- Routing 256 experts requires more bookkeeping.
- Per-expert GEMM batches are smaller (fewer tokens per expert), which means worse GEMM throughput.
- Communication in distributed training (all-to-all between experts; Lesson 34) has more parties.

DeepSeek's contributions include their own Megablocks-style kernels and the open-source DeepSeek-MoE inference library.

For 2026 deployment: vLLM, SGLang, and TensorRT-LLM all support DeepSeek-style MoE; the throughput is competitive though not always equal to comparably-sized dense models.

## What you should believe after this lesson

Three sentences:

**1. DeepSeek MoE pairs many small "routed" experts (256, top-8) with a few "shared" experts (1-2) that always activate.** Fine-grained routing gives more expressive expert combinations; shared experts handle the baseline computation that doesn't need specialization.

**2. The design produced some of the strongest open-weights LLMs in 2024-2026** (DeepSeek-V2, V3); the win over coarse-grained MoE is ~1-2% on standard benchmarks at the same active parameter count.

**3. Implementation is harder** than coarse-grained MoE — load balancing with 256 experts requires care, per-expert GEMM throughput is worse, communication costs are higher. Production support has caught up but isn't yet as mature as for Mixtral-style MoE.

## Hands-on (at home)

A small fine-grained MoE with shared experts.

```python
# fine_grained_moe.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class FineGrainedMoE(nn.Module):
    def __init__(self, d_model, d_ff_per_expert, n_routed_experts, n_shared_experts, k):
        super().__init__()
        self.n_routed = n_routed_experts
        self.k = k
        self.shared_experts = nn.ModuleList([
            nn.Sequential(nn.Linear(d_model, d_ff_per_expert), nn.SiLU(),
                          nn.Linear(d_ff_per_expert, d_model))
            for _ in range(n_shared_experts)
        ])
        self.routed_experts = nn.ModuleList([
            nn.Sequential(nn.Linear(d_model, d_ff_per_expert), nn.SiLU(),
                          nn.Linear(d_ff_per_expert, d_model))
            for _ in range(n_routed_experts)
        ])
        self.router = nn.Linear(d_model, n_routed_experts)

    def forward(self, x):
        B, N, D = x.shape
        x_flat = x.view(-1, D)
        # Shared expert path: always active.
        shared_out = sum(e(x_flat) for e in self.shared_experts)
        # Routed expert path.
        gate = self.router(x_flat)
        top_k_logits, top_k_idx = gate.topk(self.k, dim=-1)
        weights = top_k_logits.softmax(dim=-1)
        routed_out = torch.zeros_like(x_flat)
        for e in range(self.n_routed):
            em = (top_k_idx == e)
            tm = em.any(dim=-1)
            if not tm.any(): continue
            ew = (em.float() * weights).sum(dim=-1)[tm]
            eo = self.routed_experts[e](x_flat[tm])
            routed_out[tm] += ew.unsqueeze(-1) * eo
        return (shared_out + routed_out).view(B, N, D)

# Demo with small numbers for tractability.
torch.manual_seed(0)
moe = FineGrainedMoE(
    d_model=128, d_ff_per_expert=64,
    n_routed_experts=32, n_shared_experts=2, k=4
)
x = torch.randn(2, 8, 128)
out = moe(x)
print(f"input shape: {x.shape}")
print(f"output shape: {out.shape}")
print(f"per-token compute (relative): "
      f"{moe.shared_experts.__len__() + moe.k} expert FFNs of size 64")
# Compare to a Mixtral-style design with 8 large experts and top-2.
```

For the production DeepSeek-V3 inference architecture, see DeepSeek's official release.

## Further reading

- "DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models" (Dai et al, 2024) — the foundational DeepSeek MoE paper.
- "DeepSeek-V2" and "DeepSeek-V3 Technical Report" — production deployments.
- "ST-MoE" (Zoph et al, 2022) — earlier exploration of fine-grained MoE.

Next lesson: **Aux-loss-free balancing (DeepSeek-V3).** A more elegant load-balancing approach: instead of auxiliary loss, use a per-expert bias that's adjusted continuously to maintain balance. We close out the algorithmic MoE chapters before moving to expert parallelism.
