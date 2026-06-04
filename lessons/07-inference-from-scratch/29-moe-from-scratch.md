---
title: "Lesson 29 — MoE from Scratch: Gating and Top-k Routing"
date: "2026-06-04"
module: "inference-from-scratch"
order: 29
tags: ["moe", "mixture-of-experts", "routing", "gating", "sparse"]
author: "Sudipta Pathak"
prerequisites: ["28-multi-token-prediction"]
---

# Lesson 29 — MoE from Scratch: Gating and Top-k Routing

## Why this lesson exists

MoE (Mixture of Experts) is the architectural shift that defines the largest open models in 2026. Mixtral 8×7B, DeepSeek-V3, Qwen2.5-Max, Grok all use it. The pitch: replace the dense FFN in each transformer block with multiple parallel "expert" FFNs; for each token, route it to a small subset of experts; combine the experts' outputs.

The result: a model with N experts has the *parameter count* of N times a dense FFN, but the *active compute* per token of only k experts (where k << N). DeepSeek-V3 has 671B total parameters but only ~37B active per token; it costs (in compute) like a 37B dense model but has the capacity of a 671B dense model.

This lesson builds the MoE FFN from primitives: the gate that decides which experts to use, the routing, and the combination.

The lesson is reading. The Hands-on builds a minimal MoE layer.

## The architecture

Replace each transformer block's FFN with:

1. A *router* (typically a single linear layer that produces gate scores).
2. *N* expert FFNs (each is a standard SwiGLU/MLP block).
3. A combination step.

For each token:

1. Compute gate scores: `gate_logits = W_router(x)` of shape `[B, N, n_experts]`.
2. Pick the top-k experts: `top_k_indices, top_k_logits = gate_logits.topk(k)`.
3. Softmax over the top-k gate logits: `weights = softmax(top_k_logits)` of shape `[B, N, k]`.
4. For each selected expert, compute the expert's output for this token: `expert_out_i = expert_i(x)`.
5. Combine: `output = sum_i weights_i × expert_out_i`.

The key terms:
- `n_experts`: total number of experts (typically 8, 16, 32, 64, or 256).
- `k`: number of experts active per token (typically 1, 2, or 4).
- *Sparsity ratio*: `k / n_experts`. DeepSeek-V3 has 256 experts and k=8 → ~3% sparsity.

## Why MoE works

Two empirical observations:

**1. Different experts learn different "skills."** Some experts specialize in code, others in dialogue, others in math. The routing learns to send relevant tokens to relevant experts.

**2. The capacity scales with total parameters; the compute scales with active.** A 16-expert MoE has 16× the parameters of one dense FFN but only ~2× the per-token compute (if k=2). The model has 16× the knowledge-storing capacity at 2× the compute.

The catch: not all experts get used equally without intervention. The router can collapse to using only a few experts, wasting the others. Load balancing (Lesson 30) addresses this.

## The routing computation

For a batch of `B × N` tokens and `n_experts` experts:

```python
# Compute gate scores for every token-expert pair.
gate_logits = router(x)  # [B, N, n_experts]

# Pick top-k experts per token.
top_k_logits, top_k_indices = gate_logits.topk(k, dim=-1)  # both [B, N, k]

# Softmax over selected experts to get weights.
weights = top_k_logits.softmax(dim=-1)  # [B, N, k]
```

Then, for each token, dispatch it to its chosen experts:

```python
# Naive: loop over experts, gather tokens routed to each.
output = torch.zeros_like(x)
for expert_idx in range(n_experts):
    # Find tokens that selected this expert.
    expert_mask = (top_k_indices == expert_idx).any(dim=-1)
    if not expert_mask.any():
        continue
    expert_inputs = x[expert_mask]  # tokens for this expert
    expert_output = experts[expert_idx](expert_inputs)
    # Find the corresponding weights.
    expert_weights = weights[expert_mask][top_k_indices[expert_mask] == expert_idx]
    # Scatter back.
    output[expert_mask] += expert_weights.unsqueeze(-1) * expert_output
```

This is the conceptual implementation. Production MoE layers use much more sophisticated routing (Megablocks, GroupGEMM kernels) to keep things efficient.

## The dispatch problem

The expert FFN itself wants to operate on a contiguous batch of inputs (for the GEMM). If you naively dispatch token-by-token to experts, you destroy batching efficiency.

The standard production approach:
1. *Permute* tokens so all tokens going to expert 0 are together, then expert 1, etc.
2. Run each expert's GEMM on its own contiguous slice.
3. *Unpermute* outputs back to the original token order.

The permute/unpermute operations are extra memory traffic. They're handled efficiently by specialized kernels (Megablocks, NVIDIA's transformer engine).

For server inference at modest scale, the routing overhead is ~5-15% of the per-token cost. For very large MoE with hundreds of experts, the overhead is more significant; expert parallelism (Lesson 34) sharding across devices is essential.

## MoE at inference

For inference, the routing decisions are the same as training: route per-token, pick top-k experts, combine.

The KV cache and attention layers are *not* MoE-fied — they're dense. Only the FFN sublayer is the mixture. This matters for parameter accounting: the attention part is the same as a dense model, the FFN multiplies by n_experts.

Per Llama 3.1 8B with dense FFN: ~7B params total, ~5B in FFNs.
Mixtral 8×7B with MoE: ~47B params total (~5B × 8 in experts + the dense attention), ~13B active per token (top-2 of 8).

The 8× scaling of FFN parameters with the same attention parameters is why Mixtral has ~6× more total params than a similarly-named 7B dense model but only ~2× active params per token.

## What you should believe after this lesson

Three sentences:

**1. MoE replaces the dense FFN with `n_experts` parallel experts and a router that sends each token to top-k experts.** Total parameters scale with `n_experts`; active compute scales with `k`. The compute-to-capacity ratio is `k / n_experts`.

**2. The empirical advantage** is that different experts specialize in different content types; the router learns to send tokens to relevant experts. The model gains capacity without proportional compute growth.

**3. The dispatch problem** — keeping per-expert GEMMs efficient — requires specialized routing kernels (Megablocks, GroupGEMM) and dominates production implementation effort. Naive routing wastes throughput.

## Hands-on (at home)

Build a minimal MoE FFN.

```python
# moe_layer.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class Expert(nn.Module):
    """A single expert FFN — standard SwiGLU."""
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.up_proj = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)
    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

class MoELayer(nn.Module):
    def __init__(self, d_model, d_ff, n_experts, k=2):
        super().__init__()
        self.n_experts = n_experts
        self.k = k
        self.router = nn.Linear(d_model, n_experts, bias=False)
        self.experts = nn.ModuleList([Expert(d_model, d_ff) for _ in range(n_experts)])

    def forward(self, x):
        # x: [B, N, d_model]
        B, N, D = x.shape
        x_flat = x.view(-1, D)  # [B*N, D]
        gate_logits = self.router(x_flat)  # [B*N, n_experts]
        # Top-k per token.
        top_k_logits, top_k_indices = gate_logits.topk(self.k, dim=-1)  # [B*N, k]
        weights = top_k_logits.softmax(dim=-1)  # [B*N, k]
        # Initialize output.
        output = torch.zeros_like(x_flat)
        # Loop over experts (naive routing).
        for expert_idx in range(self.n_experts):
            # Mask: True for tokens that selected this expert.
            expert_mask = (top_k_indices == expert_idx)  # [B*N, k]
            token_mask = expert_mask.any(dim=-1)  # [B*N]
            if not token_mask.any():
                continue
            # Tokens for this expert.
            expert_input = x_flat[token_mask]  # [n_routed, D]
            # The weight for this expert for each routed token.
            expert_weights = (expert_mask.float() * weights).sum(dim=-1)[token_mask]  # [n_routed]
            # Compute expert output.
            expert_output = self.experts[expert_idx](expert_input)  # [n_routed, D]
            # Add weighted contribution.
            output[token_mask] += expert_weights.unsqueeze(-1) * expert_output
        return output.view(B, N, D)

# Demo.
torch.manual_seed(0)
moe = MoELayer(d_model=128, d_ff=512, n_experts=8, k=2)
x = torch.randn(2, 16, 128)
out = moe(x)
print(f"input shape: {x.shape}")
print(f"output shape: {out.shape}")

# Check that each token is routed to k=2 experts.
gate_logits = moe.router(x.view(-1, 128))
top_k_indices = gate_logits.topk(2, dim=-1).indices
print(f"first 5 tokens' chosen experts: {top_k_indices[:5].tolist()}")
```

You'll see each token gets routed to 2 experts (per the `k=2` setting). The naive routing loop is fine for a demo; production uses fused kernels.

For a fully-tuned production MoE implementation, see Mistral's reference Mixtral code or DeepSeek-V3's open-source release.

## Further reading

- "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer" (Shazeer et al, 2017) — the foundational MoE paper.
- "Switch Transformers" (Fedus et al, 2021) — Lesson 31's reference; top-1 routing.
- "Mixtral of Experts" (Jiang et al, 2024) — Mistral's MoE; popularized the top-2 8-expert design.
- "Megablocks: Efficient Sparse Training with Mixture-of-Experts" (Gale et al, 2022) — for the production routing kernel.

Next lesson: **Load balancing — auxiliary loss, expert utilization.** Naive MoE routers collapse to using only a few experts. We cover the auxiliary loss that prevents this and the related techniques.
