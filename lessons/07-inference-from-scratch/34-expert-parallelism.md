---
title: "Lesson 34 — Expert Parallelism for Inference"
date: "2026-06-04"
module: "inference-from-scratch"
order: 34
tags: ["moe", "expert-parallelism", "all-to-all", "distributed", "ep-overlap"]
author: "Sudipta Pathak"
prerequisites: ["33-aux-loss-free-balancing"]
---

# Lesson 34 — Expert Parallelism for Inference

## Why this lesson exists

A 256-expert MoE has 256 separate FFN blocks. Each block at typical d_ff might be 0.5-2B parameters. Even at INT4, a single expert is hundreds of MB; 256 experts is hundreds of GB. No single GPU holds them all.

The solution is **expert parallelism (EP)**: shard the experts across multiple GPUs. Each GPU holds a subset of experts. At each MoE layer, route tokens to whichever GPU has the relevant expert; collect outputs back. The communication primitive is **all-to-all** — a generalized exchange where each rank sends a different chunk of data to each other rank.

This lesson covers the EP design, the all-to-all communication pattern, and the overlap strategies that make production MoE inference practical.

The lesson is reading. The Hands-on sketches the all-to-all pattern.

## The shard layout

For an MoE with `n_experts` experts and EP world size `E`:

- Each GPU holds `n_experts / E` experts.
- For 256 experts, EP=8: each GPU has 32 experts.

Each GPU sees a slice of the batch (typically the same slice — *data parallelism* across the same batch — combined with EP sharding the experts within each rank).

## The all-to-all routing

For each MoE layer, on each device:

1. **Local routing**: compute gates locally on this device's tokens. Decide which expert (1 of 256) each token should go to.
2. **Permute by destination**: rearrange tokens by which device they need to go to. Tokens going to device 0's experts are grouped; tokens going to device 1's experts are next; etc.
3. **All-to-all dispatch**: send each device its tokens.
4. **Local expert FFN**: each device runs its experts on the received tokens.
5. **All-to-all combine**: send the outputs back to where they came from.
6. **Unpermute**: put outputs back in original token order.

The two all-to-all calls (dispatch and combine) are the dominant communication cost.

## All-to-all communication

In NCCL or equivalent collectives, all-to-all is the primitive where each rank sends a different message to each other rank. For `E` ranks, each rank sends `E-1` messages and receives `E-1` messages.

Total bytes per all-to-all: `O(tokens_per_step × hidden_dim)`. For a 1024-token batch and 4K hidden dim at FP16: ~8 MB per all-to-all per layer. At 60+ layers, ~500 MB of all-to-all per step.

On NVLink (within a node), this is fast (200 GB/s+). Across nodes (Infiniband or RoCE), much slower.

The all-to-all latency often dominates MoE inference time at scale.

## Overlap: hiding the all-to-all behind compute

Production MoE runtimes overlap the all-to-all with compute:
- While the previous layer's all-to-all is still in flight, start the current layer's local routing and gate computation.
- While the current layer's experts are running, dispatch the next layer's all-to-all.

This compute/communication overlap is essential for performance. Done well, the all-to-all latency is mostly hidden; the effective per-layer cost is the expert compute alone.

Done poorly, the all-to-all becomes serial with compute, and MoE inference can be 3-5× slower than the same model dense-equivalent.

## EP combined with TP and DP

In a production MoE deployment, you stack parallelisms:

- **Tensor Parallelism (TP)**: shard the attention layers (Q, K, V, O projections; FFN) across multiple GPUs within a node.
- **Expert Parallelism (EP)**: shard the experts across GPUs.
- **Data Parallelism (DP)**: replicate everything; serve more requests per second.
- **Pipeline Parallelism (PP)**: shard across layers.

DeepSeek-V3 inference uses combinations like TP=8 within a node + EP=16 across nodes + DP for throughput.

The combinations interact:
- TP and EP need different topologies (TP is NVLink-friendly within a node; EP works across nodes).
- DP multiplies the number of independent inference replicas.

## Production frameworks

vLLM, SGLang, TensorRT-LLM, and Megatron-Core all support EP. The implementations differ in efficiency; for DeepSeek-V3-scale models, you typically need fairly modern (post-2024) versions to get reasonable throughput.

For smaller MoE (Mixtral 8×7B with only 8 experts), EP isn't needed — the whole model fits on a single H100. EP becomes essential at 100+ experts or trillion-parameter scale.

## What you should believe after this lesson

Three sentences:

**1. Expert parallelism shards experts across devices**; each device holds `n_experts / EP` experts. Tokens are routed to the appropriate device via all-to-all collective communication for dispatch, then all-to-all again for combine.

**2. The two all-to-all calls per MoE layer dominate communication cost**; overlapping them with compute (so the next layer's communication starts while the current layer's experts are running) is essential for performance.

**3. EP combines with TP, DP, and PP** in production deployments. DeepSeek-V3-class models use multi-parallelism configurations (TP=8, EP=16, DP=many). vLLM, SGLang, and TensorRT-LLM support EP; small MoE (Mixtral) doesn't need it.

## Hands-on (at home)

Sketch the all-to-all pattern (conceptual; real EP needs multi-GPU).

```python
# expert_parallelism_sketch.py
import torch

# Simulate 4 ranks, each with 64 experts (total 256).
EP_size = 4
n_experts_total = 256
experts_per_rank = n_experts_total // EP_size

# Routing happens per-rank: each rank has its own tokens.
n_tokens_per_rank = 1024
d_model = 512

# Per-rank: which expert (global index 0-255) does each token go to (top-1 for simplicity)?
torch.manual_seed(0)
tokens = torch.randn(EP_size, n_tokens_per_rank, d_model)
gate_logits = torch.randn(EP_size, n_tokens_per_rank, n_experts_total)
chosen_expert = gate_logits.argmax(dim=-1)  # [EP_size, n_tokens_per_rank], values 0-255

# Convert global expert ID to (target_rank, local_expert_idx).
target_rank = chosen_expert // experts_per_rank  # which rank owns the expert
local_expert = chosen_expert % experts_per_rank   # the expert's index on that rank

# All-to-all dispatch: each rank sends its tokens to the appropriate target rank.
# (In real NCCL this is one collective call; here we simulate the routing.)
dispatched = []  # dispatched[r] = list of (origin_rank, token, expert_idx) for rank r
for r in range(EP_size):
    rank_tokens = []
    for src in range(EP_size):
        # Tokens that src is sending to r.
        mask = (target_rank[src] == r)
        for i, (t, le) in enumerate(zip(tokens[src][mask], local_expert[src][mask])):
            rank_tokens.append((src, t, le.item()))
    dispatched.append(rank_tokens)

print(f"Rank 0 received {len(dispatched[0])} tokens from across the EP group")
print(f"Rank 1 received {len(dispatched[1])} tokens")
# In a balanced MoE, this would be approximately equal across ranks.

# Each rank then runs its experts on the received tokens.
# Then all-to-all combines the outputs back to the original ranks.
# Then unpermute back to original token order.

# This is a sketch; production all-to-all uses NCCL's `all_to_all_single` or `all_to_all`.
```

For a real multi-GPU all-to-all, use `torch.distributed.all_to_all_single`:

```python
import torch.distributed as dist
# After dispatch (per-rank tokens-to-send laid out contiguously):
recv_buffer = torch.zeros_like(send_buffer)
dist.all_to_all_single(recv_buffer, send_buffer)
# recv_buffer now contains tokens from all other ranks.
```

Production EP setups use the DeepSpeed-MoE / Megablocks kernels which fuse the all-to-all with the GEMM.

## Further reading

- "GShard" (Lepikhin et al, 2020) — first paper on expert parallelism at scale.
- "Megablocks: Efficient Sparse Training with Mixture-of-Experts" (Gale et al, 2022).
- "DeepSeek-V3 Technical Report" — production EP deployment.
- DeepSpeed-MoE documentation.
- vLLM and SGLang docs on MoE serving configurations.

End of Part 5. Next: Part 6 begins with **INT8 / INT4 basics** — Module 3 covered this from the algorithm side; here we cover it from the inference-engine-internals side, including the exact GEMM kernel patterns. Then GPTQ, AWQ, SmoothQuant, FP8, and BitNet.
