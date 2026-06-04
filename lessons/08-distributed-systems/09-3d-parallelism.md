---
title: "Lesson 9 — The 3D Parallelism Grid"
date: "2026-06-04"
module: "distributed-systems"
order: 9
tags: ["3d-parallelism", "tp", "pp", "dp", "topology", "trillion-parameter"]
author: "Sudipta Pathak"
prerequisites: ["08-pipeline-parallelism"]
---

# Lesson 9 — The 3D Parallelism Grid

## Why this lesson exists

For the largest training runs (100B+ parameters), no single parallelism dimension suffices. Pure DDP runs out of memory; pure TP runs out of nodes; pure PP has a worsening bubble at scale.

3D parallelism composes all three: **TP within a node**, **PP across nodes within a "pipeline group,"** **DP across pipeline groups**. The composition exploits each dimension's strengths and works around each's weaknesses.

This lesson covers how the three dimensions compose, the topology arrangement on physical hardware, and the parameter-counting math.

The lesson is reading. The Hands-on builds the 3D parallelism configuration for a worked example.

## The 3D layout

Suppose you have `N` total GPUs. You divide them into a 3D grid:

```
N = TP × PP × DP
```

Where:
- `TP`: tensor parallelism degree.
- `PP`: pipeline parallelism degree.
- `DP`: data parallelism degree.

Each GPU has a 3D coordinate `(tp_rank, pp_rank, dp_rank)`.

For example, with 64 GPUs and `TP=8, PP=4, DP=2`:
- 8 TP-ranks form a tensor-parallel group (within a node).
- 4 PP-ranks form a pipeline-parallel group (across 4 nodes).
- 2 DP-ranks replicate the whole arrangement.

The arrangement on physical hardware:
- Each "pipeline group" is 8 nodes (PP=4 pipeline stages × DP=2 data-parallel replicas wait — no, this needs rethinking).

Let me re-set:
- DP=2 means there are 2 copies of the model.
- Each copy is split across PP=4 pipeline stages.
- Each stage is split across TP=8 tensor-parallel GPUs.
- Each GPU is one of 8 TP-ranks × 4 PP-ranks × 2 DP-ranks = 64 total.

## The topology

Mapping the 3D grid to physical hardware:

**TP dimension**: stays within a node (over NVLink). 8 GPUs of one TP group share the model's matmul shards.

**PP dimension**: spans nodes. Each pipeline stage is one node (or one TP group within a node). Adjacent pipeline stages communicate via IB.

**DP dimension**: spans even more nodes. Different DP replicas don't talk to each other except during gradient AllReduce.

For 64 GPUs in 8 nodes (8 GPUs per node):
- 8 TP-ranks per node.
- 4 pipeline stages = 4 nodes per pipeline group.
- 2 DP replicas = 8 nodes total (4 × 2).

Communication pattern:
- TP: AllReduce within a node, very fast over NVLink.
- PP: point-to-point between adjacent node pairs over IB.
- DP: AllReduce across all DP replicas (typically across the slowest network: IB across all nodes).

## The math

For Llama 405B with TP=8, PP=8, DP=2 (128 GPUs):
- Each GPU holds: 405B / (TP × PP) = 405B / 64 ≈ 6.3B parameters.
- At BF16: ~12.6 GB of parameters per GPU.
- Plus optimizer state at 12 bytes per param sharded by DP=2: 6.3B × 12 / 2 = 37.5 GB optimizer state per GPU.
- Plus activations during forward (sharded by TP=8): bounded.
- Total per-GPU memory: comfortably fits on H100.

Without 3D parallelism (pure DP at 128 GPUs):
- Each GPU would need the full 405B at BF16 + Adam state: 405B × 16 bytes = 6.5 TB. Doesn't fit on any single GPU.

The 3D approach is the only way to train models at this scale.

## When to use each dimension

The decision matrix:

**Increase TP** when:
- The current per-GPU model size is too large.
- You have NVLink bandwidth available (within-node).
- Adding TP doesn't hurt throughput (communication-overlap is fine).

**Increase PP** when:
- The model is too deep to fit even with TP.
- You have nodes to spare beyond what TP needs.
- The micro-batch count is large enough to amortize the bubble.

**Increase DP** when:
- Memory fits and you want more throughput.
- Cross-cluster fabric bandwidth supports the gradient AllReduce.
- The effective batch size after scaling is still reasonable.

Typical 2026 configurations:

- **Llama 7B/13B training**: TP=1, PP=1, DP=many (pure DDP).
- **Llama 70B training**: TP=8 (within node), PP=1, DP=many (TP + DP).
- **Llama 405B training**: TP=8, PP=8, DP=4-16 (3D parallelism).
- **DeepSeek-V3 (671B MoE) training**: TP=8, PP=16, DP+EP combination.

The exact numbers depend on cluster size and topology.

## Adding more dimensions

Beyond TP × PP × DP, modern training adds:

**EP (Expert Parallelism)**: for MoE models. Shards experts across additional GPUs. Adds an all-to-all communication pattern. Module 7 Lesson 34.

**SP (Sequence Parallelism)**: shards the sequence dimension within TP. Module 7 Lesson 47; Module 8 Lesson 10. Often called "2.5D" parallelism (TP + SP) or "4D" when combined with TP + PP + DP.

**Context parallelism**: a more aggressive sequence-parallel that works across nodes.

For Llama 3.1 405B or DeepSeek-V3 at scale, the parallelism is effectively 4D or 5D when you count all the dimensions.

## Communication cost decomposition

For a step under 3D parallelism, you have multiple collectives in flight:
- TP all-reduce (per layer, intra-node).
- PP point-to-point (per stage transition, inter-node).
- DP all-reduce (once per step, across all DP replicas).

Each goes over different network layers and overlaps with different compute:
- TP all-reduce overlaps with the next layer's compute.
- PP transitions overlap with pipeline stage's compute.
- DP all-reduce overlaps with optimizer step (or the next forward pass).

A well-tuned 3D parallel training run achieves >50% GPU utilization. A poorly-tuned one (wrong parallelism degrees, communication not overlapped) can be <20%.

## What you should believe after this lesson

Three sentences:

**1. 3D parallelism composes TP × PP × DP**: TP within a node (NVLink), PP across nodes within a pipeline group (IB), DP across pipeline groups (IB). Each dimension exploits a different network layer's strengths.

**2. The decision matrix**: increase TP when per-GPU model is too large; increase PP when the model is too deep to fit; increase DP when memory is fine and you want more throughput. Typical configurations: 70B uses TP+DP; 405B uses TP+PP+DP.

**3. Modern training adds EP (for MoE) and sequence parallelism** to the 3D grid, making it effectively 4D or 5D. The parallelism design for frontier models is a multi-axis optimization problem; mistuning costs throughput dramatically.

## Hands-on (at home)

Work out a 3D parallelism configuration for a worked example.

```python
# 3d_parallelism_calculator.py

def compute_memory_per_gpu(
    model_params_billions,
    tp_size,
    pp_size,
    dp_size,
    seq_len,
    batch_per_gpu,
    hidden_dim,
    n_layers,
    bytes_per_param=2,  # BF16
    adam_state_bytes_per_param=12,
):
    """Estimate per-GPU memory for a 3D-parallel training configuration."""
    # Each GPU's share of the model.
    params_per_gpu = model_params_billions * 1e9 / (tp_size * pp_size)
    
    # Parameter memory (BF16).
    param_mem = params_per_gpu * bytes_per_param
    
    # Gradient memory (BF16, sharded by DP if using FSDP-style).
    grad_mem = params_per_gpu * bytes_per_param / dp_size
    
    # Optimizer state (Adam mixed-precision, sharded by DP).
    optim_mem = params_per_gpu * adam_state_bytes_per_param / dp_size
    
    # Activation memory (rough; varies with TP).
    # Per layer: seq_len * batch_per_gpu * hidden * 4 (for residual, attn, ffn intermediate, etc.).
    # Sharded by TP within parallel regions.
    layers_per_gpu = n_layers / pp_size
    act_per_layer = seq_len * batch_per_gpu * hidden_dim * 4 * 2  # rough estimate
    act_mem = layers_per_gpu * act_per_layer / tp_size  # TP shards activations in parallel regions
    
    total_gb = (param_mem + grad_mem + optim_mem + act_mem) / 1e9
    return {
        'params': param_mem / 1e9,
        'grads': grad_mem / 1e9,
        'optim_state': optim_mem / 1e9,
        'activations': act_mem / 1e9,
        'total': total_gb,
    }

# Llama 70B configurations.
print("Llama 70B (8x8 = 64 GPUs):")
for tp, pp, dp in [(8, 1, 8), (8, 2, 4), (4, 2, 8)]:
    m = compute_memory_per_gpu(
        70, tp, pp, dp, seq_len=4096, batch_per_gpu=2,
        hidden_dim=8192, n_layers=80
    )
    print(f"  TP={tp}, PP={pp}, DP={dp}: {m['total']:.1f} GB/GPU")
    print(f"    params={m['params']:.1f}, grads={m['grads']:.1f}, optim={m['optim_state']:.1f}, act={m['activations']:.1f}")

print("\nLlama 405B (256 GPUs):")
for tp, pp, dp in [(8, 8, 4), (8, 4, 8), (8, 16, 2)]:
    m = compute_memory_per_gpu(
        405, tp, pp, dp, seq_len=4096, batch_per_gpu=1,
        hidden_dim=16384, n_layers=126
    )
    print(f"  TP={tp}, PP={pp}, DP={dp}: {m['total']:.1f} GB/GPU")
```

You'll see which configurations fit in 80 GB / 96 GB / 140 GB GPU memory. The right configuration is the one that maximizes throughput while fitting.

For real benchmarks, run small versions of the model with each configuration in DeepSpeed or Megatron-LM and measure tokens/sec/GPU.

## Further reading

- "Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM" (Narayanan et al, 2021) — the canonical 3D parallelism reference.
- "Training Deep Networks with Synthetic Data: Bridging the Reality Gap by Domain Randomization" — no wait, that's a different topic. The actual 3D parallelism references are in Megatron-LM and DeepSpeed documentation.
- "Llama 3.1 paper" — Section on training parallelism for 405B.
- DeepSpeed documentation on 3D parallelism.

Next lesson: **Sequence parallelism and Ring Attention.** The fourth (or fifth) parallelism dimension. Module 7 Lesson 47 was the inference view of Ring Attention; here we put it in the training and combined parallelism context.
