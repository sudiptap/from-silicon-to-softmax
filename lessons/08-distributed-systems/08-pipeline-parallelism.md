---
title: "Lesson 8 — Pipeline Parallelism: GPipe, PipeDream, 1F1B"
date: "2026-06-04"
module: "distributed-systems"
order: 8
tags: ["pipeline-parallelism", "gpipe", "pipedream", "1f1b", "micro-batches", "bubble"]
author: "Sudipta Pathak"
prerequisites: ["07-tensor-parallelism"]
---

# Lesson 8 — Pipeline Parallelism: GPipe, PipeDream, 1F1B

## Why this lesson exists

Tensor parallelism splits each matmul across GPUs. Pipeline parallelism (PP) splits the *layers* across GPUs: GPU 0 holds layers 0-9; GPU 1 holds layers 10-19; etc. A token's forward pass flows through the pipeline, stage by stage.

PP is essential for very deep models that don't fit on a single GPU even with TP and FSDP. Llama 3.1 405B can use TP=8 + PP=8 (8 nodes, each with 8 GPUs, totaling 64 GPUs) — without PP, no single node would fit the model.

The catch: pipelining creates "bubbles" — periods when some pipeline stages are idle waiting for upstream stages. The pipelining schedule (GPipe, PipeDream, 1F1B) determines how bad the bubble is and how to mitigate it.

This lesson covers PP from first principles, the three major scheduling approaches, and the practical considerations.

The lesson is reading. The Hands-on simulates a pipeline schedule.

## The basic idea

For `P` pipeline stages and a global batch split into `M` micro-batches:

```
Stage 0: Layer 0  → Layer 1  → ... → Layer L/P-1
Stage 1: Layer L/P → Layer L/P+1 → ... → Layer 2L/P-1
Stage 2: ...
Stage P-1: Layer (P-1)L/P → ... → Layer L-1
```

Each stage runs on a separate GPU. To run the forward pass:
1. Stage 0 processes micro-batch 0; sends activations to Stage 1.
2. Stage 1 processes micro-batch 0 (while Stage 0 processes micro-batch 1).
3. And so on.

After `P + M - 1` micro-batch steps, all `M` micro-batches have flowed through all `P` stages.

The activations between stages are sent via point-to-point (send/recv); typically over NVLink or IB.

## The bubble

At the start of training (or at the start of each batch), Stage 1 waits for Stage 0 to produce the first activation. During this wait, Stage 1 is idle. Same for Stage 2, Stage 3, etc.

The "pipeline bubble": the fraction of time pipeline stages are idle.

For GPipe (the simplest schedule):
- Forward: bubble of `(P-1) / (M + P - 1)` per forward.
- Backward: same bubble.
- Total bubble fraction: `(P-1) / (M + P - 1)`.

For 8 stages and 16 micro-batches: bubble = 7 / 23 ≈ 30%. Significant.

For 8 stages and 128 micro-batches: bubble = 7 / 135 ≈ 5%. Acceptable.

The bubble shrinks as you increase the number of micro-batches; more micro-batches → smaller bubble fraction.

## GPipe

GPipe (Huang et al, 2019) is the simplest schedule:
- Run all M forward micro-batches through the pipeline.
- Then run all M backward micro-batches.

Pros: simple; works with any model.
Cons: large activation memory (M × per-stage activations stored during forward, freed only during backward).

For 64 micro-batches with non-trivial per-stage activations, the memory overhead is significant.

## PipeDream and 1F1B

PipeDream (Narayanan et al, 2019) interleaves forward and backward passes to reduce activation memory.

The basic idea: as soon as a micro-batch reaches the last stage's forward, start its backward immediately. The forward continues for new micro-batches; the backward runs concurrently for completed ones.

The variant that took over: **1F1B (1 Forward 1 Backward)**. Each stage alternates: one forward, one backward, one forward, one backward...

Properties:
- Activation memory: O(P) instead of O(M). Each stage holds at most P activations at once.
- Bubble: similar to GPipe.
- More complex scheduling but the memory win is essential at scale.

1F1B is the schedule Megatron-LM uses; the standard for production PP.

## Interleaved 1F1B (virtual pipeline)

A further refinement: split each stage into multiple "virtual pipeline stages." Each GPU now runs multiple non-contiguous layer chunks.

For example, with 4 GPUs and 4 virtual pipeline rounds:
- GPU 0: layers [0, 1, 16, 17, 32, 33, 48, 49]
- GPU 1: layers [2, 3, 18, 19, 34, 35, 50, 51]
- GPU 2: layers [4, 5, 20, 21, 36, 37, 52, 53]
- GPU 3: layers [6, 7, 22, 23, 38, 39, 54, 55]

The forward pass cycles through the virtual stages multiple times per micro-batch. This *increases* the number of pipeline "stages" effectively, which shrinks the bubble.

Interleaved 1F1B is what Megatron uses for the largest training runs. The bubble at virtual_stages=4 with M=16 micro-batches is about 1/4 of the standard 1F1B bubble.

## The communication cost

Pipeline parallelism's communication: per micro-batch transition between stages, send the activation tensor (and the gradient tensor in backward).

For a stage at the activation boundary: `seq_len × batch × hidden_size × bytes` per send/recv.

For Llama 70B at seq=4096, batch=4, hidden=8192, BF16:
- Per send: 4096 × 4 × 8192 × 2 = 268 MB.
- Across pipeline stages, this happens P times per micro-batch.

This is comparable to TP's per-layer AllReduce in size, but the pattern is different (point-to-point vs collective). Point-to-point is cheaper per message than AllReduce.

## When to use PP

The clear case: when the model is too deep to fit on a single node even with TP and FSDP.

The decision:
- Model fits on a node (with TP and FSDP): no PP needed.
- Model doesn't fit on a node: TP within node + PP across nodes is the standard pattern.

For Llama 70B: doesn't strictly need PP; fits within a node with TP=8 + FSDP.
For Llama 405B / DeepSeek-V3 671B: PP is essential; one node can't hold the model.

## What PP doesn't do well at inference

PP for inference has a fundamental problem: the pipelining bubble is bad at low batch sizes.

Inference typically batches small (1-8 requests per node). The pipelining bubble at small batch sizes is huge — most stages are idle most of the time.

Continuous batching (Module 7 Lesson 41) somewhat helps by keeping the batch full, but PP for inference is rare in practice. TP is preferred for inference where parallelism is needed.

## What you should believe after this lesson

Three sentences:

**1. Pipeline parallelism splits the model's layers across GPUs**; activations flow stage-to-stage. The bubble (idle time at start/end of batch) is the central inefficiency; more micro-batches per batch shrink the bubble fraction.

**2. The progression is GPipe (simple, high activation memory) → 1F1B (interleaved forward/backward, O(P) memory) → interleaved 1F1B with virtual stages (smaller bubble at the cost of more per-stage transitions).** 1F1B and interleaved are the production standards.

**3. PP is essential for models too large to fit on a single node** even with TP and FSDP; combined with TP within a node and DP/FSDP across the system, it's the 3D parallelism (Lesson 9) that trains the largest models. For inference, PP is rare because the bubble is bad at small batches.

## Hands-on (at home)

Simulate a pipeline schedule.

```python
# pp_schedule_sim.py

def simulate_gpipe(P, M):
    """GPipe schedule: all forwards then all backwards. Print stage activity by time step."""
    # Forward phase.
    print("GPipe schedule:")
    print("Time:", " ".join(str(i).rjust(3) for i in range(2*M + 2*P)))
    for stage in range(P):
        timeline = ['.'] * (2 * M + 2 * P)
        # Forward micro-batches.
        for m in range(M):
            t = stage + m
            timeline[t] = 'F'
        # Backward micro-batches (start after all forwards).
        for m in range(M):
            t = (M + P - 1) + (P - 1 - stage) + m
            timeline[t] = 'B'
        print(f"S{stage}:", ' ' + ' '.join(c.rjust(3) for c in timeline))

def simulate_1f1b(P, M):
    """1F1B schedule: interleave forward and backward."""
    print("\n1F1B schedule:")
    print("Time:", " ".join(str(i).rjust(3) for i in range(2*M + 2*P)))
    for stage in range(P):
        timeline = ['.'] * (2 * M + 2 * P)
        # Warmup: P-stage forward only.
        n_warmup = P - stage
        for m in range(n_warmup):
            t = stage + m
            timeline[t] = 'F'
        # Steady state: 1 forward, 1 backward.
        for i in range(M - n_warmup):
            t_f = stage + n_warmup + 2 * i
            t_b = t_f + 1
            timeline[t_f] = 'F'
            timeline[t_b] = 'B'
        # Cooldown: remaining backwards.
        # (Simplified for visualization.)
        print(f"S{stage}:", ' ' + ' '.join(c.rjust(3) for c in timeline))

simulate_gpipe(P=4, M=8)
simulate_1f1b(P=4, M=8)
```

The visualization shows the bubble: empty cells (`.`) between F/B at start and end. 1F1B has a smaller bubble per the schedule structure.

For real PP, see Megatron-LM's pipeline scheduler. PyTorch's `torch.distributed.pipelining` (added in PyTorch 2.4) is the native API.

## Further reading

- "GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism" (Huang et al, 2019).
- "PipeDream: Generalized Pipeline Parallelism for DNN Training" (Narayanan et al, 2019).
- "Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM" (Narayanan et al, 2021).
- PyTorch pipeline parallelism documentation.

Next lesson: **The 3D parallelism grid.** Composing TP × PP × DP. The topology you arrange GPUs in for the largest training runs.
