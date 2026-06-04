---
title: "Lesson 4 — AllReduce Algorithms: Ring, Tree, Double Binary Tree"
date: "2026-06-04"
module: "distributed-systems"
order: 4
tags: ["all-reduce", "ring", "tree", "algorithms", "bandwidth-optimal"]
author: "Sudipta Pathak"
prerequisites: ["03-nccl"]
---

# Lesson 4 — AllReduce Algorithms: Ring, Tree, Double Binary Tree

## Why this lesson exists

AllReduce is the dominant collective in distributed ML — every DDP step does at least one. The algorithm used internally matters enormously: a poorly-chosen algorithm wastes 2× the bandwidth or doubles the latency.

NCCL picks the algorithm automatically (Lesson 3), but understanding the algorithms tells you why distributed training scales the way it does. The three families — ring, tree, double binary tree — each have a regime where they win.

This lesson derives each algorithm and the tradeoff space.

The lesson is reading. The Hands-on simulates a ring all-reduce in Python.

## What AllReduce computes

Given `N` processes, each with a tensor `x_i` of the same shape, AllReduce produces:

```
result = x_0 + x_1 + ... + x_{N-1}
```

and every process gets a copy of `result`. (The operation is summation; it can also be max, min, etc.)

The total useful work: `N × size_of_x` bytes communicated (every process needs to receive `result`). The challenge: do this efficiently on a network with limited per-link bandwidth.

The lower bound: at least `(N-1)/N × size_of_x` bytes per process must be communicated. Any AllReduce algorithm with bandwidth at this minimum is "bandwidth-optimal."

## Ring AllReduce

The ring algorithm:

1. Arrange `N` processes in a logical ring: 0 → 1 → 2 → ... → N-1 → 0.
2. Split each process's tensor into `N` equal chunks.
3. **Reduce-scatter phase** (N-1 steps):
   - At step `s`, each process sends its `(rank - s) % N`-th chunk to the next process and receives the `(rank - s - 1) % N`-th chunk from the previous, adding it to its local copy.
   - After N-1 steps, each process has the *full reduction* of one specific chunk.
4. **All-gather phase** (N-1 steps):
   - Each process now sends its complete chunk around the ring.
   - After N-1 more steps, every process has the full reduction of every chunk.

Total: `2(N-1)` send/receive steps. Each step moves `1/N` of the tensor. Total bytes per process: `2(N-1)/N × size_of_x` ≈ `2 × size_of_x` for large N. Bandwidth-optimal up to a factor of 2.

Time complexity: dominated by the bandwidth term, `2 size_of_x / bandwidth`. The latency term is `2(N-1) × α` where `α` is per-hop latency.

For large messages on high-bandwidth links: ring is excellent. The bandwidth utilization is essentially 100%.

For small messages or high N: the latency term `2(N-1) × α` dominates. A 1000-GPU ring with `α = 10 μs` would take 20 ms in latency alone — too slow for small all-reduces. This is the regime tree wins.

## Tree AllReduce

The tree algorithm:

1. Arrange processes as a binary tree.
2. **Up phase** (log N steps): from leaves to root, each parent sums its children's contributions and forwards.
3. The root has the full reduction.
4. **Down phase** (log N steps): from root to leaves, the full reduction is broadcast.

Total: `2 log N` steps. Each step's bandwidth is bounded by the link speed.

Time complexity: `log N × (size_of_x / bandwidth + α)`. The bandwidth term scales with `log N`; the latency term is the small `2 log N × α`.

Tree wins over ring at:
- Small message sizes (latency-bound).
- Very large N (where ring's N-1 hops dominate).

Tree's bandwidth efficiency: each link carries the full message size `log N` times. So total bytes per link: `size_of_x × log N`. Compared to ring's `2 × size_of_x / N`, tree is worse on bandwidth for large messages.

## Double Binary Tree

The clever trick: use *two* trees simultaneously, with different shapes. Each process is an interior node in one tree and a leaf in the other (and vice versa). The two trees split the bandwidth work between them.

The result: roughly the latency of a single tree (`log N`) with twice the bandwidth utilization (each tree carries half the data). Bandwidth efficiency approaches ring's at large N.

Double binary tree is the algorithm NCCL uses for large multi-node all-reduces. It's a sweet spot: low latency (logarithmic), reasonable bandwidth efficiency.

## Which algorithm NCCL picks

NCCL's automatic selection:
- **Small messages (< 32 KB)**: tree algorithm (latency-optimized).
- **Medium messages (32 KB - 256 KB)**: tree or hybrid.
- **Large messages (> 256 KB)**: ring (bandwidth-optimized).
- **Across nodes with many GPUs**: double binary tree.

You can override with `NCCL_ALGO=Ring` or `NCCL_ALGO=Tree` for debugging or specific tuning.

For typical ML training:
- Gradient all-reduce in DDP: medium-to-large messages (~hundreds of MB). Ring is the right choice; NCCL picks it.
- Per-layer all-reduce in TP: smaller messages (~MB). Tree-ish algorithms; NCCL picks them.
- All-reduce of optimizer state: depends on configuration.

## Hierarchical all-reduce

For multi-node clusters with NVLink within nodes and IB across nodes, a "hierarchical" all-reduce is often best:

1. *Intra-node* reduce-scatter (over NVLink): each node's GPUs collectively reduce, producing partial sums per GPU.
2. *Inter-node* all-reduce of those partial sums (over IB): each node's partial sums get combined across nodes.
3. *Intra-node* all-gather (over NVLink): every GPU in a node receives the final result.

The benefit: most of the bandwidth-heavy work (steps 1 and 3) happens over NVLink; only a fraction of the data crosses IB. The IB transfer is `1/G` of the total (where G is GPUs per node).

NCCL implements this automatically for multi-node configurations.

## What you should believe after this lesson

Three sentences:

**1. AllReduce has three algorithm families** — ring (bandwidth-optimal for large messages), tree (latency-optimal for small messages or many nodes), double binary tree (sweet spot for many-node large clusters). NCCL picks automatically based on message size and topology.

**2. The asymptotic bandwidth lower bound is `(N-1)/N × size_of_x`**; ring achieves twice this; tree pays a `log N` factor in bandwidth but `log N` instead of `N` in latency. The choice depends on which side of the latency-vs-bandwidth tradeoff dominates.

**3. Hierarchical all-reduce** (intra-node first, then inter-node, then intra-node again) lets multi-node clusters exploit NVLink within nodes and IB across nodes, dramatically reducing the cross-node bandwidth needed.

## Hands-on (at home)

Simulate a ring all-reduce in Python (single-process; no actual networking).

```python
# ring_allreduce_sim.py
import numpy as np

N = 4  # number of processes
size = 8  # tensor size

# Each process has its own tensor.
tensors = [np.arange(size) + rank * 100 for rank in range(N)]
print("Initial tensors:")
for rank in range(N):
    print(f"  rank {rank}: {tensors[rank]}")

# Verify what the correct sum is.
correct = sum(tensors)
print(f"Correct sum: {correct}")

# Split each tensor into N chunks.
chunks = [np.array_split(t, N) for t in tensors]  # chunks[rank][chunk_idx]

# Reduce-scatter phase.
for step in range(N - 1):
    # Each rank sends one chunk to next rank.
    for rank in range(N):
        next_rank = (rank + 1) % N
        send_chunk_idx = (rank - step) % N
        # next_rank adds the received chunk to its own.
        chunks[next_rank][send_chunk_idx] += chunks[rank][send_chunk_idx]
    # After the step, copies are stale; in a real ring each rank only holds its own chunks.
    # For simulation purposes, we proceed.

# After reduce-scatter, each rank holds the full reduction of one chunk.
# Rank `r` holds chunk `(r + 1) % N` summed.

# All-gather phase: each rank's complete chunk gets broadcast around the ring.
for step in range(N - 1):
    for rank in range(N):
        next_rank = (rank + 1) % N
        send_chunk_idx = (rank + 1 - step) % N
        chunks[next_rank][send_chunk_idx] = chunks[rank][send_chunk_idx].copy()

# Now each rank should have the full reduction.
result = [np.concatenate(c) for c in chunks]
for rank in range(N):
    correct_match = np.array_equal(result[rank], correct)
    print(f"rank {rank} final: {result[rank]} (correct: {correct_match})")
```

This is conceptual; the real ring all-reduce uses overlap of send and receive at each step, not the synchronous loop above. But the algorithm structure is the same.

For real benchmarks, use nccl-tests at different message sizes and observe the algorithm crossovers.

## Further reading

- "Optimization of Collective Communication Operations in MPICH" (Thakur et al, 2005) — foundational paper on collective algorithms.
- "Massively Scale Your Deep Learning Training with NCCL 2.4" (NVIDIA blog) — the double-binary-tree announcement.
- "Distributed Training of Deep Learning Models: A Taxonomic Perspective" (Mayer & Jacobsen, 2020).
- NCCL source code (github.com/NVIDIA/nccl) — the algorithms are in `src/algorithm/`.

Next lesson: **Distributed Data Parallel (DDP) from first principles.** The simplest parallelism strategy: replicate the model, split the batch, all-reduce gradients. The baseline that every other strategy improves on.
