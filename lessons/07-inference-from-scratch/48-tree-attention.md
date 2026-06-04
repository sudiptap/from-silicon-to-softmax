---
title: "Lesson 48 — Tree Attention for Branched Generation"
date: "2026-06-04"
module: "inference-from-scratch"
order: 48
tags: ["tree-attention", "best-of-n", "beam-search", "speculative", "branching"]
author: "Sudipta Pathak"
prerequisites: ["47-ring-attention"]
---

# Lesson 48 — Tree Attention for Branched Generation

## Why this lesson exists

Many advanced inference patterns involve generating *multiple candidate sequences* from a shared prefix:
- Best-of-N sampling (generate N completions, pick the best).
- Beam search (maintain B candidates per step).
- Speculative decoding with tree-shaped candidates (Medusa, SpecInfer).
- Test-time compute sampling (generate many trajectories for reasoning).

Naively, each candidate has its own KV cache and runs independently. But they share the prefix — the part of the sequence before the branching. Computing attention for the prefix N times is wasteful.

Tree Attention (related: SpecInfer, Star Attention) exploits the tree structure: compute the prefix's KV once; have all branches attend to the shared prefix; only diverge for the branch-specific tokens.

The win: O(N) cheaper than independent generation when you have N branches sharing a prefix.

This lesson covers tree attention as a generalization of prefix caching (Lesson 19) and RadixAttention (Lesson 44) to active branched generation.

The lesson is reading. The Hands-on visualizes the tree structure.

## The tree structure

A branched generation produces a *tree* of token positions:

```
PROMPT
  └── shared part
        ├── branch A: ...
        ├── branch B: ...
        └── branch C: ...
```

Or more deeply:

```
PROMPT
  └── shared part
        ├── token X
        │     ├── token X1
        │     └── token X2
        └── token Y
              └── token Y1
```

Each leaf of the tree is a token; the path from root to leaf is the candidate sequence.

For attention: each token needs to attend to all its *ancestors* in the tree. Two leaves sharing an ancestor see the same ancestor representation; the ancestor's KV is shared.

## The attention mask

Tree attention modifies the causal mask to reflect the tree structure rather than the linear position:

- For each token, the mask allows attention to its ancestors only.
- Siblings cannot attend to each other (they're on different branches).
- The mask is no longer a strict lower-triangular shape; it's tree-shaped.

```
       [PROMPT]  [shared] [token X]  [token X1] [token X2] [token Y]  [token Y1]
PROMPT  ✓
shared  ✓        ✓
token X ✓        ✓        ✓
token X1✓        ✓        ✓          ✓
token X2✓        ✓        ✓                    ✓
token Y ✓        ✓                                            ✓
token Y1✓        ✓                                            ✓        ✓
```

The mask captures: each token attends to itself and its ancestors. Siblings do not interact.

## The implementation

Production tree attention:
1. Each token has a "tree position" (its path through the tree).
2. The mask is computed from the tree positions (which tokens are ancestors of which).
3. The KV cache stores each token's K, V once; the attention computation uses the mask to scope visibility.

This is more memory-efficient than the naive approach (which would duplicate the ancestor K, V for each branch).

## Use cases

**Speculative decoding with tree candidates** (SpecInfer, EAGLE-2):
- The draft model produces a *tree* of candidate continuations (not just one chain).
- The main model verifies all of them in one pass via tree attention.
- Higher acceptance rate than linear candidates because more probability mass is covered.

**Beam search with shared prefix**:
- Maintain B beams that share the original prompt's prefix.
- Each beam has its own continuation tokens after divergence.
- Tree attention shares the prefix's KV across all beams.

**Best-of-N sampling**:
- Generate N completions from the same prompt.
- All share the prompt's KV; only the completion tokens differ.
- Tree attention is just prefix caching with N branches.

**Test-time compute (Lesson 49)**:
- Generate many reasoning trajectories for the same problem.
- Each trajectory is a branch; the problem statement is the shared prefix.

## Production support

Tree attention support in 2026 production runtimes is uneven:
- **SGLang**: native via RadixAttention (Lesson 44). Best support for tree-shaped workloads.
- **vLLM**: some support via the speculative decoding APIs (uses tree masks internally).
- **TensorRT-LLM**: tree attention for speculative decoding.
- **llama.cpp**: limited.

For agent applications and tree-search reasoning, SGLang is currently the most natural choice.

## What you should believe after this lesson

Three sentences:

**1. Tree attention generalizes prefix sharing to active branched generation** — multiple candidates share their common ancestors' KV cache; only the divergent tokens have their own KV. The attention mask is tree-shaped rather than strictly causal.

**2. Use cases include speculative decoding with tree candidates (higher acceptance than linear), beam search with shared prefix, best-of-N sampling, and test-time-compute reasoning trajectories.** Tree attention is the substrate for any inference that explores multiple continuations from a common starting point.

**3. SGLang's RadixAttention is the most natural production substrate** for tree-attention workloads in 2026; vLLM and TensorRT-LLM support it for specific cases (speculative decoding). Choose the runtime based on whether tree-shaped generation is central to your workload.

## Hands-on (at home)

A small tree-mask construction.

```python
# tree_attention_mask.py
import torch

def tree_attention_mask(parents):
    """
    parents: list where parents[i] is the index of token i's parent in the tree,
    or -1 if i is the root.
    Returns a [N, N] mask: True for positions that token i can attend to (itself and ancestors).
    """
    N = len(parents)
    mask = torch.zeros(N, N, dtype=torch.bool)
    for i in range(N):
        # Walk up the tree from i.
        j = i
        while j != -1:
            mask[i, j] = True
            j = parents[j]
    return mask

# Example tree:
# 0 (root, PROMPT)
# ├── 1 (shared)
# │    ├── 2 (token X)
# │    │     ├── 3 (token X1)
# │    │     └── 4 (token X2)
# │    └── 5 (token Y)
# │          └── 6 (token Y1)
parents = [-1, 0, 1, 2, 2, 1, 5]
mask = tree_attention_mask(parents)
print("Tree attention mask (True = can attend):")
for i in range(7):
    print(f"  token {i}: {mask[i].int().tolist()}")
# Token 3 (X1) should attend to: itself (3), parent X (2), grandparent shared (1), root (0).
# Token 4 (X2) should attend to: itself, X (2), shared (1), root (0) -- NOT X1 (sibling).
# Token 5 (Y) should attend to: itself, shared (1), root (0) -- NOT X (different branch).
```

For real tree attention, the mask is constructed dynamically as branches form, and the attention kernel uses it to skip masked positions.

## Further reading

- "SpecInfer: Accelerating Generative LLM Serving with Speculative Inference and Token Tree Verification" (Miao et al, 2023).
- "EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees" (Li et al, 2024).
- SGLang RadixAttention docs.
- "Star Attention: Efficient LLM Inference over Long Sequences" (Acharya et al, NVIDIA, 2024) — uses a different tree pattern for long-context inference.

Next lesson: **Test-time compute scaling.** When you can use multiple forward passes per problem (reasoning, best-of-N), how does total compute relate to quality? The "compute-vs-quality" frontier that o1 and R1 made famous.
