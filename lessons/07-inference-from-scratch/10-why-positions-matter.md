---
title: "Lesson 10 — Why Positions Matter"
date: "2026-06-04"
module: "inference-from-scratch"
order: 10
tags: ["positional-encoding", "permutation-invariance", "transformer"]
author: "Sudipta Pathak"
prerequisites: ["09-linear-attention"]
---

# Lesson 10 — Why Positions Matter

## Why this lesson exists

A surprise about self-attention that becomes obvious once you see it: vanilla attention has no notion of order. Permute the tokens in the input and the output permutes the same way, but the *content* of each output is unchanged. "The cat sat on the mat" and "Mat the on sat cat the" produce the same per-token outputs at each (now-shuffled) position — the model literally cannot tell them apart.

This is a problem. Language has order. Code has order. Time series has order. Everything we want transformers to do has order. The attention mechanism, on its own, doesn't.

This lesson is about the problem; the next five are about the solutions. By the end you should understand exactly what's missing from attention, why every positional encoding scheme exists, and why so many different ones have been invented.

The lesson is short. The Hands-on demonstrates the permutation-equivariance property directly.

## The permutation-equivariance property

Formally: a function `f(x_1, ..., x_n)` is *permutation-equivariant* if for any permutation `π`:

```
f(x_π(1), ..., x_π(n)) = f(x_1, ..., x_n)[π(1), ..., π(n)]
```

In words: if you permute the inputs, you get the same outputs in the permuted order.

Self-attention without positional encoding is permutation-equivariant. The proof is simple: attention computes pairwise interactions `(Q_i, K_j, V_j)` and sums them up. The "where each token sits in the sequence" doesn't enter the computation — only the content of the tokens.

You can verify this by running the same self-attention twice: once on `[A, B, C, D]`, once on `[D, B, A, C]`. The output at position `B` is identical in both cases, even though `B` is at index 1 in the first and index 1 in the second (same position) but its neighbors are different. Wait — neighbors are different but the math doesn't care about neighbors, only about content-similarities. So position-independent.

Let me restate that more precisely: take the second permutation `[A, B, D, C]`. Run self-attention on both. The output at `B` (position 1) in the first case incorporates information from A, B, C, D weighted by content similarity. In the second case, the output at `B` (still position 1) incorporates the same information from A, B, C, D weighted by the same content similarity — because the *set* of inputs is the same. The output is identical.

So attention treats the input as a *set*, not a *sequence*. For a sentence, this means the model can't distinguish "dog bites man" from "man bites dog."

## Why this isn't OK

For tasks where order matters (all language tasks; nearly all sequence tasks), the model needs to know each token's position. Some ways order matters in language:

- **Argument structure**: "The cat ate the fish" vs "The fish ate the cat."
- **Verb tense**: "I will run" vs "I had run."
- **Reference**: "Sara saw Maya. She waved." (who waved? depends on order/proximity).
- **Code**: `a / b` vs `b / a`.

If the model can't distinguish ordering, it can't learn any of these.

## The fix space

Three broad categories of solution:

**1. Inject position information into the inputs.** Add a position-dependent vector to each token's embedding before the attention layers. Sinusoidal encodings, learned encodings, and integer/binary encodings all do this.

**2. Modify the attention computation itself.** Make the dot-product score depend on the relative or absolute positions of Q and K. Shaw-style relative encodings, T5 bias, and ALiBi do this.

**3. Rotate Q and K based on position.** Apply position-dependent rotations to Q and K vectors before the dot product, so the dot-product score itself encodes relative position. RoPE does this.

Each has different properties around extrapolation (does it work at longer context than trained on?), parameter count (zero learnable vs O(N×d) learnable), and computational cost.

The arc of positional encoding research over 2017-2024 is the search for the variant with the best combination of these properties. By 2026 the field has largely settled on RoPE with various long-context scaling tricks — but the path there is informative, and each variant has scenarios where it's still preferred.

## A subtle but important point: positional encoding affects *every* token

Whatever scheme you pick, the position information has to be available to *every* token's attention computation, not just the first or the last. A single global "position vector" added at the embedding layer affects all subsequent operations, so it propagates everywhere. Per-attention-layer position modifications (RoPE, ALiBi) re-inject position information at every layer.

The choice between "inject once" and "inject every layer" matters for long-context behavior. RoPE's per-layer rotation means each layer's attention reasons about position freshly; the position information doesn't degrade through depth.

## The transformer was supposed to be position-agnostic by design

A historical note: the original transformer paper (2017) used sinusoidal position encodings — added once to the embeddings. The choice was somewhat ad-hoc; the paper acknowledges it's "one possibility" and suggests learned positional embeddings as an equivalent alternative.

The position-encoding story has been one of continuous improvement since. Each new encoding scheme addressed a specific failure mode of the previous: sinusoidal didn't extrapolate well; learned didn't either; relative encodings extrapolated better but were slow; ALiBi extrapolated for free but had ceiling on quality; RoPE became the dominant choice; long-context scaling of RoPE became the active research area as context lengths exploded.

The 2026 default: RoPE with NTK-aware or YaRN scaling for long context (Lesson 15). Most production LLMs use this.

## What you should believe after this lesson

Three sentences:

**1. Self-attention is permutation-equivariant**: it treats the input as a set, not a sequence. Without positional encoding, transformers cannot distinguish "dog bites man" from "man bites dog."

**2. Positional encoding is the auxiliary mechanism** that gives transformers a notion of order. Three broad categories: add position to input embeddings, modify attention scores based on position, or rotate Q and K based on position.

**3. The 2017 transformer used sinusoidal encodings as a placeholder**; the field has spent years iterating. By 2026 most production LLMs use RoPE with long-context scaling. The next five lessons trace this evolution.

## Hands-on (at home)

Verify that attention without positional encoding is permutation-equivariant.

```python
# permutation_invariance.py
import torch
import torch.nn.functional as F

torch.manual_seed(0)

def naive_self_attention(x):
    # x: [B, N, D]
    Q = K = V = x  # no projections, just identity, to isolate the property
    d = x.shape[-1]
    scores = (Q @ K.transpose(-2, -1)) / (d ** 0.5)
    weights = F.softmax(scores, dim=-1)
    return weights @ V

# Run on [A, B, C, D].
x = torch.randn(1, 4, 8)
out1 = naive_self_attention(x)
print("Output for [A, B, C, D]:")
print(out1.squeeze())

# Permute to [C, A, D, B] — output should permute equivalently.
perm = torch.tensor([2, 0, 3, 1])
x_perm = x[:, perm, :]
out_perm = naive_self_attention(x_perm)
print("\nOutput for [C, A, D, B]:")
print(out_perm.squeeze())

print("\nPermutation of original output (should match the permuted-input output):")
print(out1[:, perm, :].squeeze())

print("\nMatch:", torch.allclose(out_perm, out1[:, perm, :]))
```

Output: `Match: True`. Permuting the inputs permutes the outputs identically; no positional information is being captured.

To see this break a real language model: take a small LLM, embed two sentences with the same words in different orders, compare the embeddings before the first attention layer (they'll differ if positional encoding is applied). Then disable positional encoding and they'll be identical — and the model's behavior will collapse.

## Further reading

- "Attention Is All You Need" (Vaswani et al, 2017) — Section 3.5 introduces sinusoidal positional encodings.
- "On the Position Embeddings in BERT" (Wang & Chen, 2020) — analysis of different positional encoding choices.
- "RoFormer: Enhanced Transformer with Rotary Position Embedding" (Su et al, 2021) — the RoPE paper; the introduction has a clean treatment of the position-encoding problem.

Next lesson: **Absolute encodings — sinusoidal, learned, integer/binary.** The first generation of solutions. We trace how each one represents position and what each one trades away — setting up the relative-encoding paradigm shift in Lesson 12.
