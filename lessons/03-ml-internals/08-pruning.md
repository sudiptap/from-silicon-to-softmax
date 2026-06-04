---
title: "Lesson 8 — Pruning: Structured vs Unstructured, Lottery Tickets"
date: "2026-06-04"
module: "ml-internals"
order: 8
tags: ["pruning", "sparsity", "structured", "unstructured", "n-m-sparsity", "lottery-ticket"]
author: "Sudipta Pathak"
prerequisites: ["07-gptq"]
---

# Lesson 8 — Pruning: Structured vs Unstructured, Lottery Tickets

## Why this lesson exists

Pruning is, in a sentence, "set some weights to zero and skip the work they would have done." It's the obvious cousin of quantization — both shrink the model — and historically the two were peers in the compression literature. As of 2026, they are no longer peers. Quantization to INT4 ships at production scale on every major LLM; pruning, in the unstructured form that dominates the literature, almost never does. The exception — structured 2:4 sparsity on NVIDIA Ampere+ — is real, modest in win size, and rarely combined with INT4 in practice.

This lesson explains why. We cover the mechanics of unstructured and structured pruning, the lottery ticket hypothesis (a beautiful idea whose engineering implications have been underwhelming), and the practical 2026 reality: pruning is a junior partner to quantization for LLMs, useful in specific niches (medium-sized vision/CNN models, latency-critical edge deployments), and worth understanding so you know when not to reach for it.

The lesson is reading. The Hands-on prunes a real linear layer at multiple sparsity levels and shows the hardware vs accuracy tradeoff.

## The two pruning regimes

**Unstructured pruning:** set individual weights to zero, anywhere in the tensor. The simplest scheme is magnitude pruning — sort weights by absolute value, set the smallest `p%` to zero, keep the rest. Variants use gradient-based importance scores, Hessian-based importance (OBS lineage from Lesson 7), or learned masks (e.g., movement pruning).

Result: a sparse weight tensor. The non-zero values can be anywhere; the mask is arbitrary.

The accuracy story: large LLMs can be pruned to ~50% sparsity with minimal accuracy loss after a brief fine-tune. At 90% sparsity, the model is wounded but still works. At 99% sparsity, it usually collapses.

The hardware story: this is the catch. **General sparse matmul kernels on GPUs are typically slower than dense matmul up to ~95% sparsity**, because the indexing and irregular access patterns more than offset the saved multiplies. A 50%-sparse model in unstructured form runs *slower* than the dense FP16 model on a GPU. You get the memory win (half the weight bytes if you use sparse storage formats like CSR), but you lose the throughput.

The only way to get hardware speedup from unstructured sparsity is to use a runtime that has specialized sparse kernels — and these mostly don't exist for FP16 dense workloads. cuSPARSE is for very-high-sparsity scientific computing; LLM workloads don't reach that regime.

This is why unstructured pruning rarely ships: you compress the model file but not the inference cost.

**Structured pruning:** prune in a pattern the hardware can exploit. The dominant variant is **N:M sparsity** — within every block of M consecutive weights along the input dimension, exactly N are non-zero. NVIDIA Ampere onward supports 2:4 sparsity (within every 4 consecutive weights, 2 are zero) in the tensor cores natively: a 2:4-sparse matmul runs at ~2× the throughput of a dense matmul on the same chip.

Other structured patterns:
- **Block sparsity:** set whole `b × b` blocks to zero. The remaining blocks form a smaller dense matrix.
- **Channel pruning:** remove entire output channels (rows of a weight matrix). The downstream layer's corresponding input channels also disappear. Equivalent to making the model narrower.
- **Filter pruning** (CNN-specific): remove entire convolutional filters.

Structured pruning has hardware-friendly storage and computation, at the cost of less flexibility in *what* to prune — you can't selectively keep a particular weight if it falls in a block scheduled for removal.

## The 2:4 sparsity story in detail

This is the only sparsity scheme that ships at scale for transformers. On NVIDIA Ampere (A100, RTX 30xx) and later, the tensor cores have a special 2:4 path that runs at ~2× the dense throughput. The pruning recipe:

1. For each row of the weight matrix, in chunks of 4 along the input dimension, keep the 2 largest-magnitude weights and zero the other 2.
2. Re-pack the surviving 2 weights plus a small index mask (which positions they were in) into the special sparse format.
3. Fine-tune briefly (1–5% of the original training compute) to recover accuracy.

Throughput speedup: ~1.5–1.7× in practice (the 2× theoretical peak is hard to hit because other parts of the pipeline don't get faster).

Accuracy: on LLMs, 2:4 sparsity costs ~0.1–0.5 perplexity points after fine-tuning. On vision models, similar — within noise after fine-tuning.

The trouble: **2:4 sparsity doesn't compose cleanly with INT4 quantization.** The sparse tensor cores work in FP16 and BF16 modes; there's no INT4 sparse path. If you want INT4 + sparse, you're doing it in software, which negates the throughput win. So in practice, the choice is:

- INT4 weights, dense storage: ~2.5× throughput at batch 1 (mostly memory bandwidth), 4× memory reduction. Standard.
- FP16 weights, 2:4 sparse: ~1.5× throughput, 2× memory reduction. Niche.
- INT4 + 2:4 sparse: theoretically ~4× throughput, but no production kernel does this efficiently in 2026.

For most LLM deployments, INT4 quantization just dominates. Sparse-INT4 is a research frontier; it isn't shipping at production scale.

Where 2:4 sparsity does ship: high-throughput serving where the model is already INT8 (W8A8 with SmoothQuant from Lesson 3) and you want a further 1.5× throughput. NVIDIA's TensorRT-LLM supports this pattern. It's a real win for very-large-scale deployments but rarely meaningful for the laptop/edge cases this module focuses on.

## The lottery ticket hypothesis

Frankle & Carbin's 2018 Lottery Ticket Hypothesis: every dense network contains a sparse subnetwork that, if trained in isolation from its original initialization, would reach comparable accuracy. The "winning ticket" is the sparse subnetwork (mask + initial values).

The hypothesis spawned a large follow-up literature. Some highlights:

- **It's empirically true at low-to-moderate sparsity** (up to ~70%) for small image models. The winning tickets do exist and can be trained from scratch to match.
- **It generalizes poorly to large models and language models.** Finding the winning ticket at LLM scale requires iterative magnitude pruning + retraining cycles that cost as much as the original training. The "tickets" don't seem to be as universal at scale.
- **The practical engineering payoff has been small.** Even when you find the ticket, you save inference compute only via the sparse-kernel pathway, which (per the discussion above) doesn't deliver clean speedups outside structured patterns.

In 2026, the Lottery Ticket Hypothesis is fascinating science with limited deployment relevance. It is worth knowing about because:
1. It frames the model-compression problem theoretically (most weights are redundant; the question is which ones).
2. It motivates pruning algorithms that try to *find* the lottery tickets early in training.
3. It connects to the broader "are big models actually big" question that BitNet (Lesson 2) attacks from the quantization side.

## When pruning is the right answer

The honest 2026 picture: most production model-compression budgets should be spent on quantization, not pruning. But there are cases where pruning is the right move:

- **Models that need to fit a fixed compute budget per inference, not a fixed memory budget.** Phone CPU/NPU inference where the constraint is "must finish in X ms" — structured pruning shrinks the FLOPs, which directly shortens latency on these chips. Quantization helps less because phone CPUs are often INT8-native and the workload is compute-bound at small model sizes.

- **CNN-class models for vision deployments.** ResNet / EfficientNet-class models prune well (channel pruning, filter pruning) and the pruned models hit production targets that pure quantization can't reach. Most edge vision deployments use a combination: structured pruning + INT8 quantization + sometimes distillation (next lesson).

- **Encoder models for retrieval and classification.** BERT / RoBERTa-class models often have 30–50% of attention heads contributing nothing meaningful; head pruning shrinks them with no measurable accuracy loss and clear latency wins.

- **Very large MoE-style models where you want to drop entire experts.** Expert pruning is structured pruning at the expert level; it has clean hardware support and significant memory wins.

Where pruning is *not* the right answer:
- Standard decoder-only LLM inference where INT4 already exists for your runtime.
- Anything where the dense-vs-sparse kernel comparison hasn't been validated.

## A note on activation sparsity

Distinct from weight sparsity: **activation sparsity** is the observation that, post-ReLU/SiLU/GeLU, many activation values are zero or near-zero. Some specialized kernels (notably Apple's MLX paths and a few NVIDIA prototypes) can skip computation on zero activations. This shows up at inference time as "the MLP layer was 30% faster because we skipped the zero-input rows."

Activation sparsity is a runtime optimization, not a model-compression technique — it doesn't change the model on disk. But it's adjacent enough to weight sparsity that it gets confused in conversation. The relevant point: activation sparsity in modern LLMs (SwiGLU activations are mostly *not* zero — the SiLU gates make them smooth) is much lower than in old-style ReLU networks. So this lever has shrunk.

## What you should believe after this lesson

Three sentences:

**1. Unstructured pruning compresses the model file but doesn't speed up inference on standard hardware** because general sparse matmul kernels are slower than dense matmul at any realistic sparsity. Use it only when storage is the only constraint.

**2. Structured 2:4 sparsity is the one pruning pattern that ships at production scale on NVIDIA hardware** — ~1.5× throughput, ~0.1–0.5 perplexity hit, FP16/INT8 only. It doesn't compose with INT4, which is why INT4 quantization dominates for laptop/edge LLM deployments.

**3. Pruning's role in 2026 is supplementary, not primary** — it's a real lever for CNN-class vision models and BERT-class encoders, but for decoder-only LLMs the quantization story has run away with the practical compression budget.

## Hands-on (at home)

Magnitude-prune a real linear layer and observe what doesn't get faster.

```python
# magnitude_prune.py
import torch
import torch.nn as nn
import time

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(0)

# A Llama-sized FFN linear.
M, K, N = 8, 4096, 14336
W = torch.randn(N, K, device=device, dtype=torch.float16)
x = torch.randn(M, K, device=device, dtype=torch.float16)

def bench(name, fn, runs=100):
    for _ in range(5): fn()
    if device == 'cuda': torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(runs): fn()
    if device == 'cuda': torch.cuda.synchronize()
    print(f"{name:35s}: {(time.time()-t0)*1000/runs:.3f} ms")

# Dense baseline.
bench("dense FP16 matmul", lambda: x @ W.t())

# Magnitude-pruned at 50% sparsity — stored dense (zeros included).
mask = (W.abs() >= W.abs().flatten().quantile(0.5))
W_sparse = W * mask
print(f"50% sparse: {(W_sparse == 0).float().mean().item()*100:.1f}% zero")
bench("50%-sparse stored as dense FP16", lambda: x @ W_sparse.t())

# Magnitude-pruned at 90% sparsity — same story, dense kernel.
mask90 = (W.abs() >= W.abs().flatten().quantile(0.9))
W_sparse90 = W * mask90
print(f"90% sparse: {(W_sparse90 == 0).float().mean().item()*100:.1f}% zero")
bench("90%-sparse stored as dense FP16", lambda: x @ W_sparse90.t())

# Sparse storage + sparse kernel (CSR). This is where you'd hope to win.
if device == 'cuda':
    Wsp = W_sparse90.to_sparse_csr()
    bench("90%-sparse CSR matmul",
          lambda: torch.sparse.mm(Wsp, x.t().to_sparse_csr()).to_dense())
```

Two things to observe:

1. The dense-storage sparse matmul (most of the matrix is zero, stored as zero) has *identical* throughput to the original dense matmul. The hardware doesn't notice that zeros are zeros; it multiplies them anyway.

2. The actual sparse-format matmul is slower than dense FP16 at 50% sparsity, comparable around 90%, and only wins decisively at 99%+ sparsity (which is too aggressive for LLM weights). This is the kernel-economics story behind why unstructured pruning doesn't ship.

For the 2:4 sparse path — requires `torch.sparse.semi_structured` and an Ampere+ GPU — see PyTorch's tutorial; the speedup over dense FP16 is the headline ~1.5× number.

## Further reading

- "The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks" (Frankle & Carbin, 2018) — the original paper.
- "Accelerating Sparse Deep Neural Networks" (Mishra et al, NVIDIA, 2021) — the 2:4 sparsity hardware paper.
- "Movement Pruning: Adaptive Sparsity by Fine-Tuning" (Sanh et al, 2020) — a gradient-driven mask-learning approach that's elegant if you want to read further.
- "What's Hidden in a Randomly Weighted Neural Network?" (Ramanujan et al, 2020) — a striking follow-up to the lottery ticket idea.
- PyTorch's sparse tensor docs and the `torch.sparse.semi_structured` API — for the 2:4 sparsity engineering path.

Next lesson: **Distillation — From Logits to Step-by-Step.** A different compression lever entirely: instead of shrinking the weights, train a smaller model to match the larger model's outputs. This goes from classical Hinton soft-label distillation to modern step-by-step (rationale distillation, chain-of-thought distillation) that has made small reasoning models possible.
