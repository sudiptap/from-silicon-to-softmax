---
title: "Lesson 38 — SmoothQuant"
date: "2026-06-04"
module: "inference-from-scratch"
order: 38
tags: ["smoothquant", "w8a8", "activation-quantization", "outliers", "per-channel"]
author: "Sudipta Pathak"
prerequisites: ["37-awq"]
---

# Lesson 38 — SmoothQuant

## Why this lesson exists

W8A8 (INT8 weights + INT8 activations) is the throughput-maximizing inference path on Ampere+ NVIDIA hardware — 2× faster than FP16 via INT8 tensor cores. The blocker is *activation outliers*: ~0.1% of channels in LLM activations carry values 10-100× larger than the median. Per-tensor INT8 activation quantization gets crushed by these outliers; per-tensor scales are dominated by the largest value; the bulk distribution becomes useless.

SmoothQuant (Xiao et al, 2022) is the elegant fix: **migrate the outlier difficulty from activations to weights via per-channel rescaling**. The activations become well-behaved and INT8-friendly; the weights absorb the per-channel scaling and remain quantizable.

This lesson is the SmoothQuant mechanism, the inference-time integration, and the practical W8A8 deployment story.

The lesson is reading. The Hands-on applies SmoothQuant to a small linear layer and demonstrates the outlier reduction.

## The setup

Consider one linear layer: `Y = A × W^T`. The activations `A` have outlier channels; the weights `W` are well-behaved.

The math: for any per-channel diagonal scaling matrix `S`:

```
A × W^T = (A × S^-1) × (S × W^T) = A' × W'^T
```

If we pick `S` per-channel such that `A' = A × S^-1` has flat magnitude across channels, the activations become well-behaved. The weights `W' = S × W` absorb the inverse — some columns of `W'` are now larger, but weights tolerate larger range better than activations.

The picking rule:

```
s_i = (max|A_i|)^α / (max|W_i|)^(1-α)
```

where `α ∈ [0, 1]` controls the migration aggressiveness. `α = 0`: no migration. `α = 1`: aggressive (all activation difficulty moved to weights). Typical: `α = 0.5`.

The result: `A'` has more uniform per-channel range; `W'` has slightly broader per-channel range but still well within INT8 tolerance.

## The offline migration

SmoothQuant runs at conversion time (offline):

1. Compute activation statistics from calibration data (per-channel max).
2. Compute weight statistics (per-channel max).
3. Compute `S` from the formula.
4. Rescale: `W' = S × W`. Replace the weights in the checkpoint.
5. At inference: the activations passed to this layer must also be scaled by `S^-1`.

The `S^-1` activation scaling can be folded into the *previous* layer's output projection. So the activation scaling is also offline; the inference path is identical to a model that never had outliers.

## Then quantize to INT8

After SmoothQuant migration:
- Weights `W'` are quantized per-channel to INT8 (standard).
- Activations `A'` are well-behaved enough to quantize per-tensor or per-token to INT8.

The matmul becomes:

```
Y_quantized = INT8(A') × INT8(W')^T → INT32 accumulator → dequant to FP16
```

Using INT8 tensor cores at peak throughput.

## When SmoothQuant wins

The clear case: high-throughput serving on Ampere+ GPUs where INT8 tensor cores' 2× speedup over FP16 is the binding throughput constraint. Multi-tenant LLM serving where you want maximum requests-per-second.

When it doesn't matter:
- On-device single-stream inference (Module 6) where you're bandwidth-bound, not compute-bound. INT8 tensor cores are wasted; W4A16 (INT4 weights, FP16 activations) is the better choice.
- Models with low-magnitude activations where outliers aren't a problem.

SmoothQuant is part of the standard W8A8 recipe in TensorRT-LLM and vLLM's INT8 path.

## Implementation in production

Most production runtimes that support W8A8 use SmoothQuant or a close variant:
- TensorRT-LLM: explicitly applies SmoothQuant during the conversion to W8A8.
- vLLM W8A8 path: similar.
- DeepSpeed-Inference: has a W8A8 quantizer using related techniques.

The `α` hyperparameter is typically swept (0.4, 0.5, 0.6, 0.7) during calibration; the best per-layer value is chosen.

## Combining with other techniques

SmoothQuant composes:
- With AWQ (which also uses per-channel scaling) — the techniques have overlap. Some recipes pick one or the other; some combine.
- With FP8 (Lesson 39) — FP8's higher dynamic range than INT8 reduces the need for SmoothQuant, but applying it still helps.
- With KV quantization (Lesson 20) — these affect different tensors; no conflict.

## What you should believe after this lesson

Three sentences:

**1. SmoothQuant migrates outlier difficulty from activations to weights via per-channel rescaling** — `A × W^T = (A × S^-1) × (S × W^T)` with `S` picked per-channel. The activations become flat; the weights absorb the imbalance. Both then quantize cleanly to INT8.

**2. The migration is offline**: applied at conversion time. The activation rescaling can be folded into the previous layer's output projection, so the inference path is identical to a model with naturally-flat activations.

**3. SmoothQuant enables W8A8 inference** on Ampere+ GPUs, getting 2× the throughput of FP16 via INT8 tensor cores. It's the dominant approach for high-throughput multi-tenant LLM serving; less relevant for on-device single-stream where bandwidth dominates over compute.

## Hands-on (at home)

Apply SmoothQuant to a linear layer.

```python
# smoothquant_demo.py
import torch
import torch.nn as nn

torch.manual_seed(0)
# Simulate a layer with outlier activations.
N, K = 1024, 4096  # output and input dims
W = torch.randn(N, K) * 0.05  # well-behaved weights
A = torch.randn(8, K)
# Introduce outliers: 10 channels with magnitude 30x typical.
outlier_channels = torch.randperm(K)[:10]
A[:, outlier_channels] *= 30

print("Before SmoothQuant:")
print(f"  Activation max per channel range: {A.abs().max(dim=0).values.max().item():.1f}")
print(f"  Activation max per channel range (typical): {A.abs().max(dim=0).values.median().item():.2f}")
print(f"  Outlier ratio: {(A.abs().max(dim=0).values.max() / A.abs().max(dim=0).values.median()).item():.0f}x")

def smoothquant(A, W, alpha=0.5):
    # Per-channel activation max.
    a_max = A.abs().max(dim=0).values  # [K]
    w_max = W.abs().max(dim=0).values  # [K]; per input-channel of W is also K
    # Wait — W is [N, K], so per-input-channel max is .max(dim=0) = [K].
    s = (a_max ** alpha) / (w_max.clamp(min=1e-5) ** (1 - alpha))  # [K]
    return A / s, W * s  # rescaled A and W

A_smooth, W_smooth = smoothquant(A, W, alpha=0.5)
print("\nAfter SmoothQuant (α=0.5):")
print(f"  Activation max per channel range: {A_smooth.abs().max(dim=0).values.max().item():.2f}")
print(f"  Activation max per channel range (typical): {A_smooth.abs().max(dim=0).values.median().item():.2f}")
print(f"  Outlier ratio: {(A_smooth.abs().max(dim=0).values.max() / A_smooth.abs().max(dim=0).values.median()).item():.2f}x")

print("\nWeight range (slightly broader, but still INT8-friendly):")
print(f"  W max per row: max={W.abs().max(dim=1).values.max().item():.3f}, median={W.abs().max(dim=1).values.median().item():.3f}")
print(f"  W_smooth max per row: max={W_smooth.abs().max(dim=1).values.max().item():.3f}, median={W_smooth.abs().max(dim=1).values.median().item():.3f}")

# Verify the matmul is unchanged.
Y_original = A @ W.t()
Y_smooth = A_smooth @ W_smooth.t()
print(f"\nMatmul outputs match: max diff = {(Y_original - Y_smooth).abs().max().item():.6f}")
```

You should see the outlier ratio drop dramatically after SmoothQuant — from 30× to maybe 2-3×. Both A and W are now in better shape for INT8 quantization.

## Further reading

- "SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models" (Xiao et al, 2022) — the paper.
- Module 3 Lesson 3 — the broader INT8 quantization story.
- TensorRT-LLM W8A8 documentation.
- "Outlier Suppression+" (Wei et al, 2023) — an alternative to SmoothQuant.

Next lesson: **FP8 inference on Hopper/Blackwell.** The newer alternative to INT8 with native hardware support. We cover the FP8 formats (E4M3, E5M2 — recap from Module 3 Lesson 2), the kernel paths, and when FP8 wins over INT8.
