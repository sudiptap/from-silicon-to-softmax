---
title: "Lesson 3 — INT8 Quantization: Math and Recipes"
date: "2026-06-04"
module: "ml-internals"
order: 3
tags: ["int8", "quantization", "symmetric", "asymmetric", "per-channel", "smoothquant"]
author: "Sudipta Pathak"
prerequisites: ["02-precision-landscape"]
---

# Lesson 3 — INT8 Quantization: Math and Recipes

## Why this lesson exists

INT8 is the format that almost always works. The math is clean, the hardware support is universal (every accelerator from 2018 onward has INT8 tensor cores or equivalents), and the accuracy loss on most LLMs is in the noise after a half-day of calibration. If you can only learn one quantization format, learn this one — every other format in the module borrows its mental model, and most production deployments still default to INT8 weights for the parts of the model where INT4 would be too aggressive.

This lesson is the math and the practical recipes: scale-and-zero-point, symmetric vs asymmetric, per-tensor vs per-channel, weight-only vs weight-activation, and the SmoothQuant-style activation-outlier handling that turns "INT8 mostly works" into "INT8 works on the layers that previously broke."

The lesson is reading. The Hands-on does a full per-channel INT8 quantization of a Llama-class linear layer, measures the round-trip error, and shows the matmul speedup.

## The basic mapping

Given a tensor `x` of FP32 (or BF16) values, we want to store it as 8-bit integers in `[-128, 127]` (signed) or `[0, 255]` (unsigned). The mapping is affine:

```
q = round(x / s + z)
x ≈ s × (q - z)
```

where:
- `s` (scale) is a positive FP value that maps the FP range to the integer range.
- `z` (zero-point) is an integer offset that ensures `0` in the FP space maps to *something* in the integer space exactly.

`s` and `z` are stored alongside the integer tensor. The integer tensor itself is what gets loaded; `s` and `z` are tiny scalars (or per-channel vectors).

The two variants:

**Symmetric quantization** (`z = 0`):
- `s = max(|x|) / 127`, signed INT8.
- FP zero maps to integer zero. Cleaner math, easier hardware: the dequant step is `s × q`, no subtraction.
- Slight precision loss when `x`'s range is asymmetric (e.g., post-ReLU activations are all ≥ 0; symmetric INT8 wastes the negative half of the range).
- The default for weights — weights are usually roughly zero-centered.

**Asymmetric quantization** (`z ≠ 0`):
- `s = (max(x) - min(x)) / 255`, `z = -min(x) / s` (unsigned INT8).
- Uses the full integer range regardless of FP distribution.
- Better for activations after ReLU/GeLU where the distribution is one-sided.
- Costs a subtraction in the matmul accumulation.

In practice: **symmetric for weights, asymmetric for activations** is the standard recipe. Many modern recipes go symmetric for both because the simpler math composes better with fused kernels.

## How the matmul actually works

The point of an INT8 matmul is to load INT8 weights from memory and let the tensor cores do the math in higher precision. Here's the standard fused pattern.

For weights quantized symmetrically per output-channel (one scale `s_w[c]` per output channel `c`) and activations quantized per-tensor symmetrically (one scale `s_a`), the math is:

```
W[c, k] ≈ s_w[c] × q_w[c, k]      # weight is INT8
A[b, k] ≈ s_a × q_a[b, k]          # activation is INT8

(WA)[c, b] = sum_k W[c, k] × A[b, k]
           = s_w[c] × s_a × sum_k q_w[c, k] × q_a[b, k]
           = s_w[c] × s_a × INT32_acc[c, b]
```

The `INT32_acc` is what the tensor core computes natively: INT8 × INT8 → INT32 accumulator. Then a per-output-channel scalar multiplication folds in `s_w[c] × s_a`, and the result is back in FP for the next layer.

The fused pattern in production kernels: INT8 matmul → INT32 accum → scale × accum → quantize to INT8 for the next layer. No round trip to FP32 in between. This is what cuBLASLt, MLX's `quantized_matmul`, and `llama.cpp`'s INT8 paths do under the hood.

## Per-tensor vs per-channel vs per-group

The choice of *granularity* of `s` is the single biggest accuracy lever.

**Per-tensor**: one `s` for the entire tensor. Cheapest (one scalar to store), works when the value distribution is uniform, breaks when one channel has wildly different range than the others.

**Per-channel (also called per-row for weights)**: one `s` per output channel. For a weight matrix of shape `[out, in]`, you store `out` scales — usually 1–10 KB extra per layer, negligible. Handles the common case where some output features have different magnitudes than others. **This is the default for LLM weight INT8.**

**Per-group**: one `s` per group of consecutive elements along the input dimension. Group sizes of 32, 64, 128 are common. Used for INT4 (next lesson) where per-channel alone isn't enough. Also seen in INT8 for activations.

The pattern: the smaller the granularity, the more memory the scales take, the better the accuracy. For weights, per-channel is almost always sufficient at INT8. For activations, per-tensor often works if you handle outlier channels separately (next section).

## The outlier problem

There's a well-known phenomenon in LLMs from ~6B parameters upward: a small number of channels (~0.1% of features) in some layers carry activations 10–100× larger than the median. These are called *outlier features*, and they break naive per-tensor activation quantization. The `s_a` you'd compute from the per-tensor max is dominated by the outliers, so the typical-channel values get crushed into 1–2 INT8 levels and the model collapses.

Three responses:

**1. Per-channel activation quantization.** Conceptually cleanest but kernel-unfriendly: the standard INT8 matmul fuses `s_a` as a single scalar in the accumulator scaling step; per-channel `s_a` means a different scale per K dimension, which breaks the fusion. Requires kernel changes.

**2. Mixed precision: keep outlier channels in FP16.** This is the LLM.int8() approach (Dettmers et al, 2022). Decompose the matmul into INT8 × INT8 for the well-behaved channels and FP16 × FP16 for the few outlier channels, then sum. Costs ~0.5% throughput; near-zero accuracy hit. Was the default for early LLM INT8 deployments.

**3. SmoothQuant: migrate the outlier difficulty from activations to weights.** This is the most common 2024+ recipe. The insight: if `Y = A × W`, then `Y = (A / s) × (s × W)` for any per-channel diagonal `s`. Picking `s` to equalize the dynamic ranges of `A/s` and `s×W` makes both INT8-friendly. The weights absorb most of the outlier-channel scale; the activations become well-behaved. Both can be quantized per-tensor without loss.

SmoothQuant's `s_i = max(|A_i|)^α / max(|W_i|)^(1-α)` with `α ≈ 0.5` is the standard. It's a one-time, offline transformation; you bake the smoothing factor into the weights and the model is now INT8-friendly without further changes at runtime.

## Weight-only vs weight-activation INT8

There are two distinct INT8 deployments:

**W8A16 (weight-only INT8):** Weights stored as INT8, activations stay in FP16/BF16. The kernel up-casts INT8 weights to FP16 inside the tensor core, then does the matmul in FP16. Wins: 2× weight memory reduction (FP16 → INT8), no accuracy concerns from activations (still FP16). Loses: no matmul speedup from INT8 tensor cores; you're still bandwidth-bound, but loading half the bytes. **This is the default INT8 deployment for LLMs in 2025–2026.**

**W8A8 (weight-activation INT8):** Both stored and computed at INT8. Wins: 2× weight memory + uses INT8 tensor cores, so on H100/B100 you get a real matmul throughput bump. Loses: activation quantization can hurt accuracy unless you do SmoothQuant or LLM.int8(). Most useful for CNN deployments and large-batch LLM serving; less common for small-batch on-device because the matmul speedup doesn't matter (you're memory-bound anyway — Lesson 1).

The rough rule for inference:

- On-device, batch 1: W8A16 weight-only. The matmul TFLOPs don't matter; the memory bandwidth does. Quantize weights only.
- Server, large batch: W8A8 with SmoothQuant. The tensor-core throughput matters; activation quantization is worth the small accuracy hit because the compute uplift is real.
- Training: BF16 or FP8 (Lesson 2), not INT8. INT8 training exists (NVIDIA's TransformerEngine has paths) but is not the standard.

## What you should believe after this lesson

Three sentences:

**1. The INT8 mental model is `x ≈ s × q` plus a possibly-zero `z`, with the math fused so that the matmul runs as INT8 → INT32 accumulator → scale.** Everything else is a refinement of the granularity (per-tensor / per-channel / per-group) or a handling of outliers.

**2. Outlier activations are the practical obstacle, and SmoothQuant solved it in 2022 by migrating the outlier burden from activations to weights via a per-channel rescaling.** This is the default activation handling in modern INT8 LLM recipes.

**3. For on-device batch-1 inference, weight-only INT8 (W8A16) is the dominant deployment** — the activation savings of W8A8 don't matter when you're memory-bound, and W8A16 is safer accuracy-wise.

## Hands-on (at home)

A per-channel symmetric INT8 quantization of a real linear layer, round-trip error measurement, and a speed comparison against FP16.

```python
# int8_linear.py
import torch
import torch.nn as nn
import time

torch.manual_seed(0)
device = 'cuda' if torch.cuda.is_available() else 'mps'

# A linear layer roughly the size of a Llama 8B FFN gate projection.
in_features, out_features = 4096, 14336
fp16 = nn.Linear(in_features, out_features, bias=False).to(device).half()
W = fp16.weight.detach()  # [out, in], FP16

# Per-channel symmetric quantization (per output channel).
s = W.abs().amax(dim=1, keepdim=True) / 127.0      # [out, 1]
q = (W / s).round().clamp(-128, 127).to(torch.int8) # [out, in]
W_dequant = (q.float() * s.float()).half()         # FP16 reconstruction

# Round-trip error.
err = (W - W_dequant).abs()
print(f"max abs error: {err.max().item():.4e}")
print(f"mean abs error: {err.mean().item():.4e}")
print(f"max relative error in non-tiny weights: "
      f"{(err / W.abs().clamp(min=1e-3)).quantile(0.999).item():.4e}")

# Compare matmul outputs on a random batch.
x = torch.randn(8, 4096, device=device, dtype=torch.float16)
y_fp16 = x @ W.t()
y_dequant = x @ W_dequant.t()
print(f"output rel err (cosine-1): "
      f"{1 - torch.nn.functional.cosine_similarity(y_fp16.flatten(), y_dequant.flatten(), dim=0).item():.4e}")

# Storage size.
print(f"FP16 weights: {W.element_size() * W.numel() / 1e6:.1f} MB")
print(f"INT8 weights + scales: "
      f"{(q.element_size() * q.numel() + s.element_size() * s.numel()) / 1e6:.1f} MB")
```

You should see round-trip max error around 1e-3 to 1e-2 (one part in 256, which is the resolution of INT8) and cosine similarity essentially 1.0 on the matmul outputs. Weights shrink from ~117 MB to ~58 MB.

Part 2 — measure actual INT8 matmul throughput (requires bitsandbytes or torch's native INT8 paths).

```python
# int8_matmul_bench.py
import torch
import time
try:
    import bitsandbytes as bnb
    HAVE_BNB = True
except ImportError:
    HAVE_BNB = False

device = 'cuda'
M, K, N = 8, 4096, 14336
x = torch.randn(M, K, device=device, dtype=torch.float16)
W = torch.randn(N, K, device=device, dtype=torch.float16)

def bench(name, fn):
    for _ in range(5): fn()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(100): fn()
    torch.cuda.synchronize()
    print(f"{name:30s}: {(time.time()-t0)*10:.3f} ms/iter")

bench("FP16 matmul", lambda: x @ W.t())

if HAVE_BNB:
    Wi8 = bnb.nn.Int8Params(W.float(), has_fp16_weights=False, requires_grad=False).cuda()
    layer = bnb.nn.Linear8bitLt(K, N, has_fp16_weights=False).cuda()
    layer.weight = Wi8
    bench("bnb INT8 matmul (W8A16)", lambda: layer(x))
```

On a 4090 you should see W8A16 latency near or below FP16 latency at batch 8 — the matmul is memory-bound, halving the weight bytes loaded halves the latency. At batch 1024 you'd see them converge as the workload becomes compute-bound. This is the same arithmetic-intensity story from Lesson 1.

## Further reading

- "LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale" (Dettmers et al, 2022) — the original mixed-precision-with-outliers INT8 recipe.
- "SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models" (Xiao et al, 2022) — the migrate-outliers-to-weights trick.
- NVIDIA TensorRT-LLM docs on INT8 quantization — the production deployment view, including the fused dequant kernel patterns.
- `bitsandbytes` GitHub README — the standard Python entry point for INT8 LLM inference on NVIDIA.

Next lesson: **INT4 quantization — per-channel, per-group, the formats.** Same mental model, twice the compression, and the per-channel scales that worked at INT8 start breaking down — which is why per-group becomes the rule and AWQ/GPTQ become the workhorses.
