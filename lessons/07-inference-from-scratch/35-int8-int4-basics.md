---
title: "Lesson 35 — INT8 / INT4 Basics for Inference"
date: "2026-06-04"
module: "inference-from-scratch"
order: 35
tags: ["quantization", "int8", "int4", "symmetric", "per-channel", "gemm"]
author: "Sudipta Pathak"
prerequisites: ["34-expert-parallelism"]
---

# Lesson 35 — INT8 / INT4 Basics for Inference

## Why this lesson exists

Module 3 covered weight quantization from the algorithm and recipe perspective. Module 6 covered the on-device deployment perspective. This lesson is the inference-engine-internals view: what the GEMM kernel actually does when one operand is INT4 or INT8, the dequant fusion pattern, and the precision-of-accumulation gotcha that bites engineers who haven't seen it before.

For the rest of Part 6 to build on this (GPTQ, AWQ, SmoothQuant, FP8, BitNet), we need to align on what an INT8 or INT4 matmul *actually computes* at the kernel level.

The lesson is reading. The Hands-on implements a dequant-fused matmul.

## The standard W4A16 inference path

The most common production setup: weights in INT4, activations in FP16, matmul produces FP16 output.

The kernel:

```
W stored as: [N, K/8] int32_packed  (8 INT4 per int32)
scales: [N, K/group_size] fp16
zero_points: [N, K/group_size] int8 (for asymmetric)

A stored as: [M, K] fp16

for each output element [m, n]:
    accumulator = 0.0
    for k in range(K):
        group = k // group_size
        w_packed = W[n, k // 8]
        w_int4 = (w_packed >> (4 * (k % 8))) & 0xF
        w_int4 -= 8  # shift to signed range [-8, 7]
        w_fp16 = w_int4 × scales[n, group]
        accumulator += W_fp16 × A[m, k]
    output[m, n] = accumulator
```

The actual production kernel doesn't materialize the dequantized FP16 weight as an intermediate; it fuses the dequant inline with the multiply-accumulate. This is what saves the bandwidth — the INT4 weight is read from HBM once, dequantized in the SM's register file, and consumed by the GEMM.

## The accumulator precision question

The matmul accumulates many products. If the accumulator is INT16 or FP16, you risk overflow or precision loss.

Standard convention: **accumulator is FP32** (or INT32 for integer-only matmul). The intermediate accumulation has enough headroom; the result is downcast to FP16 at the end.

For INT8 weights × INT8 activations: accumulator is INT32. Sum of `K` products of two INT8 values fits in INT32 (no overflow up to K=2^17 or so).

For INT4 weights × FP16 activations: accumulator is FP32. The intermediate is float; the result is downcast to FP16 for the next layer.

If the runtime uses FP16 accumulator (rare; would be a bug for any meaningful K), you'd see precision degradation on long-K matmuls (the FFN's `K=14336` for a 7B Llama is enough to hit accuracy issues).

## The W8A8 path

Both weights and activations INT8:

```
W stored as: [N, K] int8
scales_W: [N] fp16 (per-channel)

A stored as: [M, K] int8
scales_A: [M] fp16 (per-token) or scalar (per-tensor)

for each output element [m, n]:
    int32_accumulator = 0
    for k in range(K):
        int32_accumulator += W[n, k] × A[m, k]  # INT8 × INT8 → INT32 accum
    output[m, n] = int32_accumulator × scales_W[n] × scales_A[m]  # downcast to FP16
```

This uses the GPU's INT8 tensor cores at peak throughput. NVIDIA Ampere+ has dedicated INT8 paths that are 2× faster than FP16 paths for the same compute.

W8A8 is the throughput-maximizing inference path on Hopper / Ampere. The catch: activations have outliers (Module 3 Lesson 3); naive per-tensor INT8 activation quantization destroys quality. SmoothQuant (Lesson 38) is the fix.

## The W4A16 vs W4A4 question

W4A16: weights INT4, activations FP16. Standard for memory-bandwidth-bound deployments.
- 4× weight memory reduction.
- 4× bandwidth reduction on weight reads.
- Activations stay in FP16; no activation-side problems.
- The GEMM uses FP16 tensor cores (dequantize INT4 → FP16, multiply by FP16 activations).

W4A4: weights AND activations INT4. More aggressive.
- 4× weight memory reduction PLUS 4× activation memory reduction.
- Native INT4 tensor cores (only on Blackwell and later for FP4; some research kernels exist for INT4).
- Quality cost: activation INT4 is hard; need careful per-group quantization.
- Rare in 2026 production; relevant for very-bandwidth-bound serving.

For most deployments, W4A16 is the right choice. W8A8 for high-throughput servers. W4A4 for the bleeding edge.

## Per-tensor vs per-channel vs per-group

A reminder from Module 3:

- **Per-tensor**: one scale for the whole tensor. Simple; works for small models or short tensors.
- **Per-channel (per-row)**: one scale per output channel. The standard for weight quantization.
- **Per-group**: one scale per group of N consecutive elements along the input dim. Standard for INT4 (group=128).

The finer the granularity, the more scale parameters to store and the better the accuracy. The kernel handles whichever granularity the format chose.

## The dequant fusion pattern

The key insight that makes quantized inference fast: never materialize the dequantized weight tensor in HBM. The kernel reads INT4 from HBM, dequantizes in the SM's register file (essentially free), uses the FP16-equivalent value in the GEMM, accumulates in FP32.

This means:
- **Bytes from HBM**: only the INT4 weights + scales + zero-points (small).
- **FLOPs**: same as if the weights were FP16 (the GEMM runs on FP16 internally after dequant).
- **Effective arithmetic intensity**: higher than FP16, because we're using less bandwidth for the same compute.

The dequant fusion is what makes quantization actually fast — not the smaller storage by itself, but the fact that the storage savings translate to bandwidth savings without compute slowdown.

## What you should believe after this lesson

Three sentences:

**1. The standard inference quantization recipe is W4A16**: INT4 weights with per-group scales (group=128), FP16 activations, FP16 output. The GEMM kernel reads INT4 from HBM, dequantizes inline in registers, multiplies in FP16, accumulates in FP32.

**2. W8A8 is the throughput-maximizing variant** on Ampere+ (2× faster than FP16 via INT8 tensor cores), but requires SmoothQuant-style activation handling because activations have outliers. W4A4 exists in research but is rare in production.

**3. The dequant-fused matmul pattern is what makes quantization fast** — never materialize the dequantized weight in HBM. Bytes read drop with the quantization factor; effective bandwidth utilization improves; throughput goes up.

## Hands-on (at home)

Implement a small W4A16 matmul demonstration.

```python
# w4a16_demo.py
import torch

torch.manual_seed(0)

def quantize_w4_per_group(W, group_size=128):
    """W: [N, K] fp16. Returns (W_int4_packed, scales)."""
    N, K = W.shape
    assert K % group_size == 0
    W_grouped = W.view(N, K // group_size, group_size)
    scales = W_grouped.abs().amax(dim=-1, keepdim=True) / 7.0  # [N, K//group_size, 1]
    W_quantized = (W_grouped / scales).round().clamp(-8, 7).to(torch.int8)
    return W_quantized.view(N, K), scales.squeeze(-1).to(torch.float16)

def dequantize_w4(W_int8, scales, group_size=128):
    N, K = W_int8.shape
    W_grouped = W_int8.view(N, K // group_size, group_size).float()
    return (W_grouped * scales.unsqueeze(-1)).view(N, K).half()

# Test with a Llama-sized FFN linear.
N, K = 4096, 14336
W = torch.randn(N, K).half()
W_int4, scales = quantize_w4_per_group(W, group_size=128)

# Memory comparison.
W_size_fp16 = W.element_size() * W.numel()
W_size_int4 = W_int4.numel() // 2 + scales.element_size() * scales.numel()
print(f"FP16: {W_size_fp16/1e6:.1f} MB")
print(f"INT4 + scales: {W_size_int4/1e6:.1f} MB")
print(f"Compression: {W_size_fp16/W_size_int4:.1f}x")

# Reconstruction quality.
W_rec = dequantize_w4(W_int4, scales)
err = (W - W_rec).abs().mean().item()
print(f"Mean reconstruction error: {err:.5f}")

# A W4A16 matmul (using the dequantize as a fused-in-kernel simulation).
A = torch.randn(8, K).half()
Y_quantized = A @ W_rec.t()  # The dequantized weights, then standard matmul.
Y_fp16 = A @ W.t()  # The original FP16 matmul.
print(f"Output cosine similarity: {torch.nn.functional.cosine_similarity(Y_quantized.flatten(), Y_fp16.flatten(), dim=0).item():.5f}")
```

In production, the dequant is fused into the matmul kernel — no `W_rec` materialization. Libraries like `bitsandbytes`, `auto-gptq`, and `autoawq` provide the fused kernels.

## Further reading

- Module 3 Lessons 3 and 4 — the algorithm-level coverage.
- "GPTQ" and "AWQ" papers — the algorithms that produce the W_int4 starting from a pretrained model.
- "QQQ: Quality Quattuordecillion Quantization for Large Language Models" — a recent (2024) approach unifying weight + activation + KV quantization.
- NVIDIA's CUTLASS library for quantized GEMM kernel design.

Next lesson: **GPTQ — second-order quantization.** The error-correcting algorithm that produces high-quality INT4 weights. Module 3 Lesson 7 covered it; this lesson covers the runtime kernel implications.
