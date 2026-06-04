---
title: "Lesson 51 — RMSNorm vs LayerNorm"
date: "2026-06-04"
module: "inference-from-scratch"
order: 51
tags: ["normalization", "rmsnorm", "layernorm", "block-level"]
author: "Sudipta Pathak"
prerequisites: ["50-reasoning-models"]
---

# Lesson 51 — RMSNorm vs LayerNorm

## Why this lesson exists

The original transformer (2017) used LayerNorm. The 2020-2023 generation of LLMs (Llama, Mistral, PaLM, GPT-NeoX) used RMSNorm. The change happened quietly; the practical difference is small but real.

This lesson explains both norms, the empirical justification for RMSNorm winning, and the inference-time speed implication.

The lesson is reading. The Hands-on benchmarks the two norms.

## LayerNorm

For a vector `x ∈ R^D`:

```
mean = x.mean(dim=-1, keepdim=True)
var = x.var(dim=-1, keepdim=True)
normalized = (x - mean) / sqrt(var + epsilon)
output = normalized * gamma + beta  # learnable affine
```

`gamma` and `beta` are learnable per-dimension parameters (each of shape `[D]`).

LayerNorm subtracts the mean and divides by the standard deviation, then re-scales and re-shifts via the learnable affine. It's a "complete" normalization — both the centering and the scaling.

## RMSNorm

For the same vector `x`:

```
rms = sqrt(x.pow(2).mean(dim=-1, keepdim=True) + epsilon)
normalized = x / rms
output = normalized * gamma  # only scaling, no shift
```

RMSNorm divides by the *root mean square* of the elements; no mean subtraction; no `beta` (no learnable shift).

Two simplifications relative to LayerNorm:
1. No mean subtraction (skip the `x - mean` step).
2. No `beta` learnable shift.

The result: RMSNorm has fewer parameters and slightly fewer FLOPs.

## Why RMSNorm replaced LayerNorm

The empirical finding (Zhang & Sennrich, 2019): RMSNorm achieves comparable quality to LayerNorm with slightly less compute. Convergence is similar; downstream task performance is essentially the same.

Reasons proposed:
- The mean subtraction in LayerNorm isn't critical for transformers — the activations are already roughly zero-centered after the residual stream stabilizes.
- The learnable `beta` shift can be absorbed by adjusting subsequent layers' biases; it's redundant.
- Simpler norm = slightly faster forward and backward; small but adds up over training.

The benefits at inference time:
- ~10-20% faster per norm operation (fewer ops).
- Slightly less memory.

For 32-layer transformers with norms before each attention and FFN sublayer, that's ~64 norm ops per token. The cumulative savings are small per token (~50 μs) but add up over the entire model and across the training run.

## What's used in production

The 2026 landscape:
- **RMSNorm**: Llama (all versions), Mistral, Qwen, Gemma, Phi, DeepSeek, most modern open-weights LLMs.
- **LayerNorm**: BERT, original GPT-2/3, some older models, encoder-decoder models like T5.

For new architectures: RMSNorm is the default. Almost no reason to use LayerNorm unless you're matching the architecture of an older pretrained model.

## Numerical considerations

The `epsilon` in both norms prevents division by zero:
- LayerNorm: typical `epsilon = 1e-5`.
- RMSNorm: typical `epsilon = 1e-6` (smaller because RMS is generally larger than std at low input magnitudes).

These don't usually matter; only relevant for very low-magnitude inputs (which transformers don't see in practice).

For mixed-precision training and inference, both norms compute in higher precision (FP32) to avoid catastrophic cancellation:
- The mean / variance / RMS computation runs in FP32.
- The final result is cast back to FP16 / BF16.

This is automatic in well-implemented kernels; manual reimplementations sometimes get it wrong.

## What you should believe after this lesson

Three sentences:

**1. RMSNorm is LayerNorm with the mean subtraction and the bias term removed** — `x / RMS(x) × gamma` vs `(x - mean) / std × gamma + beta`. The math is simpler; fewer parameters; slightly less compute.

**2. RMSNorm replaced LayerNorm in modern LLMs (Llama, Mistral, etc.)** because empirical performance is comparable and the slight speed advantage adds up across long training runs. There's no quality reason to prefer LayerNorm for new architectures.

**3. Both norms compute in FP32** internally to avoid mixed-precision numerical issues. Use the framework's built-in norm (`torch.nn.LayerNorm`, `torch.nn.RMSNorm` in PyTorch 2.4+) rather than reimplementing; the production kernels are tuned.

## Hands-on (at home)

Compare RMSNorm and LayerNorm.

```python
# norm_comparison.py
import torch
import torch.nn as nn
import time

class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d))
        self.eps = eps
    def forward(self, x):
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x / rms * self.gamma

D, B, N = 4096, 32, 2048
x = torch.randn(B, N, D, device='cuda', dtype=torch.float16)
ln = nn.LayerNorm(D).cuda().half()
rms = RMSNorm(D).cuda().half()

# Verify shapes match.
out_ln = ln(x)
out_rms = rms(x)
print(f"LayerNorm output: {out_ln.shape}")
print(f"RMSNorm output: {out_rms.shape}")

# Parameter count.
print(f"LayerNorm params: {sum(p.numel() for p in ln.parameters())}")
print(f"RMSNorm params: {sum(p.numel() for p in rms.parameters())}")
# LayerNorm: 2D (gamma + beta). RMSNorm: D (gamma only).

# Benchmark.
def bench(m, n=1000):
    for _ in range(10): m(x)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n): m(x)
    torch.cuda.synchronize()
    return (time.time() - t0) / n * 1000

t_ln = bench(ln)
t_rms = bench(rms)
print(f"LayerNorm: {t_ln:.3f} ms/iter")
print(f"RMSNorm:   {t_rms:.3f} ms/iter")
print(f"Speedup: {t_ln/t_rms:.2f}x")
```

On an H100 you'll see RMSNorm ~10-15% faster. Per-layer-call the difference is small; over 32 layers × all tokens it adds up.

## Further reading

- "Root Mean Square Layer Normalization" (Zhang & Sennrich, 2019) — the RMSNorm paper.
- "Layer Normalization" (Ba et al, 2016) — the original LayerNorm paper.
- PyTorch's `nn.RMSNorm` documentation (added in PyTorch 2.4).

Next lesson: **SwiGLU vs GELU.** The FFN's activation function. Llama-class models use SwiGLU; older models used GELU. The difference is small but worth understanding.
