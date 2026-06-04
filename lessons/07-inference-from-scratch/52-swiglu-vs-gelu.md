---
title: "Lesson 52 — SwiGLU vs GELU"
date: "2026-06-04"
module: "inference-from-scratch"
order: 52
tags: ["activation", "swiglu", "gelu", "ffn", "block-level"]
author: "Sudipta Pathak"
prerequisites: ["51-rmsnorm-vs-layernorm"]
---

# Lesson 52 — SwiGLU vs GELU

## Why this lesson exists

The transformer's FFN typically has two flavors:

**Standard FFN with GELU**: `down(GELU(up(x)))` — two linear layers separated by GELU activation.

**SwiGLU FFN**: `down(SiLU(gate(x)) * up(x))` — three linear layers; the gate's output gates the up projection element-wise; the result goes through the down projection.

The 2020-2023 wave of LLMs (Llama, Mistral, etc.) switched from GELU to SwiGLU. The change is small but consistent in the literature; SwiGLU is now the default.

This lesson covers both, the empirical justification, and the inference-side implication.

The lesson is reading. The Hands-on benchmarks both FFN variants.

## Standard FFN with GELU

```python
class GELU_FFN(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.up = nn.Linear(d_model, d_ff, bias=False)
        self.down = nn.Linear(d_ff, d_model, bias=False)
    def forward(self, x):
        return self.down(F.gelu(self.up(x)))
```

GELU (Gaussian Error Linear Unit): `x × Φ(x)` where `Φ` is the CDF of the standard normal. A smooth ReLU-like function.

The FFN has 2 linear layers; parameter count `2 × d_model × d_ff`.

For `d_model = 4096`, `d_ff = 16384` (4× expansion): `2 × 4096 × 16384 = 134M` parameters per FFN.

## SwiGLU FFN

```python
class SwiGLU_FFN(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.gate = nn.Linear(d_model, d_ff, bias=False)
        self.up = nn.Linear(d_model, d_ff, bias=False)
        self.down = nn.Linear(d_ff, d_model, bias=False)
    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))
```

SiLU (Sigmoid Linear Unit; also called Swish): `x × sigmoid(x)`. Close to GELU but slightly different.

The SwiGLU FFN has 3 linear layers; parameter count `3 × d_model × d_ff`. 50% more parameters than the GELU version at the same `d_ff`.

To match parameter counts, Llama-class models use `d_ff = (2/3) × 4 × d_model = (8/3) × d_model` for SwiGLU. So for `d_model = 4096`: `d_ff ≈ 10923` (rounded up to nearest convenient multiple, e.g., 11008 or 14336 depending on Llama version).

The 8/3 ratio comes from Shazeer's "GLU Variants Improve Transformer" paper — it's the ratio that matches a GELU FFN's parameter count exactly when you have 3 linear layers instead of 2.

## Why SwiGLU won

The empirical finding (Shazeer 2020 "GLU Variants Improve Transformer" + follow-ups): SwiGLU and other GLU variants (GeGLU, ReGLU) consistently outperform standard FFN-with-activation. The improvement is small (~0.5% on standard benchmarks) but consistent and reproducible.

The gating mechanism (`silu(gate) * up`) appears to give the FFN extra expressive capacity. The gate can selectively pass through or suppress certain dimensions of the up-projection; the FFN can implement more complex functions per parameter.

Trade-off:
- SwiGLU has 50% more parameters at the same `d_ff` (3 linear layers vs 2).
- Per-token compute is 1.5× the GELU FFN (3 matmuls vs 2).
- To match parameter count, you shrink `d_ff` by 2/3, recovering the original compute. The quality gain is "free" at the same compute.

For the parameter-matched comparison (the same total params): SwiGLU is consistently ~0.5% better.

## Production landscape

Models using SwiGLU:
- Llama (all versions).
- Mistral.
- Mistral.
- Qwen (all versions).
- Gemma.
- Phi.
- DeepSeek.
- Almost every modern open-weights LLM.

Models using GELU:
- GPT-2/3 family.
- BERT family.
- Some research models.

Older models. SwiGLU is the 2026 default.

## Inference implication

SwiGLU has 3 matmuls per FFN; GELU has 2. The kernel structure:

GELU FFN:
- Matmul 1: `up` (`d_model → d_ff`).
- Activation: `gelu`.
- Matmul 2: `down` (`d_ff → d_model`).

SwiGLU FFN:
- Matmul 1: `gate` (`d_model → d_ff`).
- Matmul 2: `up` (`d_model → d_ff`).
- Element-wise: `silu(gate) * up`.
- Matmul 3: `down` (`d_ff → d_model`).

For W4A16 (Module 3 / Lesson 35), SwiGLU's three matmuls each load their INT4 weights and produce FP16 outputs. The `gate` and `up` matmuls can be *fused* into a single matmul with output dimension `2 × d_ff` (concatenated outputs split into gate and up). Some production kernels do this; saves one kernel launch.

Net inference cost: SwiGLU at the parameter-matched setup is ~equivalent to GELU. The 50% more matmuls are compensated by the 2/3 smaller `d_ff`.

## What you should believe after this lesson

Three sentences:

**1. SwiGLU is `down(silu(gate(x)) * up(x))` with 3 linear layers**; GELU FFN is `down(gelu(up(x)))` with 2 linear layers. SwiGLU's gating mechanism gives slightly more expressive capacity per parameter (~0.5% benchmark improvement).

**2. For parameter-matched comparisons, SwiGLU uses `d_ff = (8/3) × d_model`** (instead of `4 × d_model` for GELU); this keeps total parameters and compute equivalent. The quality improvement is "free" at the same compute.

**3. SwiGLU is the 2026 default** for new LLMs; GELU persists in older models (BERT, GPT-2/3). Production kernels can fuse the gate and up matmuls into one to recover some of the launch overhead.

## Hands-on (at home)

Compare GELU and SwiGLU FFNs.

```python
# ffn_comparison.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import time

class GELU_FFN(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.up = nn.Linear(d_model, d_ff, bias=False)
        self.down = nn.Linear(d_ff, d_model, bias=False)
    def forward(self, x):
        return self.down(F.gelu(self.up(x)))

class SwiGLU_FFN(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.gate = nn.Linear(d_model, d_ff, bias=False)
        self.up = nn.Linear(d_model, d_ff, bias=False)
        self.down = nn.Linear(d_ff, d_model, bias=False)
    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))

D = 4096
# Parameter-matched: GELU at d_ff=4D; SwiGLU at d_ff=(8/3)D.
gelu_ff = GELU_FFN(D, 4*D).cuda().half()
swiglu_ff = SwiGLU_FFN(D, int(D * 8 / 3)).cuda().half()

print(f"GELU params: {sum(p.numel() for p in gelu_ff.parameters()):,}")
print(f"SwiGLU params: {sum(p.numel() for p in swiglu_ff.parameters()):,}")
# Should be very close.

x = torch.randn(32, 1024, D, device='cuda', dtype=torch.float16)

def bench(m, n=200):
    for _ in range(10): m(x)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n): m(x)
    torch.cuda.synchronize()
    return (time.time() - t0) / n * 1000

print(f"GELU FFN:   {bench(gelu_ff):.2f} ms/iter")
print(f"SwiGLU FFN: {bench(swiglu_ff):.2f} ms/iter")
```

Per-iteration time should be similar (parameter-matched). Quality difference is only visible after training to convergence — which requires more than a hands-on session.

## Further reading

- "GLU Variants Improve Transformer" (Shazeer, 2020).
- "Searching for Activation Functions" (Ramachandran et al, 2017) — origins of Swish/SiLU.
- "GAUSSIAN ERROR LINEAR UNITS (GELUS)" (Hendrycks & Gimpel, 2016) — origins of GELU.
- Llama, Mistral, Qwen technical reports — SwiGLU in production.

Next lesson: **Pre-norm vs post-norm.** Where the LayerNorm/RMSNorm sits relative to the residual stream determines training stability. Pre-norm became the default for stability reasons; post-norm has some theoretical advantages.
