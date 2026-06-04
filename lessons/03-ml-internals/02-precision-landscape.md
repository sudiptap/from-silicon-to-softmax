---
title: "Lesson 2 — FP16 vs BF16 vs FP8: The Precision Landscape"
date: "2026-06-04"
module: "ml-internals"
order: 2
tags: ["fp16", "bf16", "fp8", "mxfp", "ieee754", "precision", "dynamic-range"]
author: "Sudipta Pathak"
prerequisites: ["01-flop-arithmetic"]
---

# Lesson 2 — FP16 vs BF16 vs FP8: The Precision Landscape

## Why this lesson exists

Most people say "FP16" and mean "low precision." But there are at least five formats sitting between FP32 and INT8, and they are *not* interchangeable. FP16 and BF16 occupy the same 16 bits and disagree about how to spend them. FP8 splits into E4M3 and E5M2 — same width, different decisions about exponent vs mantissa, used in different stages of training. MXFP4/MXFP6/MXFP8 are entirely different beasts that bundle a shared exponent across small blocks of values.

This lesson is the catalog. We walk through the IEEE 754 mental model, then look at each format's bit layout, its dynamic range, its precision near zero, and the empirical answer to "where in the model does this format break." Once you know the catalog, the integer formats in the next two lessons make immediate sense — they're solving the same problem from a different starting point.

The lesson is reading. The Hands-on shows the bit layouts and demonstrates what each format actually represents.

## IEEE 754, fast

A floating-point number is `(-1)^s × 1.m × 2^(e - bias)`, encoded as three fields packed left-to-right: sign `s`, exponent `e`, mantissa `m`. The mantissa is the significant digits; the exponent is the scale; the sign is, well, the sign.

For a given total width, you choose how to split the bits between exponent and mantissa. More exponent bits = larger *dynamic range* (more powers of 2 you can represent). More mantissa bits = higher *precision* (more distinct values inside each power-of-2 range). It is a strict tradeoff. The whole story of FP16 vs BF16 vs FP8 is which tradeoff you picked.

The five canonical formats, all with 1 sign bit:

| Format | Total bits | Exp bits | Mantissa bits | Max value | Min normal | Smallest gap |
| ------ | ----------- | -------- | ------------- | --------- | ----------- | ------------ |
| FP32   | 32 | 8 | 23 | 3.4e38 | 1.2e-38 | ~1e-7 relative |
| FP16   | 16 | 5 | 10 | 65504 | 6.1e-5 | ~1e-3 relative |
| BF16   | 16 | 8 | 7 | 3.4e38 | 1.2e-38 | ~1e-2 relative |
| FP8 E4M3 | 8 | 4 | 3 | 448 | 1.95e-3 | ~0.06 relative |
| FP8 E5M2 | 8 | 5 | 2 | 57344 | 6.1e-5 | ~0.13 relative |

Two things to read out of that table:

**FP16 vs BF16.** Same 16 bits. FP16 has 5 exponent bits → max value 65504 → overflow city for any tensor that has even one large value. BF16 has 8 exponent bits (same as FP32!) → max value 3.4×10³⁸ → essentially the same dynamic range as FP32 → no overflow. The price BF16 pays: 7 mantissa bits vs FP16's 10. So FP16 is 8× more precise within a power-of-2 range, but FP16 overflows where BF16 doesn't even sweat.

**FP8 E4M3 vs E5M2.** Same 8 bits. E4M3 has max ~448 (small dynamic range), E5M2 has max ~57344 (FP16-like dynamic range). E4M3 has 3 mantissa bits (more precision); E5M2 has 2 (less precision).

The empirical recipe — established by NVIDIA's Hopper training papers and by every FP8 deployment since — is:

- **E4M3 for forward activations and weights.** The values are bounded by the activations and the typical norm; you want more precision and less dynamic range.
- **E5M2 for backward gradients.** Gradients have wild dynamic range (small values dominate, occasional huge ones); you want more dynamic range and tolerate less precision.

This is why FP8 *training* uses both formats. FP8 *inference* mostly uses E4M3.

## Where FP16 historically broke

The single most famous failure mode: training FP16 models without loss scaling. Gradient values were below FP16's smallest normal (6.1×10⁻⁵), so they underflowed to zero. The model trained, the loss went down, then training quietly diverged because half the gradients were silently dropped. The fix: multiply the loss by a large scaler (1024 to 65536) before backward, then divide it back out — pushing the gradients into FP16's representable range.

BF16 made this whole mess go away by giving you FP32's exponent range. You don't need loss scaling in BF16. This is why nearly every modern training run is in BF16 (or, on chips that support it, FP8 with E5M2 for grads). FP16 is a legacy format for production training — though it's still common for inference where the activations are in-range and the loss-scaling problem doesn't apply.

For *inference*, FP16 vs BF16 mostly doesn't matter for accuracy if the model was trained in either; both round to within noise of the other on most evals. But:

- If the original training was BF16, sticking with BF16 at inference is safest.
- If you're targeting hardware that prefers FP16 (older NVIDIA, some Apple paths), the conversion is usually fine; watch for activation outliers in the first and last few layers.

## FP8 in practice (2026 state)

Hopper (H100, H200) is the first generation with native FP8 tensor cores. Their peak throughput is 2× the FP16 throughput on the same chip — directly because the operand width halved, so the tensor cores can chew through twice as many elements per cycle. Blackwell takes this further to FP4. The hardware path is real and significant.

For *inference*, the practical FP8 question is: can you take a BF16 checkpoint and run it directly at FP8 without retraining? The answer, mostly:

- Activations have outliers in some channels (the famous "outlier features" — a handful of channels in some layers carry values 10–100× larger than the median). E4M3's max of 448 means these channels overflow. The fix is per-tensor or per-channel scaling: precompute a scale factor per tensor, divide before storing in FP8, multiply back during matmul accumulation.
- Weights are usually well-behaved; per-tensor FP8 weights work for most layers.
- Layer norm, softmax, residual connections: keep in BF16/FP16. The chip will up-cast for these ops.

The kernel ends up looking like: load FP8 weights → up-cast to BF16 inside the tensor core → multiply against BF16 activations (or FP8, with the per-tensor scale applied) → accumulate in FP32. The "FP8 matmul" never actually does math in FP8 directly; it stores in FP8 and computes in higher precision. This is true for INT8 and INT4 too — the format is a *storage* format. The compute happens at higher precision inside the tensor core.

This is the unifying principle of the whole module: **low precision is a memory game, not a compute game.** (You met this in Lesson 1; you're going to meet it in every subsequent lesson.)

## MXFP: block-floating-point with shared exponents

The Open Compute Project's "Microscaling" formats (MXFP4, MXFP6, MXFP8, MXINT8) are the next-generation storage format. The idea: instead of giving each value its own exponent, share an exponent across a block of 32 values. The block has one shared FP8-ish scale, and each value within the block is a small mantissa (4 or 6 or 8 bits).

A 4-bit MXFP4 element: 4 bits per value × 32 values per block + 8-bit shared exponent = 136 bits per 32 values = 4.25 bits per value. Almost as compact as INT4 but with the dynamic-range win of a shared exponent.

Blackwell adds native MXFP4 tensor cores. Hopper's FP8 will likely be remembered as the transition format; production training on Blackwell+ is expected to migrate to MXFP6/MXFP8 in 2026–2027.

For *current inference work*, MXFP4 is most interesting because it provides a hardware-accelerated path to 4-bit weights *with reasonable dynamic-range handling*, where INT4 either needs careful per-group scales (next two lessons) or relies on software paths.

## The 1-bit / ternary frontier

The far edge: BitNet (1-bit) and BitNet 1.58 (ternary: -1, 0, +1). These are not post-hoc compressions; they require training from scratch with the format baked in (during training, weights are kept in higher precision for gradient updates, but the forward pass uses the quantized form). When it works, the win is enormous: matmul becomes essentially adds, hardware can be specialized, and bandwidth pressure collapses.

As of 2026 the BitNet-class models have shown they can match same-parameter BF16 baselines up to ~7B parameters, but the ecosystem (hardware support, framework support) is still nascent. We mention it because it's the asymptote — the limit of what compression can theoretically achieve — and because it argues that the "natural" precision of LLM weights is much lower than FP16. Most of the bits in an FP16 weight tensor are not load-bearing.

## What you should believe after this lesson

Three sentences:

**1. The 16-bit format choice between FP16 and BF16 is a choice between precision and dynamic range, and BF16's dynamic range has won every battle that matters in modern ML.** Use BF16 for training; for inference, use whichever the model was trained in.

**2. FP8 inference is real on Hopper and worth it (2× throughput uplift over FP16 in many cases), but requires per-tensor or per-channel scaling to handle outlier activations.** The format is E4M3 for activations and weights; E5M2 enters only during training (gradients).

**3. All low-precision formats are storage formats; the math still happens at higher precision inside the tensor core.** This is why the central win is reducing bytes loaded, not reducing FLOPs computed — and it sets up integer quantization in the next two lessons, which uses the same principle with an integer storage format.

## Hands-on (at home)

Look at the bits.

```python
# bit_layouts.py
import struct
import numpy as np

def show_fp32(x):
    bits = struct.unpack('I', struct.pack('f', x))[0]
    s = (bits >> 31) & 1
    e = (bits >> 23) & 0xff
    m = bits & 0x7fffff
    print(f"FP32 {x:>15.6e}: {s:01b} {e:08b} {m:023b}")

def show_fp16(x):
    bits = np.float16(x).view(np.uint16).item()
    s = (bits >> 15) & 1
    e = (bits >> 10) & 0x1f
    m = bits & 0x3ff
    print(f"FP16 {x:>15.6e}: {s:01b} {e:05b} {m:010b}")

def show_bf16(x):
    fp32_bits = struct.unpack('I', struct.pack('f', x))[0]
    bf16_bits = fp32_bits >> 16
    s = (bf16_bits >> 15) & 1
    e = (bf16_bits >> 7) & 0xff
    m = bf16_bits & 0x7f
    print(f"BF16 {x:>15.6e}: {s:01b} {e:08b} {m:07b}")

for x in [1.0, 0.1, 65500, 100000, 1e-6]:
    show_fp32(x)
    show_fp16(x)
    show_bf16(x)
    print()
```

You'll see FP16 saturate to infinity at 100000 (above its 65504 max) while BF16 represents it cleanly. You'll see FP16 represent 0.1 more precisely than BF16 (more mantissa bits). The tradeoff is right there in the bits.

Part 2 — measure the FP8 overflow problem on real activations.

```python
# fp8_overflow.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "Qwen/Qwen2.5-0.5B"  # small, fits on a laptop
tok = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16).eval()

# Hook to capture activations.
acts = {}
def hook(name):
    def f(_, _i, out):
        acts[name] = out.detach().float()
    return f

# Pick a middle layer's MLP up-projection input.
for n, m in model.named_modules():
    if n.endswith("mlp.up_proj") and "10" in n:  # layer 10-ish
        m.register_forward_hook(hook(n))
        break

inputs = tok("The quick brown fox jumps over the lazy dog", return_tensors="pt")
with torch.no_grad():
    model(**inputs)

for n, a in acts.items():
    finite = a[torch.isfinite(a)]
    print(f"{n}")
    print(f"  shape: {tuple(a.shape)}")
    print(f"  max abs: {a.abs().max().item():.3f}")
    print(f"  p99 abs: {finite.abs().quantile(0.99).item():.3f}")
    print(f"  p50 abs: {finite.abs().quantile(0.5).item():.3f}")
    print(f"  FP8 E4M3 max: 448 — overflow if max > 448 without scaling: {a.abs().max() > 448}")
```

Run it on a few different prompts and a few different layer positions. You will see, on most models, that activations stay well under 448 for early layers but creep up in middle layers; on some channels, they spike. This is the per-channel-outlier story that AWQ (Lesson 6) and FP8 scaling both have to solve.

## Further reading

- IEEE 754-2008 standard — for the formal definition of the FP32/FP64/FP16 formats.
- "NVIDIA H100 Tensor Core GPU Architecture" white paper — for the canonical description of FP8 paths in hardware.
- OCP Microscaling Formats (MX) specification — for the MXFP4/MXFP6/MXFP8 layout.
- "FP8 Formats for Deep Learning" (Micikevicius et al, NVIDIA/ARM/Intel, 2022) — the joint paper that standardized E4M3 / E5M2.
- "Outlier Suppression+" and "SmoothQuant" papers — for the activation-outlier characterization that drives FP8 and INT8 quantization choices.

Next lesson: **INT8 quantization — math and recipes.** The integer cousin of the FP8 story. Same idea (storage format, computed at higher precision), simpler math (no exponent — just a scale), and ready hardware support across every accelerator from a 2018 NVIDIA chip onward.
