---
title: "Lesson 6 — AWQ: Activation-Aware Weight Quantization"
date: "2026-06-04"
module: "ml-internals"
order: 6
tags: ["awq", "quantization", "int4", "activation-aware", "salient-channels"]
author: "Sudipta Pathak"
prerequisites: ["05-calibration-qat"]
---

# Lesson 6 — AWQ: Activation-Aware Weight Quantization

## Why this lesson exists

Naive INT4 doesn't quite work on LLMs. Calibration-driven scales (Lesson 5) help, but at INT4 there's still a perplexity hit of 0.5–2 points on most models — enough that the quantized model feels worse on tasks. AWQ closed most of that gap with a single, clean insight: the weights connected to large-activation channels matter much more than the others. Protect those channels' precision; let the others quantize aggressively. The result: INT4 perplexity within ~0.1 of FP16 on Llama-class models. AWQ remains, in 2026, one of the two algorithms that ships quantized LLMs at production scale (GPTQ being the other).

This lesson is the AWQ insight, the math, and the implementation sketch. The hands-on uses `autoawq` to quantize a real model.

## The insight

Recall the matmul `Y = A × W`. For a single output element `Y[c, b] = sum_k A[b, k] × W[c, k]`, a quantization error `ΔW[c, k]` in the weight matrix contributes `A[b, k] × ΔW[c, k]` to the error in `Y`. **The contribution scales linearly with the magnitude of the corresponding activation `A[b, k]`.** A 1% error in a weight whose paired activation is 100× the typical value contributes the same to the output as a 100% error in a weight whose paired activation is the typical value.

The "salient channels" in AWQ terminology are the columns `k` (input channels) of the weight matrix where the corresponding activations `A[:, k]` are large. These are the channels worth protecting. For LLMs, calibration consistently shows that ~1% of channels carry activations 10–100× larger than the median — exactly the outlier features from Lesson 3. AWQ uses the same observation as SmoothQuant, but flips it: instead of using it to enable W8A8, it uses it to enable W4A16.

## The trick

If we could keep the salient channels' weights in FP16 and the rest in INT4, we'd be done — mixed-precision INT4 with a few FP16 channels. That works in theory but is kernel-hostile: a single matmul that loads a mix of INT4 and FP16 weights, with the format depending on column index, would be horribly slow.

AWQ's trick: use *per-channel scaling* to make the integer quantization equivalent to keeping the salient channels in higher precision, while storing every weight as INT4.

The math: for any per-channel diagonal scale `s`, we have

```
Y = A × W = (A × diag(1/s)) × (diag(s) × W) = A' × W'
```

This is the same SmoothQuant identity from Lesson 3. Pick `s` so that `s_k` is large for salient channels (large activations) and ~1 for typical channels. Then:

- `W'_k = s_k × W_k`: the salient weight columns get scaled *up*, which spreads them across more of the INT4 range — effectively increasing their precision.
- `A'_k = A_k / s_k`: the salient activations get scaled *down*. (Activations stay in FP16, so no precision is lost here.)

The quantization error of `W'`, when dequantized and applied to `A'`, becomes:

```
ΔY = A' × ΔW' = (A / s) × s × ΔW = A × ΔW
```

Mathematically identical to the un-scaled error! No free lunch, right? Wrong — the key is that **`ΔW' < ΔW` for salient channels because they now occupy more of the INT4 range, so the quantization step is finer.** The error per weight goes down for the salient channels, which is exactly the channels that contribute most to `ΔY`.

The right way to think about it: scaling `W_k` up by `s_k` *before* quantization is equivalent to using `1/s_k` of an INT4 step size for those channels. Salient channels get a finer effective grid; non-salient channels get the coarser default grid. Same total bits; better allocation of resolution.

## Picking the scales

AWQ picks `s_k` to minimize the output error of the quantized layer. The proposed form:

```
s_k = (mean(|A[:, k]|))^α
```

where `α` is a per-layer hyperparameter searched over a small grid (typically 0.0 to 1.0). `α = 0` reduces to unscaled quantization (`s = 1`). `α = 1` aggressively protects salient channels. The empirical sweet spot is around 0.5–0.7, with the exact value chosen per-layer to minimize the reconstruction MSE.

This is a one-time, offline search. For each layer:

1. Compute `mean(|A[:, k]|)` per channel from calibration data (cheap).
2. For each candidate `α` in `{0.0, 0.1, 0.2, ..., 1.0}`:
   a. Compute `s = mean_act^α`.
   b. Apply `s` and `1/s` as the smoothing transformation.
   c. Quantize the rescaled weights to INT4 per-group (group=128, symmetric).
   d. Dequantize and measure `||original_W × A - dequant_W' × (A/s) × s||²` against calibration data.
3. Pick the `α` with lowest reconstruction error.

The full algorithm is layer-by-layer; the scales are baked into the saved INT4 weight tensor. At inference time, **the scaling is no longer visible** — the runtime kernel sees just the modified INT4 weights and the per-group scales. The activation scaling `A/s` is folded into the previous layer's output (because that previous layer's last operation is also a linear, and you can multiply its output weights by `1/s`). So AWQ doesn't change the inference kernel at all; it changes the offline weight conversion.

This is the elegance of AWQ: the entire algorithm is offline. Runtime is identical to vanilla INT4. The accuracy win comes from a one-time smarter scale picking.

## Why it works (the deeper "why")

Two observations:

**1. The salient channels are *consistent* across calibration data.** If channel `k` is large for one calibration sample, it's usually large for most. This is empirical — the "outlier features" of an LLM are a stable property of the trained weights, not a random artifact. Stability lets AWQ commit to a per-layer scale offline rather than computing it dynamically per input.

**2. The smoothing-vs-quantization tradeoff is layer-dependent.** Some layers (early attention) have very uneven activation distributions and benefit from aggressive `α ≈ 0.8`. Others (late FFN) are well-behaved and don't need much smoothing (`α ≈ 0.2`). Picking `α` per-layer captures this variation. This is also why AWQ runs the grid search separately for each layer.

The combination — stable salient channels + per-layer scale tuning — is the practical reason AWQ works as well as it does.

## How AWQ compares to alternatives

**vs naive INT4 (Lesson 4):** AWQ's whole point is that naive INT4 (per-group, symmetric, MinMax-style scales) loses 0.5–2 perplexity points. AWQ recovers most of it.

**vs GPTQ (Lesson 7):** GPTQ uses a different mechanism — Hessian-aware iterative weight adjustment — to achieve similar accuracy. AWQ is faster to calibrate (one forward pass + grid search vs GPTQ's per-layer iterative inversion); GPTQ sometimes wins by a hair on perplexity. In practice they trade off:
- AWQ: 5–15 minutes to quantize Llama-7B on a 4090.
- GPTQ: 30–90 minutes for the same.
- Both end up within 0.1–0.2 perplexity of each other and 0.2 of FP16.

Most production deployments support both formats; the choice is mostly determined by what the runtime supports natively.

**vs SmoothQuant:** Same SmoothQuant identity at the core. SmoothQuant uses it to enable W8A8 (activation quantization to INT8); AWQ uses it to enable W4A16 (aggressive weight quantization while keeping activations FP16). They could be combined (W4A8 with SmoothQuant + AWQ), and some recipes do this.

**vs HQQ, OmniQuant, QuIP:** newer algorithms that compete with AWQ/GPTQ. HQQ is calibration-free (faster) but slightly worse perplexity. QuIP uses Hadamard rotations to redistribute outliers (interesting theoretical angle, less common in production). OmniQuant learns the scales via gradient descent (best perplexity but slowest). In 2026, AWQ and GPTQ remain the production defaults; the others are research-grade or niche.

## What you should believe after this lesson

Three sentences:

**1. AWQ's insight is that errors on weights paired with high-magnitude activations dominate the output error; you should give those weights more INT4 resolution.** Per-channel pre-scaling does this without changing the runtime format or kernel.

**2. The scale search is offline, per-layer, and surprisingly cheap** (one forward pass through calibration data plus a small grid search over a single hyperparameter `α`), making AWQ one of the fastest practical INT4 algorithms to run.

**3. AWQ and GPTQ are interchangeable enough in accuracy that production choice usually comes down to runtime support; pick AWQ when calibration time matters, GPTQ when the last 0.1 perplexity matters.** Both are dramatically better than naive INT4.

## Hands-on (at home)

Quantize a real model with AWQ using the `autoawq` library. Requires an NVIDIA GPU (or skip to the `llama.cpp` analog in Lesson 4 if you're on Apple).

```python
# awq_quantize.py
# pip install autoawq
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

model_id = "Qwen/Qwen2.5-1.5B"
quant_path = "Qwen2.5-1.5B-AWQ"
quant_config = {
    "zero_point": True,
    "q_group_size": 128,
    "w_bit": 4,
    "version": "GEMM",
}

# Load FP16 model.
model = AutoAWQForCausalLM.from_pretrained(model_id, device_map="cuda", torch_dtype="float16")
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Quantize (this runs the AWQ calibration + grid search internally).
# Calibration data: by default a mix from MIT/AcademicMatchin Lambda, ~128 samples.
model.quantize(tokenizer, quant_config=quant_config)

# Save.
model.save_quantized(quant_path)
tokenizer.save_pretrained(quant_path)
print(f"Saved to {quant_path}")
```

This will run for 5–10 minutes on a 4090 and produce a directory containing the quantized weights plus the AWQ-format metadata.

Quick perplexity check against the FP16 baseline:

```python
# awq_perplexity.py
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from awq import AutoAWQForCausalLM

def perplexity(model, tokenizer, n_samples=64, max_len=512, device="cuda"):
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    nlls = []
    for i in range(n_samples):
        text = ds[i]["text"]
        if not text.strip(): continue
        ids = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_len).input_ids.to(device)
        if ids.shape[1] < 2: continue
        with torch.no_grad():
            out = model(ids, labels=ids)
        nlls.append(out.loss.item() * ids.shape[1])
    return torch.tensor(nlls).sum().exp().item() / sum(ids.shape[1] for _ in nlls)

tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B")
fp16 = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B", torch_dtype=torch.float16, device_map="cuda")
awq = AutoAWQForCausalLM.from_quantized("Qwen2.5-1.5B-AWQ", device_map="cuda")

print(f"FP16 ppl: {perplexity(fp16, tok):.3f}")
print(f"AWQ INT4 ppl: {perplexity(awq, tok):.3f}")
```

You should see the AWQ INT4 perplexity within 0.1–0.3 of the FP16 — much closer than what naive INT4 would give you (which is typically 0.5–2 worse). The model also occupies 4× less memory.

For a richer comparison, also try naive INT4 (the `quantize_int4_per_group` from Lesson 4's hands-on, hooked into the model's linear layers) and see how much worse the perplexity is without the activation-aware step.

## Further reading

- "AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration" (Lin et al, 2023) — the original paper.
- `autoawq` GitHub README — the production library, with the kernel implementations.
- "SmoothQuant" paper (already cited in Lesson 3) — for the underlying SmoothQuant identity that AWQ rediscovers and uses differently.
- TensorRT-LLM and vLLM AWQ integration docs — to see what the production serving stacks expect.

Next lesson: **GPTQ — Error-Correcting Quantization.** The other workhorse INT4 algorithm. Where AWQ pre-scales the weights to align with activation magnitudes, GPTQ takes a different angle entirely: quantize one weight at a time, and propagate the rounding error into the remaining weights so the column's overall reconstruction stays accurate.
