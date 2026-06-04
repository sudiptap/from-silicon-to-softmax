---
title: "Lesson 5 — Calibration Data and Quantization-Aware Training Basics"
date: "2026-06-04"
module: "ml-internals"
order: 5
tags: ["calibration", "ptq", "qat", "minmax", "percentile", "mse"]
author: "Sudipta Pathak"
prerequisites: ["04-int4-quantization"]
---

# Lesson 5 — Calibration Data and Quantization-Aware Training Basics

## Why this lesson exists

The previous two lessons established the formats: INT8 and INT4 with per-channel or per-group scales. They left one question unanswered: **how do you pick the scale?** "max(|x|) divided by 127" is the obvious answer and the worst one. Real recipes use calibration: run the model on a representative dataset, observe the actual value distributions, and choose scales that minimize error against what the model is actually doing — not against worst-case theoretical bounds.

This lesson is about that calibration step. Three scale-selection criteria — MinMax, percentile, MSE — and when each is right. We also touch quantization-aware training (QAT), the heavier option that simulates quantization during a brief fine-tune so the model can adapt its weights to the lower precision. QAT is rarely the right choice for LLMs in 2026 (post-training methods like AWQ and GPTQ are usually sufficient), but for some edge deployments — vision models on mobile, sub-INT4 LLMs — it's still load-bearing.

The lesson is reading. The Hands-on builds a small calibration loop and compares MinMax vs percentile vs MSE scales on a real activation distribution.

## What calibration is

You start with a trained FP16 model. You want to quantize it to INT8 or INT4. For each tensor you want to quantize (weights, activations, sometimes KV cache), you need to choose a scale `s`. Weights are simple: you have the static values, so the scale comes from the weight tensor directly. Activations are harder: they depend on the input, so you have to *sample* the activation distribution.

The calibration loop:

1. Pick a calibration dataset — usually 100–1000 short text samples, ideally drawn from the model's expected deployment distribution.
2. Run the model forward on the calibration data, capturing activations at each quantization point.
3. For each capture point, compute a scale based on the observed activation distribution.
4. Quantize the model using those scales (weights too, but they don't need calibration).
5. Validate: re-run on a held-out set, measure perplexity / task accuracy / other metric, confirm the quantized model is close enough to the FP16 baseline.

Calibration is a one-time, offline step. It costs about as much as a single training epoch (cheap, since most of the model's forward time is the matmul which is fast at FP16). The output is a small per-tensor or per-channel metadata file.

## How many samples do you actually need

The empirical answer: surprisingly few. For LLMs:

- 128 samples of ~512 tokens each is usually enough for INT8 weight calibration.
- 512–1024 samples for INT4 weight calibration with AWQ/GPTQ.
- More samples don't measurably help past these counts.

For vision models (per-class activation distributions): 1024–2048 samples covering the class diversity.

The *content* of the calibration data matters more than the count. If you calibrate on English-only Wikipedia and deploy on a multilingual model, the calibration may underestimate the activation range for non-English tokens. The standard practice for general-purpose LLMs is to use a mix from C4 / Pile / WikiText — broad enough to cover the model's pretraining distribution. For task-specific deployments (e.g., a code model deployed for Python), use task-relevant calibration data.

## Three scale-selection criteria

Given an observed distribution of values `x_1, ..., x_n` at a calibration point, how do you pick `s`?

### MinMax

The naive choice: `s = max(|x|) / qmax`. Maps the absolute maximum value to the integer limit. Pros: simple, no parameters. Cons: a single outlier in calibration data fixes `s` based on that outlier, crushing the rest of the distribution. This is the failure mode that LLM.int8 / SmoothQuant address — it's particularly bad for LLM activations.

When to use: weights (where the distribution is static and well-controlled), simple/small models without outliers.

### Percentile

`s = percentile(|x|, p) / qmax`, with `p = 99.9` or `99.99` typical. Clips the top 0.1% of values to the integer limit (they get saturated to `qmax`). Pros: handles outliers gracefully — a few extreme values don't dominate the scale. Cons: the clipped values are now wrong by up to their magnitude; if the model relies on those exact values, accuracy suffers.

When to use: activations on LLMs that have known outlier patterns; basically the default for activation quantization in modern recipes.

### MSE (mean squared error minimization)

Search over candidate scales `s` and pick the one that minimizes `||x - dequantize(quantize(x, s))||²`. A direct optimization of the reconstruction error. Pros: theoretically optimal for L2-norm reconstruction. Cons: more compute (a grid search or golden-section search per tensor), and L2 reconstruction error isn't always what you want — sometimes the model is more sensitive to a few critical channels than to the bulk distribution.

When to use: when calibration time isn't a constraint and you want the best naive scale. Common in production pipelines as a default.

### Variants you'll see in practice

- **KL divergence calibration** (NVIDIA TensorRT's default for INT8 activations): minimize KL(distribution_FP16, distribution_quantized). More expensive than MSE; sometimes better.
- **Asymmetric percentile**: separate percentiles for the positive and negative tails. Useful when distributions are heavy on one side.
- **Per-token / per-sequence dynamic activation quantization** (used in some LLM int8 deployments): compute `s` at runtime from the current input, not from offline calibration. Slightly slower (extra reduction) but zero offline cost; useful for adaptive workloads.

The empirical hierarchy on LLMs: dynamic per-token > MSE > percentile (99.9) > MinMax. The accuracy delta between MSE and percentile is small (~0.05 perplexity); the delta between MSE and MinMax is large (~0.5–2 perplexity), especially at INT4.

## When PTQ isn't enough: QAT

PTQ (Post-Training Quantization) — what we've been discussing — works without modifying the weights. You compute scales from calibration data and quantize. The model never sees the quantization during training, so the weights aren't adapted to the lower precision.

QAT (Quantization-Aware Training) inserts simulated quantization into the forward pass during training:

```
forward:  x → fake_quant(x) → matmul → fake_quant(out) → ...
backward: gradients flow through the fake_quant as if it were the identity (the
          "straight-through estimator")
```

The model trains with the quantization error in the loop; the weights drift to absorb it. After training, you collapse the fake-quant into real quantization and the model retains its accuracy.

For LLMs, QAT has historically been expensive (a full pretraining or near-pretraining cost) and the win over PTQ + AWQ/GPTQ was small. So QAT was rare. In 2025–2026 the situation is shifting:

- **QAT for sub-INT4 (e.g., 2-bit) LLMs** is a real frontier; some papers (BitNet, OmniQuant variants) train from scratch with the quantization baked in.
- **QAT for small models** (sub-1B parameters) on edge devices is more common; the training cost is bearable.
- **QLoRA**, the most famous quantization-related training method, is *not* QAT — it's PTQ to NF4 of a frozen base model, with a LoRA adapter trained in FP16 on top. The base model isn't QAT-trained; the adapter compensates for the quantization error implicitly.

The practical 2026 recipe: PTQ + AWQ or GPTQ for LLMs >1B. QAT for sub-1B vision/audio models on phones. QLoRA for fine-tuning quantized LLMs.

## A complete PTQ recipe

For reference, what a real INT4 PTQ recipe looks like end-to-end:

1. **Pick the format**: GGUF Q4_K_M (for llama.cpp), AWQ-4bit (for vLLM/transformers), or GPTQ-4bit.
2. **Pick the algorithm**: AWQ (faster calibration) or GPTQ (slightly better perplexity, slower).
3. **Pick the calibration data**: 512 samples of 512 tokens from C4 or your task distribution.
4. **Run the calibration**: a single forward pass through the calibration data, capturing activation distributions; algorithm-specific scale computation.
5. **Quantize weights using the computed scales**: AWQ's per-channel scaling factor + per-group scales; GPTQ's Hessian-corrected per-group scales.
6. **Save in the chosen format**.
7. **Validate**: re-run on WikiText perplexity, MMLU, or your task-specific eval. Compare to the FP16 baseline.
8. **Iterate if needed**: if perplexity is too high, increase calibration data, try the other algorithm, or relax to a less aggressive format (Q5_K_M, INT5, INT6).

Steps 1–6 take 10–60 minutes on a 4090 for a 7B model. Step 7 takes another 10–30 minutes for a quick perplexity check or several hours for a full eval suite.

## What you should believe after this lesson

Three sentences:

**1. Calibration is the bridge between the format and the deployed model — picking scales from a representative data sample instead of from theoretical worst case.** 100–1000 samples is enough; the *content* matters more than the *count*.

**2. The hierarchy of scale-selection criteria is dynamic-per-token > MSE > percentile > MinMax, with the gap between MSE and MinMax being large (especially at INT4) and the gap between MSE and percentile being small.** Default to MSE or percentile for activations.

**3. PTQ + AWQ/GPTQ is the default for LLMs in 2026; QAT is reserved for sub-INT4 frontiers and small edge models.** QLoRA is post-hoc adapter training on top of a quantized base — close cousin but not the same as QAT.

## Hands-on (at home)

Compare the three scale-selection criteria on real activations from a small model.

```python
# calibration.py
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.manual_seed(0)
model_id = "Qwen/Qwen2.5-0.5B"
tok = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32).eval()

# A few calibration prompts.
prompts = [
    "The quick brown fox jumps over the lazy dog.",
    "In machine learning, the training loop iterates over batches",
    "Recursion is a useful tool when the structure of the problem mirrors itself",
    "The bandwidth-compute crossover happens when the matrix is large",
    "Quantization shrinks the bytes loaded per token",
    # ... add more for a real calibration set
]
all_acts = []
def hook(_, _i, out):
    # Capture the up-projection input on layer 5 — a typical activation tensor.
    all_acts.append(out.detach().float().flatten())

# Find the layer to hook.
for n, m in model.named_modules():
    if "layers.5.mlp.up_proj" in n:
        m.register_forward_hook(hook)
        break

with torch.no_grad():
    for p in prompts:
        inputs = tok(p, return_tensors="pt")
        model(**inputs)

x = torch.cat(all_acts)  # all activations from layer 5 up_proj input
print(f"calibration tensor: {x.shape}, range [{x.min():.3f}, {x.max():.3f}]")

# Three scale-selection criteria for symmetric INT8 (-127 to 127).
def quantize_dequantize(x, s):
    return (x / s).round().clamp(-127, 127) * s

def minmax_scale(x):
    return x.abs().max() / 127.0

def percentile_scale(x, p=99.9):
    return x.abs().quantile(p / 100) / 127.0

def mse_scale(x, n_candidates=100):
    s_init = x.abs().max() / 127.0
    candidates = torch.linspace(s_init * 0.3, s_init * 1.1, n_candidates)
    best_s, best_err = s_init, float('inf')
    for s in candidates:
        err = ((x - quantize_dequantize(x, s)) ** 2).mean().item()
        if err < best_err:
            best_err, best_s = err, s.item()
    return best_s

for name, s_fn in [("MinMax", minmax_scale), ("99.9th percentile", lambda x: percentile_scale(x, 99.9)),
                    ("MSE", mse_scale)]:
    s = s_fn(x)
    x_recon = quantize_dequantize(x, s)
    err = (x - x_recon).abs().mean().item()
    print(f"{name:20s}: s={s:.4e}  mean abs err={err:.4e}")
```

You will typically see MinMax give the worst reconstruction (the max value pushes the scale up and the bulk gets crushed), percentile give a much smaller error, and MSE come in slightly better than percentile. On activation tensors with outliers, the MinMax vs percentile gap is often 5–10×.

Part 2 (optional, requires a GPU) — apply each scaling strategy to the full model and measure perplexity. Use `lm-eval-harness` or a quick WikiText perplexity script. You'll see the same hierarchy reflected in the perplexity numbers.

## Further reading

- "Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference" (Jacob et al, 2017) — the foundational PTQ paper.
- "Quantizing Deep Convolutional Networks for Efficient Inference: A Whitepaper" (Krishnamoorthi, 2018) — the encyclopedia of pre-LLM PTQ and QAT.
- NVIDIA TensorRT calibration docs — the production view, with KL-divergence calibration as the default.
- "OmniQuant: Omnidirectionally Calibrated Quantization for Large Language Models" (Shao et al, 2023) — a hybrid PTQ/QAT-lite that learns the scales via gradient descent.

Next lesson: **AWQ — Activation-Aware Weight Quantization.** The first of the two algorithms that turn naive INT4 (which doesn't quite work) into deployment-ready INT4 (which does). The insight: protect the channels of the weight matrix that correspond to large activations, because errors there propagate the most.
