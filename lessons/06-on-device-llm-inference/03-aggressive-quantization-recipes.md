---
title: "Lesson 3 — Aggressive Quantization Recipes"
date: "2026-06-04"
module: "on-device-llm-inference"
order: 3
tags: ["quantization", "gguf", "awq", "gptq", "mixed-precision", "per-layer"]
author: "Sudipta Pathak"
prerequisites: ["02-picking-the-model"]
---

# Lesson 3 — Aggressive Quantization Recipes

## Why this lesson exists

Module 3 covered quantization at the math/recipe level. Module 4 Lesson 10 covered the Apple-runtime quantization options. This lesson is the operational view: given a target deployment, which quantization format do you actually pick, and how do you decide when stepping up precision is worth the bytes?

The premise: the model from Lesson 2 fits at INT4 in your device's memory budget. Q4_K_M / AWQ / GPTQ are the headline options. They differ by ~0.1–0.3 perplexity points (Module 3 Lesson 7). On a phone, what often matters more than the format choice is the *per-layer precision strategy* — which layers stay at higher precision because they're sensitive, and which can drop to INT3 or INT2 because they're not.

The lesson is reading. The Hands-on compares per-layer precision strategies on a real SLM and measures the resulting quality.

## The format menu, recapped

From Module 3 Lessons 4 and 7, the formats that matter on-device:

- **Q4_0** (llama.cpp): the simplest 4-bit format, per-block-32 symmetric, FP16 scale. Effective bits-per-weight ≈ 4.5. Fast, simple, sometimes ~0.5 perplexity worse than Q4_K_M.
- **Q4_K_M** (llama.cpp): the modern 4-bit standard. Per-superblock-256 with sub-block scales; most weights at 4-bit; embedding and output layers at Q6_K (6-bit). Effective bits ≈ 4.5–5.0. The default for most llama.cpp deployments in 2026.
- **Q4_K_S** (small) and **Q4_K_L** (large): variants of Q4_K_M that use Q4 for more vs fewer layers. Q4_K_S is more aggressive (smaller); Q4_K_L is conservative.
- **Q5_K_M**: 5-bit per-superblock. ~25% larger than Q4_K_M; usually ~0.1 perplexity better. The "if Q4 isn't quite good enough" step-up.
- **Q3_K_M / Q3_K_S**: 3-bit. ~25% smaller than Q4_K_M; usually ~0.5 perplexity worse. The "if Q4 doesn't fit" step-down. Rarely accurate enough for production but useful in extreme constraints.
- **Q2_K**: 2-bit. ~50% smaller than Q4_K_M; ~1–3 perplexity worse. Borderline-usable for some applications.
- **AWQ / GPTQ** (4-bit, group=128): Module 3 Lessons 6, 7. Comparable accuracy to Q4_K_M; slightly different memory layout.
- **MLX-native 4-bit / 8-bit / 3-bit / 2-bit**: same per-group scheme; MLX-native serialization.

For most deployments, **Q4_K_M is the default and AWQ is the closest alternative when you have AWQ-compatible runtimes.** Stepping up to Q5_K_M is the first lever when quality is borderline. Stepping down to Q3_K_M is the lever when memory doesn't fit.

## Perplexity vs size vs latency

Approximate numbers for Llama 3.2 3B on a Mac, measured against the FP16 baseline:

| Format | Size | Perplexity Δ | Tokens/sec (M3 Pro) |
| ------ | ---- | ------------ | ------------------- |
| FP16 | 6.4 GB | (baseline) | 18 |
| Q8_0 | 3.4 GB | +0.02 | 28 |
| Q5_K_M | 2.1 GB | +0.08 | 35 |
| Q4_K_M | 1.8 GB | +0.15 | 40 |
| Q4_0 | 1.6 GB | +0.30 | 42 |
| Q3_K_M | 1.4 GB | +0.50 | 45 |
| Q2_K | 1.1 GB | +1.50 | 50 |
| AWQ-INT4 | 1.8 GB | +0.10 | 40 |
| GPTQ-INT4 | 1.8 GB | +0.15 | 40 |

Patterns:
- Each step down in precision saves some memory and gains some speed; the tradeoff is perplexity.
- Q4_K_M is the knee of the curve. Q5_K_M is the safe "I want slightly better" step-up; Q3_K_M is the "I really need to fit" step-down.
- Q2_K is usually unacceptable; reserve for emergencies.
- The throughput improvements are smaller than you might expect because the kernel-launch overhead and KV cache reads also matter. Going from Q4_K_M to Q3_K_M doesn't give a 33% speedup; the speedup is closer to 10% because the weights are only one part of the bandwidth.

## Per-layer precision stepping

A significant lever that's underused: not every layer of a transformer needs the same precision. Empirically, three layer types are more sensitive than others:

**1. Embedding layer.** The token embedding table is the model's first lookup; errors here propagate through every subsequent layer. Most quant formats (including Q4_K_M) keep this at Q6 or higher.

**2. LM head (output projection).** The final projection back to vocabulary logits. Errors here directly affect the sampled token distribution. Q6 or FP16 in most modern formats.

**3. Attention layers (Q, K, V, O projections).** Some sensitivity, especially in the first and last few transformer blocks. Q4_K_M keeps these in Q4 but uses larger groups; Q5_K_M is sometimes worth the bytes.

What's least sensitive:

**4. FFN layers** (gate, up, down projections). The bulk of the parameter count and the least sensitive to precision. Q3 or even Q2 in FFN layers often costs less perplexity than the same drop in attention.

A custom recipe that exploits this: Q5 attention + Q3 FFN. Smaller than Q4_K_M (because FFN dominates parameter count), comparable perplexity (because attention stays higher), often faster (because the smaller FFN reduces bandwidth).

llama.cpp's "imatrix" feature lets you compute an importance matrix from calibration data and use it to do principled per-layer quantization. The MLX equivalent is in `mlx_lm.convert` with per-layer overrides.

## Mixed precision in practice

The most common mixed-precision recipe in 2026:

- Embedding: Q6_K or FP16.
- LM head: Q6_K or FP16.
- Attention Q/K/V/O: Q5_K (group=64).
- Attention norm and post-attention norm: FP16.
- FFN gate/up/down: Q4_K (group=128) or Q3_K (group=64) for aggressive deployments.

This is roughly what Q4_K_M does. The variations (Q4_K_S, Q4_K_L) tune which layers get the "step-up" treatment.

For custom recipes:

```bash
# llama.cpp: use --imatrix and the per-layer override syntax.
./quantize \
    --imatrix imatrix.bin \
    models/Llama-3.2-3B-Instruct-F16.gguf \
    models/Llama-3.2-3B-Instruct-custom.gguf \
    Q4_K_M
```

For MLX:

```python
from mlx_lm.utils import quantize_model_with_layer_overrides

quant_config = {
    "default": {"bits": 4, "group_size": 64},
    "model.embed_tokens": {"bits": 6, "group_size": 64},
    "lm_head": {"bits": 6, "group_size": 64},
    # ... per-layer overrides
}
```

(API surface may shift; check current docs.)

## The right format for the right runtime

A reminder from Module 5 Lesson 10: the format must be supported by your target runtime. The combinations:

- llama.cpp → GGUF Q*_K_M variants.
- MLX / mlx-lm → MLX-native 4-bit / 8-bit; GGUF read support.
- ExecuTorch → its own quant via `torchao`.
- ORT → ONNX with QDQ patterns.
- Core ML → Core ML's PTQ via `coremltools.optimize`.
- vLLM / TensorRT-LLM (server-side) → AWQ, GPTQ, FP8.

Cross-runtime portability is limited. The format choice ties you to the runtime. Pick the format that matches your deployment plan; don't quantize to AWQ if you're shipping via llama.cpp (you'd just be re-converting).

## Beyond INT4: experimental territory

A few formats worth knowing about, mostly research-grade in 2026:

**MXFP4 (Microscaling FP4)**: shared-exponent FP4 from the OCP MX spec. Hardware-accelerated on Blackwell; not yet available on most consumer chips. Promising for desktop GPUs.

**BitNet 1.58 (ternary: -1, 0, +1)**: requires training from scratch with the format baked in. Some 7B-class models have been trained; the deployment story is improving but not yet mainstream.

**HQQ**: calibration-free 4-bit / 3-bit quantization. Less accurate than GPTQ but quicker to apply. Useful when you can't run calibration.

**QuIP / QuIP#**: Hadamard-rotation-based quantization, supports very low bit-widths (2–3 bit) with better accuracy than naive approaches. Niche but worth tracking.

For mainstream deployment in 2026: Q4_K_M, AWQ, GPTQ are the safe choices. The experimental formats are for research or specific niches.

## What you should believe after this lesson

Three sentences:

**1. Q4_K_M is the knee of the precision/quality curve for on-device LLMs in 2026** — Q5_K_M is the safe step-up when quality is borderline, Q3_K_M the step-down when memory doesn't fit. Q2_K is usually unacceptable; reserve for emergencies.

**2. Per-layer precision stepping is the underused lever** — embedding, LM head, attention norms benefit from higher precision; FFN layers tolerate aggressive drops. A "Q5 attention + Q3 FFN" custom recipe often outperforms a uniform Q4 at similar total size.

**3. The format choice ties you to the runtime**: pick the format your deployment runtime supports natively (Q4_K_M for llama.cpp, MLX-native for mlx-lm, AWQ for vLLM). Cross-runtime quantization portability is poor; design the pipeline end-to-end.

## Hands-on (at home)

Compare three precision levels of the same model and measure the quality/throughput tradeoff.

```bash
# Use pre-quantized variants from HuggingFace.
huggingface-cli download bartowski/Qwen2.5-1.5B-Instruct-GGUF \
    Qwen2.5-1.5B-Instruct-Q3_K_M.gguf --local-dir gguf
huggingface-cli download bartowski/Qwen2.5-1.5B-Instruct-GGUF \
    Qwen2.5-1.5B-Instruct-Q4_K_M.gguf --local-dir gguf
huggingface-cli download bartowski/Qwen2.5-1.5B-Instruct-GGUF \
    Qwen2.5-1.5B-Instruct-Q5_K_M.gguf --local-dir gguf

# Benchmark each.
for f in gguf/Qwen2.5-1.5B-Instruct-*.gguf; do
    echo "=== $f ==="
    ls -la "$f" | awk '{print "size:", $5}'
    ./llama-bench -m "$f" -p 256 -n 128
done

# Quality check: same prompts on each, manually compare outputs.
for f in gguf/Qwen2.5-1.5B-Instruct-*.gguf; do
    echo "=== $f ==="
    ./llama-cli -m "$f" -p "Compute 23 * 47 step by step." -n 100 --temp 0
    echo ""
done
```

Read the outputs. The Q3 variant may get the arithmetic wrong (or partially right) where Q4 and Q5 succeed. The throughput numbers should rank Q3 fastest, Q5 slowest; the gap is moderate (~10–20%).

For a more rigorous quality test, run a small batch of GSM8K problems through each:

```bash
pip install lm-eval
lm-eval --model gguf --model_args path=gguf/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf \
    --tasks gsm8k --num_fewshot 5 --batch_size 1
```

Compare GSM8K scores; you'll likely see a meaningful drop at Q3 and similar scores at Q4 and Q5.

## Further reading

- llama.cpp's GGUF format documentation and the `quantize` tool's help.
- MLX `mlx_lm.convert` documentation for the per-layer override API.
- "The Curious Case of Hallucinations in Neural Machine Translation" — older but illustrative of per-layer sensitivity in transformers.
- HuggingFace's "Quantization in 4 bits" guides for the various 4-bit recipes.
- Module 3 Lessons 4, 6, 7 — the canonical references for the formats and algorithms.

Next lesson: **Mixed precision and per-channel quantization.** We continue the precision story by going deeper on the per-channel and activation-aware techniques that recover accuracy at aggressive quantization levels — the SmoothQuant family applied in the on-device context.
