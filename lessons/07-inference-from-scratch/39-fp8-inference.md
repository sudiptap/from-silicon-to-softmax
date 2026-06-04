---
title: "Lesson 39 — FP8 Inference on Hopper/Blackwell"
date: "2026-06-04"
module: "inference-from-scratch"
order: 39
tags: ["fp8", "hopper", "blackwell", "tensor-cores", "e4m3", "e5m2"]
author: "Sudipta Pathak"
prerequisites: ["38-smoothquant"]
---

# Lesson 39 — FP8 Inference on Hopper/Blackwell

## Why this lesson exists

FP8 is the floating-point cousin of INT8 — same 8 bits, but with an exponent field that gives much better dynamic range. NVIDIA Hopper (H100) introduced native FP8 tensor cores running at 2× FP16 throughput. Blackwell extends this to FP4. For server-side inference, FP8 is the dominant low-precision format in 2026 deployments.

This lesson covers FP8 formats from the inference engine view, the comparison to INT8, and the production deployment story.

The lesson is reading. The Hands-on benchmarks FP8 vs FP16 on a Hopper-class GPU.

## FP8 formats: E4M3 and E5M2

Recap from Module 3 Lesson 2:

**E4M3** (4 exponent bits, 3 mantissa bits):
- Max value: ±448.
- Min normal: ~1.9e-3.
- Higher precision (more mantissa), lower dynamic range.
- Used for activations and weights.

**E5M2** (5 exponent bits, 2 mantissa bits):
- Max value: ±57344.
- Min normal: ~6e-5.
- Lower precision, FP16-like dynamic range.
- Used for gradients during training.

For inference, E4M3 dominates. E5M2 mostly matters for training.

## The native hardware path

On Hopper, the tensor core's FP8 instructions accept FP8 inputs and produce FP32 accumulation:

```
FP8 × FP8 → FP32 accumulator → cast to FP16 output
```

Throughput: 2× FP16 (which is itself 2× FP32 on Hopper). So FP8 is 4× FP32, 2× FP16.

The H100's spec sheet: 1979 TFLOPs FP8, 989 TFLOPs FP16, 67 TFLOPs FP32. The FP8 path is the headline number.

On Blackwell, FP4 doubles this again: 4× FP16 throughput. FP4 with MXFP4 (Module 3 Lesson 2) is the next-generation high-throughput inference format.

## When FP8 wins over INT8

**FP8 advantages:**
- Better dynamic range handles outliers better than INT8. The activation-outlier problem (Module 3 Lesson 3) is much smaller; SmoothQuant is helpful but not always needed.
- Cleaner gradients (matters for training; less for inference).
- Native Hopper / Blackwell hardware support.
- Reduces the engineering complexity around per-channel scales (FP8's exponent field handles the per-channel range automatically).

**INT8 advantages:**
- Works on every accelerator from 2018 onward (Ampere, A100, consumer cards, mobile NPUs). FP8 is Hopper+ only.
- Slightly faster than FP8 on some non-Hopper hardware that has INT8 but no FP8.
- More mature tooling.

The 2026 production split:
- **Hopper / Blackwell GPUs**: FP8 is the throughput-maximizing choice. TensorRT-LLM and vLLM both default to FP8 on H100.
- **Ampere / consumer GPUs**: INT8 is the comparable path. Less throughput uplift than Hopper's FP8.
- **Other accelerators (TPU, AMD, mobile)**: typically INT8 or whatever the chip natively supports.

## The deployment recipe

For FP8 inference on Hopper:

1. **Convert weights**: cast FP16 → FP8 E4M3 per-tensor or per-channel.
2. **Activation quantization**: per-token or per-tensor E4M3 with dynamic range tracking.
3. **Matmul**: FP8 × FP8 with FP32 accumulator on the tensor cores.
4. **Downcast**: FP32 → FP16 for the next layer's input.
5. **Specific layers stay higher precision**: layer norms, softmax, the LM head. These benefit from FP16 / BF16 precision.

The conversion is much simpler than INT8 calibration because the FP8 format absorbs the dynamic range issues. A typical FP8 conversion takes minutes; INT8 calibration with SmoothQuant takes longer.

## TensorRT-LLM's FP8 path

TensorRT-LLM has the most mature FP8 inference:

```bash
trtllm-build --checkpoint_dir model --output_dir engine --use_fp8 --max_input_len 2048
```

The TensorRT-LLM engine compilation:
- Quantizes weights to FP8.
- Configures activation FP8 quantization with calibration.
- Generates kernels that use FP8 tensor cores.

Throughput on H100 for Llama 3.1 8B: ~600 tokens/s at batch 1, ~30K tokens/s at batch 64. FP16 baseline: ~300 tokens/s at batch 1, ~15K at batch 64.

vLLM has similar FP8 support; the engineering is converging.

## The KV cache angle

FP8 KV cache is a separate question (Lesson 20). FP8 KV is straightforward on Hopper because the FP8 format has native tensor-core support; storing K and V as FP8 doubles cache capacity and has clean kernel paths.

For maximum throughput: FP8 weights + FP8 KV cache + FP8 attention is the full Hopper stack. ~3-4× faster than FP16 baseline.

## Quality

FP8 quality is excellent — typically within 0.05 perplexity of FP16. Better than INT8 + SmoothQuant on most models because the format's dynamic range handles outliers natively.

For the highest-quality FP8 deployment, per-channel scales (instead of per-tensor) recover the last few hundredths of perplexity. TensorRT-LLM defaults to per-channel.

## What you should believe after this lesson

Three sentences:

**1. FP8 is the throughput-maximizing inference format on Hopper / Blackwell** — 2× FP16 throughput via native tensor cores. The E4M3 format handles activation outliers naturally; less engineering than INT8 calibration.

**2. FP8 wins over INT8 on hardware that supports it natively**; INT8 remains relevant on Ampere / consumer / mobile. The 2026 pattern: FP8 on data-center Hopper+, INT8 elsewhere.

**3. TensorRT-LLM and vLLM both support FP8 on H100** with mature kernels; the deployment is one flag. The full stack — FP8 weights, FP8 KV cache, FP8 attention — gives ~3-4× speedup over FP16 baseline.

## Hands-on (at home)

If you have access to an H100 / H200, benchmark FP8 vs FP16:

```bash
# Install TensorRT-LLM (Hopper required).
# Follow NVIDIA's installation guide.

# Build an FP8 engine.
trtllm-build --checkpoint_dir /path/to/llama-3.1-8b \
    --output_dir engine_fp8 \
    --use_fp8 \
    --max_batch_size 64

# Build an FP16 engine for comparison.
trtllm-build --checkpoint_dir /path/to/llama-3.1-8b \
    --output_dir engine_fp16 \
    --max_batch_size 64

# Benchmark.
trtllm-bench --engine_dir engine_fp8 ...
trtllm-bench --engine_dir engine_fp16 ...
```

You should see ~1.8-2× throughput from FP8.

Alternatively, with vLLM on Hopper:

```bash
vllm serve neuralmagic/Llama-3-8B-Instruct-FP8 --quantization fp8
```

(NeuralMagic publishes pre-quantized FP8 versions of popular models.)

On non-Hopper hardware, the FP8 path isn't accelerated natively; you'd see FP8 emulated at FP16 speed (no win) or fall back to INT8.

## Further reading

- "FP8 Formats for Deep Learning" (Micikevicius et al, 2022) — the canonical paper.
- NVIDIA TensorRT-LLM FP8 documentation.
- vLLM FP8 quantization documentation.
- "Hopper Architecture Whitepaper" — the hardware-side reference.
- Module 3 Lesson 2 — earlier coverage of the precision landscape.

Next lesson: **1-bit territory — BitNet b1.58 and the extreme low end.** The frontier of quantization: 1.58-bit ternary models. We close Part 6 with this aspirational direction and what it would mean for inference hardware.
