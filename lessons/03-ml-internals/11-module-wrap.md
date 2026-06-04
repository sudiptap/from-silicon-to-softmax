---
title: "Lesson 11 — Module Wrap: Picking Your Compression Budget"
date: "2026-06-04"
module: "ml-internals"
order: 11
tags: ["quantization", "wrap", "summary", "decision-tree", "budget"]
author: "Sudipta Pathak"
prerequisites: ["10-rust-gpu-frontier"]
---

# Lesson 11 — Module Wrap: Picking Your Compression Budget

## Why this lesson exists

Eleven lessons of formats, algorithms, and tradeoffs collapse into a single practical workflow: given a target device, what compression budget do you spend, in what order, with what validation? This lesson is that workflow. It also collects the mental models from the module — the things to carry into Module 4 (MLX & Apple Silicon Internals) and beyond — and is the last reading lesson of the module.

The lesson is reading. The Hands-on at the bottom is a final end-to-end script: take a real LLM (TinyLlama or Phi-3 mini), quantize it, validate, benchmark on your machine.

## The journey, replayed

We started by counting the FLOPs and bytes a transformer actually moves (Lesson 1). The key reveal: on-device LLM inference is memory-bound, not compute-bound, and quantization's win is in bytes loaded — not in FLOPs computed. Every subsequent lesson built on that.

We toured the precision landscape (Lesson 2) — the choice between FP16 and BF16 is dynamic-range vs precision; FP8's E4M3/E5M2 split serves activations and gradients respectively; MXFP shared-exponent formats sit on the path to FP4 and below.

INT8 (Lesson 3) and INT4 (Lesson 4) gave us the integer formats: the `scale × q` mental model, the granularity ladder (per-tensor → per-channel → per-group), the dequant-fused matmul pattern, and the file formats (GGUF, AWQ, GPTQ, NF4) that you encounter in the wild.

Calibration (Lesson 5) is how you pick scales that actually work — MSE / percentile / KL beat MinMax on activations by a wide margin. AWQ (Lesson 6) and GPTQ (Lesson 7) are the two production INT4 algorithms; they get to near-FP16 perplexity at 4× compression by attacking the problem from different angles (pre-scaling for AWQ, error propagation for GPTQ).

Pruning (Lesson 8) is the cousin of quantization that doesn't quite work on commodity hardware — only 2:4 structured sparsity ships at scale, and it doesn't compose with INT4. Distillation (Lesson 9) is the orthogonal lever: train a smaller model to imitate the larger one, the dominant technique for the small-reasoning-model wave. The Rust GPU frontier (Lesson 10) is the ecosystem signal: production deployment is increasingly viable outside Python; cubecl is the piece to watch for cross-platform low-precision kernels.

## What we covered, what we skipped

**Covered:**

- Per-layer FLOP and KV-cache arithmetic.
- The arithmetic-intensity / roofline framing for memory-bound vs compute-bound.
- IEEE 754, FP16, BF16, FP8 (E4M3/E5M2), MXFP4/MXFP8.
- INT8 with per-channel scales, the W8A16 vs W8A8 split.
- INT4 with per-group scales, GGUF / AWQ / GPTQ / NF4 packing layouts.
- Calibration with MinMax / percentile / MSE / KL / dynamic-per-token criteria.
- The AWQ algorithm: salient-channel-aware pre-scaling.
- The GPTQ algorithm: Hessian-driven error propagation with the Cholesky trick.
- Structured 2:4 sparsity and the broader pruning landscape.
- Hinton soft-label distillation, feature/relation variants, modern rationale distillation.
- The Rust GPU ecosystem: Burn, Candle, cubecl, where each fits.

**Skipped:**

- **Sub-INT4 / 1-bit quantization in depth.** BitNet and the OmniQuant variants for 2-bit and 3-bit weights are a research frontier; the production INT4 story is much more settled.
- **Mixed-precision training paths.** Module 3 is inference-focused; FP8 training, BF16-with-FP32-master, gradient scaling — adjacent but separate.
- **MoE-specific compression.** Expert-level quantization, expert pruning, expert distillation each have their own literature; Module 7 (Inference from Scratch) covers MoE.
- **Vision-specific compression patterns.** Channel pruning of CNNs, post-quantization fine-tuning recipes for ViT-class models — left to a specific vision module if we add one.
- **The CUTLASS / TensorRT internals for INT4/FP8 kernels.** Mentioned the existence of vendor INT4 paths; didn't reverse-engineer them.
- **Hardware-specific FP4 (Blackwell) and MXINT8 details.** Named the formats; production kernels for them are too new and too vendor-specific to write up usefully in mid-2026.

## The compression budget decision tree

A workflow that gets the right answer for most LLM deployments:

```
1. WHERE will this run?
   - Cloud GPU (H100 / A100): consider FP8 or INT8; INT4 only if memory-bound.
   - Consumer GPU (4090, 3090): INT4 is the default; INT8 if accuracy-critical.
   - Apple Silicon (M-series): INT4 via GGUF/MLX; FP16 for small models that fit.
   - Phone / NPU: INT8 quant is the floor; sometimes INT4 with vendor support.
   - Embedded / no-GPU: aggressive distillation + INT8; consider sub-1B parameters.

2. WHAT'S THE MEMORY CEILING?
   - Compute: model_size_FP16 = 2 * params / 1e9 GB
   - Compute: budget = device_RAM - OS_overhead - KV_cache_budget
   - If model_size_FP16 <= budget → consider staying at FP16
   - If model_size_FP16 / 2 <= budget → INT8 fits
   - If model_size_FP16 / 4 <= budget → INT4 fits
   - If nothing fits → distill or pick a smaller model

3. WHAT FORMAT does the RUNTIME support?
   - llama.cpp: GGUF Q4_K_M, Q5_K_M, Q8_0
   - vLLM: AWQ, GPTQ, FP8
   - TensorRT-LLM: FP8, INT8 W8A8, INT4 with vendor support
   - MLX: MLX-native 4-bit, GGUF via import
   - Transformers (HF): AWQ, GPTQ, bitsandbytes INT8/NF4

4. PICK THE ALGORITHM
   - GGUF Q4_K_M: use llama.cpp's built-in quantize tool
   - AWQ: fastest calibration, use autoawq
   - GPTQ: slightly better perplexity sometimes, use auto-gptq
   - FP8: vendor stack (TRT-LLM, vLLM-FP8)
   - NF4: only if you're also doing QLoRA fine-tuning

5. VALIDATE
   - Quick perplexity on WikiText/C4 (15-30 min)
   - Task-specific eval on representative held-out set (longer)
   - Compare to FP16 baseline: <0.3 perplexity delta is excellent, <1.0 is acceptable, >1.0 means something went wrong
   - Spot-check 20-50 prompts manually for qualitative regressions

6. KV CACHE
   - At long context, the KV cache often dominates memory: don't forget to quantize it too
   - INT8 KV cache: safe, halves memory, near-zero accuracy hit
   - INT4 KV cache: aggressive, halves again, validates more carefully

7. ITERATE
   - If perplexity hit is too large: try a different algorithm (AWQ <> GPTQ), a less aggressive format (Q4 → Q5), or smaller group size (128 → 64)
   - If you need more compression: distill to a smaller architecture first, then quantize that
   - If the runtime is the constraint: consider switching runtimes or formats
```

This is the framework. The specifics — what counts as "acceptable" perplexity, which task evals matter — are deployment-specific. The framework itself is general.

## Mental models to carry forward

Five sentences, one per major lesson:

**1. Inference at batch 1 is memory-bound.** Every compression decision should be evaluated against "does this reduce bytes loaded per token." Reducing FLOPs is a distant second.

**2. Precision is a storage choice, not a compute choice.** The tensor cores compute at higher precision than the format stored on disk. INT4 weights, FP16 activations, FP32 accumulator is the standard pattern.

**3. Granularity is the most important quantization knob.** Per-tensor → per-channel → per-group, with finer granularity buying accuracy at a small cost in storage. INT4 needs per-group; INT8 usually doesn't.

**4. Outlier activations are the practical obstacle.** SmoothQuant, AWQ, and LLM.int8 all solve variants of this. Naive recipes that ignore outliers break at INT8 for activations and at INT4 for weights.

**5. Algorithms (AWQ, GPTQ) matter more than the format.** Two different INT4 conversions of the same model can differ by 1+ perplexity points depending on whether the algorithm handled outliers well. The format file's bit layout is secondary.

## What's next

**Module 4: MLX & Apple Silicon Internals.** We pivot to the Apple side of on-device ML. Unified memory, the M-series GPU's specific quirks, MLX's design (lazy evaluation, native quantization paths, the matrix accelerators), how the ANE fits, and how to pick between MLX, Core ML, MPSGraph, and llama.cpp for a given Apple-Silicon workload. The quantization recipes from Module 3 apply directly; Module 4 makes them concrete on the Apple substrate.

**Module 5: Mobile & Edge Runtimes.** TensorFlow Lite, ONNX Runtime, ExecuTorch, NPU paths on Android (Qualcomm Hexagon, MediaTek APU). The mobile/edge runtimes consume the quantized models from Module 3 and deploy them in environments where Python is impossible.

**Module 7: Inference from Scratch.** The full LLM inference pipeline, built from primitives. This is where the matmul kernels (Module 2) + the quantization (Module 3) compose into a working production-grade inference path. KV cache management, prefix caching, scheduling, beam search and sampling, batching, prefill vs decode — all in one running system.

The matmul project that's run through all three modules is now in a clean state: CPU naive → CPU SIMD threaded → CUDA tiled → Triton → MSL → vendor BLAS (Modules 1–2), and now extended with INT8 and INT4 variants. The running benchmark on your machine should show:

- FP16 → INT8 W8A16: ~1.7–2× speedup at batch 1 on a 4090.
- FP16 → INT4 (AWQ/GPTQ): ~2.5–3.5× speedup at batch 1.
- 4× memory reduction either way.

On Apple Silicon (the lessons in Module 4 will quantify):

- FP16 → INT4 via MLX: ~1.5–2× speedup at batch 1 on an M3 Pro.
- 4× memory reduction.
- The headline benchmark: a 7B-class LLM running at usable token rates on a 16 GB MacBook.

That last result — "usable on-device LLM inference on commodity laptop hardware" — is what this entire module exists to enable. The next module makes it concrete on the Apple side.

## Hands-on (at home)

The Module 3 capstone. Take a real LLM, quantize it with each format, compare.

```python
# module3_capstone.py
# Requires: transformers, autoawq, auto-gptq, datasets, bitsandbytes (optional)
import torch
import time
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "Qwen/Qwen2.5-1.5B-Instruct"
tok = AutoTokenizer.from_pretrained(model_id)

def bench_generate(model, prompt, max_new_tokens=128):
    ids = tok(prompt, return_tensors="pt").input_ids.to(model.device)
    # Warm.
    model.generate(ids, max_new_tokens=8, do_sample=False)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t0 = time.time()
    out = model.generate(ids, max_new_tokens=max_new_tokens, do_sample=False)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    dt = time.time() - t0
    n_new = out.shape[1] - ids.shape[1]
    return n_new / dt, out

def perplexity(model, n=32, max_len=512):
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    losses, tokens = 0.0, 0
    for i in range(n):
        text = ds[i]["text"]
        if not text.strip(): continue
        ids = tok(text, return_tensors="pt", truncation=True, max_length=max_len).input_ids.to(model.device)
        if ids.shape[1] < 2: continue
        with torch.no_grad():
            losses += model(ids, labels=ids).loss.item() * ids.shape[1]
        tokens += ids.shape[1]
    return float(torch.tensor(losses / tokens).exp())

PROMPT = "Write a one-paragraph explanation of cache-blocked matrix multiplication."

paths = {
    "FP16": model_id,
    "AWQ-INT4": "Qwen2.5-1.5B-AWQ",         # from Lesson 6
    "GPTQ-INT4": "Qwen2.5-1.5B-GPTQ",        # from Lesson 7
}

for name, p in paths.items():
    print(f"\n=== {name} ===")
    try:
        m = AutoModelForCausalLM.from_pretrained(p, torch_dtype=torch.float16, device_map="cuda")
    except Exception as e:
        print(f"  skip: {e}")
        continue
    tps, _ = bench_generate(m, PROMPT)
    ppl = perplexity(m)
    print(f"  tokens/sec: {tps:.1f}")
    print(f"  perplexity: {ppl:.3f}")
    # Memory footprint.
    mem = sum(p.element_size() * p.numel() for p in m.parameters()) / 1e9
    print(f"  weights: {mem:.2f} GB")
    del m
    torch.cuda.empty_cache()
```

A successful run gives you the canonical compression vs accuracy vs throughput table for one model:

| Format | Perplexity | Tokens/sec | Memory |
| ------ | ---------- | ---------- | ------ |
| FP16 | (baseline) | (baseline) | 3.1 GB |
| AWQ-INT4 | +0.15 | ~2× | 0.8 GB |
| GPTQ-INT4 | +0.20 | ~2× | 0.8 GB |

Numbers will vary by GPU and tokenizer/model exact sizes. The pattern — INT4 within ~0.2 perplexity, ~2× throughput, ~4× memory reduction — is the takeaway.

## Further reading

- The papers cited throughout (FP8, LLM.int8, SmoothQuant, AWQ, GPTQ, OmniQuant) are the canonical references for the algorithms.
- "A Survey of Quantization Methods for Efficient Neural Network Inference" (Gholami et al, 2021) — the academic survey, now somewhat dated but still useful for the taxonomy.
- "MLPerf Inference" benchmark results — for the production-deployment view of what compression formats achieve at scale.
- HuggingFace's "Optimum" library docs — for the production integration patterns across ONNX, OpenVINO, TensorRT, and Neural Compressor.

## End of Module 3

Module 1 made the CPU fast. Module 2 made the GPU fast. Module 3 made the model small. The combination — fast hardware + small payloads — is the foundation of on-device ML.

Module 4 picks up next: **MLX & Apple Silicon Internals.** We take the unified-memory M-series SoC seriously and look at how the matrix accelerators, the GPU, and the ANE compose. The quantized models from this module land on real hardware that wasn't designed for them — and yet works remarkably well.
