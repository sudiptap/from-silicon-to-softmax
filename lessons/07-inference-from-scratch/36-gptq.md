---
title: "Lesson 36 — GPTQ: Second-Order Quantization"
date: "2026-06-04"
module: "inference-from-scratch"
order: 36
tags: ["gptq", "quantization", "second-order", "hessian", "obs"]
author: "Sudipta Pathak"
prerequisites: ["35-int8-int4-basics"]
---

# Lesson 36 — GPTQ: Second-Order Quantization

## Why this lesson exists

Module 3 Lesson 7 covered GPTQ from the algorithm side — the OBS/OBQ lineage, the Hessian-driven error propagation, the Cholesky trick that made it tractable for LLMs. This lesson is the inference-time view: what GPTQ produces, what format it ships in, and the considerations for serving GPTQ-quantized models.

The lesson is short by design (the algorithm was covered in Module 3); it focuses on the runtime/format details that matter for building inference pipelines.

The lesson is reading. The Hands-on serves a GPTQ-quantized model via vLLM and inspects its format.

## What GPTQ produces

GPTQ takes a pretrained FP16 model and produces:
- Weight tensors quantized to INT4 (or INT3, INT8 depending on bits config).
- Per-group scales (group_size=128 standard).
- Per-group zero-points (for asymmetric quantization).
- An "g_idx" tensor describing the column-permutation if "act-order" was used during calibration.

The on-disk format is the GPTQ-specific layout — supported by `auto-gptq`, vLLM, ExLlamaV2, and most modern runtimes.

## The act-order subtlety

GPTQ's act-order variant (Module 3 Lesson 7) processes weight columns in order of decreasing activation magnitude. This produces ~0.1 perplexity better than no-act-order but reorders the columns of the weight matrix.

For inference, the reordering means:
- The activation tensor needs the same column reordering applied to its input dimension before the matmul.
- The runtime handles this via the `g_idx` tensor: a permutation that tells the kernel which input column corresponds to which (post-reorder) weight column.

```python
# Pseudo-code:
def gptq_matmul_act_order(x, W_int4, scales, zeros, g_idx):
    # x: [M, K] fp16; g_idx: [K] int32 permutation
    x_reordered = x[:, g_idx]  # apply the same permutation as during calibration
    return dequant_matmul(x_reordered, W_int4, scales, zeros)
```

The `x[:, g_idx]` gather is an extra cost — typically 5-10% slower than no-act-order. The accuracy improvement is usually worth it; some runtimes offer "act_order=False" for the latency win.

## ExLlamaV2 and the fast GPTQ kernels

GPTQ's most-tuned inference kernels are ExLlamaV2's. The kernels handle:
- INT4 weight unpacking (2-per-byte).
- Per-group scale application.
- act-order permutation (when present).
- Fused dequant + GEMM.

ExLlamaV2 achieves ~30% higher throughput than baseline vLLM on GPTQ models on certain configurations. The gap shrinks each year as vLLM's kernels improve.

In 2026 the production GPTQ inference stacks are:
- vLLM (general, multi-tenant).
- TensorRT-LLM (NVIDIA-optimized, single-tenant high-throughput).
- ExLlamaV2 (single-user local hosting, peak throughput on consumer GPUs).
- llama.cpp via its GPTQ converter to GGUF.

## GPTQ vs AWQ from the inference perspective

The two algorithms produce very similar weight formats: per-group INT4 with scales and (for GPTQ) zero-points. Inference performance is essentially the same for the same hardware.

The differences are upstream:
- AWQ has a per-channel scaling factor "smoothed in" — the kernel doesn't need to apply it separately.
- GPTQ's act-order requires an extra permutation step (minor cost).

For runtime selection, you usually have one model or the other; the choice is determined by what's available pre-quantized (HuggingFace has both) and what your runtime supports best.

## What you should believe after this lesson

Three sentences:

**1. GPTQ produces INT4 weights + per-group scales + per-group zero-points + (optional) an `g_idx` permutation tensor**. The inference kernel handles the dequant fusion plus the column reordering if act-order was used.

**2. ExLlamaV2 has the most-tuned GPTQ kernels** for consumer-GPU inference; vLLM and TensorRT-LLM are competitive. Performance is close enough that the choice is usually about ecosystem fit, not raw throughput.

**3. GPTQ and AWQ are similar enough at inference** that the choice between them is determined by what's available and what your runtime supports — both produce per-group INT4 weights and the matmul kernels are nearly identical.

## Hands-on (at home)

Serve a GPTQ model via vLLM and inspect.

```bash
# Get a pre-quantized GPTQ model.
huggingface-cli download TheBloke/Llama-2-7B-Chat-GPTQ \
    --local-dir Llama-2-7B-Chat-GPTQ

# Run with vLLM.
vllm serve Llama-2-7B-Chat-GPTQ --quantization gptq

# In another terminal:
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{"model": "Llama-2-7B-Chat-GPTQ", "prompt": "Hello", "max_tokens": 50}'
```

Or with `auto-gptq` Python API:

```python
from auto_gptq import AutoGPTQForCausalLM
from transformers import AutoTokenizer

model_id = "TheBloke/Llama-2-7B-Chat-GPTQ"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoGPTQForCausalLM.from_quantized(model_id, device="cuda:0")

prompt = "Hello, how are you?"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0]))
```

To inspect the format, load the `.safetensors` file directly:

```python
from safetensors import safe_open
with safe_open("model.safetensors", framework="pt") as f:
    for key in f.keys()[:10]:
        t = f.get_tensor(key)
        print(f"{key}: shape={tuple(t.shape)}, dtype={t.dtype}")
```

You'll see keys like `model.layers.0.self_attn.q_proj.qweight` (INT32-packed INT4), `qzeros`, `scales`, and `g_idx`.

## Further reading

- "GPTQ" paper (Frantar et al, 2022).
- ExLlamaV2 GitHub (turboderp/exllamav2).
- vLLM and TensorRT-LLM GPTQ documentation.
- Module 3 Lesson 7 — algorithm-side coverage.

Next lesson: **AWQ — activation-aware quantization.** Module 3 Lesson 6 covered the algorithm; here we cover the inference kernel and format details. Then we head into SmoothQuant, FP8, and BitNet.
