---
title: "Lesson 37 — AWQ: Activation-Aware Weight Quantization"
date: "2026-06-04"
module: "inference-from-scratch"
order: 37
tags: ["awq", "quantization", "activation-aware", "salient-channels", "inference"]
author: "Sudipta Pathak"
prerequisites: ["36-gptq"]
---

# Lesson 37 — AWQ: Activation-Aware Weight Quantization

## Why this lesson exists

Module 3 Lesson 6 covered AWQ from the algorithm side: the salient-channels insight that errors on weights paired with high-magnitude activations dominate, the per-channel pre-scaling that allocates more INT4 resolution to those channels. This lesson covers AWQ's inference-time view: the kernel, the format, and the comparison to GPTQ.

The lesson is short by design; the algorithm details live in Module 3.

The lesson is reading. The Hands-on serves an AWQ model and inspects the format.

## What AWQ produces

AWQ takes a pretrained model and produces:
- Weight tensors quantized to INT4 (per-group, group=128).
- Per-group scales.
- Per-group zero-points (for asymmetric, optional).
- A per-channel pre-scaling factor that's *absorbed into the weights themselves* — no separate "rescale" step at inference.

The key difference from GPTQ: AWQ's smoothing factor is folded into the weights at conversion time. The inference kernel sees a standard per-group quantized weight tensor; the smoothing is invisible.

## The absorbed-scaling trick

The math (Module 3 Lesson 6): for a per-channel diagonal scale `s`:

```
Y = A × W = (A / s) × (s × W) = A' × W'
```

AWQ picks `s` based on activation magnitudes, then computes `W' = s × W` and quantizes `W'`. The quantized form contains the scaling.

At inference, the activations need to be scaled by `1/s` — but this can be folded into the *previous* layer's output projection (multiplying its weights by `1/s`). So the `1/s` is also absorbed into the model weights at conversion time.

Net result: at inference, the kernel sees standard per-group INT4 weights. No smoothing is applied at runtime. The smoothing was done offline.

This makes AWQ kernels nearly identical to standard per-group INT4 kernels. Less complex than GPTQ's act-order variant.

## The matmul kernel

AWQ's standard kernel:

```
W_int4 storage: [N, K/8] int32 (8 INT4 per int32, packed with a specific permutation)
scales: [N, K/group_size] fp16
zero_points: [N, K/group_size] int8

dequantize_fused_matmul(A_fp16, W_int4, scales, zero_points) → Y_fp16
```

The packing permutation: AWQ uses a specific 4-then-4 interleave (bits 0-3 of byte 0 are weight 0; bits 4-7 are weight 4; bits 0-3 of byte 1 are weight 1; etc.). This layout matches what the CUDA kernel expects.

The kernel reads INT4 weights, dequantizes inline, multiplies by FP16 activations, accumulates in FP32, downcasts to FP16. Standard W4A16 path.

## AutoAWQ and vLLM integration

The reference implementation is `autoawq`. The CUDA kernels achieve ~80-90% of theoretical peak on consumer NVIDIA GPUs.

vLLM, SGLang, and TensorRT-LLM all support AWQ format natively:

```bash
vllm serve TheBloke/Llama-2-7B-Chat-AWQ --quantization awq
```

Throughput is comparable to GPTQ on the same model; AWQ-quantized weights tend to be ~1-2% better in perplexity than GPTQ on Llama-class models (this varies; both are within noise on standard benchmarks).

## What you should believe after this lesson

Three sentences:

**1. AWQ's smoothing factor is absorbed into the weights at conversion time** — the inference kernel sees standard per-group INT4 weights with no special runtime handling. The "activation-aware" part is offline; the runtime path is identical to standard W4A16.

**2. The kernel uses a specific 4-then-4 interleaved packing** that matches the CUDA kernel's expectations. AutoAWQ and vLLM/SGLang/TensorRT-LLM all support the format natively.

**3. AWQ vs GPTQ at inference is essentially a wash**: similar throughput, similar quality (~0.1-0.2 perplexity difference on most Llama-class models, can flip per model). The choice is usually determined by what's pre-quantized and available.

## Hands-on (at home)

Run an AWQ-quantized model and inspect its format.

```bash
huggingface-cli download TheBloke/Llama-2-7B-Chat-AWQ \
    --local-dir Llama-2-7B-Chat-AWQ

# Run via vLLM.
vllm serve Llama-2-7B-Chat-AWQ --quantization awq
```

Or via `autoawq` Python:

```python
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

model_id = "TheBloke/Llama-2-7B-Chat-AWQ"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoAWQForCausalLM.from_quantized(model_id)

prompt = "Hello"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0]))
```

Inspect the AWQ format:

```python
from safetensors import safe_open
with safe_open("path/to/model.safetensors", framework="pt") as f:
    for key in list(f.keys())[:10]:
        t = f.get_tensor(key)
        print(f"{key}: shape={tuple(t.shape)}, dtype={t.dtype}")
```

You'll see `qweight` (INT32-packed INT4), `qzeros`, `scales`, but typically *not* a `g_idx` permutation (AWQ doesn't reorder columns).

## Further reading

- Module 3 Lesson 6 — the algorithm coverage.
- "AWQ" paper (Lin et al, 2023).
- autoawq GitHub repository.
- vLLM AWQ documentation.

Next lesson: **SmoothQuant.** The activation-quantization-enabler — migrate the outlier difficulty from activations to weights via per-channel rescaling. Enables W8A8 deployment. We covered the algorithm in Module 3; now we cover the inference-time integration.
