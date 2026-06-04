---
title: "On-Device LLM Inference: Module Overview"
date: "2026-06-03"
excerpt: "Serving LLMs on a single device with a memory budget that ends in MB, not GB. Aggressive quantization, KV cache strategies that survive thermal throttling, speculative decoding with draft models, LoRA hot-swap, the Small Language Model landscape, and on-device multimodal models."
module: "on-device-llm-inference"
order: 0
tags: ["on-device", "llm", "quantization", "kv-cache", "speculative-decoding", "lora", "slm", "multimodal", "overview"]
prerequisites: ["ml-internals", "mlx-apple-silicon", "mobile-edge-runtimes"]
author: "Sudipta Pathak"
---

# On-Device LLM Inference

## Why this module exists

Serving an LLM in a data center is a memory-bandwidth problem with a checkbook attached. Serving an LLM on a phone is a memory-*budget* problem with a thermal envelope, a battery counter, and no fallback. Everything that's easy on an H100 — bf16 weights, generous KV cache, no warmup constraints — breaks at the edge. Everything that mattered marginally on an H100 — quantization quality, KV cache layout, speculative decoding speedups, weight streaming — matters absolutely on a phone.

This module covers the full serving stack for that environment. By the end you should be able to take a 3B-class model, quantize it well enough to fit, run it fast enough to feel interactive, and keep it running through a 20-minute session without the device entering thermal throttling.

## How this fits

Module 6 of the depth track, the synthesis of modules 3 (ML internals & quantization), 4 (MLX & Apple Silicon), and 5 (mobile/edge runtimes). The companion module is module 7 (Inference from Scratch), which covers the same algorithmic ideas — attention, KV cache, sampling, MoE — in their full form. This module is where those ideas meet a 4GB RAM ceiling.

## The roadmap

12 tutorials.

### Setting the constraints

1. **The on-device LLM stack** — what's actually different from data-center serving; the budget table (memory, bandwidth, thermal, power), the latency budget for "feels interactive", the operator-coverage problem
2. **Picking the model** — the SLM landscape circa now: Phi-3.5, Llama 3.2 (1B/3B), Gemma Nano, Qwen 0.5B/1.5B/3B, Granite, SmolLM; the parameter / capability tradeoff

### Compression

3. **Aggressive quantization recipes** — Q4_0, Q4_K_M, Q5_K_M, AWQ, GPTQ, the MLX-native scheme; perplexity vs size vs latency tables; when to step up precision per-layer
4. **Mixed precision and per-channel quantization** — activations vs weights, the smoothquant idea, when calibration data matters
5. **Pruning and distillation as a complement** — when SLM distillation is worth the engineering cost; layer pruning vs depth pruning vs Sheared LLaMA-style

### The serving loop

6. **KV cache for tiny memory budgets** — paged KV cache on-device, KV quantization, sliding-window attention, the moment you spill to disk
7. **Memory-mapped weights and weight streaming** — `mmap` semantics, page-fault patterns, how llama.cpp does it, why this often beats explicit loading
8. **Streaming generation patterns** — token-by-token UI patterns, cancellation, partial-rollback for "stop" interactions

### Going faster

9. **Speculative decoding on-device** — using a tiny draft model (or n-gram, or self-speculation); the throughput math, the memory cost
10. **LoRA hot-swap** — keeping a base model resident and swapping persona/task adapters at request time; the merge-vs-keep-separate tradeoff

### Beyond text

11. **On-device multimodal** — Moondream, Florence-2, SmolVLM, Phi-3 Vision; vision encoder cost dominates, what to share across requests
12. **Real-time interactive use cases** — speech in (Whisper / Parakeet), speech out (TTS), code completion at typing latency, vision pipelines; end-to-end budget breakdowns

---

## What I'm filling in over time

This is the topic scaffold. Each tutorial includes a working reference implementation (typically MLX or llama.cpp), measurement code, and a "what breaks first" section — which is usually the most useful part. The goal is for someone working through this module to ship a usable on-device LLM feature, not just understand one.
