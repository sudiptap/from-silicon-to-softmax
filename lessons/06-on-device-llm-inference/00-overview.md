---
title: "Module 6 — On-Device LLM Inference"
date: "2026-06-04"
module: "on-device-llm-inference"
order: 0
tags: ["on-device", "llm", "quantization", "kv-cache", "speculative-decoding", "lora", "slm", "multimodal", "overview"]
author: "Sudipta Pathak"
prerequisites: ["ml-internals", "mlx-apple-silicon", "mobile-edge-runtimes"]
---

# On-Device LLM Inference

## Why this module exists

Serving an LLM in a data center is a memory-bandwidth problem with a checkbook attached. Serving an LLM on a phone is a memory-*budget* problem with a thermal envelope, a battery counter, and no fallback. Everything that's easy on an H100 — BF16 weights, generous KV cache, no warm-up constraints — breaks at the edge. Everything that mattered marginally on an H100 — quantization quality, KV cache layout, speculative decoding speedups, weight streaming — matters absolutely on a phone.

This module covers the full serving stack for that environment. By the end you should be able to take a 3B-class model, quantize it well enough to fit, run it fast enough to feel interactive, and keep it running through a 20-minute session without the device entering thermal throttling or running the battery from full to empty.

## How this fits

Module 6 of the depth track and the synthesis of three earlier modules: Module 3 (ML internals & quantization) provides the format recipes, Module 4 (MLX & Apple Silicon) provides the on-device substrate, and Module 5 (Mobile & Edge Runtimes) provides the runtime layer. Module 6 takes those building blocks and addresses the *operational* questions: which model to pick, how to combine the compression levers, how to manage the KV cache when RAM is the binding constraint, how to keep the user-perceived experience smooth.

The companion module is Module 7 (Inference from Scratch), which goes deep on the algorithmic ideas (attention, sampling, MoE, speculative decoding) in their full form. This module is where those ideas meet a 4 GB RAM ceiling.

The output of this module: the ability to ship a real on-device LLM feature — a chat assistant, a code completion path, an audio transcription pipeline, a vision question-answering system — knowing the failure modes, the budget math, and the optimization levers.

## The roadmap

Twelve lessons.

### Setting the constraints

1. **The on-device LLM stack** — what's actually different from data-center serving; the budget table (memory, bandwidth, thermal, power), the latency budget for "feels interactive," and the operator-coverage problem.
2. **Picking the model** — the small-language-model landscape circa mid-2026: Phi-3.5, Llama 3.2 (1B / 3B), Gemma Nano, Qwen 2.5 (0.5B / 1.5B / 3B), Granite, SmolLM. The parameter-vs-capability tradeoff, what each model is best at.

### Compression

3. **Aggressive quantization recipes** — Q4_0, Q4_K_M, Q5_K_M, AWQ, GPTQ, the MLX-native scheme; perplexity vs size vs latency tables; when to step up precision per-layer.
4. **Mixed precision and per-channel quantization** — activations vs weights, the SmoothQuant idea applied on-device, when calibration data matters and when it doesn't.
5. **Pruning and distillation as a complement** — when SLM distillation is worth the engineering cost; layer pruning vs depth pruning vs Sheared-LLaMA-style width pruning.

### The serving loop

6. **KV cache for tiny memory budgets** — paged KV cache on-device, KV quantization, sliding-window attention with attention sinks, the moment you spill to disk.
7. **Memory-mapped weights and weight streaming** — `mmap` semantics, page-fault patterns, how llama.cpp does it, why this often beats explicit loading.
8. **Streaming generation patterns** — token-by-token UI patterns, cancellation handling, partial-rollback for "stop" interactions, smooth UX over an interruptible inference loop.

### Going faster

9. **Speculative decoding on-device** — using a tiny draft model, n-gram speculation, or self-speculation (Medusa, EAGLE); the throughput math, the memory cost, when speculative decoding pays off and when it loses.
10. **LoRA hot-swap** — keeping a base model resident and swapping persona/task adapters at request time; the merge-vs-keep-separate tradeoff; the memory cost of N adapters.

### Beyond text

11. **On-device multimodal** — Moondream, Florence-2, SmolVLM, Phi-3 Vision; vision encoder cost dominates; what to share across requests.
12. **Real-time interactive use cases (module wrap)** — speech in (Whisper / Parakeet), speech out (TTS), code completion at typing latency, vision pipelines; end-to-end budget breakdowns; handoff to Module 7.

---

## What this module deliberately won't cover

- **Training and fine-tuning at scale.** Inference-only; fine-tuning is in adjacent modules.
- **Multi-tenant LLM serving** (batched serving across many users). Module 7 touches this; production server-grade serving is the vLLM / TensorRT-LLM territory.
- **Distributed inference across multiple devices.** Module 8 (Distributed Systems) covers split inference and the `exo`-style multi-Mac story.
- **The Android-NPU LLM frontier in depth.** Module 5 Lesson 8 covered Qualcomm Hexagon; we use it here but don't re-derive it.
- **App store policies, signing, OS-level integration.** Important but not ML systems.
- **Latency-vs-quality tuning of specific applications** (e.g., the precise temperature/top-p settings for a chatbot). We focus on the systems layer; the product layer is downstream.

## How to work through it

Every lesson is fully readable as prose on a phone. The hands-on sections assume access to at least one of:

- An Apple Silicon Mac for the MLX and llama.cpp Metal-accelerated paths.
- A Linux box with an NVIDIA GPU for the CUDA path (some hands-on apply directly; others can be adapted).
- A modern Android phone (2022+) for the Android-specific paths in Lessons 1, 7, and 12.

If you only have a Mac, all the lessons' hands-on sections work; the Android-specific notes describe what to expect on that platform. If you only have a Linux box with NVIDIA, the bandwidth-bound math still applies but the absolute numbers differ.

The running example for this module is a single deployment: take Llama 3.2 3B Instruct, quantize it, run it through MLX or llama.cpp on Apple Silicon, then again through llama.cpp on a Linux box for comparison. Each lesson adds one optimization or one operational pattern; the cumulative effect at the end of the module is a deployment that runs at 30+ tokens/sec on a MacBook Pro for a 20-minute interactive session without thermal throttling.

A note on tempo: this module is operations-heavy and includes more "what breaks first" content than the earlier modules. The earlier modules built mental models; this one applies them to a specific deployment shape and accumulates the practical knowledge that doesn't fit cleanly into any one algorithm.
