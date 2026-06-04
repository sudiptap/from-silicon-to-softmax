---
title: "Mobile & Edge Runtimes: Module Overview"
date: "2026-06-03"
excerpt: "The runtime layer for on-device ML across iOS, Android, embedded Linux, and microcontrollers. Core ML, ONNX Runtime, ExecuTorch, LiteRT, llama.cpp/ggml, Qualcomm QNN/Hexagon — what each one is good at, how they compose, and how to choose."
module: "mobile-edge-runtimes"
order: 0
tags: ["coreml", "onnx", "executorch", "tflite", "litert", "llama-cpp", "qualcomm", "hexagon", "on-device", "overview"]
author: "Sudipta Pathak"
prerequisites: ["ml-internals", "mlx-apple-silicon"]
---

# Mobile & Edge Runtimes

## Why this module exists

When you move ML off a data-center GPU, the framework story explodes. There's no single "PyTorch on the device." Instead there are six or seven serious runtimes, each with strengths, each with a model-format convention, each maintained by a different vendor with different incentives. Choosing wrong costs months — either in unsupported operators, in chasing the wrong accelerator, or in a runtime that won't ship to the platform you actually need.

This module is the runtime map. After it, you should be able to look at a deployment requirement — "Llama 3.2 1B on iOS with ANE acceleration", "Whisper on Android with Hexagon NPU", "TinyML on a Cortex-M4" — and instantly know which two runtimes are credible, which one's the better starting point, and what you'd lose by picking the other.

## How this fits

Module 5 of the depth track, building directly on the Apple Silicon module (4) and feeding into on-device LLM inference (6). Where module 4 went deep on one platform, this module goes wide across the edge runtime landscape.

## The roadmap

10 tutorials.

### The landscape

1. **The edge runtime map** — vendor matrix, platform matrix, model-format matrix; the runtimes you'll actually consider and the ones to skip
2. **Model formats** — ONNX, GGUF, Core ML (`.mlmodel` / `.mlpackage`), TFLite/LiteRT, ExecuTorch (`.pte`); conversion paths and what's lost in each translation

### Apple-side runtimes

3. **Core ML deep dive** — `coremltools` conversion, compute units (CPU / GPU / ANE), targeting the ANE, debugging when it falls back to CPU
4. **ONNX Runtime on iOS/macOS** — when ORT beats Core ML, execution providers, the Core ML EP

### Cross-platform runtimes

5. **ExecuTorch** — PyTorch's edge story; the `.pte` format, backend delegates, the XNNPACK / Core ML / QNN backends; what's still rough
6. **LiteRT (formerly TFLite)** — model conversion, the delegate model, GPU & NNAPI delegates, MediaPipe Tasks
7. **ONNX Runtime mobile** — the broad-coverage option; execution providers across platforms; binary size tradeoffs

### Hardware accelerators

8. **Qualcomm Hexagon NPU + QNN SDK** — the QNN execution layer, AI Hub, what you can run on Hexagon vs the Adreno GPU, the OEM-side ML story on Android
9. **llama.cpp / ggml** — the de facto on-device LLM runtime; GGUF, the backend abstraction, Metal/CUDA/Vulkan backends, why it ate the SLM serving niche

### Picking and composing

10. **The runtime decision tree** — a worked example: "I have a quantized 3B model, I need it on iOS and Android, latency budget 200ms TTFT" — building the picking matrix and defending the choice

---

## What I'm filling in over time

This is the topic scaffold. Each tutorial will include a working conversion script, a measurement script, and the gotcha list you only learn the hard way (the ops that fall back to CPU, the precision regressions, the binary size traps).
