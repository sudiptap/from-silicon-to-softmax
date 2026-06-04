---
title: "Module 5 — Mobile & Edge Runtimes"
date: "2026-06-04"
module: "mobile-edge-runtimes"
order: 0
tags: ["coreml", "onnx", "executorch", "tflite", "litert", "llama-cpp", "qualcomm", "hexagon", "on-device", "overview"]
author: "Sudipta Pathak"
prerequisites: ["ml-internals", "mlx-apple-silicon"]
---

# Mobile & Edge Runtimes

## Why this module exists

When you move ML off a data-center GPU, the framework story explodes. There's no single "PyTorch on the device." Instead there are six or seven serious runtimes, each with strengths, each with a model-format convention, each maintained by a different vendor with different incentives. Choosing wrong costs months — either in unsupported operators, in chasing the wrong accelerator, or in a runtime that won't ship to the platform you actually need.

This module is the runtime map. After it, you should be able to look at a deployment requirement — "Llama 3.2 1B on iOS with ANE acceleration," "Whisper on Android with Hexagon NPU," "TinyML on a Cortex-M4" — and instantly know which two runtimes are credible, which one's the better starting point, and what you'd lose by picking the other.

The unifying observation: the compute-and-compression substrate from Modules 1–4 doesn't change between mobile and desktop on-device. What changes is the *runtime layer* — the thing that loads the model file, dispatches kernels to the available accelerators, and exposes a programming surface to the app. Picking the right runtime layer for the platform you ship to is the half of mobile ML that no algorithm work substitutes for.

## How this fits

Module 5 of the depth track. Module 4 went deep on one platform (Apple Silicon); this module goes wide across the edge runtime landscape. It feeds directly into Module 6 (On-Device LLM Inference, where the runtime choice from this module composes with the LLM serving patterns) and Module 7 (Inference from Scratch, where the runtime-internal patterns become things you build, not consume).

The output of this module: the ability to take a deployment requirement and select a runtime quickly, with a clear understanding of the tradeoffs and the gotchas. A secondary output: a working knowledge of the model-format ecosystem (ONNX, GGUF, Core ML, LiteRT, ExecuTorch `.pte`) so you can read other people's deployments without confusion.

## The roadmap

Ten lessons.

### The landscape

1. **The edge runtime map** — vendor matrix, platform matrix, model-format matrix; the runtimes you'll actually consider in 2026 and the ones to skip.
2. **Model formats** — ONNX, GGUF, Core ML (`.mlmodel` / `.mlpackage`), LiteRT (TFLite), ExecuTorch (`.pte`); conversion paths between them and what's lost in each translation.

### Apple-side runtimes

3. **Core ML deep dive** — `coremltools` conversion, compute units (CPU / GPU / ANE), targeting the ANE, debugging when an op silently falls back to CPU.
4. **ONNX Runtime on iOS/macOS** — when ORT beats Core ML on Apple platforms, execution providers, the Core ML EP that lets ORT delegate to Core ML.

### Cross-platform runtimes

5. **ExecuTorch** — PyTorch's edge story; the `.pte` format, backend delegates (XNNPACK, Core ML, QNN, Vulkan), what's still rough vs. production-ready in 2026.
6. **LiteRT (formerly TFLite)** — model conversion, the delegate model, GPU and NNAPI delegates, MediaPipe Tasks as the high-level wrapper.
7. **ONNX Runtime mobile** — the broad-coverage option; execution providers across platforms; binary size tradeoffs.

### Hardware accelerators

8. **Qualcomm Hexagon NPU + QNN SDK** — the QNN execution layer, AI Hub, what you can run on Hexagon vs the Adreno GPU, the OEM-side ML story on Android.

### The cross-platform LLM runtime

9. **llama.cpp / ggml** — the de facto on-device LLM runtime; GGUF, the backend abstraction, Metal/CUDA/Vulkan/Hexagon backends, why it ate the small-LLM serving niche on every platform.

### Picking and composing

10. **The runtime decision tree** — a worked example: "I have a quantized 3B model; I need it on iOS and Android; latency budget 200 ms TTFT" — building the picking matrix and defending the choice. Plus the module wrap and handoff to Module 6.

---

## What this module deliberately won't cover

- **In-depth iOS or Android app development.** Bundle layouts, signing, App Store / Play Store policies — important but not ML systems.
- **Browser ML runtimes in depth.** WebGPU + ONNX Runtime Web is a real path; we mention it but don't go deep. The browser story deserves its own treatment.
- **Embedded microcontroller ML (TinyML) in depth.** TensorFlow Lite Micro and similar are real runtimes; we touch them but don't deep-dive. The constraints (RAM measured in kilobytes) make them a different discipline.
- **Vendor SDKs we won't see widely deployed** — Samsung's Exynos NPU SDKs, Huawei's HiAI — relevant in specific markets, omitted for focus.
- **Distillation / quantization workflows.** Module 3 covered these. Here we focus on what the runtime expects as input, not how to produce it.
- **The "build your own inference engine" path** — Module 7 covers it. Module 5 is about choosing among the engines that exist.

## How to work through it

Every lesson is fully readable as prose. The hands-on sections assume access to at least one of:

- A Mac for Core ML, ExecuTorch (Apple backend), ONNX Runtime on macOS.
- An Android device for LiteRT, ONNX Runtime Mobile, QNN Hexagon. A device with a Qualcomm SoC (most flagship phones 2022+) for the Hexagon-specific work.
- A Linux box for ExecuTorch with XNNPACK and ONNX Runtime, llama.cpp builds for the CPU and CUDA paths.

If you only have one of these (most likely a Mac), the prose still stands — the cross-platform sections describe what you'd see on the other platforms — and the hands-on sections marked "Apple" still work end-to-end. The decision tree at the end is platform-agnostic.

The matmul project from Modules 1–4 has done its work; it stops being the running example here because mobile/edge runtimes are about the layer above the kernel. The new running example is a small quantized LLM (1B-class) deployed across the available runtimes; we measure tokens/sec and binary footprint to inform the picking matrix.

A note on tempo: this module is breadth-first. Each lesson covers a runtime or a small group of related runtimes at the level needed to make informed choices. We don't go as deep on each as we did on MLX in Module 4 — the right amount of depth here is "enough to choose and to debug the first deployment," not "enough to contribute to the runtime."
