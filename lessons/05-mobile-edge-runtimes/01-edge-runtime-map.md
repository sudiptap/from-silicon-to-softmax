---
title: "Lesson 1 — The Edge Runtime Map"
date: "2026-06-04"
module: "mobile-edge-runtimes"
order: 1
tags: ["runtime", "edge", "mobile", "ios", "android", "landscape"]
author: "Sudipta Pathak"
prerequisites: ["00-overview"]
---

# Lesson 1 — The Edge Runtime Map

## Why this lesson exists

The first thing an ML engineer encounters when leaving the cloud is the runtime sprawl. "I want to run my model on a phone" splits into a dozen partially-overlapping options depending on the phone, the model, and the model's destination architecture. Before you can pick a runtime, you need a map: what exists, who maintains it, what platforms it covers, what kinds of models it accepts.

This lesson is that map. It's deliberately wide and shallow — every runtime in the rest of the module gets its own lesson. The point here is to plant the names, the vendor relationships, and the platform coverage in one place so you can navigate the rest with a coherent picture.

The lesson is reading. The Hands-on inventories which runtimes are installed on your dev machine and what platforms they target.

## The dimensions of choice

Five axes that determine which runtime is right:

**1. Target platform**: iOS, Android, Windows, Linux, macOS, embedded Linux, microcontroller, browser. The set of OSes and CPU architectures (ARM64, x86_64, ARMv7, Cortex-M) you must support.

**2. Target accelerator**: CPU only, mobile GPU (Adreno / Mali / Apple GPU), NPU (Hexagon, Apple ANE, MediaTek APU, Samsung NPU), DSP, custom ASIC. Whether your model needs hardware acceleration and which specific hardware is available.

**3. Model format origin**: PyTorch, TensorFlow, JAX, raw ONNX, GGUF, something custom. What the model is when it arrives at the runtime.

**4. Latency / throughput / footprint constraints**: real-time camera processing (10 ms budget), interactive LLM (200 ms TTFT), background batch (minutes per inference). Runtime binary size limits (50 MB for an app store install, 5 MB for an embedded device).

**5. Developer ecosystem**: Swift app, Java/Kotlin app, C++ embedded, Python research, JavaScript browser. The language you actually write code in.

Different runtimes optimize for different combinations. The "right" runtime is a function of where you sit on these axes.

## The serious runtimes in 2026

A roster, with one-line summaries:

| Runtime | Maintainer | Primary platforms | Model format |
| ------- | ---------- | ----------------- | ------------ |
| **Core ML** | Apple | iOS, iPadOS, macOS, watchOS, visionOS | `.mlpackage`, `.mlmodel` |
| **MLX** | Apple | macOS, iOS (preview) | MLX-native; loads from GGUF |
| **LiteRT** (TFLite) | Google | Android, iOS, Linux, embedded | `.tflite` |
| **ONNX Runtime (ORT)** | Microsoft | every major platform | ONNX (`.onnx`) |
| **ExecuTorch** | Meta | iOS, Android, Linux, embedded | `.pte` |
| **QNN SDK** | Qualcomm | Android with Snapdragon | QNN context binary |
| **llama.cpp / ggml** | Georgi Gerganov + community | macOS, Linux, Windows, Android, iOS | GGUF |
| **MediaPipe Tasks** | Google | Android, iOS, web | wraps LiteRT |
| **TensorFlow Lite Micro** | Google | microcontrollers | reduced `.tflite` subset |
| **NCNN** | Tencent | mobile (Android/iOS), embedded | NCNN format |
| **MNN** | Alibaba | mobile, embedded | MNN format |

The first eight are the ones you'll meet in production deployments. NCNN and MNN are widely used in Chinese-market deployments; less common elsewhere. TF Lite Micro is the microcontroller path.

## The vendor incentives matter

Each runtime is shaped by its maintainer's commercial interest, which is worth knowing because it affects long-term direction:

- **Apple (Core ML, MLX)**: maximize use of Apple's hardware, including the ANE. Excellent on Apple platforms; doesn't exist on Android. The "Apple deployment" path.
- **Google (LiteRT, MediaPipe)**: cover Android primarily, iOS as a courtesy. Treats NNAPI (Google's Android NPU abstraction) as the strategic accelerator layer.
- **Microsoft (ONNX Runtime)**: covers everything that runs Microsoft software (Windows, Office, Edge, etc.). The breadth-first option.
- **Meta (ExecuTorch)**: PyTorch's edge story. Strong incentive to make PyTorch models deploy without conversion friction.
- **Qualcomm (QNN)**: maximize use of Qualcomm SoCs (Hexagon NPU + Adreno GPU). Production-grade for Snapdragon Android; doesn't matter elsewhere.
- **Community (llama.cpp / ggml)**: maximize LLM inference quality and reach across all platforms. No vendor allegiance; tracks what works on real hardware.

The pattern: a runtime maintained by a hardware vendor (Apple, Qualcomm) is great on that vendor's hardware and absent elsewhere. A runtime maintained by a software vendor (Google, Microsoft, Meta) tries to cover more ground but may not hit peak performance on any specific accelerator. The community runtime (llama.cpp) chases peak performance on every platform that has motivated contributors.

## The platform matrix

Which runtimes credibly target which platforms (2026):

| Platform | Primary | Secondary | Specialized |
| -------- | ------- | --------- | ----------- |
| **iOS** | Core ML | ONNX Runtime, ExecuTorch, MLX | llama.cpp (for LLMs) |
| **macOS** | MLX, Core ML | ONNX Runtime, ExecuTorch | llama.cpp |
| **Android (Qualcomm)** | LiteRT (with QNN delegate) | ONNX Runtime, ExecuTorch | llama.cpp |
| **Android (other SoCs)** | LiteRT | ONNX Runtime, ExecuTorch | llama.cpp |
| **Windows** | ONNX Runtime | ExecuTorch, llama.cpp | DirectML EP |
| **Linux server** | ONNX Runtime, vLLM, TensorRT-LLM | ExecuTorch | llama.cpp |
| **Linux embedded (ARM)** | ONNX Runtime | LiteRT, ExecuTorch | llama.cpp, NCNN |
| **Microcontroller** | TF Lite Micro | — | — |
| **Browser** | ONNX Runtime Web, TFJS | — | llama.cpp via WASM |
| **visionOS** | Core ML | — | — |

The dominant pattern: on each major mobile OS, the platform vendor's runtime is "default" and the cross-platform runtimes are credible alternatives. For LLMs specifically, llama.cpp is the cross-platform constant.

## The model-format matrix

The model formats and what each runtime expects:

| Format | Native runtime(s) | Conversion target for |
| ------ | ----------------- | --------------------- |
| **ONNX** | ONNX Runtime | almost everyone (most runtimes can import ONNX) |
| **GGUF** | llama.cpp, MLX | LLMs across runtimes |
| **Core ML (`.mlpackage`)** | Core ML | Apple-only deployment |
| **LiteRT (`.tflite`)** | LiteRT, MediaPipe | Android-primary deployment |
| **ExecuTorch (`.pte`)** | ExecuTorch | PyTorch-origin models targeting edge |
| **QNN context binary** | QNN runtime | Hexagon-specific deployment |
| **Raw PyTorch / TF / JAX** | the framework's mobile path | starting point for conversion |

Conversion paths in 2026:

- **PyTorch → ONNX**: `torch.onnx.export`. Works for most architectures. Some recent ops (FlashAttention variants, custom kernels) may need manual handling.
- **PyTorch → Core ML**: `coremltools.convert`. Smooth path for standard architectures.
- **PyTorch → ExecuTorch (`.pte`)**: native PyTorch path; `torch.export` + `executorch.exir`.
- **PyTorch → LiteRT**: via ONNX → tf2lite typically. Multi-step; sometimes lossy.
- **TensorFlow → LiteRT**: native.
- **HF Transformers → GGUF**: `convert_hf_to_gguf.py` from llama.cpp.
- **HF Transformers → MLX**: `mlx_lm.convert`.
- **Anything → QNN**: typically TF Lite or ONNX as intermediate; QNN's own conversion tools.

The chain-of-conversions failure mode: every conversion can drop or rename an op, change the precision of an intermediate, or fail outright on an unsupported pattern. The fewer conversions you do, the fewer surprises. This is one reason "PyTorch → Core ML (one step)" tends to be more reliable than "PyTorch → ONNX → Core ML (two steps)."

## The "what should I default to" cheatsheet

For the most common 2026 deployment scenarios:

- **iOS app, vision/audio model**: Core ML.
- **iOS app, LLM**: llama.cpp via the swift bindings, or MLX via Swift integration.
- **Android app, vision/audio model**: LiteRT with NNAPI / QNN delegate.
- **Android app, LLM**: llama.cpp via NDK.
- **Both platforms, vision/audio**: ExecuTorch (PyTorch backend on Core ML for iOS, XNNPACK + QNN for Android) — one source tree, two backends.
- **Both platforms, LLM**: llama.cpp — the de-facto cross-platform LLM runtime.
- **Browser, vision model**: ONNX Runtime Web with the WebGPU EP, or TFJS.
- **Browser, LLM**: WebLLM (MLC-LLM), llama.cpp via WASM.
- **Embedded Linux (Raspberry Pi class)**: ONNX Runtime for general; llama.cpp for LLMs.
- **Microcontroller**: TF Lite Micro.
- **Linux server (production inference, not training)**: vLLM, TensorRT-LLM, ONNX Runtime, depending on model and stack.

These are "where to start"; the rest of the module is "when to pick differently."

## What you should believe after this lesson

Three sentences:

**1. The edge ML runtime landscape splits along five axes** — target platform, target accelerator, model format origin, latency/throughput/footprint, developer ecosystem — and the right runtime is a function of where you sit on these axes, not a single universal best choice.

**2. Vendor incentives shape runtime direction**: hardware-vendor runtimes (Apple's Core ML, Qualcomm's QNN) are excellent on their hardware and don't exist elsewhere; software-vendor runtimes (Google's LiteRT, Microsoft's ORT, Meta's ExecuTorch) cover more platforms but may not peak on any specific accelerator; the community runtime (llama.cpp) chases peak performance everywhere for LLMs.

**3. The cheatsheet of common 2026 deployments resolves to ~5 runtimes** — Core ML, LiteRT, ONNX Runtime, ExecuTorch, llama.cpp — with the rest serving specialized niches. Those five are what the rest of this module covers in depth.

## Hands-on (at home)

Inventory the runtimes available on your dev machine.

```bash
# inventory.sh
echo "=== Python ML runtimes ==="
for pkg in coremltools onnxruntime executorch tensorflow ai_edge_litert mlx mlx-lm llama-cpp-python; do
    if python -c "import $(echo $pkg | tr - _)" 2>/dev/null; then
        ver=$(python -c "import $(echo $pkg | tr - _) as m; print(getattr(m, '__version__', '?'))")
        printf "  %-25s installed (%s)\n" "$pkg" "$ver"
    else
        printf "  %-25s NOT installed\n" "$pkg"
    fi
done

echo ""
echo "=== Native binaries ==="
for bin in llama-cpp llama-server ollama llama-bench main; do
    if command -v $bin >/dev/null 2>&1; then
        printf "  %-25s found at %s\n" "$bin" "$(command -v $bin)"
    else
        printf "  %-25s NOT found in PATH\n" "$bin"
    fi
done

echo ""
echo "=== Platform identification ==="
uname -a
echo ""
sw_vers 2>/dev/null || cat /etc/os-release 2>/dev/null
```

Run on your machine. The output is the "starting kit" for the rest of the module; the lessons assume access to several of these, and missing entries highlight what to install before the corresponding lesson's hands-on.

Part 2 — read one model file from each major format to feel the structure.

```python
# inspect_formats.py
import os

# ONNX: protobuf, human-readable with onnx.helper.printable_graph.
try:
    import onnx
    m = onnx.load("path/to/model.onnx")
    print("ONNX model:", m.graph.name, "with", len(m.graph.node), "nodes")
except FileNotFoundError:
    print("ONNX: no test model available")

# GGUF: binary, llama.cpp's gguf-py library reads it.
try:
    from gguf import GGUFReader
    r = GGUFReader("path/to/model.gguf")
    print("GGUF model:", len(r.tensors), "tensors,",
          "architecture:", r.fields.get('general.architecture'))
except Exception as e:
    print(f"GGUF: {e}")

# Core ML: .mlpackage is a directory; the manifest tells you the structure.
mlpkg = "path/to/model.mlpackage"
if os.path.exists(mlpkg):
    for f in os.listdir(mlpkg):
        print(f"Core ML: {f}")
```

The point isn't to write production code; it's to see that each format is a different beast, and the "model file" abstraction hides this only if you let it.

## Further reading

- "Edge AI Survey" (Microsoft Research, various years) — for the academic-style landscape.
- ONNX, Core ML, ExecuTorch, LiteRT, GGUF official documentation — each has a "getting started" page that doubles as a runtime overview.
- "MLPerf Edge" benchmarks — for cross-runtime performance numbers on common models.
- "On-device AI" blog posts from Google, Apple, Qualcomm — vendor perspectives, useful for understanding incentives.

Next lesson: **Model formats.** We go deep on the formats themselves — ONNX, GGUF, Core ML, LiteRT, ExecuTorch — including conversion paths between them and what gets lost in each translation. The right format choice is upstream of the runtime choice; getting it wrong forces conversion hell.
