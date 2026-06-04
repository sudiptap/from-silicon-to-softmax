---
title: "Lesson 8 — Qualcomm Hexagon NPU + QNN SDK"
date: "2026-06-04"
module: "mobile-edge-runtimes"
order: 8
tags: ["qualcomm", "hexagon", "qnn", "npu", "snapdragon", "ai-hub", "android"]
author: "Sudipta Pathak"
prerequisites: ["07-ort-mobile"]
---

# Lesson 8 — Qualcomm Hexagon NPU + QNN SDK

## Why this lesson exists

Most flagship Android phones use Qualcomm Snapdragon SoCs, and every Snapdragon since the 600-series in the mid-2010s has had a Hexagon DSP that doubles as an ML accelerator. The Hexagon NPU on a Snapdragon 8 Gen 3 (2024) is rated at ~30 TOPS INT8 — comparable to Apple's ANE on M3-class. The Hexagon NPU on a Snapdragon 8 Gen 4 (2025) is meaningfully faster. If you're shipping ML on Android and you care about per-token-per-watt or per-image latency, the Hexagon path is the one to know.

This lesson is the Qualcomm story: what Hexagon is at the hardware level, how the QNN SDK (Qualcomm Neural Network) wraps it, the AI Hub tooling, and how QNN composes with the cross-platform runtimes (LiteRT, ONNX Runtime, ExecuTorch) that delegate to it.

The lesson is reading. The Hands-on uses the AI Hub to compile a model to Hexagon and inspect the result. (Requires a developer account; free.)

## Hexagon hardware

Hexagon began life as a DSP (digital signal processor) for cellular modem work — fast multiplication and signed arithmetic for radio signal processing. Over the 2010s, Qualcomm extended it with HVX (Hexagon Vector eXtensions) for SIMD work, then HTA (Hexagon Tensor Accelerator) and more recently HMX (Hexagon Matrix eXtensions) — the matrix-multiply units that are the ML-relevant part.

On a Snapdragon 8 Gen 3 (typical 2024 flagship):
- **Hexagon DSP**: scalar / vector unit, runs the program control flow.
- **HVX**: wide vector unit, ~1024-bit SIMD per core.
- **HMX**: matrix accelerator, tile-based matmul at INT8 / INT16 / FP16. The heavyweight ML throughput unit.
- **HTA**: legacy tensor accelerator from earlier generations; less prominent now.
- **VTCM (Vector Tightly-Coupled Memory)**: programmer-managed scratchpad, ~8 MB. Analog of CUDA shared memory or Apple threadgroup memory.

The Hexagon NPU is one or more Hexagon cores plus HVX plus HMX, working together. The HMX is what delivers the TOPS number; HVX handles the non-matmul ops; the DSP runs the program control.

Like Apple's ANE, the Hexagon NPU isn't a general-purpose compute device. It runs a specific instruction set; you reach it through a programming SDK; the runtime decides whether an op fits. Unlike the ANE, the Hexagon programming surface is more flexible — you can write custom kernels in C++ for it via the QNN SDK, where you can't write custom code for the ANE.

## QNN SDK

QNN (Qualcomm Neural Network) is Qualcomm's ML SDK. It provides:

- **A graph compiler**: takes a model (typically ONNX or TFLite as input) and produces a QNN context binary specialized for a specific Snapdragon SoC.
- **A runtime**: loads the context binary and dispatches it to Hexagon (or Adreno GPU, or CPU as fallback).
- **A backend selection layer**: HTP (Hexagon Tensor Processor — the modern HMX-driven path), HTA (legacy), GPU (Adreno), CPU.

The compilation step is per-target. A QNN binary compiled for Snapdragon 8 Gen 3 won't necessarily run optimally on Snapdragon 8 Gen 4 or on a mid-range Snapdragon 7. You typically compile multiple versions and select at app load time based on the device.

The compilation is also where most quantization decisions get baked in. QNN expects INT8 (per-tensor or per-channel) for HTP-targeted ops; the compiler converts FP16 ops to INT8 with calibration data if you ask, or assumes the model is already quantized.

A typical QNN compilation workflow:

```bash
# Convert ONNX to QNN model representation.
qnn-onnx-converter --input_network model.onnx --output_path model_qnn.cpp

# Compile to context binary for HTP (Hexagon).
qnn-context-binary-generator \
    --backend libQnnHtp.so \
    --model model_qnn.so \
    --binary_file model.bin
```

The result, `model.bin`, is the QNN context binary you ship in the app and load at runtime.

## QNN via LiteRT and ORT delegates

You rarely call QNN directly from app code. Instead, LiteRT and ORT each have a QNN delegate / EP that handles QNN integration:

**LiteRT QNN delegate**:
```kotlin
val qnnOptions = QnnDelegate.Options()
    .setBackendType(BackendType.HTP)
    .setHtpOptions(...)
val delegate = QnnDelegate(qnnOptions)
val interpreter = Interpreter(modelFile, Interpreter.Options().addDelegate(delegate))
```

**ORT QNN EP**:
```python
sess = ort.InferenceSession(
    "model.onnx",
    providers=[
        ('QNNExecutionProvider', {
            'backend_path': 'libQnnHtp.so',
            'profiling_level': 'basic',
        }),
        'CPUExecutionProvider',
    ],
)
```

Either way, the delegate / EP handles model loading, the QNN-specific compilation (cached after first run), and the dispatch at inference time. You write standard LiteRT or ORT code and add a few QNN-specific options.

## Qualcomm AI Hub

AI Hub is Qualcomm's web-based / SDK-based service for compiling and benchmarking models for Snapdragon. The pitch: upload a model, get back a QNN-compiled binary plus benchmark numbers for any Snapdragon SoC you want.

The workflow:

1. Have an ONNX or PyTorch model.
2. Use the AI Hub Python SDK to submit it:
   ```python
   import qai_hub as hub
   model = hub.upload_model("model.onnx")
   target = hub.Device(name="Samsung Galaxy S24 (Family)")
   compile_job = hub.submit_compile_job(model=model, device=target,
                                         input_specs=dict(image=(1, 3, 224, 224)))
   compiled = compile_job.get_target_model()
   ```
3. Get back a compiled `.so` (Android library) or `.bin` (QNN context binary).
4. Optionally, run benchmarks: `hub.submit_profile_job(...)` returns latency numbers per op.

AI Hub abstracts away the QNN SDK setup; it's the "easy mode" for getting Snapdragon-optimized models. The catch: it's a Qualcomm service (cloud-based compilation, with usage limits on free tier). For production deployments where you want reproducible builds in-house, the direct QNN SDK path is what you'd use.

AI Hub also publishes pre-compiled versions of many open models (vision, audio, small LLMs) on their model catalog. For prototyping, downloading a pre-compiled model is the fastest path.

## Hexagon for vision vs LLMs

The pattern from Apple's ANE applies here too:

**Vision models**: convolutions, batch norms, ReLU/GeLU. Hexagon eats these for breakfast. A typical Hexagon vision model (MobileNet, EfficientNet, YOLO) runs 5–10× faster than the CPU path and 2–3× faster than the Adreno GPU path, at much lower power.

**Audio models**: Whisper-class encoders, conv-heavy structures. Hexagon wins clearly.

**LLMs**: similar challenges as Apple's ANE. Most recent LLM ops have Hexagon support (Qualcomm has invested heavily here for the on-device LLM story they're marketing). But batch-1 decoding still suffers from kernel-launch overhead and the matmul shapes don't tile efficiently. Hexagon LLM decode is competitive with Adreno GPU LLM decode in 2026, with Hexagon winning on power and Adreno often winning on raw throughput.

The 2026 state: shipping a small LLM (1B-class) on Snapdragon via QNN/Hexagon is viable. Qualcomm's own demos and Microsoft's "AI PCs" (Snapdragon X Elite laptops) lean on this. For larger models (7B+) or for the absolute best throughput, the Adreno GPU path via OpenCL or Vulkan can match or beat Hexagon. For battery-constrained scenarios (phones), Hexagon usually wins overall.

## A note on Adreno (the Snapdragon GPU)

Snapdragon devices also have an Adreno GPU, which is the OpenGL ES / Vulkan path. For ML, Adreno is reachable via:
- LiteRT GPU delegate (OpenGL ES compute).
- ORT WebGPU or DirectML on Windows-Snapdragon devices.
- Vulkan compute directly.
- Hand-written OpenCL.

Adreno isn't bad — it's a competent mobile GPU, ~3–5 TFLOPs FP16 on flagship Snapdragons. For ML workloads, Adreno is the second-choice acceleration path on Snapdragon, behind Hexagon for the workloads Hexagon supports.

The hierarchy on a Snapdragon device for ML throughput per watt: Hexagon > Adreno > CPU. For throughput peak (when power doesn't matter): Adreno can equal or beat Hexagon on shapes the latter can't tile well.

## What you should believe after this lesson

Three sentences:

**1. The Hexagon NPU is Qualcomm's first-party ML accelerator** — combining DSP + HVX (vector) + HMX (matrix) + VTCM (scratchpad). Production-grade on every modern Snapdragon, comparable to Apple's ANE in throughput, more flexible (you can write custom Hexagon C++) but still less programmable than a GPU.

**2. The QNN SDK is the path to Hexagon**, but you rarely call it directly — LiteRT's QNN delegate and ORT's QNN EP wrap it cleanly. The compilation step bakes in the Hexagon-specific optimization; the result is a context binary specialized for a particular Snapdragon SoC family.

**3. Qualcomm AI Hub is the "easy mode" workflow** — upload a model, get back a compiled binary and benchmark numbers across Snapdragon devices. For prototyping it's the fastest path; for production builds, the direct QNN SDK path is what you'd use.

## Hands-on (at home)

The most accessible Hexagon-on-the-side experiment is via AI Hub, which is free for limited use and doesn't require a physical Snapdragon device.

```bash
pip install qai-hub
qai-hub configure --api_token YOUR_TOKEN  # signup at aihub.qualcomm.com
```

Submit a model:

```python
# aihub_demo.py
import qai_hub as hub
import torch
import torchvision.models as models

model = models.mobilenet_v3_small(weights="DEFAULT").eval()
example = torch.rand(1, 3, 224, 224)
traced = torch.jit.trace(model, example)

# Submit to AI Hub.
target = hub.Device(name="Samsung Galaxy S24 (Family)")
compile_job = hub.submit_compile_job(
    model=traced,
    device=target,
    input_specs={"image": (1, 3, 224, 224)},
)
print(f"Compile job: {compile_job.url}")

# Wait for completion and download.
compiled = compile_job.get_target_model()
print(f"Compiled model: {compiled.url}")

# Optionally, benchmark on the cloud-attached device.
profile_job = hub.submit_profile_job(model=compiled, device=target)
print(f"Profile job: {profile_job.url}")
profile = profile_job.download_profile()
print(f"Estimated inference time: {profile['execution_summary']['estimated_inference_time']} us")
```

AI Hub runs the model on actual hardware (one of their device farm Snapdragons) and returns timing data. You'll see per-op breakdown, total latency, and which ops ran on Hexagon HTP vs Adreno GPU vs CPU.

For on-device QNN work without AI Hub, you need:
- A Snapdragon Android device with developer mode enabled.
- The QNN SDK from Qualcomm's developer portal.
- LiteRT or ORT with QNN delegate / EP enabled.

The full local QNN workflow is involved; the AI Hub path is recommended for first exposure.

## Further reading

- Qualcomm QNN SDK documentation (developer.qualcomm.com) — the official SDK reference.
- "Qualcomm AI Hub" documentation and model zoo — for the cloud workflow.
- "Hexagon Processor Documentation" — for the architecture details (older but still informative).
- "Snapdragon Compute" — Qualcomm's PC-class platform docs; relevant for the Snapdragon X Elite laptops.
- Microsoft's "AI PCs" docs — for the Windows on Snapdragon ML story.

Next lesson: **llama.cpp / ggml.** The de facto on-device LLM runtime in 2026. We look at why it ate the small-LLM serving niche, how the GGUF format and ggml backend abstraction compose, and the production deployment patterns it enables across Mac, Linux, Windows, Android, iOS — making it the LLM-specific cross-platform runtime that complements the more general runtimes in the rest of the module.
