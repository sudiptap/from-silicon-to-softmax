---
title: "Lesson 7 — ONNX Runtime Mobile"
date: "2026-06-04"
module: "mobile-edge-runtimes"
order: 7
tags: ["onnx-runtime", "ort-mobile", "cross-platform", "binary-size", "execution-providers"]
author: "Sudipta Pathak"
prerequisites: ["06-litert"]
---

# Lesson 7 — ONNX Runtime Mobile

## Why this lesson exists

ONNX Runtime appeared in Lesson 4 in the Apple context. This lesson is ORT in its full cross-platform role: the option you reach for when you need one inference runtime that works on iOS, Android, Windows, Linux, embedded ARM, *and* the browser, with reasonable performance everywhere. ORT's strength is breadth. It's rarely the absolute fastest on any specific platform but it's competitive everywhere, and the same model file and the same API work across the stack.

The mobile-specific story is ORT Mobile — a stripped-down build of the runtime that trims binary size from ~30 MB (full) to ~8 MB by removing rarely-used ops and optimization paths. ORT Mobile is the production-ready cross-platform alternative to ExecuTorch and LiteRT in 2026.

The lesson is reading. The Hands-on builds an ORT inference pipeline that targets multiple execution providers on the same dev machine.

## ORT Mobile vs ORT (full)

The full ORT binary supports every op in the ONNX standard catalog plus several extensions, every execution provider, and full optimization. For desktop / server use this is fine — 30 MB doesn't matter when you have gigabytes of process memory.

For mobile apps, 30 MB of inference runtime is a hard sell when your app's total size budget might be 50 MB. ORT Mobile addresses this:

- **Reduced op set**: only ops in the "mobile ops" subset are bundled. Models using non-mobile ops fail to load. The conversion tooling has a "mobile-friendly check."
- **Reduced optimization passes**: the heaviest optimizations are stripped. The optimizer is simpler and smaller.
- **Selective EP support**: only the EPs you bundle are included. A typical mobile build has CPU + XNNPACK + the device-specific EP (Core ML on iOS, NNAPI/QNN on Android).

Sizes:
- ORT Mobile, CPU only: ~5 MB.
- ORT Mobile + XNNPACK: ~7 MB.
- ORT Mobile + Core ML EP (iOS): ~12 MB.
- ORT Mobile + QNN EP (Android): ~15 MB.

You can customize further with a "minimal build" — strip per-op overhead even more — for embedded scenarios where 5 MB is too large.

The mobile-ops check during conversion:

```python
import onnxruntime.tools.check_model_can_use_ort_mobile_pkg as checker
checker.check_model("model.onnx", "1.20", required_ops_config="mobile_ops.config")
# Errors if the model uses ops outside the mobile op set.
```

For most standard models (CNNs, transformers), the mobile op set is sufficient. For recent research architectures with novel ops, the full build is sometimes needed.

## Execution providers cross-platform

ORT's EP catalog spans every major platform:

| EP | Platforms | Accelerator |
| -- | --------- | ----------- |
| CPUExecutionProvider | all | CPU, ORT's own kernels |
| XnnpackExecutionProvider | all | CPU, XNNPACK kernels (often faster than CPU EP) |
| CoreMLExecutionProvider | iOS, macOS | Core ML, including ANE |
| NnapiExecutionProvider | Android | NNAPI (deprecated path) |
| QNNExecutionProvider | Android (Snapdragon) | Qualcomm Hexagon |
| DmlExecutionProvider | Windows | DirectML (Windows GPU abstraction) |
| OpenVINOExecutionProvider | Linux, Windows | Intel CPU/GPU/NPU |
| TensorrtExecutionProvider | Linux, Windows | NVIDIA TensorRT |
| CUDAExecutionProvider | Linux, Windows | NVIDIA CUDA |
| ROCmExecutionProvider | Linux | AMD GPU |
| WebGPUExecutionProvider | browser | WebGPU |
| WebGLExecutionProvider | browser | WebGL (older) |

The choice is per-platform. A typical mobile deployment lists 2–3 providers in priority order:

```python
# Android with Snapdragon:
providers = [
    ('QNNExecutionProvider', {'backend_path': 'libQnnHtp.so'}),
    'XnnpackExecutionProvider',
    'CPUExecutionProvider',
]

# Android, generic:
providers = [
    'NnapiExecutionProvider',  # works on any Android NPU via NNAPI
    'XnnpackExecutionProvider',
    'CPUExecutionProvider',
]

# iOS:
providers = [
    ('CoreMLExecutionProvider', {'MLComputeUnits': 'ALL'}),
    'XnnpackExecutionProvider',
    'CPUExecutionProvider',
]
```

ORT tries each in order for each op; the first EP that supports the op wins.

## When ORT is the right answer

The clear strong cases:

**1. Code-sharing across platforms.** One `model.onnx` + one `onnxruntime` dependency + the same Python (or C++ / Java / Swift) API across iOS, Android, Windows, Linux. The cross-platform inference codebase that's hardest to achieve with other runtimes.

**2. Models from non-PyTorch / non-TF sources.** scikit-learn, XGBoost, LightGBM all export to ONNX; ORT runs them. This matters when you have classical ML alongside deep learning.

**3. Server-side inference where TensorRT-LLM is overkill.** ORT's CUDA EP delivers competitive throughput for medium-traffic LLM serving without TensorRT-LLM's complexity.

**4. Browser deployment.** ORT Web with WebGPU EP is the production browser inference path in 2026.

**5. Microsoft ecosystem.** Windows apps, Office, Edge — anything that uses Microsoft's stack is ORT-default.

The weak cases:

**1. Apple-only mobile apps with vision/audio.** Core ML is more direct.
**2. Android-only apps targeting specific NPUs.** LiteRT with vendor delegate is more direct.
**3. LLM serving on NVIDIA at scale.** TensorRT-LLM is faster.
**4. PyTorch-native cross-platform.** ExecuTorch is more PyTorch-idiomatic.

## Performance positioning

ORT is rarely the fastest on any specific platform; it's almost always within 10-20% of the platform-native runtime. This is the cross-platform tax. For most production deployments the gap is acceptable; the alternative (maintaining multiple platform-specific inference codebases) costs more in engineering time than the 15% throughput would save.

The notable exceptions:

- **ORT with TensorRT EP on Linux+NVIDIA**: matches or beats native TensorRT calls (because that's what's underneath).
- **ORT with OpenVINO EP on Intel**: similar — competitive with Intel-native paths.
- **ORT on CPU**: often the fastest CPU inference for non-transformer models, thanks to aggressive load-time optimization.

The pattern: ORT is at its best when an EP wraps a vendor's own runtime; the EP overhead is small and you get vendor-native performance through ORT's portable API.

## A note on ORT Web

For browser deployment, ORT Web is the runtime to know. It runs ONNX models via WebAssembly (CPU path) or WebGPU (GPU path). Binary size is comparable to ORT Mobile after gzip.

The WebGPU EP path is, in 2026, surprisingly fast — competitive with native mobile ORT for many models. The "ML in the browser" story has become real: deployed in Edge for image enhancement, in Chrome for built-in features, in many third-party web apps.

We won't go deep on browser deployment in this module — it's adjacent but enough material for its own treatment.

## What you should believe after this lesson

Three sentences:

**1. ONNX Runtime is the cross-platform breadth runtime** — one model file, one API, every platform from iOS to embedded Linux to browser. The mobile build (~8–15 MB) is what apps ship; the full build is for desktop / server.

**2. The right ORT use case is code-sharing across platforms** where you'd otherwise maintain separate Core ML, LiteRT, and ORT codebases. The cross-platform tax is ~15% performance vs platform-native runtimes; for most deployments this is well worth the engineering savings.

**3. ORT is rarely the fastest** on any specific platform but is competitive everywhere. Reach for it when breadth matters more than peak performance; reach for the platform-native runtime when peak performance on one platform is the priority.

## Hands-on (at home)

Build an ORT inference pipeline that targets multiple EPs on the same dev machine.

```python
# ort_multi_ep.py
# pip install onnxruntime onnx torch torchvision
import onnxruntime as ort
import numpy as np
import torch
import torchvision.models as models
import time

# 1. Export a model to ONNX.
m = models.efficientnet_b0(weights="DEFAULT").eval()
x = torch.rand(1, 3, 224, 224)
torch.onnx.export(m, x, "eff.onnx", opset_version=17,
                  input_names=["input"], output_names=["output"])

# 2. List what EPs are available on this build.
print("Available EPs:", ort.get_available_providers())

# 3. Run with each in turn (those available on your platform).
input_arr = np.random.rand(1, 3, 224, 224).astype(np.float32)
results = {}

def bench(providers, name):
    sess = ort.InferenceSession("eff.onnx", providers=providers)
    actual = sess.get_providers()
    for _ in range(5):
        sess.run(None, {"input": input_arr})
    t0 = time.time()
    for _ in range(50):
        sess.run(None, {"input": input_arr})
    dt = (time.time() - t0) / 50
    print(f"  {name:30s}: {dt*1000:6.2f} ms (active: {actual})")

bench(['CPUExecutionProvider'], "CPU")
if 'CoreMLExecutionProvider' in ort.get_available_providers():
    bench([('CoreMLExecutionProvider', {'MLComputeUnits': 'ALL'}), 'CPUExecutionProvider'],
          "Core ML (ANE+GPU+CPU)")
if 'CUDAExecutionProvider' in ort.get_available_providers():
    bench(['CUDAExecutionProvider', 'CPUExecutionProvider'], "CUDA")
if 'CoreMLExecutionProvider' not in ort.get_available_providers() and 'CUDAExecutionProvider' not in ort.get_available_providers():
    print("  (no GPU/ANE EPs available; running CPU only)")
```

You'll see the ORT story play out: CPU is slow, the device-specific EP is fast, the EP overhead is small.

For mobile-specific testing, the ORT mobile binary needs to be deployed to a device. The pattern is the same; the language is Swift / Kotlin / Java; the Python pipeline above is the dev-machine equivalent.

## Further reading

- ONNX Runtime documentation (onnxruntime.ai) — the full reference.
- "ONNX Runtime Mobile" docs — for the mobile build options and ops subset.
- "ORT execution providers" reference — for the EP catalog and configuration options.
- "ONNX Runtime Web" docs — for the browser deployment path.
- Performance benchmarks at github.com/microsoft/onnxruntime/blob/main/docs/PERFORMANCE.md — for canonical performance numbers across EPs.

Next lesson: **Qualcomm Hexagon NPU + QNN SDK.** We zoom in on the dominant Android NPU and its first-party SDK. Snapdragon phones are the majority of premium Android; targeting Hexagon is how you get NPU acceleration on that hardware. The QNN SDK is the path; LiteRT and ORT both have QNN delegates that wrap it.
