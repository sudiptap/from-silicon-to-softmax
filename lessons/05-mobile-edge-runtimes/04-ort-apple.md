---
title: "Lesson 4 — ONNX Runtime on iOS/macOS"
date: "2026-06-04"
module: "mobile-edge-runtimes"
order: 4
tags: ["onnx-runtime", "ort", "execution-providers", "coreml-ep", "apple"]
author: "Sudipta Pathak"
prerequisites: ["03-coreml-deep-dive"]
---

# Lesson 4 — ONNX Runtime on iOS/macOS

## Why this lesson exists

ONNX Runtime (ORT) is Microsoft's cross-platform inference engine for ONNX models. It's available on Apple platforms — iOS, macOS, and visionOS — and is a credible alternative to Core ML in specific scenarios: when your model originated outside Apple's PyTorch / Core ML pipeline, when you want a runtime that's also deployed elsewhere (Android, Linux, Windows) for code-sharing reasons, or when ORT happens to have better support for an op your model needs.

The unifying concept that makes ORT interesting on Apple platforms is the **execution provider** (EP) abstraction. ORT itself is a graph executor; the actual kernels come from EPs. On Apple, ORT can use a CPU EP (its own kernels), an MPS EP (uses Apple's Metal Performance Shaders), or a Core ML EP (delegates to Core ML, which itself can route to the ANE). The composability lets you choose your tradeoffs.

The lesson is reading. The Hands-on runs the same ONNX model through ORT with different execution providers and compares.

## What ORT is

ONNX Runtime is a C++ inference engine with bindings in many languages (Python, Java, C#, JavaScript, Swift, Objective-C, Rust). The core is small (~10 MB binary on iOS); EPs add code for the specific accelerator backends.

ORT consumes ONNX format files. It walks the graph at load time, optimizes it (op fusion, constant folding, layout transformation), and assigns each node to an execution provider based on the EP priorities you specify.

The optimization passes are aggressive — far more than what most other runtimes do at load time. Specific passes include:
- Constant folding (replace expressions with their precomputed values).
- Op fusion (combine matmul + bias + activation into a single kernel where the EP supports it).
- Layout optimization (rearrange tensor layouts to match the EP's preferred order).
- Dead code elimination (drop ops whose outputs aren't used).
- Memory planning (pre-allocate tensor buffers to minimize allocation churn at run time).

These passes are what make ORT often faster than naive ONNX evaluation. They take a few hundred milliseconds at load time; you pay this cost once per model load.

## Execution providers on Apple

The EPs available on iOS/macOS:

**CPU EP** — ORT's own CPU kernels. Always available. Uses Accelerate.framework on Apple (which hits AMX for matmul on M-series). Reasonable performance for small models.

**MPS EP** — Uses Metal Performance Shaders for GPU acceleration. Decent performance on Apple GPU; doesn't reach the ANE. Older than the Core ML EP and somewhat less actively maintained.

**Core ML EP** — Delegates supported subgraphs to Core ML. The Core ML compiler then routes those subgraphs to ANE/GPU/CPU as it would for a native Core ML model. **This is the EP to use when you want ANE acceleration via ORT.**

**XNNPACK EP** — Cross-platform optimized CPU kernels. Available on Apple but mostly relevant on Android.

The EP priority list determines which EP gets which op. A typical Apple configuration:

```python
sess = ort.InferenceSession(
    "model.onnx",
    providers=[
        ('CoreMLExecutionProvider', {
            'ModelFormat': 'MLProgram',
            'MLComputeUnits': 'ALL',
        }),
        'CPUExecutionProvider',
    ],
)
```

ORT tries Core ML first; ops Core ML can't handle fall through to the CPU EP. The fallback is automatic (similar to Core ML's own ANE → GPU → CPU fallback).

## When ORT beats Core ML on Apple

A few scenarios where ORT is the better choice:

**1. The model originated as ONNX and Core ML conversion is broken.** Some PyTorch models export cleanly to ONNX but fail to convert to Core ML (op coverage gaps in `coremltools`). ORT lets you run the ONNX directly.

**2. You're sharing code across platforms.** ORT runs on iOS, Android, Windows, Linux, browser. If you want one inference codebase that works everywhere, ORT is the obvious choice. Core ML doesn't exist off Apple.

**3. You're integrating with an existing C++ inference pipeline.** ORT's C++ API is well-designed and stable; integrating a Core ML call from C++ is awkward (Objective-C bridging).

**4. The model uses ops with better ORT support than Core ML support.** Some specific transformer ops, some recent attention variants — ORT often catches up to new ops faster than Core ML.

**5. You want predictable behavior across OS versions.** Core ML's behavior can change between OS versions (the compiler updates, ANE firmware changes). ORT's behavior is locked to the ORT version you bundle.

When Core ML beats ORT on Apple: most of the time for app deployment, especially when ANE acceleration matters and the model has been converted cleanly. The ANE wins from a native Core ML model usually exceed what you get from ORT-via-CoreML-EP because ORT adds layers of abstraction.

## ORT on iOS specifically

Bundling ORT into an iOS app:

1. Add the `onnxruntime-objc` or `onnxruntime-c` framework to the app (via CocoaPods, SPM, or direct).
2. Include the ONNX model file as an app resource.
3. In Swift or Objective-C:
   ```swift
   import onnxruntime_objc
   let env = try ORTEnv(loggingLevel: .warning)
   let opts = try ORTSessionOptions()
   try opts.appendCoreMLExecutionProvider(with: .init())
   let session = try ORTSession(env: env, modelPath: modelPath, sessionOptions: opts)
   // Build input tensor, call session.run(...), parse output.
   ```

The binary size impact: ~15–25 MB depending on which EPs you bundle. Compared to Core ML's ~0 MB (it's part of iOS), this is the cost of ORT's cross-platform story.

## ORT mobile vs ORT (full)

ORT ships in two flavors:

- **ORT (full)**: all features, ~30 MB binary, supports all ops.
- **ORT mobile**: stripped-down build, ~8 MB, supports a curated subset of ops.

The mobile build is what most iOS/Android apps use; the full build is for desktop or server. Producing a model that works with ORT mobile requires using only the supported op set; the conversion tooling has a "mobile-friendly" check.

The choice is binary-size-driven: if 30 MB of inference runtime in your app is acceptable, use full; otherwise, use mobile and trim the model to the supported op set.

## What you should believe after this lesson

Three sentences:

**1. ONNX Runtime on Apple platforms is a credible Core ML alternative** with a more flexible EP architecture — the Core ML EP gives you ANE acceleration via ORT — and a cross-platform story that Core ML doesn't have. It's not the default for Apple-only apps but is the right choice when the model needs to also run elsewhere.

**2. Execution providers are ORT's accelerator-abstraction layer**; you list them in priority order and ORT routes each op to the highest-priority EP that supports it. On Apple, the priority chain is typically `CoreMLExecutionProvider` → `CPUExecutionProvider`.

**3. ORT does aggressive load-time optimization** (op fusion, layout transformation, constant folding) that other runtimes mostly skip — paying this cost once per model load and getting faster steady-state inference. The optimization is one reason ORT often outperforms naive ONNX evaluation on the same model.

## Hands-on (at home)

Run an ONNX model through ORT with different execution providers.

```python
# ort_apple_providers.py
# pip install onnxruntime onnx torch torchvision
import torch
import torchvision.models as models
import onnxruntime as ort
import numpy as np
import time

# 1. Export a model to ONNX.
m = models.efficientnet_b0(weights="DEFAULT").eval()
x = torch.rand(1, 3, 224, 224)
torch.onnx.export(m, x, "eff.onnx", opset_version=17,
                  input_names=["input"], output_names=["output"])

# 2. Run with different EP configurations.
def bench_with_providers(providers, name):
    sess = ort.InferenceSession("eff.onnx", providers=providers)
    print(f"\n{name}")
    print(f"  active providers: {sess.get_providers()}")
    inp = {"input": np.random.rand(1, 3, 224, 224).astype(np.float32)}
    # Warm.
    for _ in range(5):
        sess.run(None, inp)
    t0 = time.time()
    for _ in range(50):
        sess.run(None, inp)
    dt = (time.time() - t0) / 50
    print(f"  latency: {dt*1000:.2f} ms/iter")

bench_with_providers(['CPUExecutionProvider'], "CPU only")
# On macOS with onnxruntime built with Core ML EP enabled:
try:
    bench_with_providers(
        [('CoreMLExecutionProvider', {'MLComputeUnits': 'ALL'}),
         'CPUExecutionProvider'],
        "Core ML EP → CPU fallback"
    )
except Exception as e:
    print(f"Core ML EP not available: {e}")
```

Expected on M3 Pro:
- CPU only: ~30–60 ms.
- Core ML EP (with ANE): ~5–10 ms.

The Core ML EP delegates conv-heavy ops to Core ML, which routes them to the ANE. The latency lands close to what you'd see from a native Core ML deployment of the same model.

Part 2 — compare to the same model deployed via native Core ML (from Lesson 3). The native Core ML path is usually 10–30% faster than ORT-via-Core-ML-EP because of the EP abstraction overhead.

## Further reading

- ONNX Runtime documentation (onnxruntime.ai) — the execution provider catalog and platform-specific guides.
- "Execution Providers" reference — the canonical list of EPs and their capabilities.
- Apple-specific ORT build instructions — how to build ORT with the Core ML EP enabled.
- "ONNX Runtime Mobile" docs — for the binary-size-optimized build for iOS / Android.

Next lesson: **ExecuTorch.** Meta's PyTorch-edge story. The `.pte` format, the backend-delegate pattern that lets one `.pte` file target XNNPACK on CPU, Core ML on Apple, QNN on Qualcomm, and Vulkan on cross-platform GPUs. What's production-ready and what's still rough in 2026.
