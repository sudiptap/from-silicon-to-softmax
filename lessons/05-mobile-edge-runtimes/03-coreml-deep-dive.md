---
title: "Lesson 3 — Core ML Deep Dive"
date: "2026-06-04"
module: "mobile-edge-runtimes"
order: 3
tags: ["coreml", "coremltools", "ane", "compute-units", "ios", "macos"]
author: "Sudipta Pathak"
prerequisites: ["02-model-formats"]
---

# Lesson 3 — Core ML Deep Dive

## Why this lesson exists

Module 4 Lesson 11 introduced the Apple Neural Engine and mentioned that Core ML is the only way to reach it. This lesson is the practical Core ML workflow: converting a PyTorch model with `coremltools`, choosing compute units, targeting the ANE specifically, and debugging the common case where "I asked for ANE and Core ML put it on CPU." If you're shipping a vision or audio model on iOS, this is your runtime; if you're shipping an iOS LLM, this is one option among several (often not the best, per Module 4 Lesson 11) and you still want to understand it.

The lesson is reading. The Hands-on converts a real vision model, runs it across compute units, and inspects what Core ML chose.

## The `coremltools` conversion workflow

The Python entry point is `coremltools.convert`. Source side: PyTorch (most common), TensorFlow, ONNX. Target side: a `.mlpackage` directory.

A canonical PyTorch → Core ML conversion:

```python
import coremltools as ct
import torch
import torchvision.models as models

# 1. Load and trace the PyTorch model.
model = models.efficientnet_b0(weights="DEFAULT").eval()
example = torch.rand(1, 3, 224, 224)
traced = torch.jit.trace(model, example)

# 2. Convert.
ml_model = ct.convert(
    traced,
    inputs=[ct.ImageType(
        name="image",
        shape=(1, 3, 224, 224),
        scale=1/255.0,
        bias=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        color_layout="RGB",
    )],
    classifier_config=ct.ClassifierConfig(class_labels_path="imagenet_classes.txt"),
    compute_precision=ct.precision.FLOAT16,
    compute_units=ct.ComputeUnit.ALL,
    minimum_deployment_target=ct.target.iOS17,
)

# 3. Save.
ml_model.save("EfficientNet.mlpackage")
```

The pieces:

- **`traced`**: PyTorch's tracing path. `torch.jit.trace` runs the model with the example input and captures the operations. `torch.jit.script` is the alternative for models with control flow; less common for ML models.
- **`inputs`**: declared input specifications. `ct.ImageType` is the convenient image input that bundles preprocessing (scaling, bias) into the model graph itself, so the Core ML model takes a raw image and produces a class. `ct.TensorType` is the generic alternative.
- **`compute_precision`**: FP16 is the right default. FP32 is supported but wastes memory; lower precision (`ct.precision.INT8` for quantized models) requires source-side quantization.
- **`compute_units`**: which units the runtime is allowed to use. `.ALL` is the most permissive.
- **`minimum_deployment_target`**: the lowest OS version that should load the model. Newer targets unlock more recent Core ML features but exclude older devices.

The output is a `.mlpackage` directory. You can inspect it (it's just files) but you typically just load it into the Core ML runtime.

## The compute-units decision

Recapping from Module 4 Lesson 11 with the deployment perspective:

- **`.cpuOnly`**: deterministic, slow, mostly for testing.
- **`.cpuAndGPU`**: use everything except the ANE. Safer for models with non-ANE-friendly ops.
- **`.cpuAndNeuralEngine`**: allow ANE but not GPU. Rarely the right choice; the ANE-supported subset is narrow enough that you usually want GPU as fallback.
- **`.all`**: allow everything. The default for ANE-friendly models.

The compiler decides per-op. You can inspect what it chose using the Core ML profiler in Xcode:

1. Build a small Xcode project, load the `.mlpackage`.
2. Run with the Core ML profiler enabled.
3. The profiler shows a per-op breakdown of which compute unit ran each op and how long it took.

Or, in Python via `coremltools`:

```python
import coremltools as ct
ml = ct.models.MLModel("EfficientNet.mlpackage", compute_units=ct.ComputeUnit.ALL)
# Predict with a dummy image; the runtime compiles + runs.
pred = ml.predict({"image": some_image})
# Get timing/unit info via the Core ML compute plan (Core ML 7+).
plan = ml.compute_plan
for op_name, info in plan.operations.items():
    print(f"  {op_name}: ran on {info.compute_unit}, took {info.duration*1000:.2f} ms")
```

The compute plan tells you what Core ML actually did. Use this to verify the ANE is being used (or to find why it isn't).

## Why ops fall back

Core ML's compiler routes an op to the ANE only if:

1. The op type is in the ANE-supported set.
2. The tensor shapes match what the ANE's tiles handle (specific dimension constraints).
3. The dtype is FP16 or INT8 (with some BF16 on newer chips).
4. The op fits into the ANE's working memory at the chosen tile size.

If any constraint fails, the op falls back to GPU. If GPU constraints fail, it falls back to CPU. The fallback is automatic; the result is correct but slower.

Common reasons for fallback in 2026:

- **Recent transformer ops** (rotary position embeddings, certain attention masks) often don't have ANE implementations.
- **Dynamic shapes**. The ANE wants fixed shapes; if your model has variable sequence length or variable batch size, those ops go to GPU.
- **Custom layers**. Anything that `coremltools` couldn't map to a standard Core ML op becomes a "custom op" that runs on CPU.
- **Very large activations**. Tile sizes are limited; activations beyond the ANE's working set get split or sent to GPU.

The debugging workflow:

1. Convert the model.
2. Run with `compute_units=.all`, check the compute plan.
3. If you see ops on CPU/GPU that you expected on ANE, identify which ops.
4. For each problematic op, check the docs for ANE-supported variants, or refactor the model to use a friendly equivalent.
5. Reconvert and re-check.

This is the iterative loop. Most production iOS deployments go through it several times.

## Quantization in Core ML

Core ML supports several quantization paths:

**1. Post-training quantization via `coremltools`:**

```python
import coremltools.optimize.coreml as cto
config = cto.OptimizationConfig(
    global_config=cto.OpLinearQuantizerConfig(
        mode="linear_symmetric", dtype="int8", granularity="per_channel"
    )
)
quantized = cto.linear_quantize_weights(ml_model, config=config)
quantized.save("EfficientNet-INT8.mlpackage")
```

This quantizes the weights post-conversion. INT8 reduces model size by ~2× and is generally accuracy-safe for vision models.

**2. Quantization during training (QAT)** via `coremltools.optimize` applied to the PyTorch model before conversion. More complex; rarely needed for vision models.

**3. INT4 weight quantization** for LLMs (Core ML 7+). Limited but available; the Apple LLM Stable Diffusion-style deployments use this path.

**4. Palettization** — k-means-clustered weight values, similar to BitsAndBytes' approach. Useful for very small models where INT8 is still too large.

For LLMs specifically, Core ML's INT4 path exists but is less mature than llama.cpp's or MLX's. If your LLM deployment needs INT4, llama.cpp or MLX are usually better choices.

## Performance characteristics

On M3 Pro with a typical vision model (EfficientNet-B0, FP16):

| Compute unit | Latency | Notes |
| ------------ | ------- | ----- |
| CPU only | 30–60 ms | Slow baseline. |
| CPU + GPU | 5–10 ms | GPU acceleration kicks in. |
| CPU + GPU + ANE | 3–6 ms | Best; ANE handles convs. |

Power consumption:

| Compute unit | Watts under load |
| ------------ | ---------------- |
| CPU only | ~10 W |
| CPU + GPU | ~12 W |
| CPU + GPU + ANE | ~5 W |

The ANE wins on both latency *and* power for vision models. For battery-powered iOS devices the power savings are user-visible — a continuous-inference workload (camera processing) extends battery life noticeably when using the ANE.

## Deploying to an iOS app

The end-to-end shape:

1. Train and convert in Python on your dev machine (Mac or Linux).
2. Save the `.mlpackage`.
3. Drag the `.mlpackage` into Xcode; Xcode auto-generates Swift classes for the model.
4. In Swift code:
   ```swift
   let model = try EfficientNet(configuration: MLModelConfiguration())
   let prediction = try model.prediction(image: someUIImage)
   print(prediction.classLabel, prediction.classLabelProbs)
   ```
5. Build the app, deploy to device.

The Swift generated code includes typed input/output classes (one per `inputs` declaration at conversion time) and handles all the buffer marshalling. You're back to typed code; you can't see the ANE/GPU/CPU decision in your Swift code.

For runtime control of compute units in Swift:

```swift
let config = MLModelConfiguration()
config.computeUnits = .all   // or .cpuAndGPU, .cpuOnly
let model = try EfficientNet(configuration: config)
```

## When Core ML is the right answer

The case for Core ML:

- iOS / iPadOS / macOS app deployment with a vision or audio model.
- Need for ANE acceleration (only Core ML can reach the ANE).
- Tight integration with Apple platform APIs (CIImage, Vision framework, ARKit).
- Apple's deployment story (App Store, signing, OS integration) is what you're already doing.

The case against:

- Cross-platform deployment. Core ML doesn't exist on Android.
- LLM inference at the cutting edge — llama.cpp / MLX are usually faster.
- Models with operators Core ML doesn't support well (very recent research models).
- You need fast iteration on the model architecture; Core ML conversion is a step you have to redo each time.

## What you should believe after this lesson

Three sentences:

**1. `coremltools.convert` is the entry point for producing `.mlpackage` files** from PyTorch (or TF, ONNX), with the FP16 + `ALL` compute units + iOS17 deployment target being a sensible default for vision/audio models. The compute plan tells you what Core ML actually did at run time.

**2. ANE fallback to GPU/CPU happens silently when an op isn't ANE-supported** — typically recent transformer ops, dynamic shapes, custom layers, or oversized activations. Debug by inspecting the compute plan, refactoring problematic ops, and re-converting.

**3. Core ML is the dominant Apple deployment runtime for vision and audio models** with significant latency and power wins via the ANE. For LLMs on Apple platforms, MLX or llama.cpp are usually better; Core ML is a viable but not the leading option.

## Hands-on (at home)

Convert a real vision model to Core ML and inspect what Core ML did with it.

```python
# coreml_convert_and_inspect.py
# pip install coremltools torch torchvision
import coremltools as ct
import torch
import torchvision.models as models
import time
import numpy as np

# 1. Load and convert.
model = models.mobilenet_v3_small(weights="DEFAULT").eval()
example = torch.rand(1, 3, 224, 224)
traced = torch.jit.trace(model, example)

ml = ct.convert(
    traced,
    inputs=[ct.TensorType(name="image", shape=(1, 3, 224, 224))],
    compute_precision=ct.precision.FLOAT16,
    compute_units=ct.ComputeUnit.ALL,
    minimum_deployment_target=ct.target.iOS17,
)
ml.save("MobileNetV3.mlpackage")
print("converted to MobileNetV3.mlpackage")

# 2. Run with different compute units; measure latency.
for cu_name, cu in [
    ("CPU only", ct.ComputeUnit.CPU_ONLY),
    ("CPU + GPU", ct.ComputeUnit.CPU_AND_GPU),
    ("CPU + GPU + ANE", ct.ComputeUnit.ALL),
]:
    m = ct.models.MLModel("MobileNetV3.mlpackage", compute_units=cu)
    x = {"image": np.random.rand(1, 3, 224, 224).astype(np.float32)}
    # Warm.
    for _ in range(5):
        m.predict(x)
    # Bench.
    t0 = time.time()
    for _ in range(50):
        m.predict(x)
    dt = (time.time() - t0) / 50
    print(f"  {cu_name:20s}: {dt*1000:6.2f} ms/iter")
```

You should see the typical Apple Silicon pattern: ANE is fastest, GPU is close behind, CPU is much slower. On M3 Pro, the numbers might be 3 ms / 5 ms / 30 ms.

For LLM conversion (more complex), Apple's `coreml-llm-tutorial` (or community equivalents) walks through the steps. The result is rarely worth the effort vs MLX or llama.cpp, but doing it once is educational.

## Further reading

- `coremltools` documentation (apple.github.io/coremltools) — the conversion API reference and tutorials.
- "Core ML Performance Best Practices" WWDC sessions — annually-updated by Apple's Core ML team.
- "Optimizing PyTorch models for Core ML" (Apple blog) — for the PyTorch-specific conversion tips.
- `coremltools.optimize` documentation — for quantization and palettization.
- Apple ML research blog — for "Deploying X on Apple Neural Engine" articles that walk through real conversions.

Next lesson: **ONNX Runtime on iOS/macOS.** ORT is a credible alternative to Core ML on Apple platforms, particularly for models that originated outside the Apple ecosystem. We look at when ORT beats Core ML, the execution-provider abstraction, and the Core ML EP that lets ORT delegate to Core ML for ANE acceleration.
