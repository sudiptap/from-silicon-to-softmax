---
title: "Lesson 6 — LiteRT (Formerly TFLite)"
date: "2026-06-04"
module: "mobile-edge-runtimes"
order: 6
tags: ["litert", "tflite", "android", "delegate", "nnapi", "mediapipe", "tasks"]
author: "Sudipta Pathak"
prerequisites: ["05-executorch"]
---

# Lesson 6 — LiteRT (Formerly TFLite)

## Why this lesson exists

LiteRT (Lite Runtime, rebranded from TensorFlow Lite in 2024) is Google's mobile inference runtime. It's the default ML runtime on Android, ships in most Google apps and many third-party apps, and provides the broadest accelerator coverage on Android via its delegate model. If you're deploying ML on Android — especially a vision or audio model — LiteRT is almost certainly the runtime to consider first.

This lesson is the practical LiteRT view: converting a model, the delegate model that lets you target GPU and NPU, and MediaPipe Tasks as the high-level wrapper around LiteRT for common ML use cases (face detection, hand tracking, audio classification).

The lesson is reading. The Hands-on converts a small TF model to `.tflite`, runs it through the LiteRT runtime with CPU then GPU delegate, and inspects the results.

## What LiteRT is

A `.tflite` file is a FlatBuffer-serialized model file optimized for mobile / edge deployment:

- **FlatBuffer format**: zero-copy mmap loading. No deserialization step at load time.
- **Quantization-native**: per-tensor and per-channel INT8 are first-class.
- **Op set**: ~150 built-in ops covering most common ML primitives. Custom-op extension API for missing ones.

The LiteRT runtime is a C++ library, ~few MB binary. It walks the `.tflite` file, builds an interpretation graph, and executes ops via the kernel set the runtime has plus any delegates you've registered.

A typical Python load + run:

```python
import tensorflow as tf
import numpy as np

interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
interpreter.set_tensor(input_details[0]['index'], np.random.rand(1, 224, 224, 3).astype(np.float32))
interpreter.invoke()
out = interpreter.get_tensor(output_details[0]['index'])
```

The same interpreter on Android is callable from Java/Kotlin via the LiteRT Android library; on iOS via the Swift/Objective-C bindings.

## Conversion from TF / Keras

The native conversion is TF → LiteRT:

```python
import tensorflow as tf

model = tf.keras.applications.MobileNetV3Small(weights="imagenet")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]  # enables PTQ
tflite = converter.convert()
with open("mobilenet.tflite", "wb") as f:
    f.write(tflite)
```

The `Optimize.DEFAULT` enables float16 quantization or weight INT8 quantization (depending on representative dataset availability). For full INT8 (weight + activation), you provide a calibration dataset:

```python
def rep_data():
    for i in range(100):
        yield [np.random.rand(1, 224, 224, 3).astype(np.float32)]
converter.representative_dataset = rep_data
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8
tflite = converter.convert()
```

PyTorch → LiteRT is multi-step: PyTorch → ONNX → TF (via `tf2onnx`) → LiteRT. This works but is fragile; the round-trip can lose ops. The cleaner path for PyTorch-origin models is ExecuTorch (Lesson 5) or ORT (Lesson 4) on Android.

In 2026, the recommended PyTorch-to-LiteRT path is via the `ai-edge-torch` library, which compiles PyTorch directly to LiteRT without the ONNX intermediate. It's newer but increasingly the standard for PyTorch-on-Android.

## The delegate model

LiteRT's delegate model is its accelerator abstraction. By default the interpreter uses CPU kernels; delegates redirect specific ops (or whole subgraphs) to specialized backends.

Available delegates in 2026:

- **GPU delegate**: OpenGL ES / Vulkan compute on Android, Metal on iOS. Good for batched workloads and many vision models.
- **NNAPI delegate**: routes to Android's Neural Networks API, which dispatches to whatever NPU the device has (Qualcomm Hexagon, MediaTek APU, Samsung NPU, etc.). The "use the device's NPU" path.
- **Hexagon delegate** (now mostly subsumed by QNN delegate): direct Qualcomm Hexagon NPU access on older Snapdragon devices.
- **QNN delegate**: direct Qualcomm Hexagon access on newer devices via the QNN SDK.
- **Core ML delegate**: on iOS, delegates to Core ML for ANE access.
- **XNNPACK delegate**: optimized CPU kernels, often faster than the default CPU path.

Using a delegate in Python (for testing; production is in the platform language):

```python
interpreter = tf.lite.Interpreter(
    model_path="model.tflite",
    experimental_delegates=[
        tf.lite.experimental.load_delegate('libedgetpu.so.1'),  # Coral Edge TPU example
    ]
)
```

On Android (Kotlin):

```kotlin
val gpuDelegate = GpuDelegate()
val options = Interpreter.Options().addDelegate(gpuDelegate)
val interpreter = Interpreter(modelFile, options)
```

The delegate inspects each op as the model loads. Ops it supports get redirected to the delegate's kernels; ops it doesn't support stay on CPU. Like every other "supported ops" story, the gaps cause unexpected slowness when a critical op falls back.

## NNAPI: Android's NPU abstraction

NNAPI is Google's "talk to whatever NPU the device has" API. The chip vendor (Qualcomm, MediaTek, Samsung, etc.) ships a hardware abstraction layer that implements NNAPI; the OS routes NNAPI calls to that HAL.

In theory, you write code against NNAPI and it works on every Android NPU. In practice, NNAPI has been deprecated by Google as of Android 15 (2024) in favor of more direct vendor SDKs. The reasoning: NNAPI's least-common-denominator design meant it couldn't expose the best features of any specific NPU, and the vendor SDKs (QNN for Qualcomm, MediaTek NeuroPilot, etc.) consistently outperformed the NNAPI path.

So the 2026 picture: **NNAPI delegate works but is no longer the recommended path** for new development. Use vendor-specific delegates (QNN for Snapdragon) or stay on GPU/CPU.

## MediaPipe Tasks

MediaPipe Tasks is Google's high-level API for common ML use cases, built on top of LiteRT. Instead of writing model-loading and inference plumbing, you instantiate a Task object:

```kotlin
val faceDetector = FaceDetector.createFromOptions(
    context,
    FaceDetector.FaceDetectorOptions.builder()
        .setBaseOptions(BaseOptions.builder().setModelAssetPath("face_detection.tflite").build())
        .setRunningMode(RunningMode.IMAGE)
        .build()
)
val result = faceDetector.detect(MPImage.createFromBitmap(bitmap))
```

The Tasks library handles preprocessing (image scaling, normalization), the inference call, postprocessing (NMS for detection, etc.). For supported tasks — face detection, hand landmarks, pose, object detection, text classification, audio classification, gesture recognition, image segmentation, image embedding, image generation (Stable Diffusion), LLM inference — this is far less code than building it yourself.

In 2026 MediaPipe also includes:
- **LLM Inference Task**: small LLMs (Gemma, Phi) packaged for Android, with the runtime + tokenizer + generation loop bundled.
- **Image Generation Task**: Stable Diffusion on mobile.

These are alternatives to the raw LiteRT path; if your use case fits a Task, use the Task. The bundled preprocessing and postprocessing alone is worth it.

## LiteRT and quantization

LiteRT is quantization-friendly:

- **Float16 quantization**: 2× model size reduction, near-zero accuracy hit. Reliable.
- **INT8 weight quantization (dynamic range)**: weights are INT8, activations stay float at run time. Conversion doesn't need a calibration dataset.
- **Full INT8 quantization**: weights + activations both INT8. Requires representative data for calibration. ~4× model size reduction.
- **Float16 GPU delegate path**: takes a float16 model and runs on GPU; common combination.
- **INT8 NPU delegate path**: takes an INT8 model and runs on NNAPI/QNN; the standard NPU path.

The recommendation for vision models on Android: INT8 quantization + QNN delegate (or NNAPI on older devices). The 2–4× speedup over the float CPU path is real and consistent.

For LLMs on LiteRT: in 2026 still less mature than llama.cpp; the MediaPipe LLM Inference Task uses LiteRT under the hood but with significant additional logic on top.

## What you should believe after this lesson

Three sentences:

**1. LiteRT is the default Android ML runtime** — small binary, FlatBuffer-based for fast load, mature delegate model for GPU and NPU acceleration. The TF-to-LiteRT conversion is the smoothest path; PyTorch-to-LiteRT requires `ai-edge-torch` or a multi-step ONNX detour.

**2. The delegate model is the way to reach Android accelerators**: GPU delegate for batched GPU workloads, QNN delegate for Snapdragon NPU, NNAPI delegate as the legacy "any NPU" path (deprecated but still functional). Production deployments increasingly use vendor-specific delegates over NNAPI.

**3. MediaPipe Tasks is the high-level API for common use cases** — face detection, pose, object detection, text/audio classification, even LLM inference — built on LiteRT but with the preprocessing/postprocessing already bundled. If your use case fits a Task, use it; you avoid significant plumbing code.

## Hands-on (at home)

Convert a TF model to LiteRT, run with CPU and GPU delegate, compare.

```python
# litert_demo.py
# pip install tensorflow numpy
import tensorflow as tf
import numpy as np
import time

# 1. Get a small model and convert.
model = tf.keras.applications.MobileNetV3Small(weights="imagenet", include_top=True)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_data = converter.convert()
with open("mobilenetv3.tflite", "wb") as f:
    f.write(tflite_data)
print(f"converted ({len(tflite_data)/1e6:.1f} MB)")

# 2. Run on CPU.
interpreter = tf.lite.Interpreter(model_path="mobilenetv3.tflite", num_threads=4)
interpreter.allocate_tensors()
inp_idx = interpreter.get_input_details()[0]['index']
out_idx = interpreter.get_output_details()[0]['index']
x = np.random.rand(1, 224, 224, 3).astype(np.float32)
interpreter.set_tensor(inp_idx, x)

# Warm + bench.
for _ in range(5):
    interpreter.invoke()
t0 = time.time()
for _ in range(50):
    interpreter.invoke()
dt = (time.time() - t0) / 50
print(f"CPU latency: {dt*1000:.2f} ms/iter")

# 3. On Apple, you can also try the Core ML delegate via the ORT EP combo,
#    but native LiteRT GPU/Metal delegate on macOS requires extra build steps.
#    On Android device-side, the GPU delegate would be:
#    options.addDelegate(GpuDelegate())
```

This produces a `.tflite` file you can drop into an Android app. The Python path runs on CPU; the GPU delegate is Android-specific and easier to exercise on-device.

For a richer experiment on Android (if you have a device):
1. Build the Android `Image Classification` sample app from the TensorFlow examples repo.
2. Replace the bundled model with your `mobilenetv3.tflite`.
3. Run the app with the GPU delegate vs CPU; observe the latency difference.

## Further reading

- LiteRT documentation (ai.google.dev/edge/litert) — the official intro.
- "TensorFlow Lite is now LiteRT" announcement (Google, 2024) — for the rebranding context.
- MediaPipe Tasks documentation — for the high-level API surface.
- "ai-edge-torch" library — for the modern PyTorch-to-LiteRT path.
- NNAPI deprecation notice and migration guide — Google's official write-up on moving from NNAPI to vendor SDKs.

Next lesson: **ONNX Runtime mobile.** The mobile build of ORT, the binary-size tradeoffs, and the cross-platform inference story for non-LLM models. We complete the cross-platform-runtime triple (ExecuTorch, LiteRT, ORT-mobile) and then pivot to hardware-specific accelerators with Qualcomm Hexagon.
