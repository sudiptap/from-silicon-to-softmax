---
title: "Lesson 10 — The Runtime Decision Tree (Module Wrap)"
date: "2026-06-04"
module: "mobile-edge-runtimes"
order: 10
tags: ["decision-tree", "deployment", "picking-matrix", "wrap", "summary"]
author: "Sudipta Pathak"
prerequisites: ["09-llama-cpp-ggml"]
---

# Lesson 10 — The Runtime Decision Tree (Module Wrap)

## Why this lesson exists

Ten lessons of edge runtime survey collapse into a practical decision-making framework. This lesson is that framework: a decision tree backed by the comparisons accumulated through the module, plus a worked example showing how to apply it to a real deployment requirement.

It's also the module wrap. The "what we covered, what we skipped, mental models to carry forward, handoff to Module 6" lives here.

The lesson is reading. The Hands-on walks through the worked deployment example end-to-end on whatever runtime the decision tree picks.

## The journey, replayed

Module 5 went landscape → formats → runtimes → accelerators → wrap. The arc:

- Lessons 1–2 (landscape): the runtime map (which runtimes exist, who maintains them, what platforms each targets) and the model-format ecosystem (ONNX, GGUF, Core ML, LiteRT, ExecuTorch, MLX-native — how they differ and what's lost in conversion between them).
- Lessons 3–4 (Apple side): Core ML as the Apple deployment runtime with ANE access, ONNX Runtime on Apple as the cross-platform alternative with the Core ML EP for ANE access via ORT.
- Lessons 5–7 (cross-platform): ExecuTorch as PyTorch's edge story, LiteRT as Google's Android-primary runtime, ONNX Runtime mobile as the breadth-first cross-platform option.
- Lesson 8 (Android accelerator): Qualcomm Hexagon + QNN SDK for the dominant Android NPU.
- Lesson 9 (LLM runtime): llama.cpp / ggml as the de facto on-device LLM runtime, cross-platform via the ggml backend abstraction.

The synthesis: there's no universal best runtime. The right choice depends on platforms, accelerators, model origin, latency/footprint constraints, and dev ecosystem. The runtimes ecosystem has stabilized into about 5–6 production-grade options that each have a clear niche.

## The decision tree

For a deployment requirement, walk the questions in order:

```
1. WHAT KIND OF MODEL?
   ├── LLM (decoder transformer, batch 1, autoregressive)
   │    → goto LLM path (below)
   ├── Vision (CNN, ViT, detection, segmentation)
   │    → goto VISION path
   ├── Audio (Whisper, speech models)
   │    → goto AUDIO path
   ├── Tabular / classical ML (XGBoost, sklearn)
   │    → ONNX Runtime
   └── Other / research
        → start with ONNX Runtime; fall back to model's source framework

2. WHAT PLATFORM(S)?
   ├── iOS only
   │    → use platform-native runtime (LLM: llama.cpp or MLX; Vision/Audio: Core ML)
   ├── Android only
   │    → use platform-native runtime (LLM: llama.cpp; Vision/Audio: LiteRT + QNN delegate)
   ├── iOS + Android (most common)
   │    → ExecuTorch (PyTorch shop) OR llama.cpp (LLM) OR ORT (broad cross-platform)
   ├── Desktop (Win/Mac/Linux)
   │    → llama.cpp (LLM) OR ONNX Runtime (everything else)
   ├── Embedded Linux (Raspberry Pi class)
   │    → ONNX Runtime (general) OR llama.cpp (LLM)
   ├── Microcontroller
   │    → TensorFlow Lite Micro
   └── Browser
        → ONNX Runtime Web OR WebLLM / WebGPU paths

3. WHAT IS THE LATENCY / FOOTPRINT BUDGET?
   ├── Tight latency (real-time camera, <10 ms)
   │    → must hit NPU (ANE / Hexagon); use platform-native or ExecuTorch
   ├── Tight footprint (<10 MB runtime + small model)
   │    → LiteRT or stripped ORT Mobile
   ├── Moderate (interactive LLM, 200-500 ms TTFT)
   │    → most runtimes meet this; pick on other criteria
   └── Loose (background batch)
        → pick on dev ergonomics

4. WHERE DOES THE MODEL COME FROM?
   ├── PyTorch
   │    → prefer ExecuTorch (native), then llama.cpp (via GGUF for LLMs), then ONNX path
   ├── TensorFlow / Keras
   │    → prefer LiteRT (native), then ONNX
   ├── HuggingFace Transformers (LLM)
   │    → GGUF → llama.cpp, or MLX-native → mlx-lm, or ExecuTorch
   ├── ONNX (already)
   │    → ONNX Runtime is the obvious answer
   └── scikit-learn / classical ML
        → ONNX Runtime
```

The decision tree converges to one or two candidates for most requirements. The choice between the candidates usually comes down to ecosystem fit (does your team know PyTorch or TF?), binary size constraints, or specific accelerator targeting.

## A worked example

> *Requirement: deploy a 3B-parameter instruction-tuned LLM (Llama 3.2 3B, quantized to INT4) on both iOS and Android. The latency budget is 200 ms time-to-first-token (TTFT) on flagship 2024+ phones. The model is a HuggingFace Transformers checkpoint.*

Walking the decision tree:

**1. What kind of model?** LLM. Take the LLM path.

**2. What platforms?** iOS + Android. Both required. Cross-platform options: ExecuTorch, llama.cpp, ORT.

**3. What is the latency budget?** 200 ms TTFT for a 3B INT4 model on a flagship 2024+ phone. This is achievable but requires GPU or NPU acceleration; CPU alone won't make it.

**4. Where does the model come from?** HuggingFace. The conversion paths are HF → GGUF (smooth, via `convert_hf_to_gguf.py`) or HF → PyTorch → ExecuTorch (`.pte`) (less smooth but possible).

The candidates after the tree: llama.cpp and ExecuTorch.

Evaluating:

| Criterion | llama.cpp | ExecuTorch |
| --------- | --------- | ---------- |
| HF → format conversion | smooth (GGUF) | needs PyTorch intermediate |
| iOS performance (Metal) | excellent | good via Core ML backend |
| Android performance | good via Vulkan or CPU | good via XNNPACK or QNN |
| Per-platform tuning | one binary, hand-tuned per backend | one .pte, backend-delegated regions |
| Binary size on device | ~5 MB engine + 1.5 GB model | ~5 MB runtime + 1.5 GB model |
| OpenAI API server (built-in) | yes | no (build yourself) |
| Maturity | very mature | improving |
| Community | huge | growing |

**Recommendation: llama.cpp.** It wins on HF-conversion simplicity, raw performance on both platforms, and operational ergonomics. ExecuTorch would also work but adds friction without compensating advantages for this specific requirement.

Deployment shape: bundle `libllama.so` (Android) / `libllama.a` (iOS) into the app, ship the 1.5 GB Q4_K_M GGUF as a separate download (the App Store / Play Store typically penalize >150 MB installable size; download-after-install is the standard pattern), expose a Kotlin / Swift wrapper that calls llama.cpp's C++ API.

Validation:
- Build llama.cpp for the target devices.
- Convert Llama 3.2 3B to GGUF Q4_K_M.
- Run `llama-bench` on a flagship 2024 Android (Galaxy S24 with Snapdragon 8 Gen 3) and a 2024 iPhone (15 Pro with A17 Pro). Verify prefill > 30 tok/s on both (which would give ~30 ms/token prefill, comfortably under 200 ms TTFT for short prompts).
- If not meeting the budget, fall back: try smaller quantization (Q3_K_M), try a smaller model (1B instead of 3B), try GPU offload on Android (Vulkan backend) if not yet enabled.

This is the kind of analysis the module's preparation supports. Five minutes with the decision tree gives a defensible choice; the rest is execution.

## What we covered, what we skipped

Covered:

- The edge runtime map and the five axes of choice.
- Five major model formats (ONNX, GGUF, Core ML, LiteRT, ExecuTorch) and conversion paths.
- Core ML for Apple deployment with ANE access.
- ONNX Runtime on Apple as the cross-platform alternative.
- ExecuTorch as PyTorch's edge story with the backend-delegate pattern.
- LiteRT as Google's Android runtime with the delegate model.
- ONNX Runtime Mobile for cross-platform deployment.
- Qualcomm Hexagon + QNN SDK for the dominant Android NPU.
- llama.cpp / ggml for the LLM-runtime niche.
- The decision tree.

Skipped:

- **Apple Watch / watchOS deployment.** Core ML works but constraints are tighter; we didn't go deep.
- **TF Lite Micro for microcontrollers.** Mentioned as a destination; the embedded ML deeper story belongs to its own treatment.
- **NCNN and MNN.** Major in Chinese-market deployments; we didn't go deep.
- **Server-grade LLM serving** (vLLM, TensorRT-LLM, SGLang). Different niche; Module 7 covers some of this.
- **WebLLM / browser ML in depth.** Mentioned in passing; deserves its own treatment.
- **Specific NPUs we didn't cover**: Samsung Exynos NPU, Huawei NPU, Google Tensor TPU. Each is a story.
- **AI PCs (Windows on Snapdragon X Elite).** Mentioned but didn't go deep.

## Mental models to carry forward

Five sentences:

**1. The runtime layer is a real decision, not an afterthought.** Picking wrong costs months. Walk the decision tree before committing.

**2. Hardware-vendor runtimes (Core ML, QNN) are great on their hardware and absent elsewhere; software-vendor runtimes (LiteRT, ORT, ExecuTorch) are cross-platform with a tax of ~15% perf vs native; the community LLM runtime (llama.cpp) is great everywhere for its specific niche.**

**3. The format matters as much as the runtime.** Conversion paths accumulate lossiness; minimizing conversion steps means more reliable deployment. Pick a runtime whose native format is closest to your model's source.

**4. NPUs win on vision and audio; for LLMs, GPU and CPU are still often better.** This is the consistent pattern across Apple ANE and Qualcomm Hexagon as of 2026. The LLM-on-NPU story is improving but not yet dominant.

**5. For LLM deployment on consumer devices in 2026, llama.cpp is the default.** Other runtimes are credible but llama.cpp's combination of cross-platform reach, no-Python operational story, hand-tuned per-platform kernels, and active community is hard to beat.

## What's next

**Module 6: On-Device LLM Inference.** We take the runtime choices from Module 5 and build production-grade LLM inference on top of them: scheduling, token streaming, context management, integrating with apps, the operational story for "real users hitting my on-device LLM" deployments. The runtime is the substrate; the production patterns sit on top.

**Module 7: Inference from Scratch.** Build the inference pipeline from primitives. The patterns at scale: KV cache management, prefix caching, batching when it makes sense, sampling strategies, the prefill-vs-decode split, MoE routing. Most of this is framework-agnostic but uses MLX or PyTorch as the concrete substrate.

**Module 8: Distributed Systems.** Multi-device, multi-machine ML systems. For on-device, this includes split inference (model partitioned across multiple Macs or phones) and federated learning. For server, it's the NCCL / RDMA / sharding story.

The matmul project that ran from Modules 1–4 has graduated; this module was about higher-level runtime choices, not kernel-writing. The capstone here is the decision-tree exercise: take an actual deployment requirement (your own or one from your team), apply the tree, defend the choice in writing, and have a working prototype within a few days. That's the practical capability the module exists to enable.

## Hands-on (at home)

Walk through the worked example end-to-end on whatever platform you have.

```bash
# Pick a model that's small enough to deploy quickly.
MODEL_HF="meta-llama/Llama-3.2-1B-Instruct"  # 1B instead of 3B for speed
QUANT="Q4_K_M"

# 1. Get the GGUF.
git clone https://github.com/ggerganov/llama.cpp && cd llama.cpp
make LLAMA_METAL=1 -j8   # or LLAMA_CUDA=1 / LLAMA_VULKAN=1 / nothing

# Use a pre-quantized GGUF from HF for speed.
huggingface-cli download bartowski/Llama-3.2-1B-Instruct-GGUF \
    Llama-3.2-1B-Instruct-${QUANT}.gguf --local-dir models

# 2. Benchmark.
./llama-bench -m models/Llama-3.2-1B-Instruct-${QUANT}.gguf -p 128 -n 128

# Compare prefill (pp) and generation (tg) tokens/sec to your latency budget.
# Example output:
#   ggml_metal_init: allocating
#   ...
#   pp 128: 4500 tok/s   (prefill: 0.028 ms/token)
#   tg 128:  120 tok/s   (decode:  8.3 ms/token)
# For 200ms TTFT with a 100-token prompt: 100 * 0.028 = 2.8 ms — way under budget.
```

If you're targeting mobile specifically, the next step is to cross-compile llama.cpp for Android NDK or iOS Xcode and run on a device. The pattern is documented in the llama.cpp README under "Android" and "iOS" sections.

For the ExecuTorch comparison: convert the same HF model via `executorch/examples/llama`'s scripts to `.pte`, bundle the ExecuTorch runtime, run on the same device. Compare tokens/sec.

The numbers from these experiments inform the picking-matrix for *your* specific deployment context.

## End of Module 5

Module 1 made the CPU fast. Module 2 made the GPU fast. Module 3 made the model small. Module 4 made the small fast model run on Apple Silicon. Module 5 made it run on every other commodity edge platform too. The combination is on-device ML systems work as practiced in 2026.

Module 6 picks up next: **On-Device LLM Inference.** We take the runtimes from this module and build the production patterns on top of them: token streaming, context management, multi-turn conversation handling, the operational story. The runtime is the substrate; the production capabilities sit on top.
