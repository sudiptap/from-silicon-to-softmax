---
title: "Lesson 9 — llama.cpp / ggml"
date: "2026-06-04"
module: "mobile-edge-runtimes"
order: 9
tags: ["llama-cpp", "ggml", "gguf", "llm", "cross-platform", "metal", "cuda", "vulkan"]
author: "Sudipta Pathak"
prerequisites: ["08-qualcomm-hexagon-qnn"]
---

# Lesson 9 — llama.cpp / ggml

## Why this lesson exists

llama.cpp is, in 2026, the runtime that runs the most LLM inference outside of cloud providers. It runs on macOS (Metal-accelerated), Linux (CUDA, ROCm, Vulkan, CPU), Windows, Android (CPU, Vulkan), iOS (Metal), and a long tail of more obscure platforms. The format it consumes (GGUF, covered in Module 3 Lesson 4 and Module 4 Lesson 10) has become the de facto standard for distributing quantized LLM weights — hundreds of thousands of GGUF model files exist on HuggingFace alone.

The historical context: llama.cpp started in early 2023 as Georgi Gerganov's weekend project to run the first leaked Llama model on a MacBook CPU. It was supposed to be a toy. It turned out the combination of (1) carefully written quantization formats, (2) Metal kernels hand-tuned for Apple Silicon, (3) a small C++ codebase with no Python dependency, (4) aggressive community development, was the right tool for the moment when on-device LLMs became interesting. By late 2023 it was the production-grade runtime for small-LLM serving; by 2024 it had backends for every major platform; by 2026 it's the LLM-specific runtime everyone reaches for when they don't need the breadth of ONNX Runtime or the platform integration of Core ML.

This lesson is what llama.cpp is, why it won the LLM-runtime niche, and how to deploy it.

The lesson is reading. The Hands-on builds llama.cpp from source for your machine and runs a quantized LLM through it.

## What llama.cpp is

llama.cpp is a C/C++ inference engine for LLMs, built on top of the ggml tensor library. The two pieces:

**ggml**: a tensor library written in plain C, designed for inference. It provides arrays, common ops (matmul, attention, layer norm, activation), and a backend abstraction. Conceptually similar to PyTorch's `torch::Tensor` but inference-only and dependency-free.

**llama.cpp**: the LLM-specific code on top of ggml. It includes:
- Model loaders for many architectures (Llama, Mistral, Mixtral, Qwen, Gemma, Phi, ...).
- The GGUF format reader.
- The KV cache management, the attention computation, the sampling loop.
- A CLI (`main`, `llama-server`, `llama-bench`, `quantize`, etc.) for common operations.

When people say "llama.cpp," they often mean both pieces together. The split matters because ggml has been forked / used by other projects (whisper.cpp for audio, stable-diffusion.cpp for image generation), all sharing the underlying tensor library and the GGUF format.

## The backend abstraction

ggml's backend abstraction is the architectural feature that makes llama.cpp cross-platform. Backends in 2026:

- **CPU**: x86_64 (with AVX2 / AVX512), ARM (with NEON, SVE). Universal fallback.
- **Metal**: Apple Silicon GPU. Module 4 Lesson 10 covered this path.
- **CUDA**: NVIDIA GPUs on Linux / Windows. Production-grade.
- **HIP / ROCm**: AMD GPUs on Linux. Equivalent to the CUDA backend in coverage.
- **Vulkan**: cross-platform GPU. Works on Linux (NVIDIA, AMD, Intel), Windows, Android (Adreno, Mali). The "anything with a GPU" fallback.
- **OpenCL**: similar role to Vulkan; less commonly used now.
- **SYCL**: Intel GPU path (Intel Arc, Intel Data Center GPU).
- **Kompute**: Vulkan-based, another GPU abstraction.
- **CANN**: Huawei Ascend NPU.
- **Hexagon** (via dedicated backend, 2025+): direct Qualcomm Hexagon access.

For LLM inference, the priority order on a typical system:

- On macOS: Metal.
- On Linux/Windows with NVIDIA: CUDA.
- On Linux with AMD: HIP.
- On Android: Vulkan (if GPU available), CPU otherwise.
- On iOS: Metal.

The CPU backend is the universal fallback and is competitive with GPU on chips where the CPU has good SIMD (AVX-512 desktops, M-series ARM). On a high-end ARM laptop with no usable GPU, CPU + AVX2/NEON is the right answer.

## Why llama.cpp ate the small-LLM serving niche

A list, because the question comes up often:

**1. No Python.** Pure C++ binary. Drops into mobile apps, embedded systems, native desktop apps, no Python runtime to ship. Cargo-cult Python ML developers initially recoiled; the operations world embraced it.

**2. Memory-mapped models.** GGUF + mmap means the model file is loaded as virtual memory, paged in on demand by the OS. Startup is instant (the model "loads" in milliseconds because nothing actually moves; the first inference pulls in the pages it needs). For a 4 GB model on a system with 8 GB RAM, only the parts being touched are resident.

**3. Quantization-first design.** GGUF supports many quantization formats per-tensor; the runtime knows how to dequantize on the fly. The whole pipeline assumes quantized inference; it's not an afterthought retrofitted onto a float-first framework.

**4. Fast Metal kernels.** The Metal backend was hand-tuned by the maintainer over years; for small-batch LLM decoding it matches or beats MLX (Module 4 Lesson 10). The CUDA backend received similar attention.

**5. Community velocity.** New models (Llama 4, Mistral Large, Qwen 3, Gemma 3) get GGUF support within days of release. The community ports model architectures; the maintainers ensure the runtime supports them.

**6. The right scope.** llama.cpp does inference, not training. It targets decoder-only transformers, not vision or audio. This narrow focus lets it optimize harder than general-purpose runtimes.

**7. Operational ergonomics.** A single binary that takes a `--model path/to/file.gguf` and runs a chat or a completion. The `llama-server` provides a REST API compatible with OpenAI's chat completions endpoint. Drop-in for any tool that expects OpenAI.

The combination matters; no individual feature is unique, but the package is.

## The deployment shapes

llama.cpp deployments typically take one of these shapes:

**1. Command-line tool.** `./llama-cli -m model.gguf -p "Hello"`. For shell scripting, ad-hoc usage, batch processing.

**2. HTTP server.** `./llama-server -m model.gguf` — exposes an OpenAI-compatible REST endpoint on port 8080. Drops into any application stack that already speaks OpenAI's API.

**3. Embedded library (C/C++).** Link `libllama.a` or `libllama.so` into your application; call the API directly. The path for native desktop apps, mobile apps via NDK / Swift bindings, embedded systems.

**4. Language bindings.** `llama-cpp-python` for Python, `node-llama-cpp` for Node.js, `rustllama` for Rust, `llama-android` for Android Kotlin, several Swift packages for iOS. The same C++ library underneath.

**5. Higher-level wrappers.** `ollama` (a Go-based local LLM management tool that uses llama.cpp under the hood for many models), `LMStudio` (a desktop app), `text-generation-webui` (Gradio web UI). These provide the polished UX; llama.cpp is the engine.

## What llama.cpp doesn't do

Worth being explicit about the scope:

- **Not for training.** Inference only. Fine-tuning isn't a supported workflow; you train elsewhere and export to GGUF.
- **Not for vision/audio models.** Whisper.cpp and stable-diffusion.cpp are sibling projects that share ggml; the LLM-specific code in llama.cpp is just for transformers.
- **Not the absolute fastest for multi-tenant serving.** vLLM with paged attention is faster at server scale; llama.cpp targets single-user / few-user workloads.
- **Not for very large models on multi-GPU.** Some multi-GPU support exists but it's not the primary use case; PyTorch + DeepSpeed / vLLM handle that better.

The niche is clear: single-user / small-multi-user LLM inference on consumer hardware, where the model is quantized to fit in memory. For exactly this niche, llama.cpp is the runtime to beat.

## Building llama.cpp

The build is unusually simple for a C++ project:

```bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp

# CPU only.
make

# Metal (macOS).
make LLAMA_METAL=1

# CUDA.
make LLAMA_CUDA=1

# Vulkan.
make LLAMA_VULKAN=1

# HIP (AMD).
make LLAMA_HIPBLAS=1
```

Or via CMake for more control. The result is a `main` (now usually renamed `llama-cli`), `llama-server`, `quantize`, `llama-bench` and other tools in the build directory.

Build time: 1–3 minutes on a modern machine for the CPU build; longer for GPU builds (the CUDA path takes ~5 minutes).

For mobile deployment, separate build instructions exist for Android (NDK) and iOS (Xcode). The Android build produces a `libllama.so` you bundle in your app; the iOS build produces a `liblama.a` linked into the app binary.

## A working session

A short end-to-end on macOS, after building with `make LLAMA_METAL=1`:

```bash
# 1. Download a small quantized model from HuggingFace.
huggingface-cli download bartowski/Qwen2.5-1.5B-Instruct-GGUF \
    Qwen2.5-1.5B-Instruct-Q4_K_M.gguf --local-dir models

# 2. Run a one-off prompt.
./llama-cli \
    -m models/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf \
    -p "Explain entropy in three sentences." \
    -n 200 \
    -ngl 999  # offload all layers to Metal

# 3. Run the OpenAI-compatible server.
./llama-server \
    -m models/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf \
    -ngl 999 \
    --port 8080

# In another shell, hit the API:
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen-2.5-1.5b",
    "messages": [{"role": "user", "content": "Hi!"}],
    "max_tokens": 50
  }'
```

The server speaks OpenAI's chat completions API. Any client library that targets the OpenAI API works against llama.cpp's server with one URL change.

## llama.cpp on mobile

Bundling llama.cpp into a mobile app:

**Android (Kotlin)**:
```kotlin
// Via the jllama wrapper:
val model = LlamaModel.loadModel("model.gguf", ...)
val response = model.generate("Hello", maxTokens = 100)
```

**iOS (Swift)**:
```swift
// Via the LlamaContext wrapper:
let llama = try LlamaContext(modelPath: "model.gguf", contextSize: 2048)
let output = try await llama.completion(prompt: "Hello", maxTokens: 100)
```

In both cases the underlying C++ does the work; the language binding wraps it.

The mobile deployment binary size: ~5–8 MB for the runtime + your model (which is the dominant cost). A 1B INT4 LLM is ~500 MB; a 3B is ~1.5 GB. The model is what eats your app's storage.

## What you should believe after this lesson

Three sentences:

**1. llama.cpp is the de facto on-device LLM runtime in 2026** — cross-platform via the ggml backend abstraction (CPU, Metal, CUDA, Vulkan, ROCm, others), quantization-first via GGUF, no Python, mmap-friendly, hand-tuned per-platform kernels, and a vibrant community that keeps up with new model architectures.

**2. The scope is intentionally narrow**: single-user LLM inference, not training, not vision/audio (use whisper.cpp / stable-diffusion.cpp), not multi-tenant server (use vLLM). Within this scope it is hard to beat in 2026.

**3. The deployment shapes range from CLI tool to HTTP server to embedded library to mobile bindings** — pick the shape that matches your application, and the underlying engine is the same. The OpenAI-compatible server is particularly valuable for dropping into existing tooling that expects an OpenAI API.

## Hands-on (at home)

Build llama.cpp and run a model.

```bash
# 1. Clone and build (pick the variant for your platform).
git clone https://github.com/ggerganov/llama.cpp && cd llama.cpp
# macOS:
make LLAMA_METAL=1 -j8
# Linux with NVIDIA:
# make LLAMA_CUDA=1 -j8
# Linux/Windows/Android with GPU but unsure of vendor:
# make LLAMA_VULKAN=1 -j8

# 2. Download a small model.
pip install huggingface-hub
huggingface-cli download bartowski/Llama-3.2-1B-Instruct-GGUF \
    Llama-3.2-1B-Instruct-Q4_K_M.gguf --local-dir models

# 3. Run it.
./llama-cli \
    -m models/Llama-3.2-1B-Instruct-Q4_K_M.gguf \
    -p "Write a haiku about local LLMs." \
    -n 100 \
    -ngl 999

# 4. Benchmark.
./llama-bench -m models/Llama-3.2-1B-Instruct-Q4_K_M.gguf
```

The `llama-bench` output gives you the canonical performance numbers: prompt processing tokens/sec (the prefill speed) and token generation tokens/sec (the decode speed). On M3 Pro for a 1B INT4 model expect ~150-250 tok/s generation; on a 4090 expect 300-600 tok/s; on a Raspberry Pi 5 expect ~5-15 tok/s.

Then run the OpenAI-compatible server (`./llama-server -m ... -ngl 999`) and verify it speaks the API by querying with `curl` or any OpenAI client library pointed at `http://localhost:8080`.

## Further reading

- llama.cpp GitHub README (github.com/ggerganov/llama.cpp) — the primary documentation; updated frequently.
- "ggml" project documentation — the tensor library that underpins llama.cpp.
- "Why GGUF" — Georgi Gerganov's writeup on the format's design.
- The llama.cpp Discord and r/LocalLlama subreddit — the active communities where new models, optimizations, and quirks get discussed in real time.
- "Whisper.cpp" and "stable-diffusion.cpp" — sibling projects using ggml; comparing their structure to llama.cpp's is informative for understanding the shared infrastructure.

Next lesson: **The runtime decision tree.** A worked example deployment: "I have a quantized 3B model; I need it on iOS and Android; latency budget 200 ms TTFT." We work through the options, defend the choice, and produce the picking matrix that summarizes the module.
