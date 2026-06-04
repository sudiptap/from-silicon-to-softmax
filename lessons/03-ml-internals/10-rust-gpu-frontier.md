---
title: "Lesson 10 — The Rust GPU Frontier: cubecl, rust-gpu, Burn"
date: "2026-06-04"
module: "ml-internals"
order: 10
tags: ["rust", "gpu", "cubecl", "rust-gpu", "burn", "wgpu", "ecosystem"]
author: "Sudipta Pathak"
prerequisites: ["09-distillation"]
---

# Lesson 10 — The Rust GPU Frontier: cubecl, rust-gpu, Burn

## Why this lesson exists

The first nine lessons of this module were format-and-algorithm focused. This one is ecosystem focused: where the Rust-on-GPU story sits in 2026, and why it matters for low-precision ML systems specifically. The brief answer: Rust + CUDA / Rust + Metal / Rust + Vulkan is no longer a hobbyist experiment. Real frameworks exist (Burn, Candle, cubecl) that target the same problems as PyTorch + CUDA, with the advantages and disadvantages you'd expect from a non-Python compiled-language ecosystem. None of them have displaced PyTorch — and may never — but a handful of categories (cross-platform inference, embedded deployments, kernel research) increasingly favor the Rust path.

This lesson is a survey, not a tutorial. It's also the most opinionated lesson in this module, because the Rust GPU ecosystem is moving fast and any specific claim can age in months. Read this as a snapshot of "what's worth tracking" rather than "which tool to use."

The lesson is reading. The Hands-on points at a few minimal Rust + GPU examples worth running if you want to feel the texture; it's not strictly required to follow the rest of the module.

## The landscape

Three loose categories of Rust-on-GPU work:

**1. ML frameworks written in Rust.** Burn, Candle, dfdx (deprecated but historical), tch-rs (Rust bindings over LibTorch — not quite native but counts). These are the closest to "Rust PyTorch" — autograd, training, inference, layered atop GPU backends.

**2. GPU compute frameworks in Rust.** cubecl (Hugging Face / Burn team), rust-gpu (Embark Studios, Vulkan/SPIR-V focused), wgpu (cross-platform graphics + compute). These let you write kernels in Rust (or in Rust-like DSL) and run them on a GPU. Not specifically ML — general GPU compute that ML can use.

**3. Direct vendor bindings.** cuda-rs, cudarc, metal-rs. Thin layers over CUDA / Metal that let Rust drive the GPU more or less the way C++ would. Lowest abstraction; highest control; least portable.

The three categories interact: Burn uses cubecl, cubecl can target wgpu or CUDA via cudarc, etc. Picking one means buying into a stack of dependencies.

## Burn — the closest thing to "Rust PyTorch"

Burn is a Rust deep learning framework that has matured rapidly since 2023. By 2026 it offers:

- Tensors, autograd, optimizers, common layer types, transformer modules.
- Backend abstraction: same model code can run on `ndarray` (CPU), `wgpu` (cross-platform GPU), `cuda` (CUDA via cubecl), `metal` (via cubecl/wgpu), and a few experimental backends.
- Training and inference paths; serialization in a native format.
- ONNX import for bringing models in from PyTorch / TensorFlow.

What it doesn't have (compared to PyTorch):

- The vast ecosystem of model implementations, pretrained checkpoints, and research code. You will not find a Burn implementation of the latest paper a week after it dropped; you may find one months later or never.
- Distributed training (multi-node) support that matches PyTorch's DDP / FSDP.
- The full breadth of optimization passes that `torch.compile` offers.

Where Burn shines:

- **Cross-platform inference deployment.** Same code runs on a Linux server with NVIDIA, a Mac with Metal, a Windows machine with Vulkan, and a Raspberry Pi with wgpu's CPU fallback. PyTorch can do this in principle but the deployment story is painful.
- **Embedded / constrained environments.** Rust's no-std support and Burn's careful dependency choices make it feasible to run inference in environments where Python is impossible.
- **Production deployment where a Rust binary is preferred** for the usual reasons (no Python runtime, no GIL, easier ops story, predictable memory).

For Module 3 specifically, Burn supports INT8 quantized inference and is gaining INT4 support; the underlying cubecl kernels are where the action is.

## Candle — Hugging Face's minimalist take

Candle is Hugging Face's Rust ML framework, designed to be a "minimalist Rust replacement for PyTorch for inference workloads." Smaller than Burn in scope; opinionated on inference (training paths exist but are not the focus).

What Candle does well:

- **Pre-built models for common LLMs.** Llama, Mistral, Phi, Whisper, and many others have first-class Candle implementations.
- **Quantization support including GGUF.** Candle can load llama.cpp-format GGUF files directly and run them on GPU. This is a real Apple-Silicon-friendly path for LLM inference outside the Python ecosystem.
- **Compact binaries.** A Candle-based inference server can be ~5–20 MB statically linked, vs 1+ GB for a PyTorch-based deployment.

What it doesn't:

- Comprehensive training support.
- The breadth of layer types and architectures Burn supports.

The split between Burn and Candle in 2026 is roughly: Burn is the "build models from scratch in Rust" framework; Candle is the "deploy pretrained models from Rust" framework. They have overlapping but distinct use cases.

## cubecl — the GPU compute DSL

cubecl (developed by the Burn team, sponsored by Hugging Face) is the most interesting piece for this module. It's a Rust-embedded DSL for writing GPU kernels that compile to multiple backends:

- CUDA (via PTX generation through cudarc).
- WebGPU / wgpu (via WGSL generation).
- Metal (via MSL generation).
- ROCm (in progress as of 2026).

You write a kernel once in Rust syntax, and cubecl compiles it to whichever backend the user runs. The DSL covers the kinds of things you'd write in CUDA C++ or Triton — thread blocks, shared memory, warp-level intrinsics — with type safety and Rust's lifetime model on top.

The pitch versus Triton: same goal (one source, multiple GPU backends) but a different target audience. Triton is Python-first, tightly coupled to PyTorch, NVIDIA-focused. cubecl is Rust-first, framework-agnostic, multi-vendor by design. For someone who wants to write a custom INT4 matmul kernel that runs on both NVIDIA and Apple Silicon from the same source, cubecl is the obvious choice in the Rust ecosystem; Triton requires writing the Apple side separately (or using torch.compile + MLX bridging, which is not a clean path).

What cubecl gives you concretely:

- Same kernel runs on CUDA and Metal.
- Performance within ~10–20% of hand-written CUDA on the NVIDIA side; usually competitive with hand-written Metal on Apple Silicon.
- Tightly integrated with Burn; you can write a custom kernel and call it directly from a Burn model.

What it doesn't (yet):

- Full coverage of every vendor-specific intrinsic. The bleeding-edge Hopper paths (`wgmma`, async TMA) are partially supported; some still require dropping to CUDA.
- The Python integration that would let you call cubecl kernels from a PyTorch model. (This is on the roadmap but not there in 2026.)

For Module 3, cubecl matters because **it's the most credible path in the Rust ecosystem for writing custom low-precision kernels** that need to ship cross-platform. If you're targeting on-device deployment across NVIDIA, Apple, and (eventually) AMD, cubecl is the kernel-writing tool to track.

## rust-gpu — the long-standing experiment

rust-gpu (Embark Studios) lets you write GPU shaders in plain Rust syntax, compiling to SPIR-V (the Vulkan / OpenCL intermediate format). The pitch: write GPU code in the same language as your CPU code, with the same tooling.

Where it sits in 2026:

- Mature enough for production graphics work.
- Not the natural fit for ML compute — SPIR-V targets work, but the ML ecosystem (NVIDIA, Apple) prefers CUDA and Metal directly rather than going through the Vulkan layer.
- A reasonable choice if you're already in the Vulkan ecosystem (mobile graphics, cross-platform engines) and want to add ML compute.

Less relevant for this module than cubecl, but worth knowing exists.

## Direct vendor bindings

When you need maximum control and accept the maintenance burden:

- **cudarc:** Rust bindings to CUDA. The most mature option for direct CUDA work in Rust. Used internally by cubecl and Candle's CUDA backend.
- **metal-rs:** Rust bindings to Metal. The Apple-side equivalent.
- **cust:** Higher-level CUDA wrapper, less commonly used than cudarc.

These are appropriate when you're writing a kernel that depends on vendor-specific features (e.g., a Hopper-only `wgmma` matmul) and you've decided the abstraction layer isn't worth it. They're how you get full CUDA performance from Rust at the cost of platform lock-in.

## Why Rust at all for ML?

The honest case in 2026:

**1. Deployment.** Compiled binaries are easier to ship than Python+CUDA Docker images for many on-device, edge, and embedded scenarios. A Burn or Candle inference binary can be 10–50 MB; a comparable PyTorch deployment is gigabytes of base image plus dependencies.

**2. Memory and concurrency.** Rust's ownership model makes large-scale concurrent inference servers (handling many users on a multi-GPU box) easier to write correctly than equivalent Python or C++.

**3. Toolchain.** `cargo build` is a single command that produces a deployable binary; the equivalent in Python+CUDA is a manual mess of pip, conda, system CUDA installations, and driver matching. The win compounds across a deployment fleet.

**4. Specialized environments.** Browser deployment via wasm + wgpu, embedded ML on no-std targets, distributed systems where Rust's predictable performance matters.

The honest case *against*:

**1. Ecosystem.** PyTorch has 10+ years of model implementations, papers, tutorials, and community knowledge. Burn / Candle / cubecl combined have a tiny fraction of this.

**2. Research velocity.** The newest paper drops on arXiv with a PyTorch reference implementation. You will not find a Rust port for weeks-to-months, often never.

**3. Talent.** Most ML engineers know PyTorch; relatively few know Rust well enough to be productive in a Rust ML stack.

The combination means the Rust ML ecosystem is, in 2026, a strong fit for *production deployment* and a weak fit for *research and experimentation*. The two camps will likely co-exist — Python remains dominant for research; Rust grows in deployment niches.

## What you should believe after this lesson

Three sentences:

**1. The Rust ML ecosystem (Burn, Candle, cubecl) is mature enough for production inference in 2026 and is the better choice for cross-platform on-device deployment, embedded ML, and Rust-shop integrations.** It is not the right choice for research or for tracking the latest paper.

**2. cubecl is the piece worth watching for Module 3 specifically** — a Rust-embedded GPU compute DSL that compiles to CUDA, Metal, and Vulkan from one source, and is the natural place for cross-platform custom low-precision kernels in Rust.

**3. For most readers in 2026, the right stance is to know the Rust ecosystem exists and what it does well, then keep using PyTorch and Python for day-to-day ML systems work** unless you have a specific deployment pressure (binary size, no-Python target, multi-vendor portability) that the Rust path solves and Python doesn't.

## Hands-on (optional)

Build a Candle binary that runs a small quantized LLM. Requires Rust toolchain (`rustup`).

```bash
# candle_llm_demo.sh
git clone https://github.com/huggingface/candle.git
cd candle/candle-examples
cargo run --release --bin quantized -- \
    --model-id TheBloke/Llama-2-7B-Chat-GGUF \
    --weight-file llama-2-7b-chat.Q4_K_M.gguf \
    --prompt "Explain entropy in one sentence."
```

This will pull the GGUF file (~4 GB), compile the Rust binary (~3–5 minutes first run), and run quantized inference. The binary itself is ~15 MB after `cargo build --release`.

For a cubecl taste, the Burn book has a chapter "Writing custom kernels with cubecl" that walks through a small reduction kernel — same one we built in Module 2, Lesson 9 — and shows it compiling to both CUDA and WGPU.

If you don't want to install the Rust toolchain just for this lesson, that's fine. The point of this lesson is to know the ecosystem exists; the practical work on the low-precision side of this module remains primarily in Python.

## Further reading

- Burn project README and the "Burn Book" — the official intro.
- Candle examples directory on GitHub — a tour of what Candle does well (LLM inference, Whisper, Stable Diffusion).
- cubecl GitHub README and the design document — for what the DSL targets and how the backends compose.
- "Why Rust for Machine Learning?" — various community blog posts; the broad case made repeatedly with slightly different emphases.
- WonderfulWasm and the WebGPU compute story — for the browser-side angle on cross-platform Rust GPU compute.

Next lesson: **Module wrap — picking your compression budget.** A practical decision tree (target device → memory budget → format → algorithm → validation) that ties everything in this module together, plus the handoff to Module 4 where we go deep on the Apple Silicon side and start running quantized models on real on-device hardware.
