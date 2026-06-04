---
title: "Lesson 10 — Quantization on Apple Silicon"
date: "2026-06-04"
module: "mlx-apple-silicon"
order: 10
tags: ["quantization", "mlx", "gguf", "llama-cpp", "int4", "int8", "apple-silicon"]
author: "Sudipta Pathak"
prerequisites: ["09-kv-cache-unified-memory"]
---

# Lesson 10 — Quantization on Apple Silicon

## Why this lesson exists

Module 3 covered the math and the recipes for INT8 and INT4 quantization. This lesson is the Apple-Silicon-specific deployment view: which runtime supports what format, what conversion paths exist, and what numbers you should actually expect on M-series hardware. The two dominant runtimes are MLX (with its own native quantized format) and llama.cpp (GGUF). They have meaningful differences in performance, supported models, and convenience.

The lesson is also the bridge from Module 3's algorithms to Module 4's hardware. By the end you should be able to take an FP16 model, decide which Apple-Silicon runtime fits your use case, run the quantization, and have a working INT4 model that runs at the expected throughput.

The lesson is reading. The Hands-on does the same model in both runtimes and measures the throughput delta.

## The two runtimes

**MLX**, via `mlx-lm`, supports quantization to:
- INT8 (per-group, group_size=64 default, FP16 scales).
- INT4 (per-group, group_size=32 or 64 default).
- INT3, INT2 (experimental).

The format is MLX-native — not GGUF, not GPTQ, not AWQ. It uses a per-group symmetric quantization with FP16 scales packed alongside the integer weights. The conversion is one command:

```bash
mlx_lm.convert --hf-path meta-llama/Llama-3.1-8B-Instruct --quantize --q-bits 4 --q-group-size 64
```

This downloads the HF FP16 checkpoint, quantizes it, and saves the MLX-native format locally. The conversion takes 1–3 minutes for a 7B-class model on M3 Pro. There's no calibration (just MinMax-based per-group scales), so the accuracy is somewhere between naive INT4 and AWQ/GPTQ.

For better accuracy, mlx-lm also supports loading AWQ-quantized HF models (some quantizers in the community produce MLX-compatible AWQ exports).

**llama.cpp**, via the `llama.cpp` binary, supports GGUF format (covered in Module 3, Lesson 4). The Apple Silicon path uses Metal kernels written specifically for the GGUF layouts. Conversion path:

```bash
# Convert HF to GGUF FP16.
python convert_hf_to_gguf.py --outdir models meta-llama/Llama-3.1-8B-Instruct
# Then quantize.
./quantize models/Llama-3.1-8B-Instruct.gguf models/Llama-3.1-8B-Instruct.Q4_K_M.gguf Q4_K_M
```

The quantize step takes 1–5 minutes on M3 Pro. The Q4_K_M format is the standard 4-bit format for llama.cpp (Module 3 Lesson 4 covered the layout). For better accuracy, Q5_K_M trades 25% more bytes for noticeably better perplexity; Q3_K_M is more aggressive (smaller, slightly worse).

The two runtimes have different model loaders. You can't load a GGUF model in mlx-lm without converting it first; you can't load an MLX-native model in llama.cpp at all. The communities maintain pre-quantized versions of popular models on HuggingFace in both formats.

## Performance on Apple Silicon

Headline numbers for a 7B-class LLM on M3 Pro, INT4, single-stream decode:

| Runtime / format | Tokens/sec | Memory (weights) |
| ---------------- | ---------- | ---------------- |
| MLX-native 4-bit (group=64) | 35–45 | 3.7 GB |
| MLX-native 4-bit (group=32) | 30–40 | 4.0 GB |
| llama.cpp GGUF Q4_K_M | 35–50 | 3.7 GB |
| llama.cpp GGUF Q5_K_M | 25–35 | 4.5 GB |
| MLX-native 8-bit | 25–35 | 7.0 GB |
| FP16 (if it fits) | 12–18 | 14.0 GB |

The MLX vs llama.cpp comparison is close. llama.cpp often leads by a small margin (~10%) on the most common workload (single-stream chat at 7B) because the Metal kernels were aggressively hand-tuned for several years. MLX has been catching up rapidly; for some specific patterns (long context, models with non-standard architectures) MLX is now ahead.

The choice is mostly a workflow question:

- **MLX** if you want Python integration, plan to fine-tune or modify the model, or want the broader MLX ecosystem (vision models, fine-tuning, custom kernels).
- **llama.cpp** if you want a small standalone binary, the widest model coverage, or are deploying to a non-Python environment (Swift, Rust, etc., via bindings).

Both deliver competitive Apple Silicon throughput. The narrative of "llama.cpp is always faster" was true in 2023 but is no longer true in 2026.

## Accuracy

The accuracy comparison at INT4 on a held-out perplexity eval (WikiText-2, typical 7B model):

- FP16 baseline: ~6.5 perplexity.
- MLX 4-bit group=64 (no calibration): ~6.7 (+0.2).
- MLX 4-bit + AWQ quantized: ~6.55 (+0.05).
- llama.cpp Q4_K_M: ~6.6 (+0.1).
- llama.cpp Q5_K_M: ~6.55 (+0.05).
- llama.cpp Q3_K_M: ~7.0 (+0.5).

The MLX-native quantization (no calibration) is slightly worse than llama.cpp's Q4_K_M (which has some internal structure that helps), but the gap is small. For accuracy-critical applications, use Q5_K_M on llama.cpp or AWQ+MLX. For general use, the defaults of either runtime are fine.

The point: the format choices within Apple Silicon are all in a fairly narrow band (~0.2 perplexity); the bigger accuracy decisions are upstream (which model, what calibration, how much training data).

## The conversion workflow

A reproducible workflow for getting from a fresh HF model to a working Apple Silicon quantized deployment:

**Option A: mlx-lm**

```bash
# 1. Install.
pip install mlx mlx-lm

# 2. Quantize. The convert step downloads + quantizes + saves.
mlx_lm.convert \
    --hf-path meta-llama/Llama-3.1-8B-Instruct \
    --mlx-path Llama-3.1-8B-Instruct-MLX-4bit \
    --quantize --q-bits 4 --q-group-size 64

# 3. Run.
mlx_lm.generate \
    --model Llama-3.1-8B-Instruct-MLX-4bit \
    --prompt "Explain RAG in 3 sentences." \
    --max-tokens 200
```

The same model in a Python notebook:

```python
from mlx_lm import load, generate
model, tokenizer = load("Llama-3.1-8B-Instruct-MLX-4bit")
out = generate(model, tokenizer, prompt="...", max_tokens=200, verbose=True)
```

**Option B: llama.cpp**

```bash
# 1. Build.
git clone https://github.com/ggerganov/llama.cpp && cd llama.cpp
make LLAMA_METAL=1     # Apple Silicon Metal build

# 2. Convert HF to GGUF FP16, then quantize to Q4_K_M.
python convert_hf_to_gguf.py --outdir models /path/to/Llama-3.1-8B-Instruct
./quantize models/Llama-3.1-8B-Instruct-F16.gguf models/Llama-3.1-8B-Instruct-Q4_K_M.gguf Q4_K_M

# 3. Run.
./main -m models/Llama-3.1-8B-Instruct-Q4_K_M.gguf \
    -p "Explain RAG in 3 sentences." \
    -n 200 -ngl 999  # offload all layers to GPU
```

**Option C: pre-quantized from the community.** HuggingFace has both MLX-format and GGUF pre-quantized versions of most popular open models. Search for `mlx-community/<model-name>-4bit` or `TheBloke/<model-name>-GGUF`. Skip the conversion step entirely.

In 2026, Option C is what most people use. The conversion paths are useful when:
- You have a model that no one has quantized yet (a private fine-tune, a freshly-released model).
- You want a specific quantization configuration (group size, bit count) that the published version doesn't have.
- You want to validate the quantization quality yourself.

## The fine-tune-then-quantize loop

A common workflow: take a base model, fine-tune it (LoRA or full), then quantize the result.

```bash
# Fine-tune (mlx-lm supports LoRA fine-tuning).
mlx_lm.lora \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --train --data path/to/your/jsonl \
    --iters 1000 --batch-size 4

# Merge LoRA into base weights.
mlx_lm.fuse --model meta-llama/Llama-3.1-8B-Instruct --adapter-path adapters

# Quantize the merged model.
mlx_lm.convert --hf-path fused --quantize --q-bits 4
```

This is the canonical Apple Silicon fine-tuning loop. It works on an M3 Pro (18 GB unified) for 7B models with LoRA; for full fine-tuning of 7B you want M3 Max with 36+ GB.

## What about FP8?

The Module 3 FP8 discussion (Lesson 2) doesn't directly apply on Apple Silicon: the M-series GPUs (through M3) don't have native FP8 tensor cores. M4 added some FP16-via-FP8 paths but they're not as aggressive as Hopper's. For Apple Silicon, the "low-precision floating point" path is FP16 (universally supported) and BF16 (supported on M2+). Going below 16-bit, the path is integer quantization (INT8/INT4), not floating point.

For specific Apple-Silicon FP8 patterns, the situation may change post-M4 — keep an eye on Apple's WWDC announcements and the MLX release notes.

## What you should believe after this lesson

Three sentences:

**1. The two production runtimes for quantized LLM inference on Apple Silicon are MLX and llama.cpp**, with comparable performance (within ~10%) and accuracy (within ~0.2 perplexity at INT4) but different ecosystems — MLX for Python and fine-tuning, llama.cpp for standalone binaries and the widest model coverage.

**2. The default INT4 format choices (MLX 4-bit group=64, llama.cpp Q4_K_M) deliver about 35–50 tokens/sec on a 7B model on M3 Pro**, with the difference between them small enough to be a tie for most uses. The bigger accuracy levers are upstream (AWQ/GPTQ calibration vs naive scales).

**3. Pre-quantized models on HuggingFace are the default starting point in 2026** — the manual conversion path matters only for private fine-tunes or experimental configurations. The fine-tune-then-quantize loop is well-supported in mlx-lm for laptop-scale customization.

## Hands-on (at home)

Compare both runtimes on the same model.

```bash
# Install both.
pip install mlx mlx-lm
git clone https://github.com/ggerganov/llama.cpp && cd llama.cpp && make LLAMA_METAL=1 && cd ..

# Use a pre-quantized model.
# MLX side: download the MLX 4-bit version.
huggingface-cli download mlx-community/Llama-3.2-3B-Instruct-4bit \
    --local-dir Llama-3.2-3B-Instruct-MLX-4bit

# llama.cpp side: download Q4_K_M GGUF.
huggingface-cli download bartowski/Llama-3.2-3B-Instruct-GGUF \
    Llama-3.2-3B-Instruct-Q4_K_M.gguf --local-dir gguf

# Time MLX.
PROMPT="Write a short poem about Apple Silicon."
echo "=== MLX ==="
time mlx_lm.generate --model Llama-3.2-3B-Instruct-MLX-4bit \
    --prompt "$PROMPT" --max-tokens 200 --temp 0

# Time llama.cpp.
echo "=== llama.cpp ==="
time ./llama.cpp/main -m gguf/Llama-3.2-3B-Instruct-Q4_K_M.gguf \
    -p "$PROMPT" -n 200 -t 4 -ngl 999 --temp 0
```

The mlx_lm.generate and llama.cpp main both report tokens/sec at the end. Compare:
- The tokens/sec numbers (often within 10% of each other).
- The wall-clock time (includes model loading, which is the slower step in both).
- The output text (should be identical or near-identical at temp 0, modulo subtle tokenizer/sampling differences).

For a richer comparison, vary the prompt length (short vs long) and the model size (3B, 7B, 13B if you have the memory). The relative ordering varies; both runtimes have workloads where each wins.

## Further reading

- `mlx-lm` README and the conversion docs — the entry point for MLX-native quantization.
- `llama.cpp` README — for the GGUF formats and quantization options (much more variety than just Q4_K_M).
- HuggingFace `mlx-community` and `TheBloke`/`bartowski` users — pre-quantized model collections.
- "Quantizing LLMs for inference" (various blog posts) — non-Apple-specific overview that complements this lesson.
- mlx-examples LoRA fine-tuning guide — for the fine-tune-then-quantize loop in detail.

Next lesson: **The Apple Neural Engine.** What the ANE actually is and isn't, what it can run (and what it can't), the Core ML bridge, and the latency-vs-flexibility tradeoff that explains why most LLM inference still runs on the GPU even though the ANE exists and is technically much more power-efficient.
