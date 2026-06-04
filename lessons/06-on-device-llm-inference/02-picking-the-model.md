---
title: "Lesson 2 — Picking the Model: the SLM Landscape"
date: "2026-06-04"
module: "on-device-llm-inference"
order: 2
tags: ["slm", "model-selection", "phi", "llama", "gemma", "qwen", "smolllm", "granite"]
author: "Sudipta Pathak"
prerequisites: ["01-on-device-stack"]
---

# Lesson 2 — Picking the Model: the SLM Landscape

## Why this lesson exists

The first half of "running an LLM on your device" is picking which LLM. In 2026 the field of capable small language models (SLMs, loosely defined as <8B parameters) has matured to the point where the choice matters more than the implementation. A well-chosen 1B model can outperform a poorly-chosen 7B model on the task you actually care about, at 7× lower memory and latency cost.

This lesson is the field guide to the SLMs that are credible on-device deployment candidates as of mid-2026. We cover the major families, their relative strengths, and a practical matching framework: given a task and a device class, which model do you pick?

The lesson is reading. The Hands-on runs three small models on the same prompt set and compares.

## The major families (mid-2026)

The serious candidates, organized by maintainer:

### Microsoft Phi

The Phi family (Microsoft Research) is built around the thesis that *training data quality matters more than parameter count*. Phi-3.5 mini (3.8B) and Phi-3.5 small (7B) were both trained on heavily curated synthetic and filtered data, including distilled reasoning chains. The result: capabilities competitive with larger Llama / Mistral models at 1/2 to 1/4 the parameter count.

What Phi is good at:
- Reasoning tasks (math, code, structured logic). Phi-3.5 mini on MATH or GSM8K is competitive with Llama 3 8B.
- Following instructions precisely.
- Compact deployments where parameter count matters more than the absolute capability.

What it's not:
- Creative writing or open-ended conversation; can feel mechanical.
- Multilingual; primarily English.
- Long context (out-of-the-box limits are modest).

When to pick: agent-like or reasoning-heavy tasks on tight memory budgets.

### Meta Llama 3 / 3.1 / 3.2

The Llama line is the workhorse open-weights family. Llama 3.2 1B and 3B (released late 2024) are the on-device-friendly sizes; Llama 3.1 8B is the next tier up. These are dense decoder-only transformers with the standard architecture choices (RoPE, RMSNorm, SwiGLU, GQA).

What Llama 3.2 is good at:
- General-purpose chat and instruction following.
- Wide ecosystem support (every runtime supports Llama).
- Reliable, well-understood behavior; many pretrained / fine-tuned variants.

What it's not:
- The absolute strongest reasoner at its size (Phi often beats it on math).
- Not multimodal (Llama 3.2 Vision is a separate model with vision encoder).

When to pick: default for general chat / instruction following on-device.

### Google Gemma 2 / Gemma Nano

Gemma 2B and 9B (Gemma 2) and Gemma Nano (the 1.5B-class version designed for Pixel devices) are Google's open-weights line. Architecturally similar to Llama with some Google-specific choices (sliding-window attention for context efficiency).

What Gemma is good at:
- Solid general performance, similar to Llama at equivalent sizes.
- Sliding-window attention in some variants gives better long-context behavior at fixed memory.
- Tightly integrated with Google's on-device deployment story (MediaPipe LLM Inference).

What it's not:
- Less ecosystem support than Llama or Qwen.
- Updates less frequent.

When to pick: Android-first deployments via MediaPipe; or when sliding-window context efficiency matters.

### Alibaba Qwen 2.5

Qwen 2.5 is Alibaba's open-weights family, with 0.5B, 1.5B, 3B, 7B sizes (plus larger versions). Trained on a multilingual corpus with strong attention to code and math.

What Qwen is good at:
- Multilingual (especially Chinese, Japanese, Korean; strong English).
- Code generation; Qwen 2.5 Coder is a code-specific variant.
- Aggressive sizing — Qwen 2.5 0.5B is the smallest "actually useful" general LLM you can deploy.
- Strong reasoning at small sizes.

What it's not:
- Some Western evals lag (different training data emphasis).
- Slightly less ecosystem support than Llama.

When to pick: multilingual deployments, very tight budgets (the 0.5B fits anywhere), or code-focused workloads.

### IBM Granite

The Granite family (Granite 3.x) targets enterprise use cases — tool-use, structured output, code, RAG-friendly behavior. Sizes from 1B to 8B, with multiple variants (instruct, code, embedding).

What Granite is good at:
- Enterprise tasks: tool-calling, structured JSON output, retrieval-augmented generation.
- Permissive licensing (Apache 2.0).
- Granite Code variants for programming tasks.

What it's not:
- Less popular for general chat.
- Smaller community / fewer pre-quantized variants on HF.

When to pick: enterprise app deployments where the LLM is part of a tool-using agent system.

### HuggingFace SmolLM

SmolLM (135M / 360M / 1.7B) is the "how small can you go" line from HuggingFace. The 135M version isn't a real chat assistant; the 360M is borderline; the 1.7B starts to be useful.

What SmolLM is good at:
- Extreme memory budgets (microcontroller-class with the 135M).
- Educational use; the training pipeline is open and well-documented.
- Embedded scenarios where any LLM is a stretch.

What it's not:
- A general-purpose chat assistant; capabilities are limited at this scale.

When to pick: very tight memory; or research / learning.

### Mistral

Mistral 7B was the canonical small model of 2023; Mistral Nemo (12B) and Mistral Small 3 (24B) are larger; Ministral 3B is the on-device-sized one (2024).

What Ministral is good at:
- Strong general capability at the 3B size; sliding-window attention.
- Competitive with Llama 3.2 3B and Phi-3.5 mini on most evals.
- Apache 2.0 licensed.

When to pick: when you want a Llama-3.2-3B alternative.

### Apple's on-device foundation model

Apple's own on-device LLM (3B parameters), available via Apple Intelligence on iOS 18+ devices. You can't deploy it as a standalone weights file; you call it through the Foundation Models framework.

What it's good at:
- Tightly integrated with iOS, leverages the ANE more aggressively than community models can.
- Free (no API cost) for app developers within rate limits.

What it's not:
- Cross-platform.
- Customizable beyond adapters Apple supports.

When to pick: iOS-only apps where you want Apple's hardware integration and aren't bothered by being tied to Apple's specific model.

## A capability snapshot

Approximate (mid-2026) comparison on common benchmarks for the on-device-sized variants:

| Model | Params | MMLU | GSM8K | HumanEval | Multilingual |
| ----- | ------ | ---- | ----- | --------- | ------------ |
| Phi-3.5 mini | 3.8B | 69 | 82 | 65 | Weak |
| Llama 3.2 3B Instruct | 3.2B | 63 | 70 | 50 | Moderate |
| Gemma 2 2B | 2.6B | 56 | 60 | 45 | Moderate |
| Qwen 2.5 3B Instruct | 3.1B | 65 | 75 | 60 | Strong |
| Ministral 3B | 3B | 60 | 68 | 50 | Moderate |
| Granite 3.0 2B | 2.5B | 55 | 60 | 50 | Moderate |
| Llama 3.2 1B Instruct | 1.2B | 50 | 45 | 35 | Moderate |
| Qwen 2.5 1.5B Instruct | 1.5B | 55 | 60 | 45 | Strong |
| Phi-3 mini (older 3.8B) | 3.8B | 65 | 75 | 60 | Weak |
| SmolLM 1.7B Instruct | 1.7B | 45 | 35 | 25 | Weak |

The numbers shift as new versions ship; treat this table as illustrative of *order of magnitude*, not as a precise leaderboard.

Patterns:
- At ~3B, Phi-3.5 mini and Qwen 2.5 3B are the strongest.
- At ~1.5B, Qwen 2.5 1.5B leads.
- At <1B, options are limited and capabilities are correspondingly limited.

## The picking framework

A simple recipe for picking the right model:

1. **Identify the task class.** Chat, code, math, multilingual, agent/tool-use, RAG, summarization, classification?
2. **Identify the device class.** From Lesson 1 — phone / midrange tablet / Mac / desktop.
3. **Cap the parameter count to fit the device class.** Phone: ≤3B at INT4. Mac mid-tier: ≤8B at INT4 or ≤3B at FP16. Desktop: ≤30B at INT4.
4. **Pick within the cap, by task fit:**
   - Chat / general → Llama 3.2 (1B for phone, 3B for laptop).
   - Reasoning / math → Phi-3.5 mini.
   - Code → Qwen 2.5 Coder or Granite Code.
   - Multilingual → Qwen 2.5.
   - Enterprise tool-use → Granite 3.
   - iOS-only with platform integration → Apple's on-device model via Foundation Models.

This framework converges quickly; most deployments map to 1–2 candidates.

## Don't sleep on smaller variants

A frequently-overlooked observation: **the smaller variant of a family is often better for your use case than the larger.** A 1B model that fits comfortably with room for a 16K KV cache and runs at 50 tok/s is more useful than a 3B model that runs at 15 tok/s with a 2K cache cap. For latency-sensitive applications (code completion, voice assistant), the user-facing experience of the smaller, faster model is better even if its eval scores are lower.

Try the smallest viable model first. Step up if quality is the binding constraint.

## What you should believe after this lesson

Three sentences:

**1. The SLM landscape in 2026 has ~7 serious open-weights families** (Phi, Llama, Gemma, Qwen, Granite, SmolLM, Mistral/Ministral) plus Apple's on-device model. Each has a niche; no single "best" model.

**2. Pick by task class + device class**: chat → Llama, reasoning → Phi, code → Qwen Coder / Granite Code, multilingual → Qwen, enterprise → Granite, iOS-integrated → Apple's. Within the family, pick the smallest variant that meets the quality bar.

**3. Smaller variants often win on user-facing experience** because the latency/throughput improvement (2–4× faster) outweighs the eval-score decline. Try the smallest model first; step up if quality is the constraint, not the other way around.

## Hands-on (at home)

Run three small models on the same prompt set and compare.

```bash
# Install runtimes.
pip install mlx mlx-lm

PROMPTS=(
    "Explain the difference between TCP and UDP in three sentences."
    "Write a Python function to compute the nth Fibonacci number iteratively."
    "What is 23 * 47?  Show your work."
    "Translate to French: 'The quick brown fox jumps over the lazy dog.'"
)

MODELS=(
    "mlx-community/Llama-3.2-1B-Instruct-4bit"
    "mlx-community/Qwen2.5-1.5B-Instruct-4bit"
    "mlx-community/Phi-3.5-mini-instruct-4bit"
)

for model in "${MODELS[@]}"; do
    echo "=========================================="
    echo "Model: $model"
    echo "=========================================="
    for prompt in "${PROMPTS[@]}"; do
        echo "Prompt: $prompt"
        mlx_lm.generate --model "$model" --prompt "$prompt" --max-tokens 150 --temp 0.0
        echo ""
    done
done
```

Read the outputs. Note:
- Which model's code is most accurate?
- Which model handles the math correctly?
- Which model handles the French translation cleanly?
- How long does each take?

For most users running this on M3 Pro, you'll see:
- Llama 3.2 1B: fastest (~80–120 tok/s), competent on chat, weak on math.
- Qwen 2.5 1.5B: ~60–90 tok/s, strong on translation, decent on math.
- Phi-3.5 mini: ~30–50 tok/s, best on math/code, English-only.

The "best" model depends on which prompts matter for your application.

## Further reading

- Hugging Face model cards for each family — the authoritative description of capabilities and limitations.
- "Open LLM Leaderboard" (HuggingFace) — for the eval-driven view; treat with care because evals don't perfectly correlate with user-facing quality.
- LMSys Chatbot Arena leaderboard — for the human-rated chat-quality view; covers larger models but the trends apply.
- Apple's "On-Device Foundation Models" documentation (developer.apple.com) — for the iOS-specific path.
- Microsoft's Phi technical reports — the philosophy of "training data quality > parameter count."

Next lesson: **Aggressive quantization recipes.** With the model chosen, the next step is fitting it on the device. We go beyond Module 3's baseline INT4 into the per-layer-precision-stepping, mixed-precision-quantization, and runtime-specific tuning that squeezes the last bit of headroom out.
