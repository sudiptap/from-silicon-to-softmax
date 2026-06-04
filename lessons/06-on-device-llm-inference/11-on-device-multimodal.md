---
title: "Lesson 11 — On-Device Multimodal"
date: "2026-06-04"
module: "on-device-llm-inference"
order: 11
tags: ["multimodal", "vision-language", "moondream", "florence", "smolvlm", "phi-vision", "vision-encoder"]
author: "Sudipta Pathak"
prerequisites: ["10-lora-hotswap"]
---

# Lesson 11 — On-Device Multimodal

## Why this lesson exists

The text-only LLM story is the established pattern. The multimodal story — vision-language models that take images and questions and produce text answers, audio-language models that transcribe and reason — is where on-device serving in 2026 is actively growing.

The engineering shape is different from text-only. A VLM (Vision-Language Model) has two components: a *vision encoder* (typically a ViT-class image transformer) that maps an image to a sequence of "vision tokens," and a *language model* (typically a standard decoder transformer) that consumes both vision tokens and text tokens. The vision encoder's per-request cost is large; the language model's per-token cost dominates only for long responses. The optimization patterns differ accordingly.

This lesson covers the on-device VLM landscape circa mid-2026, the architectural patterns, and the engineering tricks (encoder caching, image sharing across queries) that make multimodal viable on consumer hardware.

The lesson is reading. The Hands-on runs a small VLM and measures the vision-encoder vs language-model cost split.

## What a VLM does, mechanically

A typical VLM forward pass:

1. **Image preprocessing**: resize to a fixed input size (typically 224×224, 336×336, or 448×448), normalize, tile if necessary.
2. **Vision encoding**: pass the image through the vision encoder (a ViT or CNN). Output: a sequence of *vision tokens* (typically 196 to 1024 tokens) plus an optional set of pooled features. The vision encoder is typically 50M–600M parameters.
3. **Projection**: a small MLP maps vision tokens into the language model's hidden dimension.
4. **Language model**: prepend the vision tokens to the text tokens, run the standard transformer forward pass, generate text tokens autoregressively.

The cost profile:

- **Vision encoding**: one-shot per image, takes 50–500 ms depending on encoder size and image resolution.
- **Prefill of vision tokens + text prompt**: ~ standard prefill cost on the LM portion.
- **Decode of response tokens**: standard decode rate.

For a single-image question with a 100-token response on a 3B-class VLM (typical: ~400M vision encoder + 3B language model):
- Vision: ~150 ms.
- LM prefill of 600 vision tokens + ~50 text tokens: ~150 ms.
- LM decode of 100 tokens: ~2.5 s.

Total: ~3 seconds. The decode time dominates; the vision encoder is "only" 5% of the wall-clock cost for a typical query. For very short responses (single-sentence answer to "what's in this image?"), the vision encoder becomes a larger fraction.

## The major on-device VLMs (mid-2026)

The credible deployment candidates:

**Moondream (1.86B params)**: A focused, small VLM optimized for fast inference. ~700M vision encoder + ~1B language model. Aggressive quantization-friendly design. Strong for image QA and OCR; weaker than larger models on complex reasoning.

When to use: tight memory budget, fast simple-image-QA tasks.

**Florence-2 (large: 770M, base: 230M)**: Microsoft's vision model. Originally designed as a vision-foundation model (not a chat VLM), but with reasonable image-to-text capabilities. Smaller than Moondream; can do detection, captioning, OCR. Less suited to free-form conversation.

When to use: structured vision tasks (caption, detect, OCR) where you don't need conversational answers.

**SmolVLM (256M / 500M / 2.2B)**: HuggingFace's small VLM line. Designed for on-device, with very-small variants for tight memory. Quality scales with size; the 2.2B version is competitive with Moondream.

When to use: phone-class memory budgets.

**Phi-3.5 Vision (4.2B)**: Microsoft's vision-capable Phi. ~400M vision encoder + ~3.8B language model. Strong on reasoning about images; can handle multi-image inputs. Roughly the high end of "comfortably on-device."

When to use: laptop deployments where you want reasoning quality, not just simple captioning.

**Qwen2.5-VL (3B / 7B)**: Alibaba's vision-language line. Multilingual; strong OCR; supports video. The 3B version is on-device-friendly.

When to use: multilingual VLM, especially for non-English images.

**Llama 3.2 Vision (11B / 90B)**: Meta's vision variants. The 11B fits on a Mac with INT4; the 90B is server-side. Less optimized for on-device than the dedicated SLM-VLMs above.

When to use: when you specifically need Llama's behavior; otherwise Phi-3.5 Vision or Qwen2.5-VL at similar size are usually better.

## Vision encoder cost-saving patterns

The vision encoder is fixed-cost per image; you pay it whether the model produces 10 tokens or 1000. A few engineering patterns reduce or amortize the cost:

**1. Encode once, query many times.** If the user asks multiple questions about the same image, encode the image once and cache the vision tokens. Each subsequent question only pays the LM cost. Significant win for image-based chat.

```python
vision_tokens = vision_encoder(image)  # 150 ms, once
for question in user_questions:
    # Only LM cost per question.
    response = lm.generate(vision_tokens + tokenize(question))
```

**2. Resolution scaling by task.** For OCR you need high resolution (448×448 or higher). For "what's the main subject" you can downsample to 224×224 and save 4× encoder cost. Resolution is a tunable parameter.

**3. Tiling for high-resolution images.** Some VLMs (Phi-3.5 Vision, Qwen-VL) support tiling: split a high-res image into 4-9 tiles, encode each independently, concatenate vision tokens. Trades encoder cost for resolution.

**4. Skip the vision encoder when the image is already-encoded.** For app workflows where the same image is queried by multiple users or sessions, cache the vision encoding centrally (server-side, but applicable on-device for multi-session apps).

**5. Lightweight encoders.** Some VLMs use compact encoders (Moondream's is 700M, but PaliGemma's is even smaller). For tasks that don't need pixel-precise understanding, a smaller encoder is fine.

## Vision-related quantization

The vision encoder can be quantized like any other transformer — INT4 / INT8 for weights, with calibration. The catch: vision encoders are more sensitive than language models to aggressive quantization. INT4 vision encoders sometimes produce noticeably worse downstream answers; INT8 is usually safe.

The on-device recipe in 2026:

- Vision encoder: INT8 weights (sometimes FP16).
- Vision-to-LM projection MLP: FP16 (it's tiny anyway).
- Language model: INT4 weights, INT8 KV cache (standard).

The vision encoder's memory cost at INT8 is ~350 MB for a 700M encoder. Plus a 3B INT4 LM (~1.5 GB). Plus KV cache (1–2 GB). Plus vision tokens (small). Total ~3.5 GB for a working multimodal deployment, which fits on flagship phones with care.

## Audio: a similar story

Audio-language models follow the same shape as vision-language: an audio encoder (Whisper-class, ~250M-1.5B parameters) produces a sequence of audio tokens that prepend to text tokens for an LM.

The on-device audio story in 2026:

- **Whisper** (tiny / base / small / medium): the dominant ASR (automatic speech recognition) model. Whisper-base (~70M) runs comfortably on phones; Whisper-medium (~770M) on laptops.
- **Parakeet** (NVIDIA): a competitive ASR alternative; smaller and faster than Whisper at comparable accuracy.
- **Voxtral**: a more recent audio-language model (audio in, text out, with reasoning).

The end-to-end audio chat pattern:

1. Microphone → audio encoder → audio tokens.
2. Audio tokens + system prompt → LM → response text.
3. (Optional) Response text → TTS → speaker.

The TTS step (Module 6 Lesson 12 covers this) is increasingly handled by on-device neural TTS models like Coqui, F5-TTS, or platform-provided (iOS Speech, Android TextToSpeech).

The total memory for an audio-chat deployment: Whisper-base (~150 MB INT8) + 3B LM (~1.5 GB INT4) + TTS (~100-300 MB) + KV cache (~1 GB) = ~3 GB. Tight on phones but feasible.

## What you should believe after this lesson

Three sentences:

**1. A VLM is a vision encoder + a language model bolted together** — the encoder is a one-shot per-image cost (50–500 ms); the language model is the usual per-token cost. The right model depends on resolution requirements and memory budget; Moondream and SmolVLM are the phone-friendly leaders, Phi-3.5 Vision and Qwen2.5-VL the laptop-friendly leaders.

**2. Encoder caching is the key optimization** — encode an image once, query it many times. For chat about an image this turns "every question costs an encoder call" into "first question costs an encoder, subsequent questions don't."

**3. Audio-language models follow the same pattern** with Whisper-class encoders feeding into LM; the on-device audio chat stack (mic → ASR → LM → TTS → speaker) fits in ~3 GB on a top-tier phone in 2026 and is the foundation for the always-listening voice assistant pattern.

## Hands-on (at home)

Run a small VLM and measure the encoder-vs-LM cost split.

```bash
# Install MLX VLM support.
pip install mlx-vlm

# Download a small VLM.
huggingface-cli download mlx-community/Moondream2-4bit --local-dir moondream2-mlx
```

```python
# vlm_demo.py
from mlx_vlm import load, generate
from mlx_vlm.utils import load_config
import time

model, processor = load("moondream2-mlx")
config = load_config("moondream2-mlx")

image_path = "your_image.jpg"  # any image
question = "What is shown in this image?"

# Warm.
generate(model, processor, image_path, question, max_tokens=10)

# Time the full pipeline.
t0 = time.time()
out = generate(model, processor, image_path, question, max_tokens=100, verbose=True)
total = time.time() - t0
print(f"Total time: {total*1000:.0f} ms")
print(f"Output: {out}")
```

The verbose output shows the prompt-eval and generation tok/s, separately. For Moondream2 on M3 Pro, expect:
- Vision encoder + prefill: ~150-200 ms.
- Generation: 30-50 tok/s.

For the encoder-caching pattern:

```python
# Encode once, ask multiple questions.
# (API surface varies by VLM library; this is conceptual.)
vision_features = encode_image(image_path)
for q in ["What's in this image?", "What color is the main subject?", "Is there text?"]:
    response = lm_generate_with_vision(model, vision_features, q, max_tokens=50)
    print(f"Q: {q}\nA: {response}\n")
```

The total time should be `(encoder time) + N × (LM time)`, not `N × (encoder + LM time)`. Significant savings for multi-question workflows.

## Further reading

- "Moondream: A Smaller Vision Language Model" — the Moondream project blog/papers.
- "Florence-2: Advancing a Unified Representation for a Variety of Vision Tasks" (Xiao et al, 2023).
- "Phi-3.5 Vision" technical report (Microsoft).
- "Qwen2.5-VL Technical Report" (Alibaba).
- "Llama 3.2 Vision" announcement (Meta).
- "SmolVLM" announcement (HuggingFace, 2024).

Next lesson: **Real-time interactive use cases (module wrap).** We compose the patterns from Modules 4-6 into specific deployment shapes — speech assistants, code completion, vision pipelines — with end-to-end budget breakdowns and a final decision tree. The handoff to Module 7 takes us from "deploying inference" to "building the inference engine from scratch."
