---
title: "Lesson 12 — Real-Time Interactive Use Cases (Module Wrap)"
date: "2026-06-04"
module: "on-device-llm-inference"
order: 12
tags: ["voice-assistant", "code-completion", "vision-pipeline", "tts", "wrap", "end-to-end"]
author: "Sudipta Pathak"
prerequisites: ["11-on-device-multimodal"]
---

# Lesson 12 — Real-Time Interactive Use Cases (Module Wrap)

## Why this lesson exists

Module 6 has been individual techniques: model picking, quantization, KV cache, mmap, streaming, speculative decoding, LoRA, multimodal. This lesson is the integration: we compose those techniques into three concrete real-time use cases — a voice assistant, a code completion path, a vision QA pipeline — with end-to-end budget breakdowns. Then we wrap the module and hand off to Module 7.

The unifying question across the use cases: what does the end-to-end latency budget look like, and which component does it bottleneck on? The answer is rarely "the LLM"; it's often the speech recognition, the TTS, the network round-trip, or the rendering. Knowing where the time actually goes lets you focus the optimization effort where it pays off.

The lesson is reading. The Hands-on builds a small voice-loop demo that exercises the full mic → LM → TTS chain.

## The journey, replayed

Module 6 went constraints → models → compression → serving → speed → multimodal → integration. The arc:

- Lesson 1 (the stack): what's different on-device — bandwidth-bound, thermally constrained, no failover, the budget table per device class.
- Lesson 2 (model picking): the SLM landscape; Phi for reasoning, Llama for general, Qwen for multilingual, Granite for enterprise, Apple's foundation model for iOS integration.
- Lessons 3-4 (compression): per-layer precision stepping, mixed-precision with SmoothQuant-style activation handling, calibration data choices.
- Lesson 5 (alternatives to quantization): layer pruning, distillation, when DIY distillation is worth it.
- Lessons 6-7 (memory engineering): KV cache strategies (INT8, INT4, sliding window with sinks), mmap'd weights with the page-fault profile.
- Lesson 8 (streaming UX): token-to-UI plumbing, UTF-8 buffering, cancellation, partial rollback.
- Lesson 9 (going faster): speculative decoding with draft models or n-grams, when it wins and loses on-device.
- Lesson 10 (LoRA hot-swap): one base + many adapters for multi-persona / per-user / per-task deployments.
- Lesson 11 (multimodal): vision encoder + LM, the encoder-caching pattern, the parallel audio-language story.

The synthesis: a production-grade on-device LLM deployment in 2026 composes Q4_K_M weights + INT8 KV cache + mmap'd model file + streaming with UTF-8 buffering + possibly a LoRA adapter, running through llama.cpp or MLX, hitting 30-60 tok/s for a 3B model on a Mac and 8-15 tok/s on a phone. Multimodal adds an encoder-cached vision or audio frontend.

## Use case 1: voice assistant

The end-to-end pipeline:

```
Mic → ASR (Whisper) → LM (text-only) → TTS → Speaker
     ~100 ms        ~3 s         ~500 ms
```

Budget breakdown for a "Hey assistant, what's the weather today?" → 30-word response on M3 Pro:

| Stage | Time | Notes |
| ----- | ---- | ----- |
| Mic capture | 2 s | the user actually speaks |
| ASR (Whisper-base) | 200 ms | transcribe to text |
| LM prefill (~60 tokens system+query) | 50 ms | 1.5B-class model |
| LM decode (~50 tokens response) | 1000 ms | 50 tok/s |
| TTS (Coqui or platform) | 300 ms | synthesize response audio |
| Audio playback | 4 s | the user actually listens |

User-perceived latency from end-of-speech to start-of-response-audio: ~1.6 s. This is the "how fast does it respond" metric. Good but not great; ChatGPT Advanced Voice runs at ~500 ms.

Optimizations to hit 500 ms:

1. **Streaming ASR**: don't wait for end-of-speech; transcribe progressively. Common in Whisper-streaming or specialized models.
2. **Speculative LM start**: start LM inference on partial ASR while the user is still speaking. Risk: wrong start; might need to retract.
3. **TTS-by-sentence streaming**: produce audio for the first sentence while generating the second. The user hears speech start ~300 ms into LM generation, not at the end.
4. **Smaller models everywhere**: 0.5B LM + Whisper-tiny + small TTS. Throughput-friendly; quality cost.

The bottleneck in 2026 is typically LM decode (because it dominates wall-clock). Speculative decoding (Lesson 9) and aggressive quantization (Lesson 3) help most.

For deployment: pick a runtime that supports streaming on all three components. The MediaPipe LLM Inference Task includes ASR and TTS scaffolding on Android; on iOS, Apple's Speech framework + Foundation Models + AVSpeechSynthesizer compose into the same pipeline.

## Use case 2: code completion

The end-to-end pipeline:

```
User keystroke (pause detected) → context extraction → LM (FIM-style) → render completion
                ~50 ms              ~50 ms              ~200 ms      ~10 ms
```

Code completion is much more latency-sensitive than chat. The user expects suggestions to appear within ~300 ms of stopping typing; longer and they've already typed past the suggestion point.

Budget breakdown for a typical multi-line code suggestion:

| Stage | Time | Notes |
| ----- | ---- | ----- |
| Idle detection | 100 ms | wait for typing pause |
| Context extraction | 20 ms | grab recent lines, build prompt |
| LM prefill (~500 tokens of code context) | 100 ms | a 1B-3B model is the sweet spot |
| LM decode (~50 tokens completion) | 80 ms | n-gram speculative decoding can do well here |
| Render | 10 ms | UI update |

End-to-end: ~300 ms. Achievable on a Mac; tight on a phone (code completion on phones is rare; the form factor doesn't suit it).

Optimizations:

1. **Cached KV for the file context**: keep the KV state for the file the user is editing; only re-encode the most recent changes. This is the prefix-caching technique, especially valuable in code completion where most context is stable.
2. **N-gram speculation**: code has high local repetition; n-gram acceptance rates of 60%+ are typical, giving 2× decode speedup essentially for free.
3. **Small model + retrieval**: rather than a giant model that knows everything, a small model + retrieval over the codebase (RAG-style) is often faster and more accurate for the specific codebase.
4. **Stop-on-newline-N**: most completions stop within 1-3 lines. Stop generation as soon as you have something useful; don't generate further if the user won't see it.

The model choice: Qwen 2.5 Coder 1.5B or 3B is the standard 2026 pick for on-device code completion. Specifically-trained for code; small enough to run fast; supports FIM (fill-in-middle) prompting which is what code completion actually wants.

## Use case 3: vision question answering

The pipeline:

```
Camera capture → image preprocessing → VLM (encoder + LM) → render answer
   ~10 ms          ~20 ms                ~1-2 s             ~10 ms
```

A typical photo-QA app: user takes a photo, asks "what's this?", model answers.

Budget for a single question with Moondream2 on M3 Pro:

| Stage | Time | Notes |
| ----- | ---- | ----- |
| Image capture | 10 ms | camera frame ready |
| Preprocess (resize, normalize) | 20 ms | typical 224x224 or 336x336 |
| Vision encoder | 200 ms | once per image |
| LM prefill (~700 vision tokens + question) | 200 ms | |
| LM decode (~60 tokens response) | 1500 ms | 40 tok/s |
| Render | 10 ms | |

End-to-end: ~2 s. Acceptable for "ask a question about this image"; slow if the user is asking a series of questions.

Optimizations:

1. **Encoder caching** (Lesson 11): for follow-up questions, skip the encoder. The pipeline drops to ~1.7 s for the next question.
2. **Smaller VLM**: SmolVLM 500M variant runs ~3× faster than Moondream's 1.86B; quality is lower but for simple QA may be enough.
3. **Resolution downscaling**: for "what's the main subject" you can use 224×224; for OCR you need 448×448+. Match resolution to task.
4. **Streaming**: same patterns as text-only chat. Start showing the response as it generates.

## Mental models to carry forward

Five sentences:

**1. On-device LLM serving is a budget problem more than an algorithm problem.** Most of the work is fitting models in memory, then keeping the decode loop within the user's interactive-latency tolerance.

**2. Q4_K_M weights + INT8 KV cache + mmap'd model + streaming UI is the 2026 default recipe.** Each component is small; together they're the working production pattern.

**3. The right model is the smallest one that meets the quality bar.** The latency/throughput gains from a smaller model often outweigh the quality loss for user-facing interactive applications.

**4. LoRA hot-swap turns "one base model, many personas/tasks" from a memory disaster into a clean operational pattern.** This is the on-device multi-tenancy story.

**5. The LM is rarely the only bottleneck in real-time pipelines.** Voice → ASR → LM → TTS; code completion's prefix caching; vision's encoder cost. Profile end-to-end before optimizing; the slow component is often surprising.

## What we covered, what we skipped

Covered:

- The on-device LLM constraint profile and budget tables across device classes.
- The 2026 SLM landscape and a picking framework.
- Aggressive quantization recipes including per-layer precision stepping.
- Mixed-precision and per-channel quantization with calibration-data choices.
- Pruning (layer pruning the practical winner) and distillation (mostly via off-the-shelf models).
- KV cache strategies for tight budgets including INT8 / INT4 KV and sliding window with sinks.
- mmap'd weights and the page-fault profile.
- Streaming generation patterns and UX considerations.
- Speculative decoding's on-device economics.
- LoRA hot-swap for multi-adapter deployments.
- On-device multimodal (vision and audio).
- This integration view.

Skipped:

- **Multi-tenant LLM serving on consumer hardware** (rare; Module 7 touches it).
- **Web-based LLM inference** (WebGPU + ONNX Runtime Web / WebLLM). Adjacent but separate.
- **Battery and thermal modeling in depth.** Real, important, deserves its own treatment.
- **Privacy and security of on-device LLM weights.** A whole topic.
- **The training side**: fine-tuning workflows, dataset preparation, eval pipelines for personalized models.
- **Specific app-store deployment workflows.** App / OS-vendor specific.

## What's next

**Module 7: Inference from Scratch.** Where the on-device serving stack is a deployment story (use the runtimes that exist), Module 7 is a construction story (build the inference engine). We cover attention from primitives, KV cache management at the algorithmic level, sampling strategies, MoE routing, batching schedulers, prefix caching — the things that runtime authors build. The patterns are framework-agnostic but the concrete substrate is MLX / PyTorch.

**Module 8: Distributed Systems.** Multi-device and multi-machine ML. For on-device, this includes split inference (model partitioned across multiple Macs or phones) and federated learning. For server, the NCCL / RDMA / sharding story.

**Modules 9-11**: Cluster orchestration, ML platform engineering, agents from scratch. The server-side and systems-engineering side of the curriculum.

The on-device serving capability built in this module is the foundation for the agent / RAG / multi-modal-assistant pattern that's emerging in 2026 consumer apps. The next modules are about the infrastructure that supports many such deployments at scale.

## Hands-on (at home)

Build a minimal voice loop that exercises mic → ASR → LM → TTS.

```bash
# Install dependencies (Mac-specific; adapt for other platforms).
pip install sounddevice mlx mlx-lm openai-whisper pyttsx3

# Use Apple's built-in TTS via pyttsx3 (or use a neural TTS like Coqui for quality).
```

```python
# voice_loop.py
import sounddevice as sd
import numpy as np
import whisper
import pyttsx3
from mlx_lm import load, generate
import time

# Load components.
asr = whisper.load_model("base")
lm, tokenizer = load("mlx-community/Qwen2.5-1.5B-Instruct-4bit")
tts = pyttsx3.init()
tts.setProperty('rate', 180)

def listen(duration=5, sample_rate=16000):
    print(f"[listening for {duration}s...]")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype=np.float32)
    sd.wait()
    return audio.flatten()

def transcribe(audio):
    t0 = time.time()
    result = asr.transcribe(audio.astype(np.float32))
    print(f"  [ASR: {(time.time()-t0)*1000:.0f} ms]")
    return result["text"].strip()

def think(user_text):
    messages = [
        {"role": "system", "content": "You are a brief, helpful voice assistant. Answer in 2-3 sentences."},
        {"role": "user", "content": user_text},
    ]
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    t0 = time.time()
    response = generate(lm, tokenizer, prompt=prompt, max_tokens=80, verbose=False)
    print(f"  [LM: {(time.time()-t0)*1000:.0f} ms]")
    return response

def speak(text):
    print(f"  [TTS: speaking...]")
    tts.say(text)
    tts.runAndWait()

while True:
    try:
        audio = listen(duration=4)
        text = transcribe(audio)
        if not text or len(text) < 3:
            print("  (nothing heard)")
            continue
        print(f"You said: {text}")
        response = think(text)
        print(f"Assistant: {response}")
        speak(response)
    except KeyboardInterrupt:
        print("\nGoodbye.")
        break
```

Run it. You should see the three stages' latencies reported. Typical M3 Pro numbers:
- ASR (Whisper-base): ~300 ms for 4 s of audio.
- LM (Qwen 1.5B): ~500-1500 ms depending on response length.
- TTS (pyttsx3): negligible, plays audio in real-time.

End-to-end from end-of-speech to start-of-response-audio: ~1-2 s. Comfortable for a research demo; would need streaming components for production-grade UX.

## End of Module 6

Module 1 made the CPU fast. Module 2 made the GPU fast. Module 3 made the model small. Module 4 made the small fast model run on Apple Silicon. Module 5 made it run on every other commodity edge platform. Module 6 made it run *as a real interactive deployment* — voice assistants, code completion, vision QA — within the budget envelopes consumer devices allow.

Module 7 picks up next: **Inference from Scratch.** We invert the perspective — instead of using the existing runtimes, we build the inference engine from primitives. Attention, KV cache, sampling, MoE, batching, prefix caching: the patterns that runtime authors build, and that you'd need to know if you were building a new runtime or contributing to an existing one. The skills compose with the deployment skills from Modules 4-6.
