---
title: "Lesson 50 — Reasoning Models: KV Pressure at Long Chain-of-Thought"
date: "2026-06-04"
module: "inference-from-scratch"
order: 50
tags: ["reasoning", "o1", "r1", "kv-cache", "long-context", "chain-of-thought"]
author: "Sudipta Pathak"
prerequisites: ["49-test-time-compute"]
---

# Lesson 50 — Reasoning Models: KV Pressure at Long Chain-of-Thought

## Why this lesson exists

Reasoning models (o1, R1, and their successors) generate long reasoning chains — frequently 10K-50K tokens — before the final answer. This is the most extreme case of long-context inference in 2026: the model isn't *reading* long context; it's *generating* it.

The serving implications are different from long-prompt inference. Generation creates the long context one token at a time; the KV cache grows linearly throughout the generation; serving must handle the growth gracefully.

This lesson covers the reasoning-model-specific inference challenges, the serving patterns that work, and the open questions in 2026.

The lesson is reading. The Hands-on monitors KV growth during a reasoning chain.

## The KV growth profile

A standard chat response: 50-500 tokens. KV cache grows to that size; small.

A reasoning model response: 5K-50K tokens. KV cache grows throughout the generation. By token 30K of a reasoning chain, the per-step decode is processing all 30K K, V values in attention — much heavier per step than the early-generation decode.

For Llama-3-class architecture at 32K reasoning tokens:
- KV cache size: ~3.5 GB at FP16 (single sequence).
- Per-decode-step attention reads: ~3.5 GB.
- At 200 GB/s effective bandwidth: 17 ms per step just for KV reads.
- Plus FFN weight reads (~1-2 GB at INT4 for a 7B model): another 10 ms.
- Total per step: ~25-30 ms.
- 50K tokens at 25 ms/step: 21 minutes per reasoning query.

This is the order-of-magnitude. Real numbers vary by hardware and architecture.

## What this changes for serving

**1. Per-query memory dominates.** A reasoning query consumes much more KV than a chat query. The same machine can serve fewer concurrent reasoning queries than chat queries.

**2. Decode throughput per query degrades over generation.** Early tokens of a reasoning chain are fast; later tokens are slow because attention reads more KV. The user-perceived "thinking" speed drops as the chain grows.

**3. KV cache quantization is essential.** At 32K reasoning tokens, INT8 KV is the difference between fitting and not. INT4 KV might be needed for longer chains.

**4. Sliding-window or attention-sink approaches are mostly inappropriate.** Reasoning models often need to refer back to early reasoning steps; sliding window would lose them.

## The "thinking" UX

For reasoning models, the UI traditionally shows a "thinking..." indicator while the reasoning is happening, then the final answer when complete. The reasoning chain is often hidden from the user.

Some products (OpenAI o1's web interface) show a brief summary of the reasoning. Others (the API) just expose the answer with reasoning hidden.

For the inference system, this means:
- The reasoning tokens are generated but not streamed (or streamed only to an internal channel).
- The user sees no progress for the duration of the reasoning.
- The final answer is streamed after the reasoning completes.

The user-perceived latency = reasoning time + answer time. For hard problems with long reasoning, this can be minutes.

## The cost-quality tradeoff

Reasoning models are dramatically more expensive per query than chat models:
- A chat-model query: ~$0.001 (cost of inputs + ~100 output tokens).
- An o1 query: ~$0.05-$0.50 depending on reasoning length.

API pricing reflects this. The economics work because the quality on hard problems is also dramatically higher; users pay for what they get.

For self-hosted reasoning models (R1 and its descendants), the same cost shows up as more GPU-hours per query.

## Specialized serving optimizations

Reasoning-model serving has spawned specialized optimizations:

**Hierarchical KV cache.** Keep early reasoning steps in slower memory (CPU RAM, SSD); the most recent steps in fast GPU memory. The early steps are rarely referenced after the model "moves on."

**Speculative decoding for reasoning.** Speculative decoding (Lessons 25-27) helps reasoning models more than chat models because the long generation amortizes the speculative cost better.

**Stream-suppression.** Don't stream tokens that look like internal reasoning; only the final answer. Saves UI bandwidth and gives a cleaner user experience.

**Reasoning-step caching.** If reasoning steps are sometimes reused across queries (e.g., common problem patterns), cache them in the prefix tree (RadixAttention, Lesson 44).

## The 2026 picture

In 2026, reasoning models are:
- A separate model class with their own pricing and use cases.
- Available in both API form (o1, o3, GPT-4o reasoning mode) and open-weights (R1, R1-distilled smaller variants).
- Used for: math, hard coding tasks, agent planning, scientific reasoning.
- Not used for: casual chat, simple Q&A (overkill).

The serving infrastructure is adapting:
- vLLM and TensorRT-LLM have reasoning-model-friendly modes.
- Specialized cloud providers (e.g., DeepSeek API) optimize for the reasoning workload.
- Self-hosting reasoning models requires high-memory hardware (the long KV cache).

## What you should believe after this lesson

Three sentences:

**1. Reasoning models generate 10K-50K-token chains-of-thought** before the final answer, creating KV cache pressure that dominates the inference cost. The cache grows linearly during generation; per-decode-step cost increases as the chain grows.

**2. Serving reasoning models stresses the inference stack differently** — KV cache quantization, hierarchical memory management, speculative decoding, and stream suppression are all relevant optimizations. The user-perceived latency includes the full reasoning time.

**3. The 2026 picture has reasoning models as a distinct product class** — separate pricing, separate model variants, separate serving infrastructure. They're used for hard problems where the 10-50× cost premium is justified by the quality improvement.

## Hands-on (at home)

Watch KV cache grow during a reasoning chain.

```python
# kv_growth.py
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Use a reasoning-trained smaller model.
model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"  # if available; else fall back
try:
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="cuda").eval()
except Exception:
    print(f"Model {model_id} not available; falling back to Qwen 1.5B.")
    model_id = "Qwen/Qwen2.5-1.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="cuda").eval()

prompt = "Solve step by step: A train travels at 80 km/h. How long to travel 240 km?"
ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)

# Generate with KV cache tracking.
import time
n_steps_to_track = 200
last_kv_size = 0
t_start = time.time()
out = model.generate(ids, max_new_tokens=n_steps_to_track, do_sample=False, return_dict_in_generate=True, use_cache=True)
t_total = time.time() - t_start

n_new = out.sequences.shape[1] - ids.shape[1]
print(f"Generated {n_new} tokens in {t_total:.2f} s ({n_new/t_total:.1f} tok/s)")
print(f"Final output:\n{tokenizer.decode(out.sequences[0])}")
```

For a true reasoning model, run with a much higher `max_new_tokens` (5000-10000) and watch the decode rate slow as the chain grows. The slowdown comes from the growing KV cache.

For more controlled measurement: hook into the model's forward pass and log the KV cache size at each step.

## Further reading

- DeepSeek R1 technical report (DeepSeek, 2025).
- "Scaling LLM Test-Time Compute Optimally" (Snell et al, 2024) — Lesson 49's reference.
- OpenAI o1 system card (limited technical details).
- "Distilling Reasoning into Smaller Models" — DeepSeek's R1-distilled series.
- Anthropic's Claude Reasoning announcements.

End of Part 8. Next: Part 9 begins with **RMSNorm vs LayerNorm** — the small architectural choices that everyone copies without explaining. We close Module 7 with these block-level details and the module wrap.
