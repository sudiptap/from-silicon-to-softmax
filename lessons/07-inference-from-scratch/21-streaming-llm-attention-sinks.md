---
title: "Lesson 21 — StreamingLLM & Attention Sinks"
date: "2026-06-04"
module: "inference-from-scratch"
order: 21
tags: ["streaming-llm", "attention-sinks", "sliding-window", "infinite-context", "kv-cache"]
author: "Sudipta Pathak"
prerequisites: ["20-kv-cache-quantization"]
---

# Lesson 21 — StreamingLLM & Attention Sinks

## Why this lesson exists

Sliding-window attention (Lesson 6) caps the KV cache at a fixed window size, enabling bounded-memory inference. The catch: applying it naively to a pretrained model that wasn't trained for sliding window breaks the model dramatically.

StreamingLLM (Xiao et al, 2023) discovered why and how to fix it: the first few tokens of the sequence act as *attention sinks* — heads have learned to dump excess attention probability into them. When the sinks get evicted from the sliding window, the softmax distribution becomes mis-calibrated and the model collapses.

The fix: pin the first ~4 tokens in the cache; the window slides over everything else. With this single change, a pretrained Llama-class model runs effectively-unbounded contexts with bounded KV memory.

This lesson is the diagnosis, the mechanism, and the implementation. It closes Part 3 by tying together everything we've learned about KV cache management.

The lesson is reading. The Hands-on demonstrates the collapse and the fix with a real model.

## The observation

Take a pretrained Llama / GPT-class model trained with full attention. Apply naive sliding-window attention at inference (window size W=1024, say): for each new token, attend only to the most recent 1024.

Result: the model's perplexity explodes once you generate beyond the window. Outputs become incoherent, often repetitive or random.

Why? The model was trained to attend over the entire context (which always included the first few tokens of every sequence). Attention patterns rely on those first tokens being present.

## The attention-sink hypothesis

Xiao et al's mechanistic finding: in many attention heads, the softmax has learned to put a non-trivial fraction of its probability mass on the *first few tokens* — even when those tokens have nothing semantically to do with the current query. The heads use them as "garbage dumps" for the softmax probability that would otherwise have to go somewhere.

The math: softmax forces all attention weights to sum to 1. If a head has nothing useful to attend to (e.g., a non-content token like a pause), the probability has to land somewhere. Heads learn to direct excess probability to specific positions (usually the first few tokens), which act as sinks.

When you remove the sink tokens (slide them out of the window), the softmax is forced to redistribute that probability across the remaining tokens. The distribution shifts dramatically; the attention patterns the model was relying on get disrupted; the model breaks.

## The fix

Pin the first ~4 tokens in the cache; slide the window over the rest.

```
Cache: [sink_0, sink_1, sink_2, sink_3, recent_(N-W+5), recent_(N-W+6), ..., recent_N]
```

Total cache size: `4 + (W - 4)` = `W` tokens. Same memory footprint as the naive sliding window.

The attention computation: each query attends to the 4 sinks plus the most recent `W-4` tokens. Total: `W` keys per attention step.

With this fix, a Llama-class model can run effectively forever — generating millions of tokens with bounded KV memory and coherent output. Perplexity at very long generation lengths stays close to the in-window baseline.

## Why 4 sinks?

Xiao et al found empirically that 4 is enough. Their probing analysis on Llama / GPT-2 / Falcon found that the "sink-ness" of the first few tokens drops off sharply after position 3-4. Beyond that, tokens are content-bearing, not garbage-dump.

3 sinks usually suffice; 4 is a safe default; more isn't necessary.

The first token is overwhelmingly the most important sink. `<BOS>` or whatever your tokenizer's start token is typically receives 30-50% of the "excess" attention probability across many heads.

## When attention sinks aren't needed

Models trained natively with sliding-window attention (Mistral 7B, Ministral, some Gemma variants) don't need explicit sinks. The training process baked in the right inductive bias; the model never learned to use distant tokens as sinks because it never had access to them.

For these models, vanilla sliding window works without modification.

For pretrained-with-full-attention models extended to long context via sliding window, explicit attention sinks are essential.

## Implementation in llama.cpp

The `--keep N` flag in llama.cpp implements this:

```bash
./llama-cli -m model.gguf -p "$LONG_PROMPT" --ctx-size 4096 --keep 4
```

When the cache fills:
- Keep the first 4 tokens (the sinks).
- Evict the oldest non-sink tokens.
- Continue.

This lets llama.cpp handle prompts and generations far longer than the nominal context length, with bounded memory.

In transformer libraries, the equivalent is `attention_sink_size = 4` or similar config.

## What you don't get from attention sinks

The model can run forever with bounded memory, but it can't *recall* information from beyond the window. Information that's evicted from the cache is gone.

For chat: this is fine; the recent context is what matters.
For document QA where the answer is far from the question: this breaks; the relevant context is evicted before the question arrives.
For needle-in-haystack tests: the needle is evicted if it's old enough; performance collapses.

The fundamental limit: sliding window with sinks gives bounded memory but loses long-range information. The only way around this is full attention or some other mechanism (cross-attention to an external retrieval store, etc.).

## Combining with everything else

In a production deployment of a sliding-window-friendly model:
- **Sliding window**: bounded cache, fixed `W`.
- **Attention sinks**: pin first 4 (only needed if model wasn't trained with sliding window).
- **KV quantization (Lesson 20)**: INT8 KV for further memory savings.
- **GQA (Lesson 4)**: fewer KV heads.

These all compose. The combination is what makes a chatbot run for hours on a phone or laptop with constant memory.

For Mistral 7B on M3 Pro:
- W=4096 sliding window.
- INT8 KV cache.
- GQA-8.
- Per-layer KV at INT8: `2 × 8 × 128 × 1 = 2 KB` per token.
- Per layer cache size: `2 KB × 4096 = 8 MB`.
- Across 32 layers: 256 MB. Constant regardless of generation length.

## What you should believe after this lesson

Three sentences:

**1. Sliding-window attention breaks pretrained models because the first few tokens act as "attention sinks"** — heads have learned to dump excess softmax probability into them. Evicting the sinks disrupts the attention distributions; the model collapses.

**2. The StreamingLLM fix is to pin the first ~4 tokens** ("sinks") in the cache. The window slides over everything else; the model runs effectively forever with bounded KV memory. Mistral and other sliding-window-trained models don't need this; pretrained-with-full-attention models do.

**3. Sliding window + attention sinks gives bounded memory but loses long-range information.** It's the right pattern for chat-style streaming workloads; the wrong pattern for tasks requiring fine-grained recall of distant tokens (long-document QA, needle-in-haystack).

## Hands-on (at home)

Demonstrate the collapse and the fix.

```bash
# A long prompt + a long continuation that would exceed the context window.
# Without --keep (attention sinks), naive sliding window evicts everything.
# Outputs may be incoherent past the window.

# With --keep 4: outputs stay coherent.

# Start with no sinks; observe collapse.
./llama-cli -m gguf/Llama-3.2-1B-Instruct-Q4_K_M.gguf \
    --ctx-size 1024 -n 5000 \
    -p "Write a long essay about the history of programming languages." \
    --keep 0  # no sinks

# Now with sinks; observe coherent output.
./llama-cli -m gguf/Llama-3.2-1B-Instruct-Q4_K_M.gguf \
    --ctx-size 1024 -n 5000 \
    -p "Write a long essay about the history of programming languages." \
    --keep 4  # 4 sinks
```

With `--keep 0` you should see the model degrade in coherence after generating beyond the context window. With `--keep 4` it should stay reasonable.

The effect is most pronounced on shorter context-size windows (1024 or smaller). For larger windows, the model has more headroom and degrades less dramatically.

## Further reading

- "Efficient Streaming Language Models with Attention Sinks" (Xiao et al, 2023) — the paper.
- "Attention Is All You Need" — wasn't aware of sinks, but the mechanism that creates them is built into the original softmax.
- "LongLLMLingua" and other context-compression papers — for the related "compress what's in the window" approach.
- Module 6 Lesson 6 — earlier coverage of attention sinks from the on-device deployment perspective.

End of Part 3. Next: Part 4 begins with **Greedy, temperature, top-k, top-p, min-p** — the sampling methods that turn logits into tokens. The forward pass produces a distribution; the sampling method decides which token actually comes out.
