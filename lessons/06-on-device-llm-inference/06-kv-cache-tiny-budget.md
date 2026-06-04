---
title: "Lesson 6 — KV Cache for Tiny Memory Budgets"
date: "2026-06-04"
module: "on-device-llm-inference"
order: 6
tags: ["kv-cache", "quantization", "sliding-window", "attention-sinks", "paged"]
author: "Sudipta Pathak"
prerequisites: ["05-pruning-distillation-complement"]
---

# Lesson 6 — KV Cache for Tiny Memory Budgets

## Why this lesson exists

Module 3 Lesson 1 established that the KV cache scales linearly with sequence length and often dominates memory at long context. Module 4 Lesson 9 covered KV cache layouts on Apple Silicon. This lesson is the on-device-specific story: how to manage the KV cache when memory is tight, what to do when it doesn't fit, and the strategies (quantization, sliding window with attention sinks, paging) that buy you headroom.

The headline: for a 3B-class model at long context (16K+ tokens), the KV cache can exceed the weight memory. Managing it is the difference between "this model supports 32K context" and "this model crashes at 4K context on a phone." Every on-device LLM deployment that handles non-trivial-length inputs hits this problem.

The lesson is reading. The Hands-on measures KV cache size across context lengths and shows the impact of INT8 KV quantization.

## The KV cache budget, restated

Per-token KV cache size:

```
kv_bytes_per_token = 2 (K + V) × n_layers × n_kv_heads × d_head × bytes_per_element
```

For Llama 3.2 3B (28 layers, 8 KV heads — GQA, d_head=128) at FP16:
```
kv_per_token = 2 × 28 × 8 × 128 × 2 = 114,688 bytes ≈ 112 KB
```

At various context lengths:
- 1K tokens: 112 MB
- 4K tokens: 448 MB
- 16K tokens: 1.8 GB
- 32K tokens: 3.6 GB
- 128K tokens: 14 GB

On a top-tier flagship phone with 4–6 GB available, you cap around 16K context if you spend everything on KV (and even then, there's nothing left for the weights). At 32K context, you need KV quantization or a sliding window, or both.

On a MacBook Pro M3 Pro with 10+ GB available, 32K is comfortable; 128K is the next wall.

For the 7B-class models (Llama 3.1 8B, Mistral 7B), the per-layer KV cost is higher (more layers, larger d_head sometimes) and the budget shrinks proportionally.

## INT8 KV cache: the safe first step

Quantizing the K and V tensors to INT8 cuts KV memory by 2× with negligible quality cost (~0.05 perplexity on most models). It's the safe first step when you're hitting the KV ceiling.

How it works:
- For each new K and V tensor produced during decode, quantize per-head with per-token scale factors.
- Store the INT8 values + FP16 scales in the cache.
- During attention, dequantize on the fly (the attention kernel handles this; you don't write the dequant by hand).

The bandwidth win is real: the attention kernel reads fewer bytes per attention step, which is the bottleneck during long-context decoding.

In llama.cpp:

```bash
./llama-cli -m model.gguf -p "..." -n 100 \
    --cache-type-k q8_0 --cache-type-v q8_0
```

In mlx-lm:

```bash
mlx_lm.generate --model ... --prompt "..." --kv-bits 8
```

For most workloads INT8 KV cache is a free win; use it.

## INT4 KV cache: the aggressive step

INT4 KV cache cuts further — 4× memory reduction vs FP16. The quality hit is meaningfully larger (~0.3–0.5 perplexity, sometimes more on long context). The bandwidth win is bigger.

When to use INT4 KV:
- The model must support context lengths that INT8 KV doesn't allow.
- The application tolerates some quality degradation on long-context tasks (chat, summarization where occasional missed details are OK).

When not:
- Code completion, where each token matters.
- Reasoning chains, where a single wrong token in the chain can derail the answer.

llama.cpp:
```bash
./llama-cli -m model.gguf --cache-type-k q4_0 --cache-type-v q4_0 ...
```

mlx-lm:
```bash
mlx_lm.generate --model ... --kv-bits 4
```

INT4 KV is more recent and the implementations are less mature than INT8. Test against your specific deployment before committing.

## Sliding window attention

A different angle: instead of attending to *all* prior tokens, attend to only the most recent W tokens (the "window"). Beyond the window, prior tokens are dropped from the cache.

Memory benefit: KV cache size is bounded by W regardless of total generation length. A model with a 4096-token window has a fixed KV cache size, no matter how long the conversation goes.

Quality cost: the model loses access to information outside the window. For chat (where the recent context is what matters), this is often fine. For document QA (where a fact stated 10K tokens ago might be relevant), it's catastrophic.

Models with native sliding-window attention: Mistral 7B / Ministral 3B (4K window), some Gemma variants, some Phi variants. These models are designed to work well with the window; the runtime just respects it.

Models without native sliding-window: applying a sliding window to a Llama-class model at inference time without training for it usually degrades quality significantly. The model wasn't trained to handle the K-tokens-ago disappearing.

## Attention sinks: the trick that makes sliding window work

A 2023 observation (Xiao et al, "Efficient Streaming Language Models with Attention Sinks"): when you naively apply sliding-window attention at inference time to a non-trained model, the *first few tokens* matter much more than you'd expect. Removing them causes the model to break dramatically.

The fix: keep the first ~4 tokens of the conversation pinned in the cache, plus the most recent W tokens. This "sink + window" pattern preserves the model's attention pattern around the special tokens (BOS, system prompt start) while bounding the cache size.

```
Token positions in cache:
[sink: tokens 0-3] [recent: tokens current-W to current]
```

This is what `llama.cpp`'s `--keep` flag does:

```bash
./llama-cli -m model.gguf -p "$LONG_CONVERSATION" \
    --ctx-size 2048 --keep 4
```

With `--keep 4`, the first 4 tokens stay in the cache forever; older non-pinned tokens get evicted as new ones come in. The conversation can continue indefinitely with a bounded KV cache.

For models trained with sliding-window attention this is unnecessary; the model already handles it natively. For non-sliding-window models (Llama-class), attention sinks make sliding-window inference viable.

## Paged attention on-device

Module 4 Lesson 9 discussed paged attention (vLLM's design) and noted it's overkill for single-user inference. That conclusion stands for typical mobile deployments. Paged attention becomes interesting on-device only in two scenarios:

1. **A multi-conversation app** that holds several chat sessions in memory simultaneously. Paging lets you share KV cache pages between sessions when prompts overlap (system prompt, common prefixes).
2. **A laptop-class device serving multiple users** (e.g., a developer's local LLM server hosting work for both the dev and a teammate over LAN). Same logic applies.

For single-user single-conversation on a phone, the contiguous KV cache from Module 4 Lesson 9 is simpler and just as good.

## The "spill to disk" question

What if the KV cache really doesn't fit? Some early frameworks (early 2024 experiments) tried spilling oldest KV pages to disk and reloading as needed. The result was uniformly bad: disk latency is so much higher than memory latency that any KV access requiring a disk read dominates the decode time.

The conclusion: don't try to spill KV to disk on a phone or laptop. If the cache doesn't fit, use INT8 / INT4 quantization, sliding window with sinks, or a shorter context. Disk spilling is a server-side technique (where you have NVMe at GB/s); on consumer hardware it's not viable.

## Putting it together: the on-device KV recipe

A complete strategy:

1. **Default**: contiguous cache, FP16, max context = whatever fits.
2. **Hit the memory ceiling at moderate context**: enable INT8 KV cache. 2× headroom.
3. **Still hitting the ceiling at long context**: switch to a sliding-window model (Mistral, Ministral) or apply attention-sink sliding window to a Llama-class model. Bounded cache.
4. **Extreme**: INT4 KV cache + sliding window with sinks. The most aggressive deployment.

The exact thresholds depend on device class. For a 6 GB phone running a 3B Q4 model (1.6 GB weights), you have ~4 GB for KV — about 35K tokens at FP16 or 140K at INT4. For a 4 GB phone running the same model, ~2 GB for KV, 18K at FP16. The math is the model.

## What you should believe after this lesson

Three sentences:

**1. The KV cache scales linearly with context length and often dominates memory at long context on-device** — for a 3B-class model, 16K context costs ~1.8 GB of KV at FP16, exceeding the weight memory. Managing the cache is the difference between a model that supports useful context and one that crashes early.

**2. INT8 KV cache is the safe first lever** (2× reduction, ~0.05 perplexity hit). INT4 is the aggressive second lever (4× reduction, ~0.3–0.5 perplexity hit). Sliding window with attention sinks bounds the cache to a constant regardless of generation length.

**3. Don't spill KV to disk on consumer hardware** — the latency hit dominates. If the cache doesn't fit, use quantization, sliding window, or shorter context. Disk spilling is a server-side technique that doesn't translate to phones or laptops.

## Hands-on (at home)

Measure KV cache impact on long-context inference.

```bash
# Without KV quantization (baseline).
./llama-cli -m gguf/Llama-3.2-3B-Instruct-Q4_K_M.gguf \
    --ctx-size 8192 -n 200 -ngl 999 \
    -p "$(cat long_prompt.txt)"

# With INT8 KV cache.
./llama-cli -m gguf/Llama-3.2-3B-Instruct-Q4_K_M.gguf \
    --ctx-size 8192 -n 200 -ngl 999 \
    --cache-type-k q8_0 --cache-type-v q8_0 \
    -p "$(cat long_prompt.txt)"

# With sliding window via --keep.
./llama-cli -m gguf/Llama-3.2-3B-Instruct-Q4_K_M.gguf \
    --ctx-size 2048 --keep 4 -n 5000 -ngl 999 \
    -p "$(cat very_long_prompt.txt)"
```

Use `long_prompt.txt` with ~6K tokens of text (a short story, an article, anything). Compare:
- Token throughput (`llama.cpp` reports it).
- Memory usage (Activity Monitor / RAM use during the run).
- Output quality (subjective).

You should see:
- INT8 KV: ~10–20% faster than FP16 KV at the same context (less bandwidth on attention reads); essentially the same quality.
- Sliding window with `--keep`: bounded memory regardless of total generation length; works for chat-style prompts.

For the MLX path, the analogous flags are `--kv-bits 8` and the sliding window is handled by the model architecture (use a sliding-window model like Mistral-Nemo).

## Further reading

- "Efficient Streaming Language Models with Attention Sinks" (Xiao et al, 2023) — the attention-sink paper.
- "PagedAttention: Efficient Memory Management for Large Language Model Serving" (Kwon et al, 2023) — Module 4 Lesson 9's cite.
- "KV Cache Quantization in vLLM" and "llama.cpp KV cache types" — implementation docs.
- "Mistral 7B" technical report — the sliding-window attention reference design.
- "Compressing Context to Enhance Inference Efficiency of Large Language Models" — for the LongLLMLingua and context-compression family.

Next lesson: **Memory-mapped weights and weight streaming.** A different angle on managing on-device memory: instead of loading the full model into RAM, mmap the weight file and let the OS page in what's needed. We look at how llama.cpp uses this pattern, the page-fault behavior, and why it often beats explicit loading.
