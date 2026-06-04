---
title: "Lesson 20 — KV Cache Quantization"
date: "2026-06-04"
module: "inference-from-scratch"
order: 20
tags: ["kv-cache", "quantization", "fp8", "int8", "int4", "long-context"]
author: "Sudipta Pathak"
prerequisites: ["19-prefix-caching"]
---

# Lesson 20 — KV Cache Quantization

## Why this lesson exists

The KV cache is just another tensor. Like model weights (Module 3, Module 6, Module 7 Part 6), it can be quantized to lower precision to reduce memory and bandwidth. Unlike weight quantization (offline, one-time), KV quantization is *dynamic* — the runtime quantizes K and V as they're produced during decode.

The practical math: at long context, the KV cache eclipses the weight memory (Lesson 17). KV quantization is the only way to fit very-long-context models on consumer hardware without sliding-window tricks.

This lesson covers the formats (FP8, INT8, INT4), the runtime mechanics, the quality cost, and the recipes that ship in modern serving stacks.

The lesson is reading. The Hands-on enables INT8 KV in llama.cpp and measures the throughput/memory impact.

## The setup

For each new token's K and V (each a `[n_kv_heads, d_head]` tensor), quantize per-head before storing in the cache:

```python
def quantize_kv(K_new, V_new):
    # K_new, V_new: [n_kv_heads, d_head]
    # Per-head scale (and zero-point for asymmetric).
    K_scale = K_new.abs().max(dim=-1, keepdim=True).values / 127.0  # INT8 example
    K_q = (K_new / K_scale).round().clamp(-128, 127).to(torch.int8)
    # Store K_q and K_scale.
    return K_q, K_scale
```

At attention time, dequantize on the fly:

```python
def attention_with_quantized_kv(Q, K_q, K_scale, V_q, V_scale):
    K = K_q.float() * K_scale
    V = V_q.float() * V_scale
    # Standard attention.
    return softmax(Q @ K.T / sqrt(d)) @ V
```

The dequant happens inside the attention kernel; it's fused with the matmul rather than producing intermediate FP16 tensors.

## INT8 KV cache

The safe first lever (Module 6 Lesson 6 covered the deployment view):
- Memory: 2× reduction vs FP16.
- Quality: ~0.05 perplexity hit; usually invisible on downstream tasks.
- Bandwidth: 2× reduction on attention reads.
- Speed: 10-30% faster decode at long context.

Recipe: per-head asymmetric quantization (scale + zero-point). Common across runtimes (llama.cpp, vLLM, MLX).

## INT4 KV cache

More aggressive:
- Memory: 4× reduction.
- Quality: ~0.3-0.5 perplexity hit. More visible.
- Speed: another 20-40% on top of INT8.

Recipe: per-group quantization (groups of 16-32 tokens) is more accurate than per-head global. Some runtimes use per-token quantization (each token's K and V get their own scale); higher overhead but better quality.

When to use INT4 KV: very long context where you can't fit INT8 KV. Borderline on quality; test against your specific workload.

## FP8 KV cache

The Hopper-and-later option:
- Memory: 2× reduction (FP16 → FP8).
- Quality: comparable to or slightly better than INT8 (FP8's E4M3 format handles the dynamic range better than per-head INT8 scales sometimes).
- Speed: on Hopper, FP8 has native hardware support — attention kernels operate on FP8 directly.

Recipe: store FP8 directly; the FP8 format handles per-tensor scaling internally.

FP8 KV is the right choice when:
- You have Hopper / Blackwell hardware.
- You want comparable quality to INT8 with simpler kernel paths.

vLLM, TensorRT-LLM, and SGLang all support FP8 KV on Hopper.

## When KV quantization matters

The KV cache dominates total memory at:
- Long context (>16K for typical 7B models).
- Many concurrent users (per-sequence KV adds up).
- Large GQA group count or full MHA (more K, V per token).

For these scenarios, INT8 KV is the difference between "context limited by KV cache size" and "context limited by something else." On a 24GB GPU running a 7B model:
- FP16 KV: max 16K context.
- INT8 KV: max 32K context.
- INT4 KV: max 64K context.

The compute side is also a win: bandwidth reduction → faster attention.

When KV quantization doesn't matter:
- Short context (<2K) where the cache is small anyway.
- Tight quality requirements (code completion, structured output).

## Runtime support matrix

| Runtime | FP16 KV | INT8 KV | INT4 KV | FP8 KV |
| ------- | ------- | ------- | ------- | ------ |
| llama.cpp | ✓ | ✓ | ✓ (experimental) | ✗ (no native HW most platforms) |
| vLLM | ✓ | ✓ | ✓ | ✓ (Hopper) |
| MLX-LM | ✓ | ✓ | ✓ (experimental) | ✗ |
| TensorRT-LLM | ✓ | ✓ | ✓ | ✓ |
| SGLang | ✓ | ✓ | ✓ | ✓ |

The story has stabilized in 2024-2025; most production servers support INT8 KV with one or two flags, FP8 on Hopper.

## A subtle implementation point: per-head vs per-token

Per-head: each head gets a scale tensor of shape `[1]` (or `[d_head]` for per-channel-per-head). The scale is computed at K/V production time from the head's max.

Per-token (more accurate): each token's K and V get their own scale across all heads. Slightly higher overhead (extra scale tensor per token) but better quality at INT4.

Per-group (used by some runtimes): groups of consecutive tokens share a scale. Compromise between per-head and per-token.

The choice usually doesn't matter at INT8 (quality is fine with any granularity). At INT4 it matters; per-token is usually better.

## What you should believe after this lesson

Three sentences:

**1. KV cache quantization is dynamic** — the runtime quantizes K and V as they're produced, stores the quantized form in the cache, and dequantizes on the fly during attention. The bandwidth savings on attention reads compound with the memory savings.

**2. INT8 KV is the safe lever** (~0.05 perplexity, 2× reduction); INT4 is more aggressive (~0.3-0.5 perplexity, 4× reduction); FP8 is the Hopper-and-later option with comparable quality to INT8 and native hardware support.

**3. KV quantization matters most at long context** where the cache dominates memory. For typical short-context (<2K) deployments, the FFN weight reads dominate and KV quantization is less impactful; for 16K+ deployments it's often the binding constraint.

## Hands-on (at home)

Enable INT8 KV in llama.cpp and measure.

```bash
# Use a moderately-sized model and a longer context.
./llama-cli -m gguf/Llama-3.2-3B-Instruct-Q4_K_M.gguf \
    --no-conversation \
    -c 8192 \
    -p "$(cat 4k_prompt.txt)" -n 200 -ngl 999 \
    --cache-type-k f16 --cache-type-v f16

# Measure: time, memory (Activity Monitor / nvidia-smi during the run).

# Now with INT8 KV.
./llama-cli -m gguf/Llama-3.2-3B-Instruct-Q4_K_M.gguf \
    --no-conversation \
    -c 8192 \
    -p "$(cat 4k_prompt.txt)" -n 200 -ngl 999 \
    --cache-type-k q8_0 --cache-type-v q8_0

# Compare TPS and memory.
```

For the same model, you should see:
- INT8 KV memory: ~half of FP16.
- INT8 KV tokens/sec: ~10-20% faster at this context length; bigger gain at longer.
- Quality: indistinguishable on a short qualitative test.

For INT4 KV, swap `q8_0` for `q4_0`. Quality may suffer; test on representative prompts.

## Further reading

- "KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache" (Liu et al, 2024).
- "QServe: W4A8KV4 Quantization and System Co-Design for Efficient LLM Serving" (Lin et al, 2024).
- vLLM and llama.cpp docs on KV quantization options.
- "FP8 KV Cache in TensorRT-LLM" — NVIDIA documentation for the Hopper path.

Next lesson: **StreamingLLM & attention sinks for effectively-infinite context.** We close Part 3 with the trick that combines sliding window + attention sinks to enable bounded-memory inference at arbitrary generation lengths. The mechanism we used in Module 6; now we derive it.
