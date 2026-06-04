---
title: "Inference from Scratch: Series Overview"
date: "2026-05-26"
excerpt: "A tutorial series on the full design space of modern LLM inference — every meaningful variant of attention, positional encoding, routing, sampling, quantization, and serving, built from scratch on one GPU, then taken distributed."
module: "inference"
order: 0
tags: ["inference", "llm", "attention", "moe", "kv-cache", "quantization", "serving", "overview"]
author: "Sudipta Pathak"
prerequisites: []
---

# Inference from Scratch

## Why another "from scratch" series

Most "build X from scratch" content stops at one model. You watch someone reproduce DeepSeek, or GPT-2, or Llama — and you walk away knowing how *that one model* works. The next paper drops a new attention variant, a new routing scheme, a new decoding trick, and you're back at square one.

This series takes the opposite approach. Instead of one model, we cover the whole **design space** of modern LLM inference — every meaningful variant of attention, positional encoding, routing, sampling, quantization, and serving — and we build each one from scratch so you understand the *trade-off it was invented to solve*, not just the implementation.

By the end, when the next frontier model drops, you should be able to read the architecture section in the morning and have an informed opinion by lunch.

## The teaching philosophy

Three rules guide every tutorial in this series.

### 1. Plain language first, math second

You can't intuit something you can only manipulate. Every concept starts with a sentence you could say to a colleague at a coffee machine — *"GQA is MHA with fewer K/V heads, traded for a smaller KV cache"* — before any equation shows up.

When the math arrives, we walk through it line by line, no skipped steps. No "it can be shown that." If a tensor changes shape, we say what shape, why, and what each axis means.

### 2. Build it on one GPU before you scale it

A single-GPU implementation is the unit test for your mental model. If you can't write FlashAttention for one GPU on one node, you have no business reasoning about Ring Attention across a pod.

Every topic that *can* be implemented on a single device is implemented there first — usually in plain PyTorch, sometimes a Triton kernel where it matters. You'll see the naive version, the bug, the fix, and the optimized version. In that order.

### 3. Then take it distributed

After the single-GPU version works, we rewrite it for the realistic case — tensor-parallel, sequence-parallel, expert-parallel, or all-to-all'd across the cluster. This is where most "from scratch" content drops off and where production actually lives.

You'll see the collectives (`all_reduce`, `all_gather`, `reduce_scatter`, `all_to_all`), where they sit in the forward pass, what they cost, and how to overlap them with compute. The goal is for the multi-GPU code to feel like a natural extension of the single-GPU code, not a different language.

---

## The roadmap

Nine parts, ~50 tutorials, sequenced so each one earns the next. Topics get linked here as they ship.

### Part 1 — Attention, the full family

The popular content covers GQA → MLA in the context of one model. We cover the whole genealogy and the trade-off each variant was invented to solve.

1. Self-attention from first principles — queries, keys, values, and the dot product
2. **Multi-Head Attention (MHA)** — why split, how to split, what each head learns
3. **Multi-Query Attention (MQA)** — one shared K/V, the memory-bandwidth motivation
4. **Grouped-Query Attention (GQA)** — the MHA↔MQA spectrum, Llama 2/3's choice
5. **Multi-Head Latent Attention (MLA)** — DeepSeek's low-rank KV compression
6. Sliding Window Attention — Mistral, Longformer, attention sinks
7. Cross-Attention — encoder-decoder, multimodal fusion
8. FlashAttention v1 → v2 → v3 — IO-aware attention, why softmax tiling matters
9. Linear & sub-quadratic attention — Performer, Linformer, what they trade away

### Part 2 — Positional encodings

Why transformers don't know order, and the surprisingly long arc of how we taught them.

10. Why positions matter — the permutation-invariance problem
11. Absolute encodings — sinusoidal, learned, integer/binary
12. Relative position — Shaw et al., T5 bias
13. ALiBi — linear bias, length extrapolation for free
14. RoPE — rotary embeddings derived from scratch
15. Long-context RoPE scaling — Linear, NTK-aware, YaRN, LongRoPE

### Part 3 — KV cache & memory

The KV cache is *the* central data structure of inference. Most production complexity exists to manage it.

16. Why the KV cache exists — the prefill vs decode asymmetry
17. KV cache memory layout and arithmetic intensity
18. **PagedAttention** — vLLM's virtual-memory analog for KV blocks
19. Prefix caching & cross-request KV reuse
20. KV cache quantization — FP8, INT8, INT4
21. StreamingLLM & attention sinks for effectively-infinite context

### Part 4 — Sampling & decoding

The output side of the model. Where latency is won or lost.

22. Greedy, temperature, top-k, top-p, min-p — what each one actually does
23. Beam search and why LLMs largely abandoned it
24. Constrained decoding — grammars, JSON schema, regex-guided
25. **Speculative decoding** — draft + verify
26. Medusa — multi-head speculative
27. EAGLE & lookahead decoding
28. **Multi-Token Prediction (MTP)** — DeepSeek's training-time MTP, inference-time uses

### Part 5 — Mixture of Experts

The architectural shift behind the largest open models. Everything you need to build, balance, and serve MoE.

29. MoE from scratch — gating, top-k routing
30. Load balancing — auxiliary loss, expert utilization
31. Switch Transformer (top-1) vs GShard (top-2)
32. DeepSeek MoE — fine-grained experts + shared experts
33. Aux-loss-free balancing (DeepSeek-V3)
34. **Expert parallelism for inference** — routing, all-to-all, EP overlap

### Part 6 — Quantization for inference

Squeezing the model into the memory you actually have, without breaking it.

35. INT8 / INT4 basics — symmetric vs asymmetric, per-tensor vs per-channel
36. GPTQ — second-order quantization
37. AWQ — activation-aware quantization
38. SmoothQuant — migrating outliers from activations to weights
39. FP8 inference on Hopper/Blackwell
40. 1-bit territory — BitNet b1.58 and the extreme low end

### Part 7 — Serving systems

Where the engineering you've learned meets the requests you're actually getting.

41. **Continuous batching** — Orca, why static batching is dead
42. Chunked prefill — bounding TTFT under load
43. **Prefill/decode disaggregation** — different hardware for different bottlenecks
44. **RadixAttention** — SGLang's prefix tree
45. CUDA Graphs for decode — eliminating launch overhead
46. Tensor parallelism for inference (vs training) — why TP-for-inference is its own problem

### Part 8 — Long context & test-time compute

The two frontiers pulling inference systems in opposite directions.

47. Ring Attention — sharding the sequence dimension across GPUs
48. Tree Attention for branched generation
49. Test-time compute scaling — best-of-N, majority voting, the cost curve
50. Reasoning models (o1, R1) — KV pressure when chain-of-thought runs 10k+ tokens

### Part 9 — Block-level architecture choices

The small decisions everyone copies without explaining why.

51. RMSNorm vs LayerNorm — and why everyone moved
52. SwiGLU vs GELU — the FFN that ate the transformer
53. Pre-norm vs post-norm — training stability vs representation quality
54. Tokenization for inference — BPE, SentencePiece, tiktoken, byte-level

---

## How to follow along

You can read these in order — they're sequenced for that — or jump to the part most relevant to what you're building. Every tutorial is self-contained enough that the prerequisites are listed up front, so if you land on Part 5 without reading Part 1, you'll know what you're missing.

Code lives alongside the prose. Single-GPU versions run on anything with ≥16 GB of VRAM. Multi-GPU sections assume access to at least 2× consumer GPUs (for tensor parallel) or a multi-node setup (for expert parallel and ring attention) — instructions for renting time on each are included where they apply.

If you find an error, an explanation that didn't land, or a topic that's missing — open an issue or reach out. This series is meant to be lived in, not just published.
