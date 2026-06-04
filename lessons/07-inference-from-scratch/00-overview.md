---
title: "Module 7 — Inference from Scratch"
date: "2026-06-04"
module: "inference-from-scratch"
order: 0
tags: ["inference", "llm", "attention", "moe", "kv-cache", "quantization", "serving", "overview"]
author: "Sudipta Pathak"
prerequisites: ["ml-internals", "mlx-apple-silicon", "mobile-edge-runtimes", "on-device-llm-inference"]
---

# Inference from Scratch

## Why this module exists

Most "build X from scratch" content stops at one model. You watch someone reproduce DeepSeek, or GPT-2, or Llama, and you walk away knowing how *that one model* works. The next paper drops a new attention variant, a new routing scheme, a new decoding trick, and you're back at square one.

This module takes the opposite approach. Instead of one model, we cover the whole **design space** of modern LLM inference — every meaningful variant of attention, positional encoding, routing, sampling, quantization, and serving — and we build each one from scratch so you understand the *trade-off it was invented to solve*, not just the implementation.

By the end, when the next frontier model drops, you should be able to read the architecture section in the morning and have an informed opinion by lunch.

## How this fits

Module 7 of the depth track. Modules 1–6 built the substrate (kernels, GPU architecture, compression, runtimes, deployment). Module 7 inverts the perspective: instead of using the existing runtimes, we build the inference engine from primitives. Attention, KV cache management, sampling, MoE routing, batching schedulers, prefix caching — the patterns runtime authors build, and that you need to understand if you're building a new runtime or contributing to an existing one.

The output: the ability to read a recent inference systems paper, understand the contribution in context of the broader design space, and (where appropriate) implement it from scratch in a few hundred lines of PyTorch / MLX.

## The teaching philosophy

Three rules guide every lesson:

**1. Plain language first, math second.** Every concept starts with a sentence you could say to a colleague at a coffee machine — *"GQA is MHA with fewer K/V heads, traded for a smaller KV cache"* — before any equation shows up. When the math arrives, we walk through it line by line, no skipped steps.

**2. Build it on one GPU before scaling.** A single-GPU implementation is the unit test for your mental model. If you can't write FlashAttention for one GPU on one node, you have no business reasoning about Ring Attention across a pod. Every topic that *can* be implemented on a single device is implemented there first.

**3. Then take it distributed.** After the single-device version works, we rewrite for the realistic case — tensor-parallel, sequence-parallel, expert-parallel — and show the collectives, where they sit in the forward pass, what they cost, and how to overlap them with compute.

## The roadmap

Fifty-four lessons across nine parts, sequenced so each one earns the next.

### Part 1 — Attention, the full family (9 lessons)

The popular content covers GQA → MLA in the context of one model. We cover the whole genealogy and the trade-off each variant was invented to solve.

1. **Self-attention from first principles** — queries, keys, values, dot product.
2. **Multi-Head Attention (MHA)** — why split, how to split, what each head learns.
3. **Multi-Query Attention (MQA)** — one shared K/V, the memory-bandwidth motivation.
4. **Grouped-Query Attention (GQA)** — the MHA↔MQA spectrum, Llama 2/3's choice.
5. **Multi-Head Latent Attention (MLA)** — DeepSeek's low-rank KV compression.
6. **Sliding Window Attention** — Mistral, Longformer, attention sinks.
7. **Cross-Attention** — encoder-decoder, multimodal fusion.
8. **FlashAttention v1 → v2 → v3** — IO-aware attention, why softmax tiling matters.
9. **Linear & sub-quadratic attention** — Performer, Linformer, what they trade away.

### Part 2 — Positional encodings (6 lessons)

Why transformers don't know order, and the long arc of how we taught them.

10. **Why positions matter** — the permutation-invariance problem.
11. **Absolute encodings** — sinusoidal, learned, integer/binary.
12. **Relative position** — Shaw et al., T5 bias.
13. **ALiBi** — linear bias, length extrapolation for free.
14. **RoPE** — rotary embeddings derived from scratch.
15. **Long-context RoPE scaling** — Linear, NTK-aware, YaRN, LongRoPE.

### Part 3 — KV cache & memory (6 lessons)

The KV cache is *the* central data structure of inference. Most production complexity exists to manage it.

16. **Why the KV cache exists** — the prefill vs decode asymmetry.
17. **KV cache memory layout and arithmetic intensity**.
18. **PagedAttention** — vLLM's virtual-memory analog for KV blocks.
19. **Prefix caching & cross-request KV reuse**.
20. **KV cache quantization** — FP8, INT8, INT4.
21. **StreamingLLM & attention sinks** for effectively-infinite context.

### Part 4 — Sampling & decoding (7 lessons)

The output side. Where latency is won or lost.

22. **Greedy, temperature, top-k, top-p, min-p** — what each one actually does.
23. **Beam search** and why LLMs largely abandoned it.
24. **Constrained decoding** — grammars, JSON schema, regex-guided.
25. **Speculative decoding** — draft + verify.
26. **Medusa** — multi-head speculative.
27. **EAGLE & lookahead decoding**.
28. **Multi-Token Prediction (MTP)** — DeepSeek's training-time MTP, inference-time uses.

### Part 5 — Mixture of Experts (6 lessons)

The architectural shift behind the largest open models.

29. **MoE from scratch** — gating, top-k routing.
30. **Load balancing** — auxiliary loss, expert utilization.
31. **Switch Transformer (top-1) vs GShard (top-2)**.
32. **DeepSeek MoE** — fine-grained experts + shared experts.
33. **Aux-loss-free balancing** (DeepSeek-V3).
34. **Expert parallelism for inference** — routing, all-to-all, EP overlap.

### Part 6 — Quantization for inference (6 lessons)

Squeezing the model into the memory you actually have.

35. **INT8 / INT4 basics** — symmetric vs asymmetric, per-tensor vs per-channel.
36. **GPTQ** — second-order quantization.
37. **AWQ** — activation-aware quantization.
38. **SmoothQuant** — migrating outliers from activations to weights.
39. **FP8 inference** on Hopper/Blackwell.
40. **1-bit territory** — BitNet b1.58 and the extreme low end.

### Part 7 — Serving systems (6 lessons)

Where engineering meets the requests you're actually getting.

41. **Continuous batching** — Orca, why static batching is dead.
42. **Chunked prefill** — bounding TTFT under load.
43. **Prefill/decode disaggregation** — different hardware for different bottlenecks.
44. **RadixAttention** — SGLang's prefix tree.
45. **CUDA Graphs for decode** — eliminating launch overhead.
46. **Tensor parallelism for inference** (vs training) — why TP-for-inference is its own problem.

### Part 8 — Long context & test-time compute (4 lessons)

The two frontiers pulling inference systems in opposite directions.

47. **Ring Attention** — sharding the sequence dimension across GPUs.
48. **Tree Attention** for branched generation.
49. **Test-time compute scaling** — best-of-N, majority voting, the cost curve.
50. **Reasoning models** (o1, R1) — KV pressure when chain-of-thought runs 10k+ tokens.

### Part 9 — Block-level architecture choices (4 lessons)

The small decisions everyone copies without explaining why. Last lesson serves as module wrap.

51. **RMSNorm vs LayerNorm** — and why everyone moved.
52. **SwiGLU vs GELU** — the FFN that ate the transformer.
53. **Pre-norm vs post-norm** — training stability vs representation quality.
54. **Tokenization for inference + module wrap** — BPE, SentencePiece, tiktoken, byte-level; the close of Module 7 and handoff to Module 8.

---

## What this module deliberately won't cover

- **Training-side concerns**: optimizer choice, learning rate schedules, gradient checkpointing — covered elsewhere. We focus on what matters for *inference*.
- **Pretraining datasets and data pipelines.** Adjacent and important but a different curriculum.
- **Reinforcement learning from human feedback (RLHF)** in depth. The inference-time consequences (longer outputs, reasoning chains) are covered; the training algorithms aren't.
- **Application-layer concerns**: prompt engineering, agent design, RAG architectures. Module 11 covers agents.
- **Detailed comparison of specific production runtimes.** We cover the patterns runtimes use; the runtime-by-runtime comparison was Module 5.

## How to work through it

Every lesson is fully readable as prose. Hands-on sections require:

- A GPU for the kernel-level lessons (Part 1 lesson 8 FlashAttention; Part 3 PagedAttention; Part 7 CUDA Graphs). A consumer NVIDIA GPU (3080+ or any datacenter card) works; Apple Silicon works for most of the non-CUDA-specific parts via MLX.
- 2× GPUs for the distributed-pattern hands-on (Part 1 tensor parallel; Part 5 expert parallel; Part 8 Ring Attention). Cloud rentals on Lambda / RunPod / Modal are fine.
- Most lessons have a single-device hands-on that works on any reasonable laptop.

A note on tempo: this is the largest module in the curriculum (54 lessons). Read it in the part-order it presents; each part builds the conceptual base for the next. Parts 1, 3, and 7 are the dense ones; Parts 4, 6, and 9 are lighter; Parts 2, 5, and 8 are intermediate. Budget the time accordingly.

The capstone: by the end of the module, you should be able to build a simple but real LLM inference engine — KV-cache management, continuous batching, prefix caching, speculative decoding — in ~2000 lines of PyTorch or MLX. The last lesson points at a reference implementation.
