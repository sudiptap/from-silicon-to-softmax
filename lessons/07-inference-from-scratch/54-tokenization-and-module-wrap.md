---
title: "Lesson 54 — Tokenization for Inference + Module Wrap"
date: "2026-06-04"
module: "inference-from-scratch"
order: 54
tags: ["tokenization", "bpe", "sentencepiece", "tiktoken", "wrap", "module-summary"]
author: "Sudipta Pathak"
prerequisites: ["53-pre-norm-vs-post-norm"]
---

# Lesson 54 — Tokenization for Inference + Module Wrap

## Why this lesson exists

Tokenization sits at the edges of the inference system: convert user text to token IDs (encode), convert generated token IDs back to text (decode). It's not a hot path computationally — milliseconds per query — but it's the place where subtle bugs and surprises live.

This lesson covers the major tokenizer families (BPE, SentencePiece, tiktoken, byte-level) and the inference-side considerations. Then we wrap Module 7.

## The tokenizer families

**BPE (Byte-Pair Encoding)**: GPT-2/3, Llama 1. Starts with character-level vocabulary; merges most-frequent pairs iteratively until reaching the target vocab size. The merges form a dictionary; encoding applies them greedily.

**SentencePiece**: T5, Llama, many multilingual models. A variant of BPE that operates on Unicode characters directly (no language-specific pre-tokenization). Subword-style splits; handles non-Latin scripts cleanly.

**tiktoken**: OpenAI's tokenizer for GPT-4 / GPT-4o. BPE with specific design choices (handling of whitespace, multi-byte characters). Very fast Rust implementation.

**Byte-level BPE**: GPT-2 onward. Operates on bytes (256 base tokens) rather than characters; no out-of-vocabulary issues even for unseen scripts or emojis.

The 2026 landscape:
- Open-weights LLMs: SentencePiece (Llama, Mistral, Qwen) or byte-level BPE (GPT-NeoX, OPT).
- OpenAI: tiktoken.
- HuggingFace: a wrapper supporting all formats.

## What matters at inference

**1. Speed.** Encoding/decoding 100 tokens takes microseconds with a tuned tokenizer. With a slow Python implementation, it can take milliseconds — small but visible on per-request latency.

**2. Special tokens.** Chat templates use special tokens like `<|im_start|>`, `<|im_end|>`, `<|user|>`. These have to be encoded correctly; getting the chat template wrong produces garbage outputs.

**3. Streaming decoding.** When generating tokens one at a time, decoding the latest token in isolation may produce broken UTF-8 (Lesson 8 covered the buffer pattern). The tokenizer must support incremental decoding.

**4. Token-level constraints.** For constrained decoding (Lesson 24), the constraint engine needs token-level lookup tables ("which tokens advance this regex/grammar"). The tokenizer's vocabulary determines what's possible.

## The chat template

Every modern instruction-tuned LLM has a *chat template* — the exact formatting of the system/user/assistant turns. For Llama 3:

```
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_message}<|eot_id|><|start_header_id|>user<|end_header_id|>

{user_message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

```

For Qwen:

```
<|im_start|>system
{system_message}<|im_end|>
<|im_start|>user
{user_message}<|im_end|>
<|im_start|>assistant
```

The exact tokens differ between models; using the wrong template produces poor outputs. HuggingFace's `tokenizer.apply_chat_template` handles this if your tokenizer config is correct.

For inference, getting the chat template right is often more important than any other tokenization detail. Many "the model produces garbage" issues are template bugs.

## The tokenizer vocabulary

Vocabulary sizes for modern LLMs:
- Llama 3: 128K tokens.
- Qwen 2.5: 152K tokens.
- GPT-4o: ~200K tokens.

Larger vocabulary means fewer tokens per text (better compression) but bigger embedding table and LM head. The vocab size is a hyperparameter chosen at pretraining; you can't easily change it.

The LM head's matmul cost is proportional to vocab size. For 128K vocab × 4096 hidden, the LM head is 524M parameters. At INT4, ~260 MB. Not huge but non-trivial.

## When tokenization bites

A list of common issues:

**Whitespace handling.** "Hello" and " Hello" tokenize differently in many tokenizers. The leading space matters; it usually becomes part of the next token rather than a separate token.

**End-of-sequence (EOS).** Each tokenizer has an EOS token (often `<|endoftext|>` or `<|eot_id|>`). Forgetting to set the right EOS in generation causes the model to keep going past where it should stop.

**Special token escapes.** If user input contains text like `<|im_start|>`, naïvely tokenizing it would insert a special-token boundary the user didn't intend. Most tokenizers handle this by treating special tokens as atomic only in certain contexts; this varies.

**Decoding bug: partial UTF-8.** Streaming decode of one token may produce invalid UTF-8 bytes (the character spans tokens). Module 7 Lesson 8 covered the buffer pattern.

**Tokenization mismatch.** If you tokenize input differently from how the model was trained, performance degrades dramatically. Always use the model's official tokenizer config.

## Module 7 wrap

Fifty-four lessons across nine parts. The journey:

**Part 1 (Attention, 9 lessons)**: from self-attention's basic dot-product through the MHA → MQA → GQA → MLA evolution, sliding window, cross-attention, FlashAttention v1-v3, and linear/sub-quadratic alternatives. The full genealogy of "how attention has been refined."

**Part 2 (Positional encodings, 6 lessons)**: why positions matter (the permutation-invariance problem); absolute encodings; relative encodings; ALiBi; RoPE; long-context RoPE scaling. The arc from "transformers don't know order" to "RoPE + YaRN handles 128K+ context cleanly."

**Part 3 (KV cache, 6 lessons)**: why the cache exists (prefill vs decode asymmetry); layout and arithmetic intensity; paged attention; prefix caching; KV quantization; StreamingLLM with attention sinks. The central data structure of inference and how to manage it.

**Part 4 (Sampling & decoding, 7 lessons)**: greedy / temperature / top-k / top-p / min-p; beam search and why LLMs abandoned it; constrained decoding; speculative decoding; Medusa; EAGLE & lookahead; MTP. How logits become tokens.

**Part 5 (MoE, 6 lessons)**: gating and top-k routing; load balancing; Switch vs GShard; DeepSeek's fine-grained MoE; aux-loss-free balancing; expert parallelism. The architecture behind the largest open models.

**Part 6 (Quantization for inference, 6 lessons)**: INT8/INT4 basics; GPTQ; AWQ; SmoothQuant; FP8 on Hopper; BitNet. Compressing the model to fit memory and bandwidth.

**Part 7 (Serving systems, 6 lessons)**: continuous batching; chunked prefill; PD disaggregation; RadixAttention; CUDA Graphs; TP for inference. How requests turn into throughput.

**Part 8 (Long context & test-time compute, 4 lessons)**: Ring Attention; Tree Attention; test-time compute scaling; reasoning models. The frontiers.

**Part 9 (Block-level choices, 4 lessons)**: RMSNorm; SwiGLU; pre-norm; tokenization. The small decisions that everyone copies.

The synthesis: building a working LLM inference engine from scratch in 2026 requires understanding all of the above. The patterns compose: GQA + RoPE + paged attention + chunked prefill + continuous batching + INT4 + FlashAttention + speculative decoding. That's not nine independent optimizations; it's one integrated system.

## What we covered, what we skipped

Covered: everything in the roadmap.

Skipped:
- **Training-time concerns** (gradients, optimizers, distributed training algorithms). The training side belongs to a different curriculum.
- **Specific model architectures in depth** (Llama internals, DeepSeek implementation details). We covered the patterns; the per-model details are in technical reports.
- **Hardware-specific kernel design** for accelerators other than NVIDIA. Apple's MLX kernels were Module 4; other accelerators are Module 5.
- **Application-layer concerns**: agents, RAG, tool use, multimodal app design. Module 11.

## Mental models to carry forward

Five sentences that capture the bulk of Module 7:

**1. Decode is bandwidth-bound at batch 1; every optimization is a variation on "load fewer bytes per token."** Quantization, GQA, MLA, KV cache management, batching all attack this.

**2. The KV cache is the central data structure**; most production complexity is KV cache management. Paged attention, prefix caching, KV quantization, sliding window all manage this.

**3. Attention's `O(N²)` cost in `N` is the central architectural challenge at long context.** FlashAttention attacks the implementation; sliding window and linear attention attack the algorithm. Modern long-context LLMs combine: FlashAttention + RoPE scaling + KV quantization + sometimes sliding window with sinks.

**4. Continuous batching + chunked prefill + paged attention is the modern serving stack.** Throughput improvements over naive batching are 2-23×; latency control via chunked prefill is essential for multi-tenant.

**5. New paradigms (MoE, reasoning models, speculative decoding) keep shifting what's possible.** The inference systems field is still evolving rapidly in 2026; build the foundation but expect to keep learning.

## What's next

**Module 8: Distributed Systems.** The full network-layer story for ML — RDMA, InfiniBand, NCCL collectives, the multi-GPU communication primitives that underpin all of distributed inference and training. Module 7 used these (TP, EP, Ring Attention); Module 8 derives them.

**Module 9: Cluster Orchestration.** Kubernetes for ML, the scheduler, the resource isolation. The infrastructure layer that sits between bare metal and inference engines.

**Module 10: ML Platform Engineering.** The end-to-end ML platform: experiment tracking, model registry, CI/CD for models, monitoring. The operational layer.

**Module 11: Agents from Scratch.** The application layer on top of LLM inference — tool use, planning, multi-step reasoning, the agent architectures.

## End of Module 7

Module 1 made the CPU fast. Module 2 made the GPU fast. Module 3 made the model small. Module 4 made on-Apple-Silicon fast. Module 5 made on-every-platform fast. Module 6 made on-device deployments real. Module 7 built the inference engine itself — every component, every variant, every tradeoff. By the end of Modules 1-7 you have the full vertical stack from silicon to softmax for *single-system* LLM inference.

Module 8 picks up at the network: the multi-system story. The infrastructure for serving billions of requests, training trillion-parameter models, and the communication primitives that make it possible.
