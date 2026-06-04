---
title: "Lesson 9 — Speculative Decoding On-Device"
date: "2026-06-04"
module: "on-device-llm-inference"
order: 9
tags: ["speculative-decoding", "draft-model", "medusa", "eagle", "n-gram", "throughput"]
author: "Sudipta Pathak"
prerequisites: ["08-streaming-generation"]
---

# Lesson 9 — Speculative Decoding On-Device

## Why this lesson exists

Autoregressive decoding has a structural inefficiency: each token requires loading all of the model's weights through memory to compute that one token. The GPU's compute units (which can do 100s of TFLOPs in parallel) are idle most of the time, waiting for weights to arrive from HBM / unified memory. Speculative decoding sidesteps this by *proposing* multiple future tokens via a small fast model (the "draft model"), then *verifying* them in parallel with one forward pass of the large model. If most proposals are accepted, you get multiple tokens per main-model forward pass — directly improving throughput.

The technique is well-known on server-side LLM inference (vLLM and TensorRT-LLM both support it). On-device it's more nuanced because the second-model memory cost competes for the same scarce RAM, and the draft model's overhead can outweigh the savings on small main models. This lesson is the on-device-specific view: when speculative decoding pays off, the variants (draft model vs n-gram vs self-speculation), and the operational tradeoffs.

The lesson is reading. The Hands-on runs llama.cpp with and without speculative decoding to measure the on-device speedup.

## The basic idea, restated

Autoregressive decoding:
1. Compute logits for the next token given all prior tokens.
2. Sample a token.
3. Append to the sequence; repeat.

Each step requires one forward pass of the model. For a 3B-class model, one forward pass is ~25 ms on M3 Pro — the bandwidth cost of loading 1.5 GB of weights at ~200 GB/s effective.

Speculative decoding:
1. Use a small draft model to generate `k` candidate next tokens (e.g., 4 tokens).
2. Run one forward pass of the main model on those `k` tokens *in parallel* (as if they were a single multi-token batch).
3. The main model produces `k` logits; compare to the draft's proposals to verify.
4. Accept the longest prefix of draft tokens that matches what the main model would have sampled. The rest are rejected; the main model's first divergent token replaces the rejected ones.

Outcome: at most `k+1` tokens generated per main-model forward pass (the `k` accepted draft tokens plus the main model's one verification token after the divergence point).

If the draft model agrees with the main model 80% of the time and you propose `k=4`:
- 4 draft tokens accepted: `(0.8)^4 ≈ 0.41` probability. 5 tokens per main pass.
- 3 accepted: `(0.8)^3 × 0.2 = 0.10`. 4 tokens per main pass.
- 2 accepted: ~ 0.13. 3 tokens.
- 1 accepted: ~ 0.16. 2 tokens.
- 0 accepted: ~0.20. 1 token.

Expected: ~2.7 tokens per main-model pass. About 2.7× the throughput, *if the draft model is free.*

## The draft-model cost

The draft model isn't free. On-device, it costs:

- **Memory**: the draft model's weights need to be resident too. For a 3B main + 0.5B draft (1/6 the size), that's 100 MB extra at INT4.
- **Compute**: each speculative step runs the draft model `k` times, sequentially. A 0.5B model at INT4 takes ~6 ms per forward; `k=4` is 24 ms of draft time per step.
- **Bandwidth**: the draft model competes for the same memory bandwidth as the main model.

The net throughput:
- Without spec: 1 token per 25 ms = 40 tok/s.
- With spec (k=4, 80% acceptance, ~24 ms draft + ~30 ms main with k+1=5 tokens parallel): 2.7 tokens per 54 ms = 50 tok/s.

Net win: ~25%. Real but modest.

The arithmetic gets better when:
- The main model is larger (proportional draft cost shrinks).
- The acceptance rate is higher (more tokens per main pass).
- The draft model is much faster than the main (1/10 the size is the rule of thumb).

The arithmetic gets worse when:
- The main model is small (the draft savings are a smaller fraction of the workload).
- The acceptance rate is low (the draft work is mostly wasted).
- Memory pressure forces the draft to evict main-model pages.

## When speculative decoding wins on-device

The clear wins:

**1. Large main models on capable hardware.** Llama 3.1 70B on a Mac Studio with M2 Ultra. The main model is huge; even a moderate draft (Llama 3.2 1B) is tiny by comparison. 1.5–2× speedup is common.

**2. Code generation with predictable patterns.** Code has high local predictability (variable names repeated, syntax patterns). A draft model — or even an n-gram (see below) — has high acceptance rates. Cursor and similar tools use speculative decoding for completion.

**3. Long generations with low diversity.** A summarization task that's mostly paraphrasing existing text has high draft-model acceptance.

The clear losses:

**1. Small main models** (1B–3B). The draft model's overhead is a large fraction of the workload; the gains are small or negative.

**2. Highly diverse generation** (creative writing, agent reasoning). Low draft acceptance; the technique wastes more compute than it saves.

**3. Very tight memory** budgets where the draft model competes for KV cache space.

## Variants beyond "use a smaller model"

**N-gram speculation.** Instead of a draft model, use a lookup table of "given these N preceding tokens, what's the most likely next token?" Build the table from a corpus once; query it at every speculative step. Costs almost no memory and almost no compute; works well when the generation has predictable local patterns (code, structured output).

In llama.cpp this is enabled with `--draft-n` and operates on a frequency table.

**Self-speculation (Medusa).** Add small "Medusa heads" to the main model — additional output heads that predict the second, third, ... future tokens directly from the same hidden state. No separate model; just extra parameters trained into the main model. Verification is the same forward-pass technique. Throughput gains are similar to using a small draft model, without the memory cost of a separate model.

**Self-speculation (EAGLE).** A more elaborate version: train a lightweight network on top of the main model's hidden states to predict candidate next tokens. Smaller than a separate model; more accurate than Medusa.

**Tree speculation (SpecInfer).** Propose a *tree* of candidate continuations (e.g., 3 candidates for next token, 2 for the token after each, etc.) and verify in parallel. Higher acceptance probability than linear speculation because you cover more of the probability mass.

For on-device in 2026, **n-gram speculation is the most practical** because it has near-zero overhead. The Medusa/EAGLE approaches need model retraining and are mostly used in pre-shipped variants of specific models (e.g., Medusa-equipped Llama variants).

## Speculative decoding in llama.cpp

```bash
# With a draft model.
./llama-cli \
    -m main/Llama-3.1-70B-Instruct-Q4_K_M.gguf \
    --draft-model draft/Llama-3.2-1B-Instruct-Q4_K_M.gguf \
    --draft-tokens 4 \
    -p "Write a story about..."

# N-gram speculation (no draft model needed).
./llama-cli \
    -m main/Llama-3.1-70B-Instruct-Q4_K_M.gguf \
    --draft-n 3 \
    -p "..."
```

llama.cpp tracks the acceptance rate and reports it at the end. A high acceptance rate (>70%) indicates speculation is winning; low (<40%) means it's losing.

In mlx-lm (mid-2026), speculative decoding support is improving but less stable than llama.cpp's; expect API changes.

## The "stop trying to outsmart the OS" caveat

Speculative decoding adds complexity. The win on-device for small-to-medium models is modest (10–30%). Before adding speculation, ask:

- Is the latency / throughput actually the binding constraint? Or is the user fine with the current speed?
- Are there simpler wins available — better quantization, KV cache improvements, larger group sizes?
- Does the operational complexity (managing two models, the higher debugging surface) cost more in maintenance than the throughput is worth?

For deployments where the main model is small (≤3B) and the workload is interactive chat, often the answer is "don't bother with speculation; ship the simpler thing." For deployments with large models or high-throughput batch workloads (code completion at scale), speculation is real and worth implementing.

## What you should believe after this lesson

Three sentences:

**1. Speculative decoding's win on-device is proportional to the main-to-draft size ratio** — large main + tiny draft = significant speedup (1.5–2×); small main + small draft = modest or negative speedup. The arithmetic favors larger models running on more capable hardware.

**2. N-gram speculation is the most practical variant for typical on-device use** because it has near-zero overhead and works well on code / structured / repetitive workloads. Medusa-style self-speculation requires pre-trained variants but avoids the separate-model memory cost.

**3. Most small-model on-device deployments don't benefit much from speculative decoding** — the overhead eats the savings. Reach for it when the main model is large (7B+) or when you've already squeezed every other optimization out.

## Hands-on (at home)

Measure speculative decoding's effect on your machine.

```bash
# Get a big-ish main and a small draft.
huggingface-cli download bartowski/Llama-3.1-8B-Instruct-GGUF \
    Llama-3.1-8B-Instruct-Q4_K_M.gguf --local-dir gguf
huggingface-cli download bartowski/Llama-3.2-1B-Instruct-GGUF \
    Llama-3.2-1B-Instruct-Q4_K_M.gguf --local-dir gguf

# Baseline (no speculation).
./llama-bench -m gguf/Llama-3.1-8B-Instruct-Q4_K_M.gguf -n 256

# With speculative decoding.
./llama-cli \
    -m gguf/Llama-3.1-8B-Instruct-Q4_K_M.gguf \
    --draft-model gguf/Llama-3.2-1B-Instruct-Q4_K_M.gguf \
    --draft 4 \
    -p "Write a 200-word essay on the history of memoization in computer science." \
    -n 200 \
    --temp 0
```

The second invocation reports the accepted-draft-tokens count and the effective tok/s. On M3 Pro you'll see something like:
- Baseline 8B: ~15 tok/s decode.
- With 1B draft (k=4): ~20–25 tok/s decode, ~60-75% acceptance rate.

For the 3B case (smaller main), the win is smaller; for the 70B case (much larger main), the win is bigger.

For n-gram speculation:

```bash
./llama-cli -m gguf/Llama-3.1-8B-Instruct-Q4_K_M.gguf --draft-n 3 -p "..." -n 200
```

N-gram works best for repetitive / structured generation (code, tables, formulaic text); it adds little for diverse / creative generation.

## Further reading

- "Fast Inference from Transformers via Speculative Decoding" (Chen et al, 2023) — the foundational paper.
- "Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads" (Cai et al, 2024).
- "EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty" (Li et al, 2024).
- "SpecInfer: Accelerating Generative LLM Serving with Speculative Inference and Token Tree Verification" (Miao et al, 2023).
- llama.cpp's speculative decoding docs.

Next lesson: **LoRA hot-swap.** A different operational technique: instead of shipping multiple full models for multiple personas / tasks, keep the base model resident and swap small LoRA adapters at request time. We look at the memory math, the merge-vs-keep-separate tradeoff, and when LoRA hot-swap is the right answer.
