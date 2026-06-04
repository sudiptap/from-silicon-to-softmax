---
title: "Lesson 41 — Continuous Batching"
date: "2026-06-04"
module: "inference-from-scratch"
order: 41
tags: ["continuous-batching", "orca", "serving", "throughput", "static-batching"]
author: "Sudipta Pathak"
prerequisites: ["40-bitnet"]
---

# Lesson 41 — Continuous Batching

## Why this lesson exists

Continuous batching (introduced by Orca, Yu et al, 2022; mainstreamed by vLLM) is the single biggest server-side throughput optimization in LLM serving. Going from static batching to continuous batching delivers 2-23× throughput improvement on production workloads — without any change to the model.

The mechanism: instead of waiting for a fixed batch to complete before starting another, continuously *swap in* new requests and *kick out* finished ones, keeping the GPU busy with a full batch at all times.

This lesson covers static batching's problem, continuous batching's solution, and why it changed LLM serving permanently.

The lesson is reading. The Hands-on simulates the throughput difference.

## Static batching's problem

The naive approach: collect N requests; run them as one batch; return results when all N complete.

The problem: LLM requests have *highly variable* output lengths. One request generates 20 tokens; another generates 500. With static batching:
- All N requests start together.
- The 20-token requests finish quickly but can't return until the 500-token requests finish.
- The GPU's effective batch size shrinks as requests finish; the batch is "padded" with idle slots.
- When the last (500-token) request finishes, the whole batch returns.

GPU utilization plummets. For a typical chat workload (high variance in output lengths), static batching keeps the GPU at 20-40% effective utilization.

## Continuous batching's solution

Instead of running fixed batches to completion:

1. Maintain a "running batch" of in-progress requests on the GPU.
2. Each step: run one decode iteration on all in-progress requests (in parallel).
3. After each step: check if any requests finished (hit EOS or max_tokens). If so, return them to the user and free their cache.
4. Check if any new requests are waiting. If so, run their prefill (Lesson 16) and add them to the running batch.
5. Repeat.

The GPU is never under-utilized. As requests finish, new ones immediately take their place. The batch stays full.

## The KV cache implication

For continuous batching to work, the KV cache management must support:
- **Adding new sequences** to the batch (allocating KV space for prefill).
- **Removing finished sequences** (freeing their KV space).
- **Mixed prefill + decode** in the same step (some sequences are prefilling; others are decoding).

This is exactly what PagedAttention (Lesson 18) was designed to enable. Without paged allocation, dynamic batch composition would be impractical because of memory fragmentation.

Orca's original implementation predated paged attention; it used token-level scheduling with separate compute kernels for prefill and decode. vLLM combined Orca's scheduling with paged attention for the modern production design.

## Mixed prefill + decode in one step

A subtle implementation point: a single forward pass through the model can mix prefill tokens (from new requests) and decode tokens (from in-progress requests). The kernel handles them in a single batched pass.

The "batch" passed to the model:
- Some sequences contribute N prefill tokens (their full prompt).
- Other sequences contribute 1 decode token (their next position).

The attention kernel and the FFN treat them uniformly; the attention computation respects per-sequence boundaries (one sequence's prefill doesn't attend to another's decode).

This unified batching is what gives continuous batching its full throughput. The alternative — separate prefill and decode passes — leaves GPU idle between phases.

## Throughput numbers

The Orca paper reported 2-23× throughput improvement over static batching on synthetic LLM serving workloads.

Production deployments:
- vLLM with continuous batching: ~2× throughput over a static-batch baseline at high concurrency.
- TGI (Hugging Face Text Generation Inference): similar gain.
- TensorRT-LLM: native continuous batching with similar gains.

The win scales with workload variance. For low-variance workloads (all requests the same length), static batching is fine; for chat-like workloads (high variance), continuous batching is essential.

## Implementation complexity

Continuous batching is *the* infrastructure change in LLM serving:
- Per-step scheduler: decide which requests to add, which to remove, what to run next.
- Per-sequence KV cache management (via paged attention).
- Dynamic memory management for the running batch.
- Multi-tenant safety (one request's failure shouldn't crash the batch).

The implementation in vLLM is ~50K lines of code; the core continuous-batching loop is maybe 5K lines but it touches everything.

## The wait-list problem

When the GPU is full (max concurrent sequences), new requests wait. The scheduler's job: when a sequence finishes, immediately admit the highest-priority waiting request.

Priorities:
- **Time-based**: oldest waiting first (FIFO).
- **Length-based**: prefer short requests (they finish fast; better tail latency).
- **Cost-based**: prefer high-paying customers (real production policy).

Most production deployments use FIFO with some preemption for long-running requests that exceed limits.

## What you should believe after this lesson

Three sentences:

**1. Static batching wastes GPU on variable-length LLM requests** — short requests block on long ones; effective batch size drops as the batch ages. Continuous batching swaps requests in and out per step, keeping the GPU at full utilization.

**2. The implementation requires paged-attention-style KV management** to dynamically allocate and free per-sequence cache. The unified mixed prefill+decode step is what enables full GPU utilization across phases.

**3. Continuous batching delivers 2-23× throughput improvement** over static batching on real workloads. It's the foundation of every modern LLM serving stack (vLLM, TGI, TensorRT-LLM, SGLang); LLM serving at scale would be much more expensive without it.

## Hands-on (at home)

A simulation of static vs continuous batching throughput.

```python
# batching_sim.py
import random

# Simulate request lengths (some short, some long).
random.seed(0)
n_requests = 100
lengths = [random.choice([20, 50, 100, 300, 500]) for _ in range(n_requests)]

# Static batching: process in batches of fixed size; longest determines batch time.
def static_batch_time(lengths, batch_size=8):
    total_tokens_processed = 0
    total_time_steps = 0
    for i in range(0, len(lengths), batch_size):
        batch = lengths[i:i+batch_size]
        # The batch finishes when its longest sequence does.
        # During those time steps, all sequences in the batch use a slot.
        max_len = max(batch)
        total_tokens_processed += sum(batch)  # actual useful tokens
        total_time_steps += max_len * batch_size  # GPU slot-time used (longer + idle padding)
    return total_tokens_processed, total_time_steps

# Continuous batching: at each step, the GPU processes batch_size sequences;
# as soon as one finishes, a new one is admitted.
def continuous_batch_time(lengths, batch_size=8):
    pending = lengths[:]
    in_progress = []  # list of (request_id, remaining_length)
    total_tokens = 0
    total_time_steps = 0
    while pending or in_progress:
        # Fill the batch to batch_size.
        while pending and len(in_progress) < batch_size:
            in_progress.append([0, pending.pop()])
        # One step: each in-progress request advances by 1 token.
        for entry in in_progress:
            entry[1] -= 1
            total_tokens += 1
        total_time_steps += batch_size  # all batch_size slots are used
        # Remove finished.
        in_progress = [e for e in in_progress if e[1] > 0]
    return total_tokens, total_time_steps

s_tok, s_time = static_batch_time(lengths, batch_size=8)
c_tok, c_time = continuous_batch_time(lengths, batch_size=8)
print(f"Static:     {s_tok} tokens in {s_time} slot-steps. Utilization: {s_tok/s_time*100:.1f}%")
print(f"Continuous: {c_tok} tokens in {c_time} slot-steps. Utilization: {c_tok/c_time*100:.1f}%")
print(f"Throughput improvement: {(c_tok/c_time) / (s_tok/s_time):.2f}x")
```

You'll see continuous batching achieves near-100% utilization while static batching stays around 40-60% on variable-length workloads. The ratio matches the production-reported speedups.

For real benchmarks, run vLLM in both modes (it supports static batching via a config flag for comparison) on a chat-like workload.

## Further reading

- "Orca: A Distributed Serving System for Transformer-Based Generative Models" (Yu et al, 2022) — the foundational continuous-batching paper.
- "Efficient Memory Management for LLM Serving with PagedAttention" (Kwon et al, 2023) — vLLM's combination of paged attention + continuous batching.
- TGI architecture docs.
- TensorRT-LLM serving documentation.

Next lesson: **Chunked prefill.** A complementary technique: instead of processing a long prompt's prefill in one giant kernel, break it into chunks interleaved with decode steps. Bounds TTFT under load.
