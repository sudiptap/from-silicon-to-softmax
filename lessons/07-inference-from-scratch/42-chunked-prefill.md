---
title: "Lesson 42 — Chunked Prefill"
date: "2026-06-04"
module: "inference-from-scratch"
order: 42
tags: ["chunked-prefill", "ttft", "serving", "scheduling", "head-of-line-blocking"]
author: "Sudipta Pathak"
prerequisites: ["41-continuous-batching"]
---

# Lesson 42 — Chunked Prefill

## Why this lesson exists

Continuous batching (Lesson 41) handles the throughput side of LLM serving. But it has a tail-latency problem: a request with a very long prompt (say, 10K tokens) blocks all in-progress decoding while its prefill runs. Other users see a 1-2 second freeze in their stream as the long prompt is processed.

Chunked prefill (Agrawal et al, 2024; "Sarathi" paper, popularized in production by vLLM and SGLang) addresses this: break the long prompt into chunks (typically 512 or 1024 tokens each); process one chunk per step alongside ongoing decode tokens; spread the prefill cost across many steps so no single step is dominated by it.

The result: decode tokens for other users continue smoothly; the long-prompt user's TTFT increases slightly (split prefill is a bit slower than single prefill) but tail latency for everyone else improves dramatically.

The lesson is reading. The Hands-on simulates the TTFT-vs-throughput tradeoff.

## The problem without chunked prefill

In standard continuous batching:
- Each step runs decode for in-progress requests AND prefill for new requests.
- New request prefill is *atomic*: the entire prompt is processed in one step.
- A 10K-token prefill is one big chunked matmul in one step.
- That step takes 100-300 ms (depending on hardware).
- All in-progress decode requests pause for that step.

For a chat application where users are mid-stream, a 200 ms freeze every time someone submits a long context is jarring.

## The chunked prefill mechanism

Configure a `chunk_size` (typical: 512 or 1024 tokens). When a new request arrives:

1. Split its prompt into chunks of `chunk_size`.
2. Each step: include one prefill chunk from any waiting request (alongside the per-step decode work).
3. The prefill takes `ceil(prompt_length / chunk_size)` steps instead of 1.
4. Per step, the work is bounded: at most `chunk_size` prefill tokens + however many decode tokens are in progress.

The total prefill cost is similar (slightly higher due to overhead from splitting), but it's spread across multiple steps. No single step is dominated by one long prefill.

## The TTFT vs steady-state-latency tradeoff

Chunked prefill trades:

- **Higher TTFT for the long-prompt user**: a 10K-token prompt with chunk_size=512 takes 20 steps to prefill instead of 1. If each step is 20 ms, TTFT goes from ~200 ms to ~400-600 ms.

- **Lower jitter for everyone else**: in-progress decode steps continue at their normal rate. No 200 ms freeze when a long prompt arrives.

For interactive chat applications, the lower jitter usually matters more than the (still acceptable) TTFT increase. For batch-style workloads where TTFT is the primary metric, chunked prefill may not be worth it.

## The chunk-size choice

Smaller chunks (`chunk_size = 128`):
- Finer granularity; smoother decode for everyone else.
- Higher per-step overhead (more chunks per prefill).
- Prefill takes longer overall.

Larger chunks (`chunk_size = 2048`):
- Closer to standard continuous batching's behavior.
- Lower overhead per prefill.
- Still some jitter for in-progress decode when a chunk runs.

The sweet spot: 512-1024 for typical chat workloads. Tune based on workload profile.

## Chunked prefill + Sarathi's scheduling

The Sarathi-Serve paper (Agrawal et al, 2024) formalized chunked prefill with a scheduling algorithm:

1. Maintain in-progress decode and queued prefill.
2. Each step: pack the batch with decode tokens + prefill tokens up to the GPU's compute budget.
3. The compute budget is set so each step has a bounded duration (e.g., 20 ms target).

The scheduler dynamically picks how many decode tokens and how many prefill chunk tokens to run, trading off TTFT and decode latency.

vLLM, SGLang, and TensorRT-LLM all support chunked prefill with similar scheduling. It's standard in production.

## When chunked prefill matters

The clear wins:
- Multi-tenant LLM serving where users have variable prompt lengths.
- Chat applications where smooth streaming for in-progress users matters.
- Long-context deployments (8K, 32K, 128K) where prefill dominates per-request work.

When less critical:
- Single-user inference (no other users to block).
- Workloads where all prompts are short (~500 tokens or less).
- Batch processing where TTFT doesn't matter.

For server-side production: enable chunked prefill by default. The cost is minimal; the benefit on tail latency is real.

## What you should believe after this lesson

Three sentences:

**1. Chunked prefill breaks long-prompt prefills into chunks** (typically 512-1024 tokens) processed across multiple steps. This bounds per-step duration; in-progress decode continues smoothly without freezing when a long prompt arrives.

**2. The trade is mild TTFT increase for the long-prompt user vs lower jitter for everyone else.** For interactive chat workloads the trade is clearly worth it; for batch workloads it may not be needed.

**3. Chunked prefill + continuous batching + paged attention is the production serving stack** in 2026. vLLM, SGLang, and TensorRT-LLM all combine these; the synergy is what makes high-throughput, low-jitter LLM serving possible.

## Hands-on (at home)

A simulation of chunked vs unchunked prefill impact on tail latency.

```python
# chunked_prefill_sim.py

# Simulate a stream of requests; some have long prompts.
requests = [
    {"id": 0, "prompt_len": 100, "output_len": 50, "arrival_step": 0},
    {"id": 1, "prompt_len": 200, "output_len": 100, "arrival_step": 5},
    {"id": 2, "prompt_len": 5000, "output_len": 80, "arrival_step": 10},  # long prompt
    {"id": 3, "prompt_len": 150, "output_len": 60, "arrival_step": 12},
    {"id": 4, "prompt_len": 100, "output_len": 40, "arrival_step": 20},
]

def simulate_without_chunked_prefill(requests, max_steps=200):
    in_progress = []  # [(id, prompt_remaining_for_prefill, output_remaining)]
    waiting = list(requests)
    step_times = []
    decode_jitter = []
    
    for step in range(max_steps):
        # New arrivals.
        arrived = [r for r in waiting if r["arrival_step"] <= step]
        waiting = [r for r in waiting if r["arrival_step"] > step]
        for r in arrived:
            in_progress.append([r["id"], r["prompt_len"], r["output_len"]])
        
        if not in_progress: continue
        
        # If any request needs prefill, do it (atomically). Skip decode this step.
        prefills = [e for e in in_progress if e[1] > 0]
        if prefills:
            # Pick the first one; do its full prefill this step. Time proportional to prompt_len.
            this_prefill = prefills[0]
            step_duration = this_prefill[1]  # proportional to prefill tokens
            this_prefill[1] = 0  # done
            # Decode steps this step: ZERO (the step is consumed by prefill).
            decode_jitter.append(step_duration)
        else:
            # All decode step.
            step_duration = 1  # 1 ms per decode step (relative units)
            decode_jitter.append(0)
            for e in in_progress:
                e[2] -= 1
            in_progress = [e for e in in_progress if e[2] > 0]
        
        step_times.append(step_duration)
    
    return decode_jitter

def simulate_with_chunked_prefill(requests, chunk_size=512, max_steps=200):
    in_progress = []
    waiting = list(requests)
    decode_jitter = []
    
    for step in range(max_steps):
        arrived = [r for r in waiting if r["arrival_step"] <= step]
        waiting = [r for r in waiting if r["arrival_step"] > step]
        for r in arrived:
            in_progress.append([r["id"], r["prompt_len"], r["output_len"]])
        
        if not in_progress: continue
        
        # If any prefill remaining, do at most chunk_size tokens.
        prefills = [e for e in in_progress if e[1] > 0]
        prefill_tokens_this_step = 0
        for e in prefills:
            take = min(e[1], chunk_size)
            e[1] -= take
            prefill_tokens_this_step = take
            break
        
        # Also do decode for in-progress decode-only requests.
        decode_count = 0
        for e in in_progress:
            if e[1] == 0 and e[2] > 0:
                e[2] -= 1
                decode_count += 1
        in_progress = [e for e in in_progress if e[1] > 0 or e[2] > 0]
        
        # Step duration: bounded by chunk_size + decode count.
        step_duration = max(prefill_tokens_this_step // 10, decode_count, 1)
        # Jitter for decode: how much over baseline (1 ms) was this step?
        decode_jitter.append(step_duration - 1)
    
    return decode_jitter

no_chunk = simulate_without_chunked_prefill(requests)
with_chunk = simulate_with_chunked_prefill(requests, chunk_size=512)
print(f"Without chunked prefill: max decode jitter = {max(no_chunk)}")
print(f"With chunked prefill:    max decode jitter = {max(with_chunk)}")
```

You should see the unchunked version has a huge jitter spike when the 5000-token prefill arrives; the chunked version spreads it over ~10 steps with much smaller per-step jitter.

For real measurements, use vLLM with `--enable-chunked-prefill` vs without and measure P99 token-to-token latency on a mixed workload.

## Further reading

- "Sarathi-Serve: Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve" (Agrawal et al, 2024).
- vLLM documentation on chunked prefill.
- "DistServe" (Zhong et al, 2024) — Lesson 43's reference; takes the idea further with prefill/decode disaggregation.

Next lesson: **Prefill/decode disaggregation.** Different hardware for different bottlenecks. Prefill is compute-bound; decode is bandwidth-bound. Run them on different GPUs (or different GPU types) and connect with a fast network.
