---
title: "Lesson 43 — Prefill/Decode Disaggregation"
date: "2026-06-04"
module: "inference-from-scratch"
order: 43
tags: ["pd-disaggregation", "prefill", "decode", "distserve", "splitwise"]
author: "Sudipta Pathak"
prerequisites: ["42-chunked-prefill"]
---

# Lesson 43 — Prefill/Decode Disaggregation

## Why this lesson exists

Lessons 16-17 established that prefill is compute-bound and decode is bandwidth-bound. These have fundamentally different optimal hardware profiles. Continuous batching and chunked prefill (Lessons 41-42) try to balance the two on the same GPU; prefill/decode (PD) disaggregation goes further: run prefill on one set of GPUs (or one GPU type) and decode on another, connected by a fast network.

The idea: pick prefill hardware that's compute-optimized (cheap H100s, or A100s with high compute density). Pick decode hardware that's bandwidth-optimized (newer chips with higher HBM bandwidth, or larger memory). Each does its own thing well; the KV cache transfers between them.

This pattern (DistServe 2024; Splitwise 2024) is the cutting edge of large-scale LLM serving. It's overkill for most deployments but transformative for very-high-throughput, very-low-latency production.

The lesson is reading. The Hands-on sketches the architecture.

## The architecture

A PD-disaggregated serving stack:

1. **Prefill cluster**: GPUs optimized for compute. When a request arrives, runs the prefill (potentially with chunked prefill within each prefill node). Produces the KV cache for the entire prompt.
2. **KV transfer**: the prefill cluster sends the request's KV cache to the decode cluster over a high-bandwidth interconnect (NVLink, RDMA over Infiniband).
3. **Decode cluster**: GPUs optimized for memory bandwidth. Receives the KV cache; runs all decode steps; streams tokens to the client. When done, frees the KV cache.

The clusters communicate via a scheduler that decides which prefill node handles each new request and which decode node receives the result.

## Why this helps

Two wins:

**1. Each cluster runs at its optimal point.** Prefill clusters max compute utilization without being slowed by decode's bandwidth wait. Decode clusters max bandwidth utilization without being interrupted by prefill bursts.

**2. Better cost efficiency.** Compute-optimized GPUs (cheap consumer-grade for prefill) can be paired with bandwidth-optimized GPUs (H200, B100 for decode). The aggregate cost per request can drop significantly compared to running everything on uniform hardware.

DistServe reported ~3-4× throughput improvement over uniform-hardware continuous-batching at similar latency.

## The KV transfer challenge

Moving the KV cache between clusters is non-trivial:
- A 4K-token prompt for a 7B model has ~2 GB of KV cache (FP16).
- Transferring 2 GB over 200 Gbps Infiniband: ~80 ms. That's a meaningful chunk of the request's total latency.

Optimizations to reduce transfer cost:
- **Layer-wise transfer**: start sending earlier layers' KV while later layers are still being prefilled.
- **Compression in transit**: transfer INT8 KV (Lesson 20) instead of FP16.
- **NVLink between racks** (uncommon but real for top-tier deployments).
- **Co-location** of prefill and decode on the same machine for short prompts (avoid transfer entirely when it's not worth it).

The transfer cost limits how aggressive disaggregation can be. For short prompts the transfer dominates and you don't disaggregate; for long prompts (where prefill is the binding cost) it pays off.

## The scheduler

The scheduler decides:
- Which prefill node to send each request to (load balancing).
- When prefill finishes, which decode node to send the KV to.
- How to handle decode finishes (free KV memory on the decode node).
- How to handle prefill node failures (retry, with KV not yet committed).

This is sophisticated distributed-systems territory. DistServe and Splitwise both have detailed scheduler designs.

## Production maturity

As of mid-2026:
- DistServe is research-grade, with open-source implementation.
- Splitwise (Microsoft) is deployed at Azure scale.
- Anthropic and OpenAI likely have internal PD-disaggregated deployments (not publicly described).
- vLLM and SGLang have started adding PD-disaggregation features.

For mainstream deployments, the engineering complexity is significant. PD-disaggregation is a real win only at large scale (many concurrent users, multi-rack deployments).

## What you should believe after this lesson

Three sentences:

**1. Prefill and decode have different optimal hardware profiles**: prefill wants compute density, decode wants memory bandwidth. PD disaggregation runs them on different clusters, each optimized for its phase.

**2. The KV cache transfer cost is the main implementation challenge** — ~2 GB per medium-sized request is meaningful at network speeds. Layer-wise transfer, KV quantization, and selective co-location are the mitigations.

**3. PD disaggregation is large-scale serving territory**: DistServe/Splitwise patterns deployed at hyperscale, not yet mainstream in self-hosted deployments. For most deployments, continuous batching + chunked prefill on uniform hardware is sufficient.

## Hands-on (at home)

There's no straightforward single-machine demonstration of PD disaggregation; the value comes from cross-machine resource allocation. The conceptual sketch:

```
┌────────────────┐   KV cache transfer   ┌──────────────────┐
│ Prefill node 1 │ ─────────────────────► │ Decode node 1    │
│ (8x A100)      │                        │ (4x H200, MIG)   │
│ batch: prefill │   ◄──── new request    │ batch: decode    │
└────────────────┘                        └──────────────────┘
                                                 │
                                                 │ stream tokens
                                                 ▼
                                              user
```

For real PD-disaggregation experimentation, see DistServe's GitHub or Microsoft's Splitwise paper. Running a multi-node PD-disaggregated cluster requires NVLink + Infiniband setup; not a quick weekend project.

For the conceptual breakdown:

```python
# pd_disaggregation_sketch.py
class PrefillNode:
    def __init__(self, model, gpu_type='A100'):
        self.model = model
        self.kv_caches = {}

    def handle_request(self, request_id, prompt):
        kv = self.model.prefill(prompt)
        self.kv_caches[request_id] = kv
        return kv

class DecodeNode:
    def __init__(self, model, gpu_type='H200'):
        self.model = model
        self.in_progress = {}

    def receive_kv(self, request_id, kv):
        self.in_progress[request_id] = kv

    def step(self):
        new_tokens = {}
        for rid, kv in self.in_progress.items():
            new_token, kv = self.model.decode_step(kv)
            self.in_progress[rid] = kv
            new_tokens[rid] = new_token
        return new_tokens

class Scheduler:
    def __init__(self, prefill_nodes, decode_nodes):
        self.prefill = prefill_nodes
        self.decode = decode_nodes

    def route_request(self, request_id, prompt):
        # Pick a prefill node (load-balance).
        prefill_node = self.prefill[0]  # simplified
        kv = prefill_node.handle_request(request_id, prompt)
        # Transfer KV to a decode node.
        decode_node = self.decode[0]  # simplified
        # In reality: this is a network transfer of multiple GB.
        decode_node.receive_kv(request_id, kv)
        return decode_node

# This is conceptual; production code is thousands of lines of distributed systems work.
```

## Further reading

- "DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving" (Zhong et al, 2024).
- "Splitwise: Efficient Generative LLM Inference Using Phase Splitting" (Patel et al, 2024).
- DistServe GitHub repository.
- Anthropic's engineering blog (occasional commentary on internal serving architecture).

Next lesson: **RadixAttention.** SGLang's prefix-tree approach to cross-request KV sharing — the generalization of prefix caching (Lesson 19) to a more flexible data structure.
