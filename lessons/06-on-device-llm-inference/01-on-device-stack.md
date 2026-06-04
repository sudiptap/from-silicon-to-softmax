---
title: "Lesson 1 — The On-Device LLM Stack"
date: "2026-06-04"
module: "on-device-llm-inference"
order: 1
tags: ["on-device", "llm", "budget", "thermal", "interactive-latency"]
author: "Sudipta Pathak"
prerequisites: ["00-overview"]
---

# Lesson 1 — The On-Device LLM Stack

## Why this lesson exists

Before the rest of the module gets practical, we need to set the constraints. An LLM running on a phone or laptop is not just "an H100 inference, but slower." The constraint profile is fundamentally different: memory in single-digit GB, bandwidth in hundreds of GB/s instead of TB/s, thermal envelope of 5–15 watts sustained, no horizontal scaling, no failover. The optimizations that pay off are different. The failure modes are different. The user-perceived quality bar is different.

This lesson is the constraint sheet. By the end, you should be able to look at a deployment requirement — "run X model on Y device, latency Z" — and predict whether it's feasible before you spend a day measuring.

The lesson is reading. The Hands-on builds the budget calculator that quantifies the constraints for your specific deployment target.

## What's different from data-center serving

The headline comparison, M3 Pro MacBook vs an H100 server:

| Axis | M3 Pro MacBook | H100 server |
| ---- | -------------- | ----------- |
| Memory | 18 GB unified | 80 GB HBM3 + 256+ GB host RAM |
| Memory bandwidth | 300 GB/s | 3000 GB/s |
| Compute (FP16) | ~7 TFLOPs | ~700 TFLOPs |
| Power budget | ~25 W package | ~700 W |
| Thermal sustainability | needs throttling under sustained load | engineered for sustained load |
| Concurrent users | 1 (the human) | 100s with batching |
| Failover | none | the whole rack |
| Latency floor (kernel launch) | ~50 µs | ~5 µs |

The "1 vs 100s of concurrent users" line is the biggest single difference. On the server, you batch many users' requests together to amortize the memory load of the weights — at batch 32, each user gets 1/32 the per-token wall-clock cost they'd see at batch 1. On a phone there is only one user; you cannot batch your way out of the bandwidth wall.

This means **on-device inference is fundamentally bandwidth-bound for the workloads we care about**, in a way that even server-side single-stream isn't quite. Every Module 3 lesson built up to this point — memory dominates, FLOPs are secondary, quantization wins because it reduces bytes moved. Module 6 takes that observation as the load-bearing premise and runs with it.

Other practical differences:

- **No failover.** If the inference loop crashes, the user sees it. There's no second device to take over.
- **Cold start matters.** The user opens the app; they want a response within a second or two. The model can't take 30 seconds to load.
- **Thermal limits cap sustained throughput.** A 4090 in a desktop can run at peak for hours; an iPhone running an LLM at peak overheats in 2–3 minutes and throttles to ~70% of peak performance. Sustained interactive use means designing for the post-throttle envelope, not the burst envelope.
- **Battery counts.** A 30-minute chat session that drains 30% of the battery is a product failure. Apps with high power consumption get uninstalled.

## The budget table

The numbers you must respect, by device class (as of mid-2026):

**Top-tier flagship phone (iPhone 16 Pro, Galaxy S24 Ultra, Pixel 9 Pro):**
- Memory available to app: 4–6 GB (out of 8–12 GB total).
- Memory bandwidth: 100–200 GB/s (varies by chip and platform).
- Sustained power: 4–6 W.
- Practical model size: ≤3B parameters at INT4 (≤1.5 GB).
- KV cache budget: 1–2 GB.

**Mid-range phone (2023+ Snapdragon 7-class, 6–8 GB RAM):**
- Memory available: 2–4 GB.
- Memory bandwidth: 50–80 GB/s.
- Sustained power: 3–4 W.
- Practical model size: ≤1.5B parameters at INT4 (≤750 MB).
- KV cache budget: 500 MB-1 GB.

**MacBook Air M3 (8 GB):**
- Memory available: 4–5 GB (more if you close other apps).
- Bandwidth: 100 GB/s.
- Sustained power: 8–10 W (fanless).
- Practical model size: ≤7B at INT4, with care.
- KV cache budget: 1–3 GB.

**MacBook Pro M3 Pro / M3 Max (18–96 GB):**
- Memory available: 10+ GB easily, model size is rarely the binding constraint.
- Bandwidth: 300–800 GB/s.
- Sustained power: 25–60 W (active fan).
- Practical model size: up to 70B at INT4 on M3 Max 64 GB+.
- KV cache budget: 5–20 GB.

**Desktop Linux + RTX 4090 (24 GB VRAM):**
- VRAM: 24 GB.
- Bandwidth: 1000 GB/s.
- Sustained power: 350 W under load.
- Practical model: 14B at FP16 or 30B at INT4.

The progression: smaller devices have tighter everything. The "headroom" on a desktop GPU disappears on a phone. A model that runs comfortably on a 4090 may not even fit on a phone, let alone run fast enough.

## The "feels interactive" latency budget

Human-perceived interactivity for a chat LLM:

- **<100 ms first token (TTFT)**: feels instant. Premium experience.
- **100–500 ms TTFT**: clearly responsive. Good experience.
- **500 ms–1 s TTFT**: noticeably delayed. Acceptable.
- **>1 s TTFT**: feels slow. Users start to lose faith.
- **>3 s TTFT**: users assume it's broken.

Tokens-per-second for the streaming generation (after TTFT):

- **>30 tok/s**: faster than the user can read. Premium.
- **15–30 tok/s**: matches comfortable reading speed. Good.
- **8–15 tok/s**: slower than reading. Tolerable for short responses.
- **<5 tok/s**: feels labored. Users wait, not read.

For a typical interactive chat, the target is "<500 ms TTFT, >15 tok/s decode." Both bars need to be hit; failing either one breaks the experience.

For non-chat use cases the bars shift:
- Code completion: <100 ms TTFT (no streaming visible to the user); tokens/sec doesn't matter as long as a 30–80-token completion arrives quickly.
- Voice assistant: <300 ms TTFT (to start TTS rolling); >20 tok/s decode to keep up with TTS.
- Background summarization (no real-time UI): minutes are fine.

## The operator-coverage problem

A subtle one that bites in practice: the model architecture you pick might not be fully supported by the runtime you pick.

Examples:
- A model uses a custom attention pattern (sliding-window with attention sinks, or grouped-query attention with non-standard group counts). The runtime's attention kernel doesn't know about it. The runtime falls back to a slower path, or refuses to load.
- A model uses MoE with 8 experts. The runtime supports MoE but only for 4 or 16 experts. You either re-architect or pick a different runtime.
- A model uses a novel quantization scheme (1.58-bit BitNet, MXFP4). The runtime quantization formats don't include it. You either re-quantize to a supported format or pick a different runtime.

Operator coverage is the silent killer of on-device deployments. The model converts, the file loads, the first run produces something, but the throughput is 1/3 of what the math says — because the model is hitting an unoptimized path for one critical op.

How to defend against this:

1. **Pick widely-supported architectures.** Llama, Mistral, Qwen, Gemma. These are the "shapes" every major runtime targets. Avoid bleeding-edge research architectures unless you're committed to writing the runtime support yourself.
2. **Benchmark early.** Don't pick a model based on its eval scores alone; benchmark on the target runtime before committing.
3. **Use mature runtimes.** llama.cpp, MLX, ExecuTorch (mostly). These have invested in coverage. Less-mature runtimes will have more gaps.

## What you should believe after this lesson

Three sentences:

**1. On-device LLM serving is bandwidth-bound, single-stream, thermally-constrained, and power-constrained** — the opposite of every assumption that holds in data-center serving. The optimizations that pay off are bandwidth-saving (quantization) and bytes-resident (KV cache management); FLOPs are secondary.

**2. The budget table varies by 100× across device classes** — a phone has 4 GB of usable memory and 50–200 GB/s of bandwidth; a Mac has 18+ GB and 300–800 GB/s; a desktop GPU has 24+ GB and 1000+ GB/s. Pick the device class first; design the deployment for that class.

**3. "Feels interactive" means <500 ms TTFT and >15 tok/s decode** for typical chat. Both bars must be hit; failing either breaks the user experience. Operator-coverage gaps in the runtime are the silent latency killer — benchmark before committing.

## Hands-on (at home)

Build a budget calculator for your specific deployment.

```python
# on_device_budget.py
def estimate_budget(model_params_billions, quantization_bits, n_layers, d_model, n_heads_kv, d_head,
                    target_context, peak_bandwidth_gbs, contention_factor=0.7, efficiency=0.7):
    """Estimate memory and per-token latency for an LLM deployment."""
    # Model weight memory.
    weight_bytes = model_params_billions * 1e9 * quantization_bits / 8
    # KV cache for target_context tokens.
    kv_bytes = 2 * n_layers * n_heads_kv * target_context * d_head * 2  # FP16 KV
    # Per-token bandwidth: weights + KV cache reads dominate.
    bytes_per_token = weight_bytes + kv_bytes
    # Effective bandwidth.
    eff_bw = peak_bandwidth_gbs * 1e9 * contention_factor * efficiency
    decode_time_per_token = bytes_per_token / eff_bw
    tokens_per_sec = 1.0 / decode_time_per_token
    return {
        'weight_gb': weight_bytes / 1e9,
        'kv_gb': kv_bytes / 1e9,
        'total_gb': (weight_bytes + kv_bytes) / 1e9,
        'decode_ms_per_token': decode_time_per_token * 1000,
        'tokens_per_sec': tokens_per_sec,
    }

# Llama 3.2 3B Instruct on M3 Pro at INT4, 2K context.
b = estimate_budget(
    model_params_billions=3.2, quantization_bits=4,
    n_layers=28, d_model=3072, n_heads_kv=8, d_head=128,
    target_context=2048,
    peak_bandwidth_gbs=300,  # M3 Pro
)
print("Llama 3.2 3B @ INT4 on M3 Pro, 2K context:")
for k, v in b.items():
    print(f"  {k}: {v:.2f}")

# Same model on a top-tier phone (Galaxy S24, ~100 GB/s).
b = estimate_budget(
    model_params_billions=3.2, quantization_bits=4,
    n_layers=28, d_model=3072, n_heads_kv=8, d_head=128,
    target_context=2048,
    peak_bandwidth_gbs=100,
)
print("\nLlama 3.2 3B @ INT4 on Galaxy S24, 2K context:")
for k, v in b.items():
    print(f"  {k}: {v:.2f}")
```

Expected output:
- Mac: ~1.6 GB weights, ~140 MB KV, ~25 tok/s.
- S24: ~1.6 GB weights, ~140 MB KV, ~8 tok/s.

The phone is roughly 3× slower than the Mac for the same model — entirely the bandwidth ratio (100 vs 300 GB/s). The formula is the model.

Run with different model sizes, quantization levels, and context lengths to feel out the budget envelope. The numbers won't be exactly what you measure (kernel-launch overhead, prefill amortization, sampling overhead all add) but they're within 30% — close enough to know whether a deployment is feasible before you build it.

## Further reading

- "Efficiently Scaling Transformer Inference" (Pope et al, 2022) — for the data-center half of the comparison.
- "MLPerf Tiny" benchmarks — for the embedded/microcontroller end of the spectrum.
- Apple, Qualcomm, Google ML developer documentation — each has a "designing for on-device" section that approximates this budget table from their hardware's POV.
- "On-Device Foundation Models" (Apple Intelligence WWDC sessions, 2024–2025) — Apple's specific framing of the on-device LLM constraints.

Next lesson: **Picking the model — the SLM landscape.** Now that the budget is set, we look at which small language models are credible deployment candidates in mid-2026, what each is best at, and how to pick.
