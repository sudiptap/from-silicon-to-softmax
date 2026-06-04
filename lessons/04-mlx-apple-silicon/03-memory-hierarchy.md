---
title: "Lesson 3 — The Memory Hierarchy on Apple Silicon"
date: "2026-06-04"
module: "mlx-apple-silicon"
order: 3
tags: ["memory-hierarchy", "cache", "slc", "threadgroup-memory", "bandwidth"]
author: "Sudipta Pathak"
prerequisites: ["02-unified-memory"]
---

# Lesson 3 — The Memory Hierarchy on Apple Silicon

## Why this lesson exists

Lesson 2 established that there's one shared memory pool. This lesson is about everything between that pool and the compute units. Apple Silicon has a multi-level cache hierarchy that's distinct from both standard x86 desktop CPUs and from NVIDIA GPUs. The P-core L1 is unusually large. The L2 is shared across cores in a cluster. The GPU has its own L2 plus a programmer-managed threadgroup memory analog to CUDA's shared memory. And the System Level Cache (SLC) sits at the bottom, shared by everyone — a chip-wide last-level cache that absorbs traffic before it hits LPDDR5.

The lesson sketches each level, what it costs to access, and what patterns matter for ML kernels. The most important point: the SLC's existence means streaming-heavy access patterns get more help than they would on a non-SLC architecture, which makes some CUDA optimizations less load-bearing on Apple Silicon and gives some patterns a free-ish win that wouldn't work on NVIDIA.

The lesson is reading. The Hands-on profiles a tiled matmul kernel on Apple Silicon to show the cache effects.

## The hierarchy, top to bottom

For an M3 Pro as a concrete example:

| Level | Per-unit size | Latency (cycles) | Visible to |
| ----- | ------------- | ---------------- | ---------- |
| P-core L1 (instructions) | 192 KB | ~4 | one P-core |
| P-core L1 (data) | 128 KB | ~4 | one P-core |
| E-core L1 (data) | 64 KB | ~4 | one E-core |
| P-cluster L2 | 16 MB | ~14 | all P-cores in the cluster |
| E-cluster L2 | 4 MB | ~14 | all E-cores in the cluster |
| GPU per-core register file | ~64 KB | ~1 | one GPU thread (subdivided) |
| GPU threadgroup memory | 32 KB | ~5 | one threadgroup |
| GPU L1 | 16–32 KB | ~10 | one GPU core |
| GPU L2 | ~6 MB | ~50 | all GPU cores |
| System Level Cache (SLC) | 16 MB | ~80 | the entire SoC (CPU, GPU, ANE, …) |
| LPDDR5 DRAM | 18 GB | ~300 | everyone |

A few things stand out vs an x86 + NVIDIA workstation:

**P-core L1 is huge.** 128 KB L1d is ~2× a typical Intel/AMD desktop core. This means a single P-core can hold ~16K FP64 elements or ~32K FP32 elements in L1 — enough for medium working sets to stay completely cached. For ML preprocessing (NumPy operations, tokenization, custom Python loops) this matters: a lot of the working set fits in L1 and never goes to DRAM.

**P-cluster L2 is shared.** 16 MB shared across the 6 P-cores. This is the level where parallel threads on the CPU side benefit from sharing data — a common batch passed across threads stays in L2.

**SLC is unique to Apple Silicon.** A 16 MB chip-wide cache that sits in front of DRAM. Everyone benefits — CPU loads that miss L2 may hit SLC; GPU loads that miss GPU-L2 may hit SLC. The SLC is opaque to programmers (there's no API to control it), but it changes the practical cost of "loading from DRAM": often you don't, you load from SLC.

**GPU threadgroup memory is smaller than CUDA shared memory.** 32 KB per threadgroup on Apple GPU vs 48–164 KB on NVIDIA SMs. This means tiled matmul kernels that use very large shared memory tiles on NVIDIA need to be retuned with smaller tiles on Apple. The MLX and MPSGraph kernels handle this automatically; if you're writing custom MSL, the tile size is a per-architecture parameter.

## What the SLC does for ML kernels

The SLC's practical effect: kernels that fit in 16 MB or less effectively see ~16 MB of "extra cache" beyond their L2. For ML inference, this most often shows up in:

**KV cache reads during decode.** A short-context KV cache (say, 2 KB per layer × 32 layers = 64 KB per token, × 512 tokens = 32 MB) doesn't fit in SLC, but the *recently-accessed* portion (the last few tokens, the prefix being attended to most heavily) often does. This is a noticeable speedup over a non-SLC architecture.

**Small model inference.** A quantized 1B-parameter model at INT4 is ~500 MB, far larger than SLC. But the *frequently-accessed* layers (the first and last few, the LM head) cycle through SLC and run faster than the bandwidth math would predict.

**Attention computation in FlashAttention-style kernels.** The intermediate tile of attention (Q × K^T) for a single head is small (~16 KB for typical dimensions) and lives in threadgroup memory anyway. But the streaming reads of K and V — particularly the tail tokens at long context — get some help from SLC.

The SLC is opaque, so you can't "target" it explicitly. The right mental model: assume that accesses to a working set under ~10–12 MB (leaving headroom for OS and other processes) get an extra cache level for free. This isn't worth optimizing for; it's worth being aware of when you're surprised by Apple Silicon performance numbers.

## GPU memory: registers, threadgroup, cache

Within the GPU, the hierarchy mirrors NVIDIA's at a high level:

- **Registers** per thread (effectively per ALU): used for live values. The number is limited; if a kernel uses too many registers per thread, occupancy drops.
- **Threadgroup memory**: 32 KB per threadgroup, programmer-managed. The Apple analog of CUDA shared memory. Same usage pattern: load a tile from DRAM into threadgroup memory, multiple threads collaborate on it, write the result back.
- **GPU L1 / L2**: read-only and write-through caches for global memory. Help with reuse of recently-accessed data.
- **System Level Cache** (same as above).
- **DRAM** (unified with CPU's).

The key MSL-vs-CUDA mapping:

| CUDA term | MSL term |
| --------- | -------- |
| Block / thread block | Threadgroup |
| Warp (32 threads) | SIMD group (32 threads on Apple too) |
| Shared memory | Threadgroup memory |
| Constant memory | (no direct equivalent — use `constant`-qualified args) |
| Texture memory | Sampler-bound textures |
| Global memory | Buffer (`device` qualified) |

Module 2's Lessons 5 and 7 went deep on the kernel-writing details. This lesson is about the implications for kernel performance specifically on Apple Silicon: smaller threadgroup memory means smaller tile sizes; the SLC means streaming patterns get extra help; the unified memory means no host-device transfer overhead at kernel launch.

## CPU/GPU shared L2: not quite

A common question: do the CPU and GPU share any cache level above SLC? The answer: no. The P-cluster L2 is CPU-only; the GPU L2 is GPU-only. The first level they share is the SLC.

This means: a tensor that the CPU just computed and wrote to DRAM (or to SLC) is not in any GPU cache. When the GPU first reads it, it has to fetch from SLC or DRAM, paying the corresponding latency. There's no "warm handoff" between the two like there is, say, between two cores in the same CPU cluster.

In practice: this doesn't matter much for typical workloads (the CPU→GPU handoff is amortized across the whole kernel). It does matter for very small kernels where the handoff latency is a large fraction of the runtime — but that's not a common ML scenario.

## How this changes kernel-writing decisions

Three concrete differences from CUDA when you're writing custom Metal kernels (Lesson 8 goes deeper):

**1. Use smaller tiles.** A tile of 128×128 FP16 needs 32 KB — exactly the threadgroup memory limit. CUDA tiles of 64×64×2 = 16 KB are a closer match for Apple's threadgroup memory. Some MLX kernels use even smaller tiles (32×32×4 = 4 KB) to allow multiple threadgroups per core.

**2. Don't over-rely on shared memory.** On NVIDIA, the optimal kernel often pushes shared memory to its limit. On Apple Silicon, the SLC + GPU-L2 hierarchy means that some patterns that would lose without aggressive shared-memory use on NVIDIA do fine on Apple Silicon — the caches absorb the difference. Profile before assuming you need to manually tile something to threadgroup memory.

**3. Plan for occupancy.** Apple GPU cores have similar register pressure tradeoffs as NVIDIA SMs. A kernel that uses too many registers per thread reduces occupancy and underutilizes the SIMD units. Apple's tooling (Xcode's Metal Performance HUD, the kernel analyzer) reports these metrics; aim for ~50% theoretical occupancy as a starting point.

## What you should believe after this lesson

Three sentences:

**1. Apple Silicon's memory hierarchy mirrors x86 and NVIDIA at the level of "registers → L1 → L2 → DRAM" but adds two distinctive features**: an unusually large P-core L1 (good for CPU-side preprocessing) and a chip-wide System Level Cache (good for working sets up to ~10 MB).

**2. GPU threadgroup memory is smaller (32 KB) than NVIDIA shared memory (48–164 KB)**, which means CUDA tiled-kernel sizes need to be reduced when porting; the MLX framework handles this automatically, but custom MSL kernels need to choose tile sizes for the Apple constraint.

**3. The SLC is the unique architectural feature** — opaque to programmers but real, giving streaming and KV-cache-heavy workloads a noticeable boost over what the DRAM-bandwidth math would predict. Account for it qualitatively when reasoning about performance, but don't try to target it explicitly.

## Hands-on (at home)

Profile a tiled matmul kernel to see the cache effects on Apple Silicon.

```python
# cache_effects.py
# pip install mlx
import mlx.core as mx
import time

device = mx.gpu  # default

# Sweep matmul sizes to see where cache effects kick in.
sizes = [256, 512, 1024, 2048, 4096, 8192]
results = []

for s in sizes:
    a = mx.random.normal((s, s), dtype=mx.float16)
    b = mx.random.normal((s, s), dtype=mx.float16)
    mx.eval(a, b)

    # Warm.
    for _ in range(3):
        c = a @ b
        mx.eval(c)

    # Bench.
    n_iters = max(3, 50 // (s // 512))
    t0 = time.time()
    for _ in range(n_iters):
        c = a @ b
        mx.eval(c)
    dt = (time.time() - t0) / n_iters

    flops = 2 * s ** 3
    tflops = flops / dt / 1e12
    mem_per_iter = 2 * (a.nbytes + b.nbytes + c.nbytes) / 2  # weights+activations both read once
    print(f"size={s:5d}  time={dt*1000:7.2f} ms  {tflops:6.2f} TFLOPs  "
          f"working set: {(a.nbytes + b.nbytes + c.nbytes)/1e6:6.1f} MB")
    results.append((s, tflops))
```

Expected pattern on M3 Pro:
- 256×256: working set 384 KB. Fits in GPU L1 + a bit of L2; very high TFLOPs (or zero — too small to amortize launch cost).
- 1024×1024: working set 6 MB. Fits in SLC. High TFLOPs.
- 4096×4096: working set 96 MB. Way over SLC, hits DRAM. TFLOPs plateau at the compute-bound ceiling.

The transition point where the working set exceeds SLC is the place where the kernel becomes truly bandwidth-bound. Above that, the kernel runs at peak compute (about 7 TFLOPs FP16 on M3 Pro for matmul); below it, the kernel runs faster because most loads hit caches.

Part 2 — examine what happens at sizes just inside vs just outside SLC.

```python
# slc_boundary.py
import mlx.core as mx
import time

# SLC is ~16 MB. Each matrix at FP16 is s*s*2 bytes. Two matrices.
# Crossover for two matrices in SLC: 2 * s*s*2 = 16 MB → s ≈ 1448.

for s in [1024, 1280, 1500, 1700, 2000]:
    a = mx.random.normal((s, s), dtype=mx.float16); mx.eval(a)
    b = mx.random.normal((s, s), dtype=mx.float16); mx.eval(b)
    for _ in range(3): mx.eval(a @ b)
    t0 = time.time()
    for _ in range(20): mx.eval(a @ b)
    dt = (time.time() - t0) / 20
    working = (a.nbytes + b.nbytes) / 1e6
    tflops = 2 * s**3 / dt / 1e12
    print(f"s={s:5d} working={working:5.1f} MB  {tflops:6.2f} TFLOPs")
```

You should see a discontinuity around s=1500 — TFLOPs drop noticeably as the working set spills out of SLC. The exact numbers depend on chip variant and what else is running on your machine.

## Further reading

- "Apple Silicon CPU Optimization Guide" by Dougall Johnson — has the most accessible discussion of P-core L1/L2 latencies.
- Anandtech M-series reviews — for the canonical cache hierarchy reports each generation.
- Apple's "Metal Best Practices Guide" — Section "Choose an Appropriate Resource Storage Mode," which is the official discussion of the cache attributes.
- "Memory hierarchy of the Apple M-series" — Maynard Handley's reverse-engineering documentation (PDF; widely available).
- "Performance Optimization on Apple Silicon" WWDC sessions — typically released annually; the most recent has SLC commentary.

Next lesson: **Metal & MPS — the programming surfaces.** We pivot from hardware to APIs. Metal is the low-level GPU programming layer; MPS is Apple's high-level neural-network ops library; MPSGraph is the autodiff layer on top of MPS. Each has its place in a stack, and the choice depends on what you're building and how much you need to control.
