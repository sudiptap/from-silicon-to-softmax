---
title: "Lesson 1 — The M-Series SoC at a Glance"
date: "2026-06-04"
module: "mlx-apple-silicon"
order: 1
tags: ["apple-silicon", "soc", "p-cores", "e-cores", "ane", "amx", "slc"]
author: "Sudipta Pathak"
prerequisites: ["00-overview"]
---

# Lesson 1 — The M-Series SoC at a Glance

## Why this lesson exists

Most people who own an M-series Mac know the headline number — "M3 Pro, 12-core CPU, 18-core GPU, 18 GB unified memory" — and stop there. The headline hides almost everything that matters for ML systems. The CPU isn't one kind of core; it's two (performance and efficiency cores). The "18-core GPU" is one cluster of one kind of compute unit, organized in a way that's nothing like an NVIDIA SM. There's also an ANE (Neural Engine) the headline mentions only in marketing materials, an AMX matrix coprocessor the headline doesn't mention at all, and a system-level cache shared across the whole chip that the OS uses opaquely.

This lesson is the map. We walk the die — what blocks exist, what each one is for, what's reachable from ML code, and what isn't. The next two lessons go deeper on memory; this lesson lays out the compute units.

The lesson is reading. The Hands-on identifies your specific M-series chip and reports its key parameters.

## The 30-second tour

A typical M-series SoC (M3 Pro as a concrete example) packs:

- **CPU complex**: 6 performance ("P") cores + 6 efficiency ("E") cores. Two distinct microarchitectures, sharing a memory controller and the SLC.
- **GPU complex**: 18 GPU cores, each containing 128 ALUs (so 2304 ALUs total). Organized into clusters that share some on-chip memory.
- **Neural Engine (ANE)**: 16 cores, ~38 TOPS at INT8. A separate accelerator with its own memory paths.
- **AMX (Apple Matrix coprocessor)**: a wide matrix multiply unit attached to the P-core cluster. Not in the public Metal API, but used by Accelerate.framework and MLX under the hood.
- **Media engines**: ProRes encoders/decoders, HDR pipelines. Not ML-relevant.
- **Secure Enclave, ISP, display, USB controllers, …**: peripheral; off-topic.
- **Memory**: 18 GB of LPDDR5 unified memory connected over a wide on-package bus (300 GB/s on M3 Pro, ~400 GB/s on M3 Max, ~800 GB/s on Ultra variants).
- **System Level Cache (SLC)**: ~16 MB shared across the whole package, used as a last-level cache by CPU, GPU, and ANE.

Every M-series variant scales these numbers: M3 (8/10/10), M3 Pro (12/18/16), M3 Max (16/40/16), M3 Ultra (24/80/32) for CPU/GPU/ANE counts. The microarchitectures are shared across variants in a generation; only the count differs.

## The CPU: two kinds of cores

Apple Silicon's CPU is heterogeneous. The two core types:

**P-cores (performance)**: wide out-of-order, deep pipeline, big caches (192 KB L1d per core on M3-class), high clock (~4 GHz). Optimized for sequential or moderately-parallel work where latency per operation matters. ML-relevant cases: single-threaded preprocessing, the orchestration code around model inference, dense linear algebra via Accelerate that uses AMX (more on this in a moment).

**E-cores (efficiency)**: narrower, in-order or modestly out-of-order, smaller caches (64 KB L1d), lower clock (~2 GHz). Optimized for background work where energy efficiency dominates over latency. ML-relevant cases: background data loading, log writers, monitoring threads. You usually don't want compute-heavy ML code on E-cores.

The OS scheduler decides which core a thread runs on. You can hint preferences via `pthread_set_qos_class_self_np` (Quality of Service classes), but ultimate placement is the kernel's call. The relevant QoS classes:

- `USER_INTERACTIVE`: highest, P-core preference.
- `USER_INITIATED`: P-core preference under load.
- `UTILITY`: mix.
- `BACKGROUND`: E-core preference.

For ML inference threads, `USER_INITIATED` is the right pick — the thread is doing user-visible work and should land on a P-core.

In practice: the OS gets it right most of the time. You don't have to think about it for typical Python ML code. Where it matters is when you're writing a custom inference server with explicit thread pools; the QoS hints help the scheduler keep the inference work on P-cores.

## AMX: the matrix coprocessor

This is the unit most people don't know exists. The AMX (Apple Matrix Extensions) is a wide matrix-multiply engine sitting next to the P-core cluster. It's accessed via private instructions (not in the public ISA documentation) and used by Apple's Accelerate.framework for dense linear algebra.

What AMX does:

- 512-bit-wide vector multiplies (vs the 128-bit NEON on the CPU side).
- Outer-product accumulation into a 1 KB register file.
- Effectively a "tensor core for the CPU."

Throughput is significant: an M1's AMX does ~1 TFLOPs FP32 (vs the ARM cores' ~250 GFLOPs aggregate). On M2 and M3, the AMX gets wider and faster.

How to reach it: there's no public API. You get AMX implicitly when you call `Accelerate`'s `BLAS` or `LAPACK` routines (or `cblas_sgemm`, or NumPy when linked against Accelerate). MLX uses AMX for some CPU paths. Custom code can't directly emit AMX instructions through any documented mechanism.

For Module 4 specifically, the AMX matters in two ways:

1. It's why CPU matmul on Apple Silicon is unexpectedly fast — Accelerate's BLAS hits the AMX, and a single P-core can match a moderate Intel desktop CPU on matmul throughput.
2. MLX's CPU backend uses it, so MLX-on-CPU is much faster than NumPy-with-OpenBLAS on a same-clock comparison.

The lesson: if your code does dense linear algebra on Apple Silicon CPU, use Accelerate (or MLX). Don't roll your own; you won't reach AMX.

## The GPU: not an NVIDIA GPU

The Apple GPU shares ideas with mobile GPUs more than with NVIDIA's data-center GPUs. The structure (M3-class):

- **GPU cores** (a "core" in Apple's terminology) are organized in clusters of ~10–18 per generation/variant.
- Each core has 128 ALUs (FP32/INT32-capable) arranged in 4 SIMD-groups of 32 lanes each.
- Each core has 32 KB of "threadgroup memory" (the Apple-Silicon analog of CUDA shared memory).
- Cores share an L2 cache; the GPU complex shares the SLC with the CPU.

Compared to an NVIDIA SM:
- Similar SIMD width (32 lanes both).
- Similar threadgroup-memory size (NVIDIA shared mem is 48–164 KB depending on generation; Apple is 32 KB, smaller).
- **No native tensor core** on M1/M2. M3 added the first matrix-multiply instructions (the "SMMA" path). M4 expanded this. The hardware-accelerated matmul path on Apple GPU is much newer and much less aggressive than NVIDIA's tensor cores.
- **Unified memory means no separate VRAM**. The GPU reads from the same LPDDR5 bus the CPU uses. This is the headline difference and the topic of Lesson 2.

Practical throughput:
- M3 Max GPU: ~17 TFLOPs FP16, ~400 GB/s memory bandwidth.
- M3 Pro GPU: ~7 TFLOPs FP16, ~300 GB/s bandwidth.
- An RTX 4090 by comparison: ~150 TFLOPs FP16, ~1 TB/s.

So the M3 Max GPU is roughly 1/10 the absolute throughput of a 4090 but at 1/10 the power. For on-device LLM inference where the workload is memory-bound (Module 3 Lesson 1), the throughput ratio is similar to the bandwidth ratio — i.e., the M3 Max is about 2.5–4× slower than the 4090 on LLM decode, not 10× slower. Per-watt, Apple is dramatically more efficient.

## The Neural Engine (ANE)

The ANE is a fixed-function neural network accelerator. It's not a general-purpose compute device; it runs a specific set of operations efficiently and falls back to CPU/GPU for unsupported ops.

What the ANE does well:
- INT8 / FP16 convolutions and matmuls.
- Activation functions, batch norm, layer norm.
- Common transformer ops at FP16.

What the ANE doesn't do:
- Custom kernels. No programmable surface.
- General compute. It's not a GPU; you can't write arbitrary code for it.
- BF16 (until M4-generation; older ANE is FP16 only).

How to reach it: Core ML is the public API. You compile a model to a `.mlpackage`, load it via `coremltools` or Swift, and choose `MLComputeUnits.all` (lets the OS schedule between CPU/GPU/ANE). The runtime decides per-op whether the ANE can run it; if not, falls back.

The ANE's marketing claim is ~38 TOPS on M3. Real-world LLM inference rarely hits this number because of two things:
1. **LLM matmul shapes don't fit the ANE's tile sizes well** for autoregressive decode (small batch). The ANE shines on conv-heavy workloads (vision models) and large-batch matmul.
2. **The op coverage gap**. A typical LLM has ops the ANE doesn't support (rotary position embeddings, certain norms, custom attention masks); these fall back to GPU, and the back-and-forth dominates latency.

Production rule in 2026: vision models → ANE wins decisively. Audio models (Whisper) → ANE wins, often with cleverness around chunking. LLMs → GPU via MLX usually wins; ANE-only LLM inference exists (some specialized models) but is rare. We come back to this in Lesson 11.

## The Secure Enclave, ProRes, etc.

These are real but off-topic:
- **Secure Enclave**: cryptography, key storage, biometrics.
- **ProRes engines**: hardware-accelerated video codec for ProRes / H.264 / H.265 / AV1.
- **ISP (Image Signal Processor)**: camera pipeline.
- **Display engine**: drives the screens.
- **Thunderbolt, USB controllers**: peripherals.

For ML inference, you'll never touch any of these directly. Worth knowing they exist so you understand why an idle M3 still shows non-zero power draw.

## What you should believe after this lesson

Three sentences:

**1. The M-series SoC is more heterogeneous than the "X-core CPU + Y-core GPU" headline suggests** — two CPU core types, an undocumented matrix coprocessor (AMX), a programmable GPU, and a fixed-function neural accelerator (ANE), all sharing a unified memory pool.

**2. The GPU is the dominant compute surface for general-purpose ML and especially for LLM inference** — ANE wins on conv-heavy vision models but is rarely the right tool for autoregressive LLM decode. AMX is hidden behind Accelerate.framework and rarely matters directly.

**3. The headline differences from NVIDIA are unified memory (no VRAM), smaller absolute throughput per chip (~1/10 of a 4090), better perf-per-watt by ~5–10×, and a different (smaller) on-chip memory hierarchy that changes which CUDA patterns transfer over and which don't.**

## Hands-on (at home)

Identify your specific Mac's chip and report its key parameters.

```bash
# chip_identity.sh
sysctl -a | grep -E "machdep.cpu.brand_string|hw.perflevel|hw.memsize"
system_profiler SPHardwareDataType | grep -E "Chip|Total Number of Cores|Memory"
system_profiler SPDisplaysDataType | grep -E "Chipset Model|Total Number of Cores"
```

A sample output on M3 Pro:

```
machdep.cpu.brand_string: Apple M3 Pro
hw.perflevel0.physicalcpu: 6      # P-cores
hw.perflevel1.physicalcpu: 6      # E-cores
hw.memsize: 19327352832            # 18 GB
Chip: Apple M3 Pro
Total Number of Cores: 12 (6 performance and 6 efficiency)
Memory: 18 GB
Chipset Model: Apple M3 Pro
Total Number of Cores: 18         # GPU cores
```

Now use MLX to report the actual peak memory bandwidth on your machine.

```python
# memory_bandwidth.py
# pip install mlx
import mlx.core as mx
import time

# A large array; copy it many times to saturate memory bandwidth.
N = 1024 * 1024 * 64  # 64 MFP16 elements = 128 MB
a = mx.random.normal((N,), dtype=mx.float16)
mx.eval(a)

# Warmup.
for _ in range(3):
    b = a + 1
    mx.eval(b)

t0 = time.time()
runs = 50
for _ in range(runs):
    b = a + 1
    mx.eval(b)
dt = time.time() - t0

# Two reads (a) + one write (b) = 3 array passes per iter.
bytes_moved = 3 * a.nbytes * runs
print(f"effective bandwidth: {bytes_moved / dt / 1e9:.1f} GB/s")
```

Expected results:
- M3: ~80 GB/s effective (chip peak ~100 GB/s).
- M3 Pro: ~200 GB/s (peak ~300 GB/s).
- M3 Max: ~280 GB/s (peak ~400 GB/s).
- M3 Ultra: ~500 GB/s (peak ~800 GB/s).

"Effective" runs ~70% of peak because we're not perfectly streaming. The peak is what the bus supports; the effective is what real code achieves. This is the number that gates LLM inference throughput — every subsequent lesson works against this ceiling.

## Further reading

- "Apple M1 / M2 / M3 architecture" — Anandtech reviews of each generation are the deepest published technical analyses outside Apple itself.
- "Apple Silicon" Wikipedia entry — for the canonical reference of per-chip core counts and bandwidth numbers across variants.
- Dougall Johnson's "Apple Silicon CPU Optimization Guide" — for the deep CPU microarchitecture details, especially around the P-cores and AMX.
- Asahi Linux project documentation — for community-reverse-engineered details on the GPU and the GPU MMU.
- "Inside Apple's Neural Engine" (various conference talks; the WWDC sessions on Core ML Performance are the most authoritative).

Next lesson: **Unified memory architecture.** The single most important architectural choice on Apple Silicon. We look at what "unified memory" actually means at the hardware level, the cost model that replaces CUDA's `cudaMemcpy` mental model, and why a CUDA pattern that pre-copies tensors to GPU often *hurts* on Apple Silicon.
