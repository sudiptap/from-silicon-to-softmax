---
title: "Lesson 3 — Apple Silicon GPU Architecture"
date: "2026-06-03"
module: "gpu-computing"
order: 3
tags: ["apple-silicon", "metal", "tbdr", "unified-memory", "simd-group"]
author: "Sudipta Pathak"
prerequisites: ["02-nvidia-architecture"]
---

# Lesson 3 — Apple Silicon GPU Architecture

## Why this matters

Apple's GPU is the single most important GPU for on-device AI. Hundreds of millions of devices ship with it. Its programming model is similar enough to NVIDIA's that the mental shifts you just absorbed transfer cleanly. Its hardware is different enough that "the NVIDIA way" can be wrong on Apple Silicon — especially around memory, where the unified architecture changes the cost model substantially.

This lesson covers what's the same, what's different, and what those differences imply for fast kernels. After this lesson you have a working architectural picture for *both* of the GPUs you'll actually touch in this curriculum: NVIDIA for data-center and frontier LLM training, Apple for on-device serving.

## Concept

### TBDR: tile-based deferred rendering

Apple's GPU originated as a mobile design. Mobile GPUs use **tile-based deferred rendering (TBDR)**: the screen is broken into small tiles (32×32 pixels), and rendering happens *per tile*, entirely in fast on-chip memory before the result is committed to main memory. This was a thermal/power decision for phones — write to fast SRAM, batch commits to slow DRAM.

The architectural fingerprint of TBDR persists in Apple's compute GPUs:

- Threadgroup memory (Apple's name for shared memory) is generous and fast.
- Per-tile temporary storage is a first-class concept.
- Memory bandwidth to *unified DRAM* is more constrained than you'd expect from a discrete GPU at similar TFLOPs.

The implication for compute: Apple GPUs favor kernels that load a tile into threadgroup memory and operate on it many times before writing back. Roughly the same pattern as cache-blocked CPU matmul or shared-memory CUDA matmul — perhaps even more strongly so.

### The hardware layout

An M-series Mac has *one* GPU "die" that's part of the SoC (system on chip), alongside the CPU cores, the Neural Engine, the secure enclave, media engines, and the unified memory controller. The GPU is structured as:

- **GPU cores** — Apple's term for what NVIDIA calls SMs. An M1 had 8, M1 Max had 32, M2 Max had 38, M3 Max had 40, M4 Max has up to 40. Apple says each core can run ~24 threads concurrently (the "threads per core" varies by generation).
- **ALUs per core** — each core has 128 ALUs (FP32 capable). 32 SIMD-group sized at 32. So one M3 Max GPU has ~5,120 FP32 ALUs total.
- **Tensor accelerators per core** — recent generations (M3+) have dedicated matrix-multiply hardware called "Dynamic Caching" + the ANE (Neural Engine) sitting separately. We'll come back to ANE in Module 4.

For comparison: an RTX 4090 has 16,384 FP32 cores; an M3 Max has 5,120. The 4090 wins in absolute FP32. But the M3 Max draws ~50W vs the 4090's ~300W. And on memory-bandwidth-bound workloads (most LLM inference at modest batch sizes), the gap is much smaller than the FLOPs ratio implies.

### Memory: unified, shared, but bandwidth-bounded

This is the deepest architectural difference from NVIDIA.

The CPU, GPU, and Neural Engine on an Apple Silicon SoC all access the **same physical DRAM** through the same memory controller. There is no `cudaMemcpy`. Any pointer the CPU produces is directly addressable by the GPU. The "transfer to GPU" step in a CUDA program becomes a no-op on Apple Silicon.

This is the "zero-copy" property. For *correctness* it's lovely. For *performance* it's mixed:

- **No data transfer cost** for CPU→GPU handoff. Massive win for pipelines that frequently do small CPU/GPU dispatches.
- **Total memory bandwidth is shared.** The GPU isn't competing with itself for HBM (as on NVIDIA); it's competing with the CPU, the ANE, and the media engines. Bandwidth budget is a system-level resource, not a GPU resource.
- **Bandwidth numbers are lower in absolute terms.** M3 Max: ~400 GB/s. M4 Max: ~546 GB/s. H100: ~3 TB/s. Apple GPUs lose the bandwidth arms race by a factor of ~5–8.

The right framing: Apple GPUs are *bandwidth-constrained relative to their compute*. The arithmetic intensity needed to be compute-bound is lower (~50–80 FLOPs/byte), which means *more* kernels are compute-bound on Apple than on NVIDIA. Conversely, kernels that are bandwidth-bound on NVIDIA are even more bandwidth-bound on Apple.

The right kernel patterns: aggressive use of threadgroup memory (load tile once, reuse many times), avoid round-trips through unified DRAM, keep working sets small per dispatch.

### SIMD groups and the execution model

Apple's analog to a warp is a **SIMD-group** of 32 threads. Same SIMT semantics: 32 threads execute the same instruction at the same time, divergence is serialized, all the patterns from NVIDIA apply.

A "thread" in Apple's parlance is similarly a virtual concept; a **threadgroup** (= block in CUDA) is a unit that runs on one GPU core and can communicate via threadgroup memory and threadgroup barriers.

The terminology is intentionally similar to OpenGL/Vulkan compute conventions. If you've written compute shaders for any graphics API, the Metal compute API will feel familiar.

### Tensor accelerators and matrix engines

Apple has been progressively adding matrix-multiply acceleration:

- **Apple AMX** — a CPU-side coprocessor, *not* on the GPU. Accessible via Accelerate.framework / `cblas_sgemm`. Discussed in Module 1.
- **ANE (Apple Neural Engine)** — a separate accelerator on the SoC, designed for inference. Accessible only via Core ML; we'll cover it in Module 4. ~18 TOPs of int8.
- **GPU matrix instructions** — newer generations (M3+) have `simdgroup_matrix_multiply_accumulate` style operations in Metal that look much like NVIDIA's tensor core MMA. Lower throughput than NVIDIA tensor cores but in the right ballpark for on-device workloads.

The picture: for inference-heavy workloads, Apple offers multiple paths (CPU+AMX, GPU+SIMD, GPU+matrix instructions, ANE). The "right one" depends on the model shape, the precision, and the latency target. Module 4 is the comparison.

### What you can't do on Apple GPUs

A short list of things that are CUDA standard but missing or different on Metal:

- **Direct equivalent of `cp.async.bulk` / TMA.** Metal has its own async copy primitives but they're less powerful and less documented.
- **Cooperative groups** — CUDA has rich abstractions for sub-warp / multi-warp / multi-block coordination. Metal's equivalents are sparser.
- **Profiling depth.** Nsight Compute on NVIDIA is the gold standard for kernel-level profiling. Xcode's Metal debugger is good but less comprehensive.
- **Open architectural docs.** NVIDIA publishes detailed whitepapers; Apple does not. Most of what we know about Apple GPU internals comes from Asahi Linux work, Dougall Johnson's reverse engineering, and tribal knowledge.

If you're writing the same kernel for both platforms, plan to develop on NVIDIA (better tooling, better docs) and port to Apple second.

## Code walkthrough

A way to query Apple GPU properties similar to how `cudaGetDeviceProperties` works on NVIDIA. In Swift / Objective-C through Metal:

```swift
import Metal

guard let dev = MTLCreateSystemDefaultDevice() else {
    fatalError("No Metal device")
}

print("Device: \(dev.name)")
print("Family Apple9: \(dev.supportsFamily(.apple9))")
print("Has unified memory: \(dev.hasUnifiedMemory)")
print("Max threadgroup memory: \(dev.maxThreadgroupMemoryLength) bytes")
print("Max threads per threadgroup: \(dev.maxThreadsPerThreadgroup)")
print("Working set size: \(dev.recommendedMaxWorkingSetSize / 1024 / 1024) MB")
print("Registry ID: \(dev.registryID)")

// SIMD-group size — usually 32 on current Apple GPUs.
let lib = try! dev.makeDefaultLibrary(bundle: .main)
// More properties available via Metal's reflection APIs.
```

Save and run as part of a small Mac app or with `swift run` in a package. Or, the easier path: use MLX from Python:

```python
import mlx.core as mx
print("Default device:", mx.default_device())
# MLX exposes some properties via internal APIs that vary by version.
# For full details, check System Information → Graphics/Displays in macOS.
```

`system_profiler SPDisplaysDataType` in a terminal gives you the GPU-side architectural numbers (cores, name, etc.) without writing code.

## Mental model & pitfalls

Single sentence: **Apple GPUs are smaller, share memory with the CPU and Neural Engine, and are more bandwidth-constrained than NVIDIA — which makes threadgroup-memory discipline matter even more, while the unified memory removes the entire CPU↔GPU transfer problem.**

Pitfalls when porting CUDA mental models to Apple:

- **Assuming high memory bandwidth.** 400 GB/s vs 3 TB/s. Bandwidth-bound kernels need redesigning.
- **Ignoring the ANE.** For inference-only workloads at FP16/INT8, the Apple Neural Engine often beats both the GPU and the CPU. Going GPU-only on Apple is sometimes the wrong choice. Module 4 covers when to use which.
- **Reaching for `cudaMemcpy` analogs.** Don't. Just pass the pointer. Allocations from CPU-side Swift/Python are automatically GPU-addressable.
- **Profiling with the wrong tool.** Xcode's GPU debugger and the "Metal System Trace" template in Instruments are the right tools. Don't try to use Nsight (NVIDIA-only) or assume `perf` covers GPU.
- **Trusting marketing TFLOPs.** Apple's headline GPU numbers are often peak ALU throughput, which a real kernel may achieve only at the highest occupancy. The matrix-instruction throughput on newer chips is the more relevant number for ML inference.

## Hands-on (at home)

If you don't have an Apple Silicon Mac, this lesson is reading-only — come back to it when you do.

1. **Confirm your GPU.** `system_profiler SPDisplaysDataType` in a terminal prints architectural details (chipset, cores, metal family).

2. **Install MLX.**

```bash
pip install mlx mlx-lm
```

3. **A first matmul on the Apple GPU**:

```python
import mlx.core as mx
import time

n = 4096
a = mx.random.uniform(shape=(n, n)).astype(mx.float16)
b = mx.random.uniform(shape=(n, n)).astype(mx.float16)

# Warm up
c = mx.matmul(a, b)
mx.eval(c)

t0 = time.time()
for _ in range(20):
    c = mx.matmul(a, b)
mx.eval(c)
dt = (time.time() - t0) / 20

flops = 2 * n**3
print(f"{dt*1000:.2f} ms  {flops / dt / 1e12:.1f} TFLOPs FP16")
```

Expected on M3 Max: ~8 ms per 4096-cube FP16 matmul → ~17 TFLOPs FP16.

On M2 Pro: ~25 ms → ~5–6 TFLOPs FP16.

On M1: ~40 ms → ~3–4 TFLOPs FP16.

These are the *real* numbers you have to budget against on Apple Silicon. They're well below NVIDIA H100 (~500 TFLOPs FP16 with tensor cores), which is expected: different power envelope, different price point. They're also impressive for the wattage — an M3 Max draws ~50W at peak GPU; an H100 draws 700W. Per watt, Apple is competitive.

4. **Compute Apple's arithmetic intensity ceiling.**

For M3 Max: ~17 TFLOPs FP16 / 400 GB/s = ~42 FLOPs/byte. Compare to H100's ~330 FLOPs/byte. **Apple GPUs become compute-bound at a much lower arithmetic intensity** — which means more kernels can saturate their FLOPs ceiling without needing tensor cores. It also means bandwidth-bound kernels are *even more* bandwidth-bound. Both effects matter for fast on-device inference.

## Further reading

- Asahi Linux project's "Asahi GPU drivers" pages — by far the deepest open documentation of M-series GPU internals.
- "Apple Silicon GPU Compute" — various blog posts and conference talks; Marius Magnusson, Dougall Johnson, and others.
- Apple's *Metal Performance Shaders Documentation* — vendor-supplied kernels that hit hardware optimal paths. Worth knowing what they cover.
- "Programming Apple's GPU with Metal" — Apple's developer documentation.
- MLX source code on GitHub — production-quality Metal/MLX kernels, with the architectural assumptions baked into the comments.

Next lesson: CUDA basics. Now that you have the mental and architectural picture for both NVIDIA and Apple, we write actual kernels. We start on NVIDIA because the toolchain is more mature.
