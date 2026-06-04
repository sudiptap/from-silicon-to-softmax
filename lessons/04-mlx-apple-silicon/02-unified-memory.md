---
title: "Lesson 2 — Unified Memory Architecture"
date: "2026-06-04"
module: "mlx-apple-silicon"
order: 2
tags: ["unified-memory", "zero-copy", "bandwidth", "iommu", "cost-model"]
author: "Sudipta Pathak"
prerequisites: ["01-m-series-soc"]
---

# Lesson 2 — Unified Memory Architecture

## Why this lesson exists

The single biggest architectural difference between Apple Silicon and a CUDA workstation isn't the GPU's microarchitecture — it's that there is no separate GPU memory pool. The 18 GB of RAM in an M3 Pro is one pool shared by the CPU, GPU, ANE, and Secure Enclave alike. No `cudaMemcpy`. No host-to-device transfer. No "your tensor must be on the right device" exceptions to chase.

This sounds like a strict win, and on many workloads it is. But it changes the performance cost model in ways that break habits trained on CUDA. Two examples we'll work through: the CUDA pattern of "pre-load the model weights onto GPU, then iterate" doesn't even mean anything on Apple Silicon — there's no "onto GPU" to move them to. More subtly, the CUDA pattern of "make sure your tensor is contiguous on device" matters less because there's no copy step, but bandwidth contention between CPU and GPU sharing the same memory bus introduces a new cost dimension that CUDA programmers don't think about.

This lesson is the cost model. By the end you should be able to look at a tensor operation and reason about its memory cost in Apple-Silicon terms, not in translated-from-CUDA terms.

The lesson is reading. The Hands-on measures the CPU↔GPU bandwidth contention directly.

## What "unified memory" means at the hardware level

The M-series SoC has a single on-package memory controller wired to LPDDR5 DRAM chips that sit on the same package as the SoC die. Every compute unit (CPU, GPU, ANE) has a physical path to this memory controller. There is no separate DRAM for the GPU. The "memory bus" is one bus, shared.

A tensor in memory has a single physical address. That address is reachable from a CPU load instruction, a GPU memory read, and (for supported types) an ANE memory read — without any data movement.

In CUDA terms: every allocation is automatically `cudaMallocManaged` (unified memory), but with the wrinkle that there's no actual page migration happening behind the scenes — the page lives where it was allocated and stays there.

Two consequences:

**1. No host-device transfer cost.** Operations that would require `cudaMemcpyHostToDevice` or `cudaMemcpyDeviceToHost` in CUDA simply don't exist on Apple Silicon. A NumPy-style preprocessing step in Python can hand its output directly to MLX or to a Metal kernel with no copy. This is a real, often-large win for end-to-end pipelines.

**2. CPU and GPU share memory bandwidth.** The aggregate memory bandwidth (300–800 GB/s depending on chip variant) is split across whoever is reading. If the CPU and GPU are both running matmul-heavy workloads, the total bandwidth they get is the chip's peak — not the sum of "CPU bandwidth" and "GPU bandwidth," because there is no such sum. The pattern that CUDA programmers use of "load the next batch on CPU while GPU computes the current batch" doesn't give the same overlap; both sides are pulling from the same pool.

## The mental model shift

A few CUDA patterns and what they become on Apple Silicon:

| CUDA pattern | Apple Silicon equivalent |
| ------------ | ------------------------ |
| `model.to('cuda')` to move weights to GPU | No-op. Weights live in unified memory; the GPU can read them where they are. |
| `cudaMemcpyAsync` to overlap copy with compute | Not applicable. There's no copy. |
| Pin host memory with `cudaMallocHost` for fast transfers | Not applicable. |
| "GPU memory usage" as a separate metric from RAM | One number: total memory pressure. |
| Pre-fetch the next batch to GPU memory | Pre-compute the next batch on CPU; result is already where GPU wants it. |
| Worry about PCIe bandwidth as a bottleneck | Worry about LPDDR5 bandwidth shared with CPU. |
| Two-process / two-GPU workloads keep their tensors isolated | Two processes contend for the same memory bus; pinning to different processors doesn't help bandwidth-wise. |

The pattern of "model fits in GPU VRAM" becomes "model fits in RAM, minus what other processes need, minus what the OS and graphics layer want." A 16 GB MacBook running a 7B model at FP16 (~14 GB) is technically possible but practically miserable because the OS and browser leave less than 14 GB free; the model swaps to disk and inference latency explodes.

## When "zero-copy" isn't quite zero

A subtle point: while there's no data copy between CPU and GPU, there is sometimes a *page table* operation when memory needs to be made accessible to a different unit. On older Apple Silicon (M1), the GPU and CPU had slightly different page table requirements (caching behavior, attribute bits), and transitioning a buffer between "CPU-only-resident" and "GPU-accessible" required a small kernel call that flushed caches.

On M2 onward, this overhead is essentially gone — buffers are accessible to all units by default with no transition penalty. But there's still a small cost when you allocate a buffer through Metal (versus malloc) because Metal sets up its own bookkeeping. The difference is microseconds, not milliseconds, but for very short kernels it can be a noticeable fraction.

The MLX framework abstracts all of this: an `mx.array` lives in unified memory accessible by both CPU and GPU streams. The framework picks the right backend for each operation; you don't manage the placement.

## The bandwidth contention story

This is the real new cost dimension. Suppose you have an M3 Pro (300 GB/s peak bandwidth). You're running an LLM inference workload on the GPU that uses ~70% of peak bandwidth (210 GB/s). If, simultaneously, a CPU process starts a memory-bandwidth-heavy workload (say, decompressing a large video stream at 100 GB/s), the GPU's bandwidth drops correspondingly — total memory traffic can't exceed the chip's peak, so the two workloads share.

This shows up as: your LLM token rate drops when other heavy CPU work is happening on the same machine. The drop is proportional to how much bandwidth the other workload consumes.

There are a few practical implications:

**1. Idle the OS during benchmarks.** If you're measuring LLM token rate, close the browser, the music player, the video conferencing app. They cost real bandwidth.

**2. Don't run two LLM inference processes on the same Mac.** Even if both fit in memory, they're now sharing 300 GB/s. The throughput per request drops, often super-linearly because of cache contention on top of bandwidth contention.

**3. CPU preprocessing should be lean.** A common CUDA optimization is to do heavy preprocessing on CPU while GPU computes. On Apple Silicon, heavy CPU preprocessing competes for memory bandwidth and can slow down the GPU workload.

**4. The ANE has its own memory paths.** This is one place where running on ANE genuinely doesn't compete with GPU bandwidth. For workloads where the ANE can handle the model (vision), running on ANE while CPU/GPU do other things is a real composition win.

## Reading the cost model

A back-of-envelope formula for memory-bound workloads on Apple Silicon:

```
effective_time = bytes_to_move / (peak_bandwidth × contention_factor × efficiency)
```

Where:
- `bytes_to_move`: total memory traffic (model weights loaded per token + activations + KV cache reads).
- `peak_bandwidth`: chip spec (300 GB/s on M3 Pro).
- `contention_factor`: 1.0 if you're alone; ~0.5 if a heavy CPU workload runs alongside; varies.
- `efficiency`: how close to peak the workload achieves. Empirically 0.6–0.8 for well-tuned MLX kernels.

For an INT4 7B Llama on M3 Pro doing autoregressive decode:
- bytes per token ≈ 3.5 GB (model weights at INT4) + ~50 MB (KV cache reads for short context).
- 3.5 GB / (300 GB/s × 1.0 × 0.7) ≈ 17 ms per token ≈ 60 tokens/second.

This is approximately what people measure in practice. The formula is the cost model. When your measured number is much lower than the prediction, the answer is usually one of: contention (other workloads), inefficiency (kernel using less than 70% of peak), or you forgot something (KV cache, scale tensors, activations).

## What's different about Apple Silicon's bandwidth

The LPDDR5 bus on M3 Pro at 300 GB/s sounds low compared to an H100's 3000 GB/s HBM3. The headline ratio is 10×. But for *inference*, the relevant number is what's reachable from a real model:

- H100 HBM bandwidth: 3000 GB/s. Real LLM decode: ~2000 GB/s effective.
- M3 Pro bandwidth: 300 GB/s. Real LLM decode: ~200 GB/s effective.

So the real-world ratio is closer to 10×, matching the headline. What you *don't* get on Apple Silicon is the ability to scale up — there's one chip per machine; the H100 can be scaled to 8-way in a node.

The other thing you get on Apple Silicon that's hard to get on H100: large memory at moderate cost. An M3 Max with 128 GB unified is ~$4K; that puts a 70B-parameter model into FP16 territory and a 200B-class model into INT4 territory. An H100 with 80 GB HBM is ~$30K. The bandwidth ratio favors the H100 by 10×; the dollar-per-GB ratio favors the Mac by ~10× in the other direction. This is why Apple Silicon has carved out a niche for large-model on-device inference: not because it's the fastest, but because it's the *only* commodity hardware that holds the model in memory at the price point.

## What you should believe after this lesson

Three sentences:

**1. Unified memory means no host-device transfer cost** and the CUDA mental model of "weights live on GPU, activations are copied across" simply doesn't apply. The framework handles this transparently; you stop thinking about device placement.

**2. The new cost dimension is bandwidth contention** — CPU and GPU share the same memory bus, so anything running on the CPU competes with the GPU for bandwidth. Lean preprocessing and avoiding background memory-heavy workloads matter for benchmark integrity and real-world latency.

**3. The cost model `time = bytes / (bandwidth × contention × efficiency)` predicts LLM decode latency well**, and the per-chip bandwidth (300–800 GB/s) is the single number that bounds it. Per-watt and per-dollar Apple wins; per-chip absolute throughput, NVIDIA wins.

## Hands-on (at home)

Measure the CPU/GPU bandwidth contention directly.

```python
# bandwidth_contention.py
# pip install mlx
import mlx.core as mx
import numpy as np
import time
import threading

# GPU-bound workload: large matmul that's bandwidth-heavy.
def gpu_workload(stop_event, duration_s):
    # Memory-bound matmul: a single 4096-D vector against a big weight.
    W = mx.random.normal((4096, 14336), dtype=mx.float16)
    x = mx.random.normal((1, 4096), dtype=mx.float16)
    mx.eval(W, x)
    t0 = time.time()
    iters = 0
    while time.time() - t0 < duration_s and not stop_event.is_set():
        y = x @ W
        mx.eval(y)
        iters += 1
    dt = time.time() - t0
    bytes_per_iter = W.nbytes  # the bandwidth-dominating term
    return iters / dt, bytes_per_iter * iters / dt / 1e9  # iter/s, GB/s

# CPU-bound workload: NumPy memcopy that consumes bandwidth.
def cpu_memcopy_loop(stop_event):
    arr = np.random.randn(1024 * 1024 * 32).astype(np.float32)  # 128 MB
    while not stop_event.is_set():
        _ = arr.copy()

# Baseline: just the GPU workload.
print("Baseline (GPU only):")
ips, gbps = gpu_workload(threading.Event(), 3.0)
print(f"  GPU matmul: {ips:.1f} iter/s, {gbps:.1f} GB/s effective")

# With CPU memcopy contention.
print("\nWith concurrent CPU memcopy:")
stop = threading.Event()
cpu_thread = threading.Thread(target=cpu_memcopy_loop, args=(stop,))
cpu_thread.start()
ips, gbps = gpu_workload(stop, 3.0)
stop.set()
cpu_thread.join()
print(f"  GPU matmul: {ips:.1f} iter/s, {gbps:.1f} GB/s effective")
```

Expected: the GPU effective bandwidth drops 20–40% when the CPU is also hammering memory, depending on chip variant. On M3 Pro, the baseline might be 200 GB/s and the contended number 130–160 GB/s. This is the contention factor in action.

Part 2 — verify that there's no host-device transfer cost.

```python
# zero_copy.py
import mlx.core as mx
import numpy as np
import time

# Create a large array in NumPy (CPU-resident in NumPy's mental model).
size = 1024 * 1024 * 32  # 128 MB
np_arr = np.random.randn(size).astype(np.float32)

# Convert to MLX. This is the moment a CUDA workflow would pay a copy cost.
t0 = time.time()
mx_arr = mx.array(np_arr)
mx.eval(mx_arr)
dt_convert = time.time() - t0
print(f"np→mx convert: {dt_convert*1000:.2f} ms ({np_arr.nbytes/dt_convert/1e9:.1f} GB/s)")
# On Apple Silicon, this is mostly the bookkeeping cost of MLX wrapping the
# array; the actual data isn't moved if the layout matches. Real "copy" cost
# would be ~0.5 ms for 128 MB at memory speed.

# Now use the MLX array. Should be cheap.
t0 = time.time()
y = mx.sum(mx_arr)
mx.eval(y)
dt_use = time.time() - t0
print(f"mx.sum on the array: {dt_use*1000:.2f} ms")
```

You'll see the convert step is small (microseconds for the wrapping; maybe a few ms if MLX has to materialize a copy because of layout). The "GPU access" of the array via `mx.sum` is essentially the cost of the reduction itself, not a transfer cost. This is the unified-memory win made concrete.

## Further reading

- Apple's "Metal documentation" — specifically the sections on "Storage modes" for `MTLBuffer`, which spells out the cache attribute combinations available.
- "Anandtech: Apple's M1 SoC review" — the deepest published analysis of the M1 memory hierarchy and unified-memory implications.
- "The Cost of Heterogeneous Memory" (academic, various authors) — the general academic framing of unified memory as a cost-model question.
- MLX source: `mlx/core/allocator.h` and `mlx/backend/metal/allocator.cpp` — for how MLX wraps Metal allocators.
- "Maynard Handley's documentation on Apple Silicon" (gumroad/leaked PDFs) — the most detailed reverse-engineered docs on the memory controller and SLC.

Next lesson: **The memory hierarchy on Apple Silicon.** We zoom out from unified-memory-as-architectural-choice to the actual cache hierarchy: L1/L2/SLC, what each one caches, how they interact between CPU and GPU, and what this means for kernel-writing decisions.
