---
title: "Lesson 1 — The GPU Mental Model"
date: "2026-06-03"
module: "gpu-computing"
order: 1
tags: ["gpu", "simt", "warps", "occupancy", "mental-model"]
author: "Sudipta Pathak"
prerequisites: ["bare-metal"]
---

# Lesson 1 — The GPU Mental Model

## Why this matters

A GPU is not just a "faster CPU." It is a *different kind* of processor that's optimal for a different shape of problem. CPUs are designed to make one thread as fast as possible — deep pipelines, branch prediction, large caches, out-of-order execution, all in service of one instruction stream. GPUs are designed to make ten thousand threads run *acceptably* fast in parallel — shallow pipelines, modest per-thread cache, but enormous bandwidth and tens of thousands of in-flight threads.

The two designs have different efficiency frontiers. A workload that *can* be expressed as ten thousand independent computations will run 100× faster on a GPU. A workload that *can't* — anything with serial dependencies, complex control flow, or per-thread data — runs slower on a GPU than on a CPU, despite the raw silicon being more capable.

This lesson is the mental shift. Once you have it, everything in the rest of the module — CUDA, Metal, Triton, FlashAttention — is a fluent application of the same model. Without it, every line of GPU code looks like CPU code badly transliterated, and you'll fight the hardware on every kernel.

## Concept

### Throughput-oriented design

Take one GPU number and one CPU number for context. An NVIDIA H100 has ~80 *billion* transistors and pulls 700 watts at peak. A high-end x86 CPU has ~30 billion transistors and pulls 250 watts. The H100 has 132 SMs (Streaming Multiprocessors), each with 128 FP32 cores → 16,896 FP32 cores total. The CPU has maybe 32 cores. So 500× more "cores," at 4× the power. Per-core, GPU cores are much weaker — but they're a different bargain.

The bargain is **throughput, not latency**. A GPU is built to keep tens of thousands of in-flight threads busy by switching between them. When one thread stalls (waiting for memory, waiting for division, waiting for anything), the SM switches to another ready thread instantly — zero context-switch cost, because all the registers are in a giant per-SM register file with a "current thread" pointer. The CPU's trick of "out-of-order execution to hide latency" is replaced by the GPU's trick of "swap threads to hide latency." Same goal, different mechanism.

The implication: GPUs are happiest with **a lot of parallel work**. Tens of thousands of threads is normal; hundreds of thousands is routine; millions for a big matmul is fine. Anything less is leaving the chip idle, the same way a CPU with 30% IPC is leaving execution units idle.

### SIMT: thirty-two threads, one instruction

NVIDIA's execution model is **SIMT**: Single Instruction, Multiple Threads. Threads are grouped into **warps** of 32. All 32 threads in a warp execute the *same instruction* on different data, in lockstep. This is exactly like SIMD on a CPU — 32-lane vector operation — but exposed in the programming model as if they were 32 independent threads.

When the threads in a warp need to take different control-flow paths (e.g., `if (x > 0)`), the hardware **serializes**: it runs the `if` body on a subset of threads with the other threads masked out, then runs the `else` body on the other subset with the first set masked out. The total cost is the sum of both paths. This is called **warp divergence**, and it's the GPU's version of "branch misprediction is expensive" — except the cost is per warp, not per branch.

Apple Silicon GPUs use a similar model. The unit there is called a **SIMD-group** rather than a warp; the size is 32 on Apple GPUs (was 32 in M1, varies in newer chips). The semantics are the same: 32 threads executing in lockstep, divergent paths serialized.

### Threads, blocks, grids

A CUDA kernel launches **a grid of blocks**, where each **block** contains **threads**. Numbers are configurable per launch:

```
total_threads = grid_dim × block_dim
```

A typical launch: `grid_dim = (256, 1, 1)`, `block_dim = (256, 1, 1)` → 65,536 threads.

Blocks are the *scheduling unit* — one block is assigned to one SM and stays there for its lifetime. Threads within a block can communicate via **shared memory** (a small fast SRAM, per-block) and can synchronize with each other (`__syncthreads()`). Threads in *different* blocks cannot communicate except via global memory.

The 3D shape `(x, y, z)` for blocks and grids is purely a convenience for indexing into 1D / 2D / 3D problems. Internally everything is linearized.

Apple Silicon calls these **threadgroups** (= blocks) and **grids** (= grids). Threadgroup memory ≈ shared memory. The picture is identical.

### Memory hierarchy

A typical NVIDIA GPU:

| Level | Size | Latency | Bandwidth | Programmer-managed? |
| --- | -----: | -----: | --------: | ------ |
| Registers (per thread) | 256 × 32-bit | 1 cycle | — | Yes (implicitly) |
| Shared memory (per SM) | 100 KB | ~10 cycles | many TB/s | Yes |
| L1 cache (per SM) | shared with shared memory | ~20 cycles | many TB/s | No |
| L2 cache (chip-wide) | 40–60 MB | ~200 cycles | ~5 TB/s | No |
| HBM (global memory) | 40–80 GB | 400+ cycles | 2–3 TB/s | No |

What stands out:

- **Registers are huge.** A CPU has ~32 named registers; a GPU thread has 256 (and the SM has tens of thousands of physical registers split among threads). This makes register-resident accumulators the norm for fast kernels.
- **Shared memory is fast and small.** Think of it as a *programmer-managed L1*. You explicitly load data into it from global memory, then operate on it. This is the GPU's equivalent of cache blocking on CPUs — except you control the cache.
- **Global memory is fast in bytes/sec but slow in cycles.** HBM gives terabytes per second of bandwidth, but each *access* costs hundreds of cycles. The trick to fast GPU code: amortize each global memory access over many compute operations by reusing data from shared memory or registers.

Apple Silicon's GPU memory hierarchy is structurally similar, with one massive difference: **unified memory**. The "global memory" of the GPU is the *same physical memory* as the CPU. No `cudaMemcpy`. No PCIe transfer. This changes the cost model substantially and is the topic of Module 4.

### Coalesced memory access

When 32 threads in a warp issue 32 memory accesses, the hardware tries to **coalesce** them: if the addresses are contiguous (each thread reads its index × 4 bytes, say), the 32 accesses are served by *one* 128-byte memory transaction. If the addresses are scattered, the hardware issues 32 separate transactions — 32× the memory traffic for the same compute.

This is the GPU equivalent of "stride-1 access is cache-friendly" from Module 1. Threads in a warp should read adjacent memory addresses. The right way to lay out your data is to make this happen naturally.

### Occupancy

The SM has a budget of registers and shared memory. A kernel that uses lots of registers per thread (call it a "fat" thread) lets the SM keep fewer threads in flight. A kernel that uses very little per thread can have many more threads in flight, giving the SM more options for hiding latency.

**Occupancy** is the ratio of active warps on an SM to the maximum the SM could support. High occupancy is generally good — more options to hide latency — but the relationship isn't monotonic. Past a point, higher occupancy means each thread is register-poor and spills to local memory (a slow per-thread global memory area). The right occupancy is workload-dependent; many production kernels run at 25–50% occupancy and beat 75%-occupancy versions because of register usage.

### One sentence

**A GPU is a machine for running an army of threads with a generous memory bandwidth budget, in exchange for tight per-thread budgets on registers, fast cache, and divergent control flow.**

## Code walkthrough

Conceptual — we'll see real kernel code in Lessons 4 and 7. For now, the *shape* of GPU code:

```
// CPU code: one thread does all the work
for i in 0..N {
    c[i] = a[i] + b[i];
}

// GPU code: N threads each do 1/N of the work
__global__ void add_kernel(float* a, float* b, float* c) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    c[i] = a[i] + b[i];
}

// Host launches grid of N threads:
int N = 1_000_000;
add_kernel<<<(N + 255) / 256, 256>>>(a_gpu, b_gpu, c_gpu);
```

What changed conceptually:

- The `for` loop disappeared. The parallelism *is* the loop — one thread per iteration.
- Each thread computes its own `i` from its grid coordinates.
- We launch N threads (rounded up to a multiple of the block size).

This is the canonical GPU pattern: take a loop, turn each iteration into a thread, launch them all at once. For "embarrassingly parallel" code (vector add, element-wise multiply, ReLU), this is essentially the whole story.

Matmul is harder — it's not embarrassingly parallel. Each output cell needs a reduction over the inner dimension. The naive port is "one thread per output cell, each thread does the inner-loop reduction." It works. It also leaves 90% of the GPU's potential on the table. Lessons 4–5 show why and how to do better.

## Mental model & pitfalls

Single sentence: **A CPU optimizes the speed of one thread; a GPU optimizes the throughput of many threads. If your problem has parallel work for tens of thousands of threads, the GPU is faster by 100×. If not, it's slower.**

Pitfalls when porting CPU code to GPU:

- **Thinking of "a thread" as a CPU thread.** GPU threads are a unit of parallelism, not a unit of independence. They run in lockstep within a warp; divergence is expensive; per-thread state is tiny.
- **Reading memory the way a CPU loop would.** A loop `for i in 0..N { read a[i] }` on CPU is great. The same access pattern across 32 GPU threads in a warp depends on which thread reads what — if `thread_id = i` then it's coalesced; otherwise it's not.
- **Branchy code.** If 16 threads in a warp take one path and 16 take another, the hardware runs both paths sequentially. For mathematically uniform code, this isn't a real concern; for data-dependent branching, it bites.
- **Launching too few threads.** A kernel launch of 128 threads on a GPU with 16,000 cores is leaving 99% of the chip idle. The minimum launch for "the GPU isn't wasting itself" is roughly the SM count × max threads per SM ≈ tens of thousands.
- **Allocating GPU memory per launch.** `cudaMalloc` is slow (~50 µs). Reuse buffers. Same lesson as the CPU allocator from Module 1.
- **Ignoring kernel launch overhead.** A kernel launch costs ~5–10 µs on NVIDIA, often more elsewhere. If your kernel takes less time than that to execute, you're being eaten by overhead. Either fuse small kernels into bigger ones (Lesson 8) or use CUDA Graphs.

## Hands-on (at home)

This lesson's hands-on is to set up the GPU toolchain you'll use for the rest of the module. No code yet; just the toolchain check.

### If you have NVIDIA GPU access (local or cloud)

1. **Install CUDA Toolkit.** On Linux: `nvidia-cuda-toolkit` or the official runfile from developer.nvidia.com. On WSL2 + Windows, NVIDIA's CUDA-on-WSL instructions.

2. **Verify with `nvidia-smi`** — should show your GPU and driver version. If this doesn't work, the toolchain isn't fully set up.

3. **Verify with `nvcc --version`** — should show the CUDA compiler version. Generally 12.x+ is current.

4. **Run a "hello world" CUDA program** — the canonical:

```cuda
#include <stdio.h>
__global__ void hello() {
    printf("Hello from thread %d, block %d\n", threadIdx.x, blockIdx.x);
}
int main() {
    hello<<<2, 4>>>();
    cudaDeviceSynchronize();
    return 0;
}
```

```bash
nvcc -o hello hello.cu && ./hello
```

You should see 8 lines (2 blocks × 4 threads). If the order looks scrambled, that's expected — threads don't have a defined print order.

### If you have an Apple Silicon Mac

1. **Install Xcode** if not already (large download; can take a while).

2. **Verify Metal is available** — every Apple Silicon Mac has it. Confirm with `xcrun --sdk macosx --show-sdk-path` (returns a path).

3. **Optional: install MLX** (the Apple Silicon ML framework we'll use later):

```bash
pip install mlx
```

Test:

```python
import mlx.core as mx
x = mx.ones((1024, 1024))
y = mx.matmul(x, x)
mx.eval(y)
print(y[0, 0])  # should print 1024.0
```

If this runs, you have a working MLX install — and a working GPU pathway, since MLX uses the GPU by default.

### If you have neither

Cloud GPU options for an evening of CUDA practice:

- **Lambda Cloud** — billed per second; H100 instances are ~$2/hr.
- **Modal** — serverless GPU; pay only for kernel execution time; great for one-off experiments.
- **RunPod** — A100/H100 instances; spot pricing under $1/hr for older cards.
- **Vast.ai** — auction-priced consumer cards; cheapest but quality varies.

For ARM/Metal practice without a Mac: harder. macOS-in-VM works but the GPU passthrough is dodgy. The honest answer is "this curriculum's on-device modules really benefit from owning Apple Silicon hardware." If you're committed to learning the on-device stack, an M-series Mac mini is the minimum viable kit.

### Confirm one number

Run a simple bandwidth test:

```cuda
// bandwidth.cu
__global__ void copy_kernel(float* dst, const float* src, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = src[i];
}

int main() {
    int N = 1 << 26;  // 64M floats = 256MB
    float *src, *dst;
    cudaMalloc(&src, N * sizeof(float));
    cudaMalloc(&dst, N * sizeof(float));
    cudaMemset(src, 1, N * sizeof(float));
    cudaDeviceSynchronize();

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 50; i++) {
        copy_kernel<<<(N + 255) / 256, 256>>>(dst, src, N);
    }
    cudaDeviceSynchronize();
    auto dt = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t0).count() / 50;

    double gbps = N * sizeof(float) * 2 / dt / 1e9;  // *2: read + write
    printf("%.1f GB/s\n", gbps);
    return 0;
}
```

Expected:
- H100: ~3000 GB/s.
- A100: ~1800 GB/s.
- 3090/4090: ~900 GB/s.
- M-series GPU (M2 Pro): ~250 GB/s (via Metal — different code, similar idea).

This number is your **memory bandwidth ceiling**. Every kernel in the rest of this module is bounded by it. Knowing what it is means you'll instantly know when your code is hitting the wall vs underutilizing.

## Further reading

- *Programming Massively Parallel Processors* (Kirk & Hwu) — the classic CUDA textbook. The first 3 chapters are exactly this mental model.
- "An Even Easier Introduction to CUDA" — Mark Harris (NVIDIA blog). Single best CUDA quick-start.
- "GPU Architecture and Programming Models" — many lectures online; CMU 15-418 (Carnegie Mellon's parallel computing course) is excellent.
- Apple's Metal Programming Guide — for the parallel concepts on the Apple side; lighter than CUDA's docs but accurate.

Next lesson: NVIDIA GPU architecture in detail. SMs, warps, tensor cores, the memory hierarchy in numbers.
