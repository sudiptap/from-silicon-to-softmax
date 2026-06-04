---
title: "Lesson 2 — NVIDIA GPU Architecture"
date: "2026-06-03"
module: "gpu-computing"
order: 2
tags: ["nvidia", "ampere", "hopper", "blackwell", "sm", "tensor-core", "hbm"]
author: "Sudipta Pathak"
prerequisites: ["01-gpu-mental-model"]
---

# Lesson 2 — NVIDIA GPU Architecture

## Why this matters

CUDA is the language; the NVIDIA microarchitecture is what your code is actually running on. The same kernel can perform very differently on Ampere (A100) and Hopper (H100) because the tensor core formats changed, the SM register file size doubled, and the L2 cache more than doubled. Knowing the architecture means you can predict the right block size, the right tile shape, and the right precision for the silicon you have.

This lesson is the architecture tour. You don't need to memorize it. You do need to know which numbers vary across generations and which are stable, because every published benchmark assumes you know.

## Concept

### The Streaming Multiprocessor (SM)

The SM is the unit you should think of as "one core" for budgeting purposes. Each SM has:

- **A set of CUDA cores** (FP32 ALUs) that run the basic per-thread arithmetic. On Hopper, an SM has 128 FP32 cores.
- **Tensor Cores** — specialized matrix-multiply-accumulate units. On Hopper, 4 of them per SM, each doing 256 FP16 × FP16 + FP32 MACs per cycle. This is the "headline TFLOPs" hardware.
- **A large register file** — 64K registers (256 KB) on Hopper, split among the threads in flight on the SM.
- **A unified L1 / shared memory** — 256 KB total on Hopper, configurable how much is L1 vs shared.
- **Schedulers** — typically 4 warp schedulers per SM, each capable of issuing 1 warp per cycle.

Total chip: number of SMs × per-SM compute. H100: 132 SMs. A100: 108 SMs. RTX 4090: 128 SMs (consumer variant with weaker tensor cores). The per-SM picture is similar; the chip-wide picture scales with SM count.

### Tensor Cores: the big-FLOPs hardware

CUDA cores are general — any FP32 operation. **Tensor Cores** are specialized for matrix multiply-accumulate (MMA): they take small fixed-shape matrices, multiply them, and accumulate into a third matrix, all in one instruction. The shapes have grown across generations:

- Volta (V100): 4×4×4 FP16 MMA.
- Ampere (A100): 16×8×16 FP16, plus 16×8×8 TF32, plus FP64 MMA, plus sparse-MMA support.
- Hopper (H100): 64×8×16 FP16/BF16, FP8 with E4M3 / E5M2 formats, much wider asynchronous MMA via the new "warp-group MMA" (`wgmma`).
- Blackwell (B100): even wider; FP4 support; further async improvements.

The headline FLOPs numbers ("989 TFLOPs FP8 on H100") refer to tensor cores at peak. **CUDA-core FP32 alone gives you maybe 5–10% of this number.** The difference is what you're trading away by writing a kernel that uses CUDA cores instead of tensor cores. For dense matmul, tensor cores are the right tool — but they constrain your operand shapes and precisions. We'll come back to this in Lessons 4–5.

### The memory hierarchy in numbers (H100 example)

```
Registers (per thread):   max 255 × 32-bit            ~1 cycle
                                                       
Shared memory (per SM):   228 KB usable               ~20 cycles
L1 data cache (per SM):   shared with shared (256K)   ~20 cycles
                                                       
L2 cache (chip-wide):     50 MB                       ~150 cycles
HBM3 (global):            80 GB, 3 TB/s              ~400 cycles
                                                       
Host RAM (over PCIe5):    ~ system RAM, ~64 GB/s     ~1000+ cycles
```

A100's numbers are similar but smaller: 40 MB L2, 1.5–2 TB/s HBM.

What stands out for fast kernels:

- **L1/shared has many TB/s aggregate bandwidth** across all SMs. Inner loops that fit in shared memory go at roughly 10× the speed of inner loops bound by HBM.
- **HBM is fast at 3 TB/s but slow at 400-cycle latency.** Long-latency, high-throughput memory: amortize hits by reusing data many times once loaded.
- **PCIe transfers are the worst.** Get data to the GPU before you need it; never round-trip per kernel.

### Warps and warp schedulers

The SM has 4 warp schedulers. Each cycle, each scheduler picks one *ready* warp (= 32 threads) from its assigned warps and issues one instruction. Each instruction completes 32 lanes of work.

What makes a warp "not ready":
- Waiting on a memory load that hasn't returned.
- Waiting on a tensor-core MMA in flight.
- Stalled by a `__syncthreads()` for other warps.

If your kernel runs few warps per SM (low occupancy), the scheduler runs out of ready warps to issue, the SM goes idle, and your throughput drops. This is why warps per SM > 8 is generally a healthy target — but as discussed in Lesson 1, the relationship isn't monotonic.

### What changed across generations

A short cheat-sheet of the architectural shifts you'll see referenced in papers and code:

- **Volta → Ampere**: bigger L2, TF32 format (faster than FP32 with similar accuracy for training), sparse tensor cores.
- **Ampere → Hopper**: massive jump in tensor core size, FP8 support, async warp-group MMA (`wgmma`), TMA (tensor memory accelerator — async bulk copies between HBM and shared memory), thread block clusters (multiple SMs can share data).
- **Hopper → Blackwell**: FP4 support, more aggressive sparsity, expanded async operations, NVLink5.

For LLM inference circa now: H100 and B100 are the common production targets; consumer cards (4090) are competitive for small-model inference but lack the FP8 / FP4 tensor core throughput.

### The async revolution (Hopper+)

The single biggest programming-model change in the last few generations is **async**. Hopper introduced TMA (`cp.async.bulk`) for asynchronous tile copies between HBM and shared memory, and async tensor-core MMA. Together they allow the kernel to *overlap* memory transfers with compute — explicitly, at the warp level — rather than relying on the scheduler to hide latency by switching warps.

This is the foundation of the **producer/consumer pattern**: one set of warps fetches tiles into shared memory; another set runs MMA on tiles already there. They communicate via a small synchronization mechanism. The kernel structure looks more like a pipeline than a loop. Most modern fast kernels (CUTLASS 3.x's main path, FlashAttention 3, Triton on Hopper) use this pattern.

We'll see the structure in Lesson 5 and the matmul lessons. The mental model: explicit async is the GPU equivalent of how SIMD + multi-issue lets CPUs overlap work — but now it's the kernel author who has to set up the overlap.

## Code walkthrough

A concrete piece of architectural data: how to ask the GPU what it is, from inside a CUDA program.

```cuda
#include <cstdio>
#include <cuda_runtime.h>

int main() {
    int dev;
    cudaGetDevice(&dev);
    cudaDeviceProp p;
    cudaGetDeviceProperties(&p, dev);

    printf("Device: %s\n", p.name);
    printf("Compute capability: %d.%d\n", p.major, p.minor);
    printf("SMs: %d\n", p.multiProcessorCount);
    printf("Max threads per SM: %d\n", p.maxThreadsPerMultiProcessor);
    printf("Max threads per block: %d\n", p.maxThreadsPerBlock);
    printf("Shared memory per SM: %zu KB\n", p.sharedMemPerMultiprocessor / 1024);
    printf("Shared memory per block: %zu KB\n", p.sharedMemPerBlock / 1024);
    printf("Registers per SM: %d\n", p.regsPerMultiprocessor);
    printf("L2 cache: %zu MB\n", p.l2CacheSize / 1024 / 1024);
    printf("Memory clock: %d MHz\n", p.memoryClockRate / 1000);
    printf("Memory bus width: %d bits\n", p.memoryBusWidth);
    printf("Memory bandwidth: %.1f GB/s\n",
        2.0 * p.memoryClockRate * (p.memoryBusWidth / 8) / 1e6);
    printf("HBM size: %.1f GB\n", p.totalGlobalMem / 1e9);
    printf("Warp size: %d\n", p.warpSize);

    return 0;
}
```

Compile and run on whatever GPU you have. The output is the *exact* numbers your kernels are budgeting against. Examples:

**H100 PCIe** would say: 80 GB HBM, 50 MB L2, 132 SMs, 2048 threads per SM, 228 KB shared per SM, ~3000 GB/s memory bandwidth.

**RTX 4090** would say: 24 GB GDDR6X, 72 MB L2, 128 SMs, 1536 threads per SM, 100 KB shared, ~1000 GB/s memory bandwidth.

**Old A100 40GB** would say: 40 GB HBM, 40 MB L2, 108 SMs, 2048 threads per SM, 164 KB shared, ~1500 GB/s memory bandwidth.

Memorize the bandwidth number for the GPU you're using; it's the denominator of "how much memory traffic can my kernel afford."

## Mental model & pitfalls

Single sentence: **The SM is the budget unit; tensor cores are the compute hardware that gets you headline FLOPs; the memory hierarchy is what every fast kernel is structured around; async transfers (Hopper+) are how modern fast kernels overlap memory and compute.**

Pitfalls:

- **Treating "TFLOPs" as a single number.** TF32 ≠ FP16 ≠ BF16 ≠ FP8. The headline number on a marketing slide is usually the smallest precision the tensor cores support, at maximum sparsity. Real workloads run at a fraction. Know which number you're comparing against.
- **Conflating CUDA-core throughput with tensor-core throughput.** Writing a kernel in plain CUDA without tensor-core intrinsics gives you the CUDA-core number (5–10× lower). Many naive kernels do exactly this and look slow.
- **Ignoring HBM/PCIe distinction.** PCIe-attached cards (most H100 configurations, all consumer cards) have a slow link to the CPU. SXM-attached (NVLink-attached) cards in DGX-style nodes have faster GPU-GPU links. For single-GPU work, PCIe vs SXM matters less; for multi-GPU it's everything.
- **Targeting too few SMs.** Launching with `grid_dim < SM count` means some SMs are idle from launch onward. The minimum sensible launch is `grid_dim = SM count × 4` or so, to give the scheduler room.
- **Overusing shared memory.** Every kilobyte of shared memory per block reduces how many blocks can fit on an SM. Below a threshold, occupancy collapses. Inspect with `--ptxas-options=-v` during compile.

## Hands-on (at home)

Light week — most of this lesson is reference material.

1. **Run the `cudaGetDeviceProperties` program** above on whatever GPU you have access to. Save the output to a file (`gpu-info.txt`). You'll refer back to these numbers throughout the rest of the module.

2. **Look up your GPU's official specs** on NVIDIA's website. Compare to what `cudaGetDeviceProperties` reports. Numbers should match closely (consumer cards may report differently for game vs compute modes).

3. **Compute the "FLOP/byte ratio"** for your GPU: `peak_FP16_TFLOPs / peak_HBM_GB_per_sec`. For H100 SXM: 989 TFLOPs FP8 / 3000 GB/s ≈ 330 FLOPs/byte. For A100: 312 TFLOPs FP16 / 2000 GB/s ≈ 156 FLOPs/byte. This is the *arithmetic intensity* a kernel needs to hit peak. Below it: memory-bound. Above it: compute-bound. Used in the next several lessons.

4. **(If you don't have local NVIDIA access)** Read the H100 architecture whitepaper from NVIDIA. It's the single best resource for understanding modern NVIDIA GPUs end to end. The Hopper whitepaper covers TMA, wgmma, and the async model in detail.

## Further reading

- "NVIDIA Hopper Architecture In-Depth" — NVIDIA developer blog (and the whitepaper). Authoritative.
- "Dissecting the NVIDIA Volta GPU Architecture" — Citadel paper that started the genre of microarchitectural reverse-engineering for NVIDIA GPUs. Subsequent generations have similar papers.
- Chips and Cheese GPU deep-dives — independent measurements; useful sanity check against vendor claims.
- CUTLASS documentation — NVIDIA's library that exposes these architectural features. Reading the comments in their main MMA paths is an education in itself.

Next lesson: Apple Silicon GPU architecture. Different design, different constraints, same general mental model. Then we get to writing kernels.
