---
title: "Lesson 5 — CUDA Memory Hierarchy"
date: "2026-06-03"
module: "gpu-computing"
order: 5
tags: ["cuda", "shared-memory", "registers", "coalescing", "bank-conflicts", "tiled-matmul"]
author: "Sudipta Pathak"
prerequisites: ["04-cuda-basics"]
---

# Lesson 5 — CUDA Memory Hierarchy

## Why this matters

Lesson 4's naive matmul hit a few thousand GFLOPs on a 4090 — already 100× the best CPU number from Module 1. It is also running at roughly 5% of what the same hardware can do. The gap is exactly the same gap we closed on CPU in Lesson 5 of Module 1: data reuse. The GPU's HBM has fast bandwidth (terabytes per second) but slow latency (hundreds of cycles per access) and finite throughput (every byte goes over the same bus). If each output cell of the matmul triggers `K` global memory reads of `A` and `K` reads of `B`, we are saturating HBM long before we saturate the FMA pipelines.

The fix is shared memory — a small, fast on-chip SRAM that the *programmer* explicitly controls. We load tiles of `A` and `B` into shared memory, then have many threads reuse that data many times before going back to global memory. Same insight as cache blocking, more explicit because we're hand-managing the cache.

This is the most important single lesson in Module 2. The shared-memory tiled matmul is the canonical CUDA kernel — every fast kernel you'll read uses this pattern as a substrate. After this lesson the matmul jumps from ~5% to ~30%+ of peak, and you have the framework to read CUTLASS, FlashAttention, and most production CUDA code.

## Concept

### The hierarchy, applied to threads

Each level of the GPU memory hierarchy is accessible to different scopes:

| Memory | Scope | Size | Latency | Programmer-managed? |
| ----- | ----- | ----: | ------: | ----- |
| Registers | per thread | 255 × 32-bit max | 1 cycle | Implicitly (via local variables) |
| Shared memory | per block | 100+ KB | ~20 cycles | Yes — declared with `__shared__` |
| L1 cache | per SM | shared with shared | ~20 cycles | No — hardware-managed |
| L2 cache | chip-wide | 40–60 MB | ~150 cycles | No — but hints exist (`cudaStreamSetAttribute`) |
| Global memory (HBM) | chip-wide | 24–80 GB | ~400 cycles | No |

For our matmul, the win is in the *gap* between shared memory (~20 cycles) and global memory (~400 cycles). Each global load amortized over many shared-memory accesses brings the *effective* memory cost down by 10–20×.

### The tiled matmul recipe

The high-level structure: pick a tile size, say 32×32. Each thread block will compute one 32×32 tile of `C`. Inside the block, we walk along the `K` dimension in chunks of 32 (the "K-tile"). For each chunk:

1. Cooperatively load a 32×32 sub-block of `A` into shared memory.
2. Cooperatively load a 32×32 sub-block of `B` into shared memory.
3. Synchronize all threads in the block (`__syncthreads()`).
4. Each thread computes its partial sum, reading from shared memory.
5. Synchronize again before loading the next K-tile.

Why this works: each tile of `A` and `B` is loaded **once** from global memory but read **32 times** inside the block — once per row of `C` we compute. We've reduced the global memory traffic by a factor of 32.

### Cooperative loads and how threads share work

A block of 32×32 threads has 1024 threads. A 32×32 tile of `A` has 1024 floats. The natural mapping: each thread loads one float. Concretely:

```cuda
__shared__ float sA[32][32];
__shared__ float sB[32][32];

int t_row = threadIdx.y;
int t_col = threadIdx.x;
int block_row = blockIdx.y * 32;
int block_col = blockIdx.x * 32;

// Each thread loads one element of A and one of B per K-tile.
for (int k_tile = 0; k_tile < K; k_tile += 32) {
    sA[t_row][t_col] = A[(block_row + t_row) * K + (k_tile + t_col)];
    sB[t_row][t_col] = B[(k_tile + t_row) * N + (block_col + t_col)];
    __syncthreads();
    
    // ... compute over sA, sB ...
    
    __syncthreads();
}
```

The two `__syncthreads()` are necessary: the first ensures all threads have loaded their float before any thread tries to use the shared memory; the second ensures no thread starts overwriting shared memory before all threads are done using the current tile.

### Coalesced access — making the loads cheap

When 32 threads in a warp load from global memory, the hardware tries to **coalesce** the 32 accesses into one or two 128-byte memory transactions. The condition: the 32 addresses must be contiguous (and aligned, on older hardware).

In the load above, `threadIdx.x` is the fastest-varying index. So threads in a warp (which differ by `threadIdx.x`) read addresses that differ by 1 float — perfectly contiguous. The hardware coalesces. One transaction per warp per load. 

If we had instead written `sA[t_col][t_row] = A[(block_row + t_col) * K + (k_tile + t_row)]`, threads in a warp would differ by `K` floats — strided by thousands of bytes. Each access would be a separate transaction. **The same algorithm written wrong is 32× slower.** This is the GPU equivalent of "stride-1 vs stride-1024" from Module 1 Lesson 2.

The rule: arrange your indexing so threads in a warp read consecutive memory addresses.

### Bank conflicts — the shared-memory pitfall

Shared memory is divided into **banks** — 32 of them on current architectures. Each bank can service one access per cycle. If 32 threads in a warp access 32 different banks, all 32 reads complete in parallel. If two or more threads access the same bank with different addresses, the accesses serialize.

Banks are interleaved: word 0 is in bank 0, word 1 in bank 1, …, word 31 in bank 31, word 32 in bank 0 again. So accessing `sA[i][threadIdx.x]` for varying `i` (column-walk) is conflict-free — each thread hits a different column = different bank. Accessing `sA[threadIdx.x][i]` for the same `i` across threads (row-walk) is also conflict-free — each thread reads `i*32 + threadIdx.x*32*32`, which is `threadIdx.x` words apart mod 32 = different banks.

But accessing `sA[threadIdx.x][threadIdx.x]` (broadcast-pattern with stride 33) creates a 32-way bank conflict. So does any pattern where 32 threads hit `addr % 32 == k` for the same `k`.

A common fix: pad the shared array to a width of 33 (or any size coprime to 32) so successive rows fall on different banks:

```cuda
__shared__ float sA[32][33];
```

This wastes a few hundred bytes but eliminates one of the most common bank-conflict patterns.

### Registers — the fastest memory

Local variables in a kernel live in **registers**, the fastest memory on the GPU. Reads are essentially free (1 cycle, in the instruction itself). Writes are free.

Each thread has a budget — up to 255 registers on current hardware. If you exceed that budget, the compiler **spills** to local memory, which is a slow per-thread global-memory area. Spills are disastrous for performance.

Implication for matmul: in a tiled kernel, you want each thread to compute *multiple* output cells, accumulating in register-resident variables. This amortizes shared-memory access (which is ~20 cycles) over many FMAs. The pattern is to have each thread own, say, an 8×8 sub-tile of the output, with 64 register accumulators.

We'll write a tile-per-thread version below. The full register-tiling story is what gets cuBLAS to peak; we'll go halfway.

## Code walkthrough

The shared-memory tiled matmul, one float per thread (the simpler version).

```cuda
#define TILE 32

__global__ void matmul_tiled(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int K, int N
) {
    __shared__ float sA[TILE][TILE];
    __shared__ float sB[TILE][TILE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE + ty;
    int col = blockIdx.x * TILE + tx;

    float acc = 0.0f;

    // Walk along K in tile-sized chunks.
    for (int k_tile = 0; k_tile < K; k_tile += TILE) {
        // Cooperative load. Each thread loads one element.
        // Guard for problem sizes not multiples of TILE.
        if (row < M && k_tile + tx < K)
            sA[ty][tx] = A[row * K + (k_tile + tx)];
        else
            sA[ty][tx] = 0.0f;

        if (k_tile + ty < K && col < N)
            sB[ty][tx] = B[(k_tile + ty) * N + col];
        else
            sB[ty][tx] = 0.0f;

        __syncthreads();

        // Compute partial sum over the K-tile.
        #pragma unroll
        for (int k = 0; k < TILE; ++k) {
            acc += sA[ty][k] * sB[k][tx];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = acc;
    }
}
```

A walkthrough of what this is doing.

**Block size = 32×32.** That's 1024 threads per block — the maximum on most CUDA hardware. One block fully owns one 32×32 tile of `C`.

**Two `__shared__` arrays** for the A and B tiles. Together they're 32×32×4 + 32×32×4 = 8 KB per block — well within the per-block shared memory limit. We could have multiple blocks per SM; the SM has 100 KB+ of shared memory, so 8 KB per block allows 12+ blocks per SM, which is great for occupancy.

**The cooperative load.** Each of the 1024 threads loads one float from `A` and one from `B`. Threads in the same warp (differing only in `threadIdx.x` for warps assigned to a row of the block) access contiguous memory in both loads → coalesced.

**The inner loop reads from shared memory.** `sA[ty][k]` is read by all threads in row `ty` for varying `k` — same address, broadcast-pattern, single bank access (modern GPUs have a free broadcast). `sB[k][tx]` is read by all threads in column `tx` for varying `k` — `k*32 + tx` words, no bank conflict because `tx` is the warp's fast axis. The inner loop is clean.

**`#pragma unroll`** tells the compiler to unroll the inner loop. With `TILE = 32`, the inner loop becomes 32 explicit FMAs in the assembly, which lets the compiler schedule them aggressively. Without the unroll, the compiler often does this anyway when `TILE` is a constant, but the hint is cheap insurance.

**`__syncthreads()` placement.** First sync after loading, before reading. Second sync after the inner loop, before the next iteration overwrites the shared tiles. Both are mandatory.

### Performance numbers

On a 4090 with a 1024-cube problem:

- Naive (Lesson 4): ~3 TFLOPs.
- Tiled (this lesson): ~15–20 TFLOPs.
- cuBLAS: ~80 TFLOPs.

We went from 5% of peak to 20–25% of peak. Big jump, still 4× away from cuBLAS. The remaining gap is:

- **Larger tiles + register tiling.** Each thread should compute an 8×8 micro-tile of C, with 64 register accumulators. This amortizes shared-memory access by 8×.
- **Tensor cores.** Plain CUDA cores cap at ~80 TFLOPs FP32. Tensor cores hit 300+ TFLOPs FP16 / 600+ FP8. Switching to tensor-core MMA is required for the headline numbers.
- **Double buffering.** Load the next K-tile while computing the current one. Overlaps memory with compute.
- **Async copies (Hopper+).** TMA + `cp.async.bulk` for explicit overlap.

We won't write all of this by hand in this module — Triton (Lesson 6) lets us get most of the way there with much less code, and that's the pragmatic choice for most engineers most of the time.

### Register tiling — the next step (sketch)

To illustrate the register-tile pattern, here's what the inner part of an 8×8 per-thread version would look like:

```cuda
// 8x8 outputs per thread; block is 16x16 threads → 128x128 output tile per block.
float acc[8][8] = {0};

for (int k_tile = 0; k_tile < K; k_tile += TILE_K) {
    // ... cooperative load of larger sA and sB tiles ...
    __syncthreads();

    for (int k = 0; k < TILE_K; ++k) {
        // Load 8 elements of A's column k into registers.
        float a_reg[8];
        for (int i = 0; i < 8; ++i)
            a_reg[i] = sA[ty * 8 + i][k];

        // Load 8 elements of B's row k.
        float b_reg[8];
        for (int j = 0; j < 8; ++j)
            b_reg[j] = sB[k][tx * 8 + j];

        // 64 FMAs.
        for (int i = 0; i < 8; ++i)
            for (int j = 0; j < 8; ++j)
                acc[i][j] += a_reg[i] * b_reg[j];
    }
    __syncthreads();
}
// Store 8x8 outputs back to C.
```

Each thread holds 64 floats of accumulator + 16 floats of A/B working set in registers — 80 registers per thread for accumulators alone, plus some bookkeeping. Inside the 255-register budget on Ampere/Hopper, and reaches ~50–60% of peak. This is the kernel shape every fast CUDA matmul has.

## Mental model & pitfalls

Single sentence: **Tile the problem so each tile fits in shared memory; have each thread compute multiple output cells from register-resident accumulators; you've now done the GPU equivalent of cache blocking + SIMD micro-kernels.**

Pitfalls:

- **Non-coalesced loads.** Index expressions where `threadIdx.x` *isn't* the fastest-varying global memory offset. Cost: 32× the memory traffic. Sniff by reading the indexing carefully and asking "as `threadIdx.x` increments by 1, does the global address increment by 1 float?"
- **Bank conflicts.** Especially patterns where threads read along the slow axis of a `__shared__` 2D array. Pad rows to a coprime stride (e.g., 33 instead of 32) when conflicts are unavoidable.
- **Forgetting `__syncthreads()`.** Race condition: some threads read stale shared memory while others overwrite it. Symptoms: wrong answers, sometimes; rarely a hang. Always: hard to debug.
- **Mid-warp `__syncthreads()` in divergent code.** If only some threads in a warp execute the sync, behavior is undefined. Keep sync calls outside any data-dependent conditional.
- **Spilling registers.** If your kernel uses too many local variables, the compiler stashes some in slow "local memory." Watch for it with `--ptxas-options=-v` at compile time: it tells you registers per thread and local memory usage.
- **Overusing shared memory.** Each KB of shared memory per block reduces the number of blocks per SM. Past a threshold, occupancy drops and the SM scheduler runs dry. Keep an eye on the budget.
- **Block size that's not a multiple of 32.** Threads beyond a warp boundary still consume scheduler slots but contribute partial warps that waste resources. Always use block sizes that are multiples of 32.

## Hands-on (at home)

1. **Implement `matmul_tiled`** as above. Compile with `nvcc -O3 -arch=sm_XX --ptxas-options=-v -o matmul matmul.cu`. The `-v` flag prints register / shared memory usage. You should see something like "Used 38 registers, 8192 bytes smem."

2. **Bench against the naive version.** Expected speedup: 5–8× on most modern GPUs.

3. **Sweep tile size.** Try TILE = 16, 32, 64. With TILE = 64 you'd need 4×4×4×1024 = 64KB shared memory per block, which exceeds the per-block limit on most cards — you'll need a block size that doesn't have one thread per element. Often TILE = 32 with 32×32 blocks is the sweet spot for the simple version.

4. **Try register tiling.** Implement the 8×8-per-thread version sketched above. Expect ~2× over the basic tiled version. Look for register spills (`Used 80+ registers` in the ptxas output is OK; `0 bytes stack frame` is what you want — anything else means spills).

5. **Read CUTLASS' main matmul path.** Just glance through. The structure you've just built is what CUTLASS' template machinery generates for many configurations.

6. **Optional bank-conflict experiment.** Change `sA[TILE][TILE]` to `sA[TILE][TILE]` and add a deliberate stride-32 access pattern. Use Nsight Compute to see the bank conflict counters jump (`shared_load_bank_conflict`). Then add the `+1` padding and watch them go to zero.

## Further reading

- *CUDA C Programming Guide*, chapters on shared memory and memory coalescing — authoritative.
- CUTLASS tutorial paths — readable C++ that mirrors what we just wrote, generalized.
- Mark Harris's "Optimizing Parallel Reduction in CUDA" — applies to reductions, but the techniques (coalescing, bank conflicts, register usage) transfer.
- Simon Boehm's blog post "How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance: a Worklog" — the single best walkthrough of going from naive to near-cuBLAS step by step. Highly recommended.
- NVIDIA's CUTLASS GEMM tutorial — explicit, deep, modern.

Next lesson: Triton. Same matmul, ~30 lines of Python, gets to cuBLAS-class performance with substantially less code. The future of "fast GPU code by hand" runs through Triton.
