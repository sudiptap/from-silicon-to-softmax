---
title: "Lesson 9 — The Reduction Problem"
date: "2026-06-03"
module: "gpu-computing"
order: 9
tags: ["reduction", "warp-shuffle", "softmax", "layernorm", "cuda"]
author: "Sudipta Pathak"
prerequisites: ["08-kernel-fusion"]
---

# Lesson 9 — The Reduction Problem

## Why this matters

A reduction is the operation that takes a collection of numbers and combines them into one: sum, max, mean, dot product, the denominator of softmax, the running stats of LayerNorm. Reductions look trivial. They aren't, on GPUs. The naive approach — one thread sums everything serially — uses 1/10000th of the hardware. The "correct" approach — a tree reduction across threads, then blocks — is harder than you'd guess on first try, with a surprising number of pitfalls (warp divergence, bank conflicts, race conditions, the last-warp problem).

This lesson covers the reduction patterns every GPU programmer ends up writing. Softmax depends on them. LayerNorm depends on them. FlashAttention depends on them. A clear mental model for reductions makes the rest of fast inference legible.

## Concept

### Why naive doesn't work

Naive "one thread sums an array":

```cuda
__global__ void sum_kernel_naive(float* in, float* out, int n) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        float s = 0;
        for (int i = 0; i < n; ++i) s += in[i];
        *out = s;
    }
}
```

Launches 1 thread. Uses 1/16000th of the GPU. Equivalent to a slow CPU. The whole point of a GPU is parallelism; this kernel rejects it.

### The tree reduction

The standard pattern: every thread does its share of the work, then threads in a block combine their partial sums in a tree, then blocks combine their partial sums in a final pass.

```
Level 0: thread i loads in[i] into its register
Level 1: threads 0..N/2 add their value to thread (i + N/2)'s value
Level 2: threads 0..N/4 add ...
...
Level log(N): thread 0 has the sum
```

Each level halves the number of active threads but doubles each survivor's accumulation. Total time: O(log N) steps, where N is block size.

A naive implementation:

```cuda
__shared__ float s[BLOCK];
s[threadIdx.x] = in[blockIdx.x * BLOCK + threadIdx.x];
__syncthreads();

for (int stride = BLOCK / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride) {
        s[threadIdx.x] += s[threadIdx.x + stride];
    }
    __syncthreads();
}

if (threadIdx.x == 0) out[blockIdx.x] = s[0];
```

This works. It's also slow. The issue is the `if (threadIdx.x < stride)` — once `stride` drops below the warp size (32), half the warp is idle, then 3/4, then 7/8. Warp divergence costs us as we get to the leaves of the tree.

### Warp shuffles — the modern reduction primitive

Modern GPUs (Kepler and beyond on NVIDIA, all Apple GPUs) have **warp shuffle** instructions that let threads in a warp exchange register values directly, without going through shared memory.

```cuda
val = __shfl_down_sync(0xFFFFFFFF, val, offset);
```

This says: each thread sends its `val` to the thread `offset` positions below it. Costs ~5 cycles, no shared memory, no syncs, no divergence.

The warp-level sum looks like:

```cuda
__device__ float warp_reduce_sum(float val) {
    val += __shfl_down_sync(0xFFFFFFFF, val, 16);
    val += __shfl_down_sync(0xFFFFFFFF, val, 8);
    val += __shfl_down_sync(0xFFFFFFFF, val, 4);
    val += __shfl_down_sync(0xFFFFFFFF, val, 2);
    val += __shfl_down_sync(0xFFFFFFFF, val, 1);
    return val;  // lane 0 has the sum
}
```

Five shuffle-and-add steps, ~25 cycles. The whole warp's data is reduced to lane 0. No shared memory, no `__syncthreads()`, no warp divergence (every thread executes every shuffle; only the result of lane 0 is meaningful at the end).

### The block reduction in two phases

```cuda
__device__ float block_reduce_sum(float val) {
    static __shared__ float s[32];  // up to 32 warps in a 1024-thread block
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;

    val = warp_reduce_sum(val);     // Phase 1: each warp reduces its 32 values

    if (lane == 0) s[wid] = val;    // Lane 0 of each warp writes to shared
    __syncthreads();

    if (wid == 0) {
        val = (threadIdx.x < blockDim.x / 32) ? s[lane] : 0.0f;
        val = warp_reduce_sum(val); // Phase 2: warp 0 reduces the 32 partials
    }
    return val;  // thread 0 of block has the block sum
}
```

Two-phase reduction:
1. Each warp reduces its 32 values to one (in lane 0).
2. The first warp picks up all the per-warp sums and reduces them.

Result: thread 0 of the block has the block sum, in ~50 cycles total.

### Cross-block: the last problem

For a multi-block reduction, you have two options.

**Option 1: two-pass.** Launch a kernel that produces one partial sum per block. Then launch a second kernel that sums those partials. Total: 2 kernel launches. Clean, simple, and the right answer when N is large.

**Option 2: atomic add to a single global location.** Each block, after computing its partial, does `atomicAdd(out, block_sum)`. Total: 1 kernel launch but all blocks contend on one atomic. Fast when block count is small (≤100s); slow when block count is large.

In practice: use two-pass for large reductions; use atomic for small block counts or when you need the result inside the same kernel for some downstream computation.

### Numerical stability — Kahan and friends

Summing many FP16 values can lose precision rapidly. Two strategies:

- **Accumulate in FP32 even if inputs are FP16.** Always. The hardware supports FP32 accumulators in tensor cores precisely for this reason.
- **Pairwise / tree sums** (what we're doing) already have better numerical stability than serial sums; the maximum error grows as O(log N) instead of O(N).
- **Kahan compensated summation** is overkill for most ML reductions and rarely worth the extra cost.

For LayerNorm specifically, computing the variance has a classic stability pitfall: `Var(X) = E[X²] - E[X]²` cancels badly when X is roughly mean-zero with small variance. The two-pass formula (compute mean first, then variance from `(X - mean)²`) is numerically safe and cheap.

### Softmax — the canonical reduction-heavy kernel

Softmax is two reductions and a normalization:

```
m = max(x)              # reduction 1 (numerical stability)
s = sum(exp(x - m))     # reduction 2
y = exp(x - m) / s      # element-wise
```

A fused softmax kernel does this all in one pass over the input — load the row, compute max via warp/block reduction, compute exp + sum via second reduction, write the normalized output. Two reductions, one global memory write of output, no intermediate writes.

**The "online softmax" trick** combines the max and sum into a single streaming pass:

```
m = -inf
d = 0
for x in stream:
    new_m = max(m, x)
    d = d * exp(m - new_m) + exp(x - new_m)
    m = new_m
# Now m is the max, d is the denominator.
```

You process each element once and update both running max and running denominator together. This is the heart of FlashAttention's algorithm — it lets you compute softmax across tiles without ever materializing the full row.

## Code walkthrough

A complete sum kernel with the two-phase block reduction and atomic cross-block aggregation. Pragmatically fast for any input size.

```cuda
#include <cuda_runtime.h>

__device__ float warp_reduce_sum(float val) {
    val += __shfl_down_sync(0xFFFFFFFF, val, 16);
    val += __shfl_down_sync(0xFFFFFFFF, val, 8);
    val += __shfl_down_sync(0xFFFFFFFF, val, 4);
    val += __shfl_down_sync(0xFFFFFFFF, val, 2);
    val += __shfl_down_sync(0xFFFFFFFF, val, 1);
    return val;
}

__device__ float block_reduce_sum(float val) {
    __shared__ float s[32];
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;

    val = warp_reduce_sum(val);
    if (lane == 0) s[wid] = val;
    __syncthreads();

    if (wid == 0) {
        val = (threadIdx.x < blockDim.x / 32) ? s[lane] : 0.0f;
        val = warp_reduce_sum(val);
    }
    return val;
}

__global__ void sum_kernel(const float* __restrict__ in, float* __restrict__ out, int n) {
    // Each thread loads multiple elements (grid-stride loop) for cleanliness.
    float val = 0.0f;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = idx; i < n; i += stride) {
        val += in[i];
    }

    float block_sum = block_reduce_sum(val);

    if (threadIdx.x == 0) {
        atomicAdd(out, block_sum);
    }
}
```

Three things worth highlighting.

**Grid-stride loop.** Each thread processes multiple elements (`i += stride` jumps by the total launched threads). This decouples the launch configuration from the input size — you can use a fixed grid (256 blocks of 256 threads, say) for any N. Simpler launching, often faster than the "exactly one element per thread" pattern.

**Warp shuffles inside `warp_reduce_sum`.** Five shuffles. No shared memory used by the warp-level part. Fast.

**Atomic at the end.** Each block contributes its sum via `atomicAdd`. With 256 blocks, that's 256 atomic contentions — small enough to be quick. For 10000+ blocks, use the two-pass approach instead.

### A softmax kernel sketch

```cuda
__global__ void softmax_kernel(const float* __restrict__ in, float* __restrict__ out, int N) {
    // One block per row. Each thread handles N / blockDim.x columns.
    int row = blockIdx.x;
    int tid = threadIdx.x;

    float local_max = -INFINITY;
    for (int i = tid; i < N; i += blockDim.x) {
        local_max = fmaxf(local_max, in[row * N + i]);
    }
    float row_max = block_reduce_max(local_max);
    __syncthreads();

    float local_sum = 0.0f;
    for (int i = tid; i < N; i += blockDim.x) {
        local_sum += expf(in[row * N + i] - row_max);
    }
    float row_sum = block_reduce_sum(local_sum);
    __syncthreads();

    float inv_sum = 1.0f / row_sum;
    for (int i = tid; i < N; i += blockDim.x) {
        out[row * N + i] = expf(in[row * N + i] - row_max) * inv_sum;
    }
}
```

`block_reduce_max` is the same pattern as `block_reduce_sum` with `fmaxf` in place of `+`. We make three passes over the row:
1. Find the row max.
2. Compute the sum of exponentials.
3. Write the normalized output.

This reads the input three times. A more advanced version uses the online softmax trick to do it in one pass — that's a key optimization for FlashAttention's inner loop.

## Mental model & pitfalls

Single sentence: **Reductions on GPU are a tree: warp-level shuffles, then per-warp shared-memory exchange, then per-block global aggregation. Get this pattern memorized; it shows up in every interesting kernel.**

Pitfalls:

- **`__shfl_down_sync` mask.** The first argument is the active-lane mask. For full-warp shuffle, use `0xFFFFFFFF` (all 32 bits set). If you call shuffle from inside a divergent branch, the mask must reflect which lanes are participating, or you get wrong answers silently.
- **The active-warp count.** Block-reduce assumes every warp in the block has a meaningful value to contribute. If the block size isn't a multiple of 32, the last warp has dead threads that must contribute neutral values (0 for sum, -inf for max).
- **Atomic contention.** `atomicAdd` with 10K blocks all hitting one location is a serialization. Use multi-bucket atomics, two-pass reductions, or block-level aggregations to spread the contention.
- **Numerical instability** in variance / softmax. Use stable formulations.
- **Old CUDA shuffle.** Pre-CUDA-9.0 used `__shfl_down` without the sync; behavior was different on Volta+. Always use the `_sync` variants.
- **Apple Silicon equivalents.** Metal has `simd_shuffle_down(val, offset)` and friends. Same idea, same patterns, different namespace.

## Hands-on (at home)

1. **Implement `sum_kernel`** above. Compile with nvcc, run on an input of 100M floats.

2. **Compare to `torch.sum`** (which calls cuBLAS under the hood). Your kernel should be within 30% — most of the rest is launch overhead and the careful tuning cuBLAS does.

3. **Time the warp shuffle vs the shared-memory tree reduction.** Replace `warp_reduce_sum` with a shared-memory version doing the same job. Measure. Shuffle should be 2–3× faster for the warp-level step.

4. **Write a fused softmax kernel** as in the code walkthrough. Bench against `torch.softmax`.

5. **Implement the online softmax variant.** Single pass, one reduction (the running max/denominator pair). Measure: should be ~30% faster than the three-pass version because of fewer reads.

6. **(Optional) Reduction in Triton.** Compare:

```python
@triton.jit
def sum_kernel(x_ptr, out_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    s = tl.sum(x)
    tl.atomic_add(out_ptr, s)
```

Triton's `tl.sum` compiles to the same kind of warp-shuffle tree. The code is much shorter; the performance is competitive.

7. **(Optional, harder) Implement an MSL warp shuffle reduction** using `metal::simd_shuffle_down`. Run it on Apple Silicon. Identical pattern, different symbols.

## Further reading

- Mark Harris's "Optimizing Parallel Reduction in CUDA" — the canonical NVIDIA tutorial. Walks through every optimization in detail.
- Justin Luitjens, "Faster Parallel Reductions on Kepler" (NVIDIA developer blog) — the warp shuffle introduction.
- "Online normalizer calculation for softmax" (Milakov & Gimelshein) — the FlashAttention precursor paper.
- *CUDA C Programming Guide*, shuffle intrinsics section.
- The FlashAttention paper (Lesson 10) — applies the online softmax to an attention kernel.

Next lesson: FlashAttention. With reductions, fusion, and tiled matmul in the bag, we can now read — and understand — the most consequential kernel in modern LLMs.
