---
title: "Lesson 6 — Triton: Same GPU, 10× Less Code"
date: "2026-06-03"
module: "gpu-computing"
order: 6
tags: ["triton", "kernel", "matmul", "block-programming", "openai", "ml-systems"]
author: "Sudipta Pathak"
prerequisites: ["05-cuda-memory-hierarchy"]
---

# Lesson 6 — Triton: Same GPU, 10× Less Code

## Why this matters

Triton is the most consequential thing that has happened to GPU kernel programming in the LLM era. It's a Python-embedded DSL — looks like NumPy with a different decorator — that compiles down to PTX (the NVIDIA virtual ISA) and is, for many kernel shapes, as fast as hand-tuned CUDA at a fraction of the code volume. FlashAttention, by Tri Dao, is written in Triton. Most of the fast attention variants, the FP8 inference kernels, the fused MoE kernels — all of them live in Triton today.

The reason: Triton finds the right balance between "I want control over the GPU" and "I don't want to spend a week per kernel." A CUDA matmul kernel that hits 60% of peak is hundreds of lines of careful work. The equivalent Triton matmul is ~50 lines, hits 70–80% of peak out of the box, and is portable across NVIDIA GPU generations because the Triton compiler does the microarchitecture-specific code generation. The cost: less control at the extreme end. For 95% of kernels you'd ever want to write, Triton is the right answer.

This lesson is the Triton tour: what it is, what its programming model gives you, what it takes away, and a working matmul kernel side by side with the CUDA version from Lesson 5.

## Concept

### What Triton is

Triton is a domain-specific language and compiler for writing GPU kernels. From the user's perspective, you write a Python function decorated with `@triton.jit`. Inside the function, you write what looks like block-level NumPy code. The Triton compiler turns it into optimized PTX (or, recently, AMD's equivalent).

The level of abstraction is interesting. CUDA's unit of programming is the **thread** — you write code for one thread and the hardware runs many copies. Triton's unit is the **block** (a programmer-defined tile size). You write code for one block, and the compiler figures out how to execute it across threads. The compiler handles SIMT-level details (warp organization, shared memory, register allocation) so you don't have to.

The vocabulary:

- A Triton **program** is one instance of a kernel running on one block of data. Identifiable by its program ID (`pid`), analogous to `blockIdx` in CUDA.
- Operations are vectorized: `tl.load`, `tl.dot`, `tl.store` work on entire tiles, not single elements.
- Pointer arithmetic is explicit (you compute the pointers for a tile yourself, give them to `tl.load`/`tl.store`).
- Masks (boolean tiles) handle boundaries naturally.

If you've used NumPy on slices, the mental model translates: you're operating on rectangular blocks of memory, and the compiler decides how those operations map to threads.

### What you gain

- **No `__syncthreads()`.** The compiler inserts synchronizations as needed.
- **No bank-conflict pondering.** Triton's data placement is handled by the compiler.
- **No coalescing math.** Triton lays out the memory access patterns from your high-level expressions.
- **Portable across NVIDIA GPUs.** Auto-tunes block sizes and other knobs per architecture.
- **Python integration.** Same module as your model code; no separate compilation step.

### What you give up

- **Last 5–15% of performance.** A truly hand-tuned CUDA kernel can still beat Triton, especially with very specialized tensor-core / TMA paths on Hopper. For most kernels this gap is small enough not to matter.
- **Fine-grained control.** If you want to do something the compiler doesn't anticipate (a fancy data permutation, a per-warp specialization), you may hit walls.
- **CUDA-only(ish) ecosystem.** Triton runs on NVIDIA primarily; AMD support is mature in recent versions; Apple/Metal support is not a thing. For cross-vendor portability, MLX or candle are better choices.

### The Triton matmul, conceptually

You launch `(M/BLOCK_M) × (N/BLOCK_N)` programs. Each program computes one `BLOCK_M × BLOCK_N` tile of output. Inside the program:

```
acc = zeros((BLOCK_M, BLOCK_N))
for k in range(0, K, BLOCK_K):
    a = load A[m_offsets, k_offsets]   # BLOCK_M × BLOCK_K
    b = load B[k_offsets, n_offsets]   # BLOCK_K × BLOCK_N
    acc += a @ b                        # tile matmul, uses tensor cores
store C[m_offsets, n_offsets] = acc
```

That's the whole structure. Triton's compiler handles: cooperative loads into shared memory, tensor-core MMA instructions, register allocation, sync, the works.

## Code walkthrough

The full matmul kernel.

```python
import triton
import triton.language as tl
import torch

@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Which output tile do we compute?
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Row and column indices for this tile.
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Pointers to A and B tiles. A is (BLOCK_M, BLOCK_K), B is (BLOCK_K, BLOCK_N).
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Walk along K.
    for k in range(0, K, BLOCK_K):
        # Masks to handle non-multiple sizes at boundaries.
        a_mask = (offs_m[:, None] < M) & ((k + offs_k)[None, :] < K)
        b_mask = ((k + offs_k)[:, None] < K) & (offs_n[None, :] < N)

        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        # Tile-level matmul — uses tensor cores when shapes/precisions allow.
        acc += tl.dot(a, b)

        # Advance to next K-tile.
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # Store the result.
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


def matmul(a, b, BLOCK_M=128, BLOCK_N=128, BLOCK_K=32):
    M, K = a.shape
    K2, N = b.shape
    assert K == K2
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    return c
```

Reading this, a few things to internalize:

**`tl.constexpr`** marks compile-time constants. The block sizes are baked into the generated code, which lets the compiler unroll and optimize aggressively. Changing block sizes recompiles the kernel.

**`tl.arange(0, BLOCK_M)`** produces a tile of indices, not a single value. All subsequent expressions on it are tile operations. The compiler emits the right code for the right number of threads.

**Pointer construction with broadcasts.** `offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak` is a 2D pointer tile, where rows correspond to M and columns to K. NumPy-style broadcasting works.

**`tl.dot(a, b)`** is the matmul — and this is where the magic is. For FP16/BF16 inputs and FP32 accumulator on supported hardware, the compiler emits tensor-core MMA. For FP32 inputs, it emits CUDA-core FMAs. You don't choose — the compiler does.

**Masking** for boundaries. `mask=a_mask, other=0.0` says: where the mask is false, return 0 instead of loading. Equivalent to the `if` guards in the CUDA version, but expressed declaratively.

**`triton.cdiv(M, BLOCK_M)`** is ceiling division — standard "number of tiles" math.

### Performance comparison

For a 4096×4096×4096 matmul in FP16 on an RTX 4090:

| Implementation | TFLOPs FP16 | Lines of code |
| ---- | ---: | ---: |
| Naive CUDA (Lesson 4 equivalent in FP16) | ~10 | ~50 |
| Shared-memory tiled CUDA (Lesson 5) | ~50 | ~120 |
| Triton matmul (above) | ~150 | ~30 |
| Optimized Triton (tuned block sizes, auto-tuned) | ~250 | ~60 |
| cuBLAS via `torch.matmul` | ~280 | ~1 |

So at this problem size, Triton gives you 90% of cuBLAS in 30 lines and 95% with auto-tuning in 60 lines. The remaining 5% is the gap between "Triton's compiler" and "NVIDIA's hand-tuned cuBLAS engineers." For most use cases, the 5% gap is worth it for the ability to *modify the kernel* — fused activations, custom output scaling, weird input shapes.

### Auto-tuning

The block sizes (`BLOCK_M`, `BLOCK_N`, `BLOCK_K`) matter for performance. Triton lets you decorate a kernel with `@triton.autotune` and a set of candidate configs:

```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        # ... more configs ...
    ],
    key=['M', 'N', 'K'],  # cache by problem shape
)
@triton.jit
def matmul_kernel(...): ...
```

At first call for a new shape, Triton tries each config, benchmarks, and caches the winner. Subsequent calls reuse the choice. `num_stages` controls pipeline depth (more stages = more shared memory used for overlapping the next K-tile load with the current compute). `num_warps` controls how many warps per block.

The wisest move: copy the auto-tune block from the official tutorial. The configs in there are good defaults.

## Mental model & pitfalls

Single sentence: **Triton lets you write block-level numerical code and the compiler turns it into a fast SIMT kernel — for matmul, attention, and most reduction-shaped kernels, this is the productivity win you want.**

Pitfalls:

- **Treating Triton like NumPy.** It looks like NumPy. It is not NumPy. There's no broadcasting between tiles of different shapes; you need explicit reshapes / broadcasts via `[:, None]` / `[None, :]`. Operations across program boundaries don't compose; each program is independent.
- **Forgetting the mask.** When a tile extends past the edge of a matrix, you must mask the load or you read garbage / segfault.
- **Hot-loading.** First call into a Triton kernel triggers compilation, which takes seconds. Don't time the first call.
- **Mismatched dtypes for `tl.dot`.** The compiler chooses tensor-core paths based on input dtypes. FP16 in → FP32 accumulator → tensor core. FP32 in → FP32 accumulator → CUDA core (much slower). For best matmul performance, your inputs should be FP16 or BF16.
- **Excessive block sizes.** Block sizes too large overflow shared memory or registers. The compiler will tell you; pay attention.
- **Auto-tune cache invalidation.** Triton caches auto-tune choices by the `key=[...]` arguments. If those are wrong, you may reuse a slow config from a different shape. Include any argument that affects the optimal config.

## Hands-on (at home)

1. **Install Triton.** It ships with PyTorch 2.x:

```bash
pip install torch triton
```

2. **Run the matmul kernel** above on FP16 inputs. Compare to `torch.matmul`:

```python
a = torch.randn(4096, 4096, device='cuda', dtype=torch.float16)
b = torch.randn(4096, 4096, device='cuda', dtype=torch.float16)

# Warmup
for _ in range(3):
    _ = matmul(a, b)
    _ = a @ b
torch.cuda.synchronize()

import time
t0 = time.time()
for _ in range(20):
    c1 = matmul(a, b)
torch.cuda.synchronize()
print(f"Triton: {(time.time()-t0)/20*1000:.2f} ms")

t0 = time.time()
for _ in range(20):
    c2 = a @ b
torch.cuda.synchronize()
print(f"cuBLAS: {(time.time()-t0)/20*1000:.2f} ms")

print(f"max abs diff: {(c1-c2).abs().max().item():.4f}")
```

The diff should be small (FP16 reduction order differs; ~10⁻¹ is fine). Triton should be ~80–95% of cuBLAS speed.

3. **Add auto-tuning** and re-bench. You should see another ~10–30% bump.

4. **Read the official Triton matmul tutorial** at `triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html`. It includes a "swizzling" trick (reordering tile assignments for better L2 cache behavior) that we omitted for clarity but matters at scale.

5. **Fused matmul + activation.** Modify the kernel to compute `C = ReLU(A @ B + bias)` in one pass. This is the *whole point* of writing Triton kernels — you don't pay separate memory passes for the matmul, the bias add, and the ReLU. Three operations, one global memory write of `C`.

6. **(Optional) Read FlashAttention's Triton implementation.** The `flash-attention` repo contains Triton versions of the kernel. Once you understand this matmul, FlashAttention is "the same idea, but for `softmax(QK^T)V` with online softmax." It's beautifully readable.

7. **(Optional) AMD Triton.** Recent Triton versions target AMD MI300X with the same code. If you have AMD GPU access, run the same kernel; performance is competitive with hand-written ROCm.

## Further reading

- Triton's official tutorials — go through the matmul, the layer norm, the dropout, and the FlashAttention tutorials in order.
- "Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations" (Tillet et al.) — the original paper. Read after the tutorials.
- Tri Dao's FlashAttention paper and GitHub — the canonical example of Triton in the LLM world.
- "How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance: a Worklog" (Simon Boehm) — comparison material; the CUDA arc Triton compresses.

Next lesson: Metal Shading Language. Same matmul, Apple Silicon style. Triton doesn't run there yet, so we go back to writing kernels by hand — but with the mental shortcuts you now have from CUDA.
