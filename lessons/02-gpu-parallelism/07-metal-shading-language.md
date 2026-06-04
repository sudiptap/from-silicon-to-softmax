---
title: "Lesson 7 — Metal Shading Language"
date: "2026-06-03"
module: "gpu-computing"
order: 7
tags: ["metal", "msl", "threadgroup", "simdgroup", "apple-silicon", "matmul"]
author: "Sudipta Pathak"
prerequisites: ["06-triton"]
---

# Lesson 7 — Metal Shading Language

## Why this matters

If you want fast GPU code on Apple Silicon, you write Metal Shading Language (MSL). MLX uses it under the hood. PyTorch's MPS backend uses it. Every iOS or Mac app doing GPU compute uses it. Knowing how to write a kernel directly in MSL is the equivalent skill, on Apple, of knowing how to write a CUDA kernel on NVIDIA.

The good news: if you understand the CUDA matmul from Lesson 5, you understand the MSL matmul. The execution model is the same (SIMT, lockstep within a SIMD-group, threadgroup-private fast memory, cooperative loads, barrier syncs). The vocabulary changes — Apple calls things by different names — but the structure transfers directly.

This lesson is shorter than the CUDA lessons because most of the conceptual content is already in your head. The new material is the API surface, the syntax differences, and the one or two place Apple's design diverges meaningfully (no tensor cores in the NVIDIA sense; SIMD-group matrix ops as the closest equivalent; the unified-memory programming model).

## Concept

### The Metal terminology, side by side with CUDA

| CUDA | Metal | Notes |
| ---- | ----- | ----- |
| Kernel (`__global__`) | Compute function (`kernel`) | Same idea: a function many threads run in parallel. |
| Thread | Thread | Same. |
| Warp (32 threads) | SIMD-group (32 threads on current Apple GPUs) | Same. |
| Block | Threadgroup | Same. |
| Grid | Grid | Same. |
| `blockIdx` | `threadgroup_position_in_grid` | Same. |
| `threadIdx` | `thread_position_in_threadgroup` | Same. |
| `__shared__` memory | `threadgroup` memory | Same. |
| `__syncthreads()` | `threadgroup_barrier(mem_flags::mem_threadgroup)` | Same. |
| `__device__` | `kernel` / `void` (no separate marker) | MSL uses C++-style functions. |
| `__restrict__` | `[[no_alias]]` or none | Use sparingly; the compiler is usually fine. |

The differences are mostly cosmetic. The mental model from CUDA Lesson 5 maps over directly.

### Metal's C++ flavor

MSL is a dialect of C++14. You get:

- Function templates.
- Operator overloading.
- `auto` type inference.
- C++ math library functions (`metal::math::sqrt`, etc.).

You don't get:

- Full C++ standard library (no `std::vector`).
- Exceptions.
- Virtual dispatch (mostly).
- Recursion (Metal kernels can't recurse).

### Compiling and dispatching Metal kernels

Two paths in practice:

**Path A — embed source, compile at runtime.** Your host program (Swift, Objective-C, Rust via metal-rs) calls Metal APIs to compile the MSL source string at program startup. Fast iteration; one binary for all GPUs.

**Path B — precompile.** Compile MSL ahead of time with `xcrun -sdk macosx metal` to produce `.metallib` files. Faster startup; required for shipping iOS apps; the standard production path.

For learning and experimentation, Path A. For production, Path B.

### From Python via MLX

MLX exposes a Metal kernel custom-op API. You write the kernel as a Python string, and MLX compiles + runs it. This is the fastest path to "I want to test a Metal kernel" if you don't want to set up a Swift project. We'll use this for hands-on.

### What Apple has instead of tensor cores

NVIDIA's tensor cores are dedicated MMA hardware accessible through `mma.sync` PTX or `tl.dot` in Triton. Apple's analog has shifted over generations:

- **M1/M2**: no GPU-side matrix instructions. AMX on the CPU side; ANE separately. GPU matmul is plain SIMD FMAs.
- **M3+**: `simdgroup_matrix_multiply_accumulate` (SMMA) instructions in MSL. 8×8×8 FP16 MMA per SIMD-group per cycle. Apple's first generation of GPU matrix accelerators.
- **M4**: extended SMMA, similar shape.

For matmul on M3+, using SMMA gives ~2–3× over plain FMA. For broad portability across M-series, plain FMA is the safest choice. Below we write the plain FMA version; SMMA is in the further-reading section.

### Unified memory implications

The key practical implication: you allocate a `MTLBuffer` from the host, hand the pointer to the kernel, and that's it. The GPU sees the same memory the CPU does. No `cudaMemcpy`. No host/device split for correctness purposes.

You still need to consider where the *physical pages* live (for newer Macs the OS handles this transparently; for older or constrained scenarios you can hint with `MTLResourceOptions`). For our matmul, we don't think about it.

## Code walkthrough

The shared-memory tiled matmul, in MSL. Compare directly to the CUDA version in Lesson 5.

```metal
#include <metal_stdlib>
using namespace metal;

#define TILE 32

kernel void matmul_tiled(
    device const float* A          [[buffer(0)]],
    device const float* B          [[buffer(1)]],
    device       float* C          [[buffer(2)]],
    constant int& M                [[buffer(3)]],
    constant int& K                [[buffer(4)]],
    constant int& N                [[buffer(5)]],
    uint2 tg_pos                   [[threadgroup_position_in_grid]],
    uint2 t_pos                    [[thread_position_in_threadgroup]]
) {
    threadgroup float sA[TILE][TILE];
    threadgroup float sB[TILE][TILE];

    const uint tx = t_pos.x;
    const uint ty = t_pos.y;
    const uint row = tg_pos.y * TILE + ty;
    const uint col = tg_pos.x * TILE + tx;

    float acc = 0.0f;

    for (int k_tile = 0; k_tile < K; k_tile += TILE) {
        // Cooperative load with bounds check.
        if (row < (uint)M && (uint)(k_tile + tx) < (uint)K)
            sA[ty][tx] = A[row * K + (k_tile + tx)];
        else
            sA[ty][tx] = 0.0f;

        if ((uint)(k_tile + ty) < (uint)K && col < (uint)N)
            sB[ty][tx] = B[(k_tile + ty) * N + col];
        else
            sB[ty][tx] = 0.0f;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Inner loop, register-resident accumulator.
        for (int k = 0; k < TILE; ++k) {
            acc += sA[ty][k] * sB[k][tx];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row < (uint)M && col < (uint)N) {
        C[row * N + col] = acc;
    }
}
```

Comparing to the CUDA version:

- **`device const float* A [[buffer(0)]]`** — the `device` storage class marks pointers to global memory. `[[buffer(0)]]` is the binding index, telling Metal which input buffer this is. CUDA had implicit position-based binding; Metal makes it explicit.
- **`threadgroup float sA[TILE][TILE]`** — exactly the same idea as `__shared__`.
- **`threadgroup_barrier(mem_flags::mem_threadgroup)`** — verbose name for the same idea as `__syncthreads()`.
- **`constant int& M [[buffer(3)]]`** — Metal's way of passing scalar constants; the `constant` storage class is for small read-only data shared across threads.

The kernel body is structurally identical. If you understood Lesson 5, this reads like a translation.

### Dispatching from MLX

MLX makes it easy to invoke an MSL kernel from Python:

```python
import mlx.core as mx
from mlx.core import fast

KERNEL = """
#include <metal_stdlib>
using namespace metal;

#define TILE 32

kernel void matmul_tiled(
    device const float* A          [[buffer(0)]],
    device const float* B          [[buffer(1)]],
    device       float* C          [[buffer(2)]],
    constant int& M                [[buffer(3)]],
    constant int& K                [[buffer(4)]],
    constant int& N                [[buffer(5)]],
    uint2 tg_pos                   [[threadgroup_position_in_grid]],
    uint2 t_pos                    [[thread_position_in_threadgroup]])
{
    // ... body as above ...
}
"""

# Compile and wrap.
kernel = fast.metal_kernel(
    name="matmul_tiled",
    source=KERNEL,
    input_names=["A", "B"],
    output_names=["C"],
    ensure_row_contiguous=True,
)

def matmul_msl(a, b):
    M, K = a.shape
    _, N = b.shape
    grid = ((N + 31) // 32, (M + 31) // 32, 1)
    threads_per_tg = (32, 32, 1)
    c = kernel(
        inputs=[a, b],
        template=[("scalar", "float")],
        grid=grid,
        threadgroup=threads_per_tg,
        output_shapes=[(M, N)],
        output_dtypes=[a.dtype],
    )[0]
    return c
```

(Exact API varies slightly across MLX versions; check `mlx.core.fast.metal_kernel` documentation for the current call signature.)

### Performance

For a 4096-cube matmul on M3 Max:

| Implementation | TFLOPs FP16 |
| --- | ---: |
| Naive MSL (one thread per output, no shared memory) | ~3 |
| Tiled MSL (above) | ~10 |
| MLX's built-in `mx.matmul` (uses SMMA) | ~17 |
| MPSGraph `matmul` | ~16 |
| Accelerate via `cblas_hgemm` | not available; AMX is FP32 only |

So tiled MSL gets us ~60% of the way to MLX/MPS' built-in. The remaining gap is SMMA — the matrix instructions Apple added in M3 that MLX uses automatically. Adding SMMA to our kernel would close most of the rest.

### Using SMMA (sketch)

```metal
#include <metal_simdgroup_matrix>

simdgroup_float8x8 sa, sb, sc;
simdgroup_load(sa, &sA[0][0], TILE);
simdgroup_load(sb, &sB[0][0], TILE);
simdgroup_multiply_accumulate(sc, sa, sb, sc);
simdgroup_store(sc, &C[row*N + col], N);
```

The compiler-recognized matrix types (`simdgroup_float8x8`, etc.) map to the M3+ matrix instructions. Using them requires `metal::simdgroup_matrix` headers and a different layout discipline — see Apple's documentation and the MLX source for working examples.

For this curriculum's purposes, knowing that SMMA exists and is the missing piece is the key insight; mastering its API is left for when you genuinely need it.

## Mental model & pitfalls

Single sentence: **Metal Shading Language is CUDA with a Apple-flavored syntax and one fewer hardware features (no Hopper-tier TMA, no FP8) — write the same tiled matmul; rely on MLX or MPSGraph when you need the matrix-accelerator path.**

Pitfalls:

- **Forgetting `[[buffer(N)]]` indices.** Buffer bindings must match the order you set them in host code. Off-by-one is silent and produces garbage.
- **`threadgroup_barrier(mem_flags::mem_threadgroup)`** is the syntax; using `metal::threadgroup_barrier(mem_flags::mem_threadgroup)` from the wrong namespace, or forgetting the flag, has subtly different behavior.
- **Indexing with `int` vs `uint`.** `thread_position_in_threadgroup` returns `uint`; mixing with signed `int` for bounds math triggers warnings that are sometimes correctness issues. Be consistent.
- **Apple's matrix instructions aren't portable to M1/M2.** Code using SMMA won't run on older Macs. If portability matters, write the plain-FMA kernel and let MLX/MPSGraph handle the SMMA path on supported devices.
- **Unified memory doesn't mean free.** Even though there's no copy, the GPU's effective bandwidth to DRAM is what it is (~400 GB/s on M3 Max). The CPU and GPU contend for the same bus. For high-throughput pipelines, batch your GPU work.
- **MPS graph caching pitfalls.** If you go via MPSGraph (a different API), graph compilation is slow; cache the compiled graph. (For raw kernel use this isn't an issue.)

## Hands-on (at home)

Requires an Apple Silicon Mac.

1. **Install MLX and the MLX-LM tools** if you haven't yet:

```bash
pip install mlx mlx-lm
```

2. **Run the tiled MSL kernel via MLX.** Use the code skeleton above. Verify correctness against `mx.matmul`.

3. **Bench.** A 4096-cube FP16 matmul. Expected on M3 Max: ~10 TFLOPs for the tiled kernel; `mx.matmul` is the speed-of-light reference.

4. **Watch the GPU.** Open Activity Monitor → Window → GPU History (or use `sudo powermetrics --samplers gpu_power -i 100 -n 5`). You should see the GPU pegged near 100% during the bench.

5. **Try a non-tile-aligned size.** A 4097×4096 matrix. Mask logic should handle it; verify the result has no garbage.

6. **(Optional) Add SMMA.** Read Apple's `simdgroup_matrix` documentation. Add a SMMA path inside the inner loop. Measure the speedup — expect a meaningful jump on M3+ that brings you close to `mx.matmul`.

7. **(Optional) Profile in Xcode.** Build a small Mac app that loads your `.metallib`, dispatches the kernel, and run it under Xcode's GPU debugger. The "Capture GPU frame" command shows timing per kernel and per memory operation. This is the Apple-side equivalent of Nsight Compute.

## Further reading

- *Metal Shading Language Specification* (developer.apple.com) — the language reference. Dry but authoritative.
- *Metal Programming Guide* (developer.apple.com) — broader API + concepts.
- MLX's `mlx/backend/metal/kernels/` source on GitHub — production Metal kernels for matmul, attention, GEMM, etc. The most concrete examples available.
- Apple's WWDC 2023 / 2024 sessions on Metal compute and machine learning — yearly summaries of new features.
- Dougall Johnson and Asahi Linux writeups on Apple GPU internals — what's documented at the hardware level.

Next lesson: kernel fusion. Now that we can write a kernel on both NVIDIA and Apple, the most consequential optimization left isn't "making one kernel faster" — it's "merging several kernels into one."
