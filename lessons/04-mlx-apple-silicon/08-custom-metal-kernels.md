---
title: "Lesson 8 — Custom Metal Kernels from MLX"
date: "2026-06-04"
module: "mlx-apple-silicon"
order: 8
tags: ["metal", "msl", "custom-kernels", "mlx", "threadgroup-memory", "simdgroup"]
author: "Sudipta Pathak"
prerequisites: ["07-mlx-internals"]
---

# Lesson 8 — Custom Metal Kernels from MLX

## Why this lesson exists

MLX has good coverage of standard ML primitives — matmul, attention, normalization, activation functions — and the lazy fusion + `mx.compile` story handles most non-standard combinations. But when you want a truly custom kernel — a fused INT4 dequant + matmul with a specific layout, a novel attention variant, a custom reduction with non-standard masking — you need to drop to Metal Shading Language.

The good news: MLX provides a clean way to integrate custom MSL kernels into the lazy graph. You write the kernel in MSL (covered in Module 2 Lesson 7), declare it to MLX, and call it like any other MLX op. It composes with the rest of the graph, participates in lazy evaluation, and can be auto-differentiated if you provide the gradient (less commonly needed for inference).

This lesson is the integration pattern. It assumes you already know MSL (from Module 2). The focus here is on the MLX-specific wrapping and what changes when your kernel lives inside an MLX graph.

The lesson is reading. The Hands-on writes a custom fused activation kernel and benchmarks it against the MLX-native composition.

## The MLX custom-kernel API

The entry point is `mx.fast.metal_kernel`. Pseudo-API:

```python
import mlx.core as mx

kernel = mx.fast.metal_kernel(
    name="my_kernel",
    input_names=["x", "scale"],
    output_names=["out"],
    source="""
        uint idx = thread_position_in_grid.x;
        out[idx] = x[idx] * scale[0];
    """,
    header="#include <metal_stdlib>\nusing namespace metal;",
)

def call_kernel(x, scale):
    return kernel(
        inputs=[x, scale],
        output_shapes=[x.shape],
        output_dtypes=[x.dtype],
        grid=(x.size, 1, 1),
        threadgroup=(256, 1, 1),
    )[0]
```

What MLX provides:

- Marshals input arrays to Metal buffers. No copies — uses the unified-memory pointers directly.
- Generates the boilerplate kernel signature from `input_names` and `output_names`.
- Schedules the dispatch on the MLX GPU stream so it composes with surrounding ops.
- Hands the output back as an `mx.array` that participates in the lazy graph.

What you write: the kernel body, in MSL syntax, assuming the inputs and outputs are bound to the names you declared.

## A worked example: fused SiLU + multiply

A common LLM op pattern is `silu(x) * y` — the gate-then-multiply that appears in SwiGLU FFN layers. MLX can fuse this via `mx.compile`, but let's write it as a custom kernel to see the workflow.

```python
import mlx.core as mx

silu_mul_source = """
    uint idx = thread_position_in_grid.x;
    if (idx >= n[0]) return;
    float xi = float(x[idx]);
    float yi = float(y[idx]);
    float silu = xi / (1.0 + exp(-xi));
    out[idx] = (T)(silu * yi);
"""

# T is templated; MLX picks based on the actual dtypes at call time.
silu_mul_kernel = mx.fast.metal_kernel(
    name="silu_mul",
    input_names=["x", "y", "n"],
    output_names=["out"],
    source=silu_mul_source,
    header="""
        #include <metal_stdlib>
        using namespace metal;
        template <typename T> kernel void silu_mul_T(
            device const T* x [[buffer(0)]],
            device const T* y [[buffer(1)]],
            device const int* n [[buffer(2)]],
            device T* out [[buffer(3)]],
            uint thread_position_in_grid [[thread_position_in_grid]]
        );
    """,
)

def silu_mul(x: mx.array, y: mx.array) -> mx.array:
    n = mx.array([x.size], dtype=mx.int32)
    return silu_mul_kernel(
        inputs=[x, y, n],
        output_shapes=[x.shape],
        output_dtypes=[x.dtype],
        grid=((x.size + 255) // 256 * 256, 1, 1),
        threadgroup=(256, 1, 1),
    )[0]

# Compare to MLX-native composition.
x = mx.random.normal((1024 * 1024,), dtype=mx.float16)
y = mx.random.normal((1024 * 1024,), dtype=mx.float16)
mx.eval(x, y)

a = silu_mul(x, y)         # custom kernel
b = mx.sigmoid(x) * x * y  # MLX native, will fuse via eager fusion
mx.eval(a, b)

# Verify correctness.
err = mx.abs(a - b).max().item()
print(f"max abs error: {err}")
```

(The exact API surface may shift slightly between MLX versions; check the `mx.fast.metal_kernel` docstring on your installed version.)

The custom kernel is one Metal dispatch with one load each of `x` and `y` and one store of `out`. The MLX-native version is also one dispatch after eager fusion. The throughput should be essentially identical — which is the point: MLX's fusion is good enough that custom kernels rarely beat native composition on standard patterns.

Where custom kernels win is on patterns MLX's fusion can't handle: kernels that mix matmul-style and element-wise work, kernels with non-standard memory access patterns, kernels that exploit SIMD-group reductions, kernels for novel quantization formats.

## A real-world case: the MLX INT4 matmul

The `mlx.quantized` module's INT4 matmul is a custom Metal kernel ~500 lines of MSL. Its structure:

```
threadgroup {
    // Tile of unpacked weights (group_size=128, INT4 → FP16 in shared mem).
    threadgroup half tile_w[BM * BK];
    // Tile of activations.
    threadgroup half tile_a[BK * BN];
}

// Each thread computes BMperT × BNperT outputs.
half acc[BMperT][BNperT] = 0;

for k_tile in 0..K/BK:
    // Cooperative load: each thread loads BLOCKSIZE/N_threads ints4 weights,
    // unpacks two-per-byte, multiplies by the per-group scale, writes to tile_w.
    barrier();
    // Tile multiply: compute acc += tile_w[:, :] @ tile_a[:, :]
    for j in 0..BK:
        for ii in 0..BMperT:
            for jj in 0..BNperT:
                acc[ii][jj] += tile_w[my_row+ii, j] * tile_a[j, my_col+jj]
    barrier();

// Write out.
for ii, jj: out[global_row+ii, global_col+jj] = acc[ii][jj];
```

The kernel does in one dispatch what would be three separate ops naively (unpack INT4 → FP16, dequantize with scales, multiply with activations). The fusion saves two intermediate writes and the corresponding reads. On a real LLM workload, the custom INT4 matmul is ~2× faster than the naive "unpack to fp16, then matmul" sequence.

This is the canonical example of when custom MSL pays off: when the operation crosses MLX's fusion boundary (matmul + element-wise dequant + load-from-quantized-format) and would otherwise require multiple dispatches.

## SIMD-group operations

The Apple GPU has SIMD groups of 32 threads each (the same width as NVIDIA warps). MSL provides intrinsics for collective operations within a SIMD group:

- `simd_sum(x)`: returns the sum of `x` across all 32 lanes.
- `simd_max(x)`, `simd_min(x)`, `simd_prefix_inclusive_sum(x)`: corresponding patterns.
- `simd_broadcast(x, lane)`: broadcast lane `lane`'s value to all 32 lanes.
- `simd_shuffle(x, lane)`: every lane reads the `x` from lane `lane`.

These are essential for high-performance reductions. The FlashAttention kernel uses `simd_sum` for the row-wise softmax reduction; without it, you'd need shared memory + barriers, which is much slower.

The MLX `mx.fast.metal_kernel` source body can use these intrinsics directly. There's no MLX abstraction over them; you write them as you would in any MSL code. The Module 2 Lesson 9 (reduction problem) hands-on shows the SIMD-shuffle reduction pattern; the same code transplants directly into an MLX custom kernel.

## When NOT to write a custom kernel

A reality-check list:

- **The operation is in MLX already.** Common matmul, attention, conv, norm. Don't write your own; MLX's version is usually faster and you're not maintaining it.
- **`mx.compile` already handles it.** A chain of element-wise ops, or matmul + bias + activation. Try `mx.compile` first; if it gives you 80% of the win for 5% of the work, ship it.
- **The kernel is for one-off research.** A novel attention variant for a paper, but you'll only run it a few times. The development cost of a custom kernel (debug cycles, performance tuning) is significant.
- **You're early in optimization.** Profile first. The kernel you're considering writing may not be on the critical path.

When custom kernels are worth it:

- **A specific fused pattern that ships in a production library.** mlx-lm's INT4 matmul, custom attention variants for the Whisper kernel.
- **A novel algorithm that touches the hardware in a non-standard way.** SIMD-group-based reductions that don't map to MLX's primitives.
- **A learning exercise.** Writing a custom kernel and watching it beat (or fail to beat) MLX native is the fastest way to understand the framework's fusion limits.

## What you should believe after this lesson

Three sentences:

**1. MLX's `mx.fast.metal_kernel` provides a clean way to inject custom MSL into the lazy graph** — buffers come from unified memory with no copies, the kernel composes with surrounding MLX ops, and the result is a regular `mx.array` you can keep using.

**2. The real wins for custom kernels are in fusion patterns MLX can't reach** — particularly anything involving non-standard memory layouts (quantized formats, packed weights) or cross-pattern fusion (matmul + element-wise quant). The MLX INT4 matmul kernel is the canonical example.

**3. SIMD-group intrinsics are essential for high-performance reductions** and are available directly in MSL inside an MLX kernel; the FlashAttention-style attention kernels rely on them, and any non-trivial custom kernel for ML workloads ends up using them.

## Hands-on (at home)

Write a custom Metal kernel and call it from MLX. We'll do a fused "scale and threshold" operation — multiply by a scalar, then clamp to a range — as a tractable example.

```python
# custom_kernel.py
import mlx.core as mx

scale_clamp_source = """
    uint idx = thread_position_in_grid.x;
    if (idx >= n[0]) return;
    float xi = float(x[idx]);
    float scaled = xi * float(scale[0]);
    float clamped = max(float(lo[0]), min(float(hi[0]), scaled));
    out[idx] = (T)clamped;
"""

scale_clamp_kernel = mx.fast.metal_kernel(
    name="scale_clamp",
    input_names=["x", "scale", "lo", "hi", "n"],
    output_names=["out"],
    source=scale_clamp_source,
)

def scale_clamp(x: mx.array, scale: float, lo: float, hi: float) -> mx.array:
    return scale_clamp_kernel(
        inputs=[
            x,
            mx.array([scale], dtype=x.dtype),
            mx.array([lo], dtype=x.dtype),
            mx.array([hi], dtype=x.dtype),
            mx.array([x.size], dtype=mx.int32),
        ],
        output_shapes=[x.shape],
        output_dtypes=[x.dtype],
        grid=((x.size + 255) // 256 * 256, 1, 1),
        threadgroup=(256, 1, 1),
    )[0]

# Verify correctness.
x = mx.random.normal((1024 * 1024,), dtype=mx.float32)
y_custom = scale_clamp(x, 2.0, -1.0, 1.0)
y_native = mx.clip(x * 2.0, -1.0, 1.0)
mx.eval(y_custom, y_native)
err = mx.abs(y_custom - y_native).max().item()
print(f"max abs error: {err}")  # should be ~0 (floating-point noise)
```

Bench the two:

```python
import time
def bench(fn, n=200):
    for _ in range(3): mx.eval(fn())
    t0 = time.time()
    for _ in range(n): mx.eval(fn())
    return (time.time() - t0) * 1000 / n

print(f"custom:  {bench(lambda: scale_clamp(x, 2.0, -1.0, 1.0)):.3f} ms")
print(f"native:  {bench(lambda: mx.clip(x * 2.0, -1.0, 1.0)):.3f} ms")
```

On M3 Pro you'll see they're within 5–10% of each other; the eager fusion does its job. If you intentionally break fusion (use `mx.eval` between the multiply and clip), you'll see the custom kernel pull ahead.

Part 2 — read the MLX quantized matmul kernel. It's at `mlx/backend/metal/kernels/quantized.metal` in the MLX source. ~500 lines, well-commented, and is the reference implementation for fused INT4 matmul on Apple Silicon. Reading it gives you the template for any quantized-matmul work you might want to do.

## Further reading

- MLX docs: "Custom Metal Kernels" — the official guide to `mx.fast.metal_kernel`.
- MLX source: `mlx/backend/metal/kernels/` — the production kernels, including `quantized.metal`, `sdpa.metal` (FlashAttention), and `gemm.metal`.
- "Optimizing Metal Performance Shaders for Machine Learning" WWDC sessions — for Apple's own perspective on writing fast Metal kernels.
- Apple's "Metal Best Practices Guide" — section on compute kernel optimization.
- `mlx-examples` repo has several "custom kernel" examples worth reading.

Next lesson: **KV cache strategies on unified memory.** We pivot from kernel-writing to a specific architectural question: how the unified-memory model changes the design space for KV caches in autoregressive decoding, and what patterns work on Apple Silicon that don't on CUDA (and vice versa).
