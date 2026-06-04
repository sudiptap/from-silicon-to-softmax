---
title: "Lesson 4 — Metal & Metal Performance Shaders"
date: "2026-06-04"
module: "mlx-apple-silicon"
order: 4
tags: ["metal", "mps", "mpsgraph", "msl", "api", "framework"]
author: "Sudipta Pathak"
prerequisites: ["03-memory-hierarchy"]
---

# Lesson 4 — Metal & Metal Performance Shaders

## Why this lesson exists

The Apple GPU has three layers of programming surface that you'll meet in any non-trivial ML workload, and they're often confused for each other:

- **Metal** is the low-level GPU API (analog: CUDA driver/runtime + nvcc + PTX assembly). You write shaders in Metal Shading Language (MSL), dispatch them via the Metal API, and manage buffers and resources directly.
- **Metal Performance Shaders (MPS)** is a high-level library of optimized neural-network ops (analog: cuDNN / cuBLAS / NCCL bundled). Convolutions, matrix multiplications, normalization, attention. You call MPS primitives instead of writing kernels.
- **MPSGraph** is Apple's computation-graph layer on top of MPS (analog: a thin TensorFlow-style graph builder with autodiff). Build a graph of MPS operations, optionally with derivatives; execute as one big compiled blob.

PyTorch's MPS backend, MLX, and Core ML all sit on top of one or more of these. Knowing which layer is which lets you reason about why a given operator is fast, slow, or absent.

The lesson is reading. The Hands-on dispatches a Metal kernel from Swift (or Python via PyObjC) and shows the layers.

## Metal: the low layer

Metal is the macOS/iOS GPU API. It's a C-style API plus a Swift/Objective-C wrapper plus the Metal Shading Language (MSL) for the shaders themselves.

A Metal compute pipeline:

1. Create an `MTLDevice` (handle to the GPU).
2. Create a `MTLCommandQueue` (analog to a CUDA stream — a serial dispatch queue for work).
3. Compile MSL source to a `MTLLibrary`, then create a `MTLComputePipelineState` for a specific kernel function.
4. Allocate `MTLBuffer`s for inputs/outputs (analog: `cudaMalloc`, but the buffer is in unified memory).
5. Create a `MTLCommandBuffer` from the queue, then a `MTLComputeCommandEncoder` from the buffer.
6. Set the pipeline state, set buffers as kernel arguments, dispatch threads (`dispatchThreadgroups`).
7. End encoding, commit the buffer. Wait for completion if needed.

The MSL kernel itself looks like CUDA-style code with different attribute syntax:

```metal
// kernel.metal
#include <metal_stdlib>
using namespace metal;

kernel void vector_add(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* c       [[buffer(2)]],
    uint idx              [[thread_position_in_grid]]
)
{
    c[idx] = a[idx] + b[idx];
}
```

The `[[buffer(N)]]` attributes bind arguments to the buffers the host passed in; `[[thread_position_in_grid]]` is the analog of CUDA's `blockIdx.x * blockDim.x + threadIdx.x`.

When you write a custom kernel for ML — say, a fused INT4 dequant + matmul kernel that doesn't exist in MPS — this is the layer you're working at. Module 2 Lesson 7 covered MSL itself; Module 4 Lesson 8 covers the specific case of writing custom kernels and calling them from MLX.

For most ML work you don't write Metal directly. You write Python (PyTorch, MLX) that calls a library that ultimately dispatches Metal. The custom-kernel case is the exception.

## Metal Performance Shaders (MPS)

MPS is a framework of pre-built, optimized neural-network operations. The roster includes:

- **MPSMatrixMultiplication**: GEMM. The analog of cuBLAS.
- **MPSCNNConvolution**: 2D convolutions, with multiple algorithms (direct, winograd, FFT depending on shape).
- **MPSCNN..**: layer norm, batch norm, softmax, pooling, activation functions.
- **MPSImage..**: image processing primitives (not neural-network specific, but ML-adjacent).
- **MPSNDArray** + matrix-multiply functions: the more recent (post-iOS 14) API for arbitrary-rank tensors, used by everyone modernizing onto MPS.

MPS exposes its operations as classes you instantiate and execute via a command encoder, similar to Metal but at a higher level. You don't write the kernel; you parameterize the pre-written one.

What MPS provides that you couldn't easily replicate by hand:

- Auto-selection of algorithms based on the shape (e.g., conv falls back from winograd to direct for shapes where winograd loses).
- Hardware-aware tile sizes for the specific Apple GPU you're running on.
- The matrix-multiply path that hits Apple's tensor-core-equivalent instructions on M3+ (SMMA, the matrix-multiply intrinsics).

What MPS doesn't:

- Custom fusions. If you want matmul + bias + activation as one kernel, MPS sometimes fuses it (MPSGraph does, more reliably), sometimes doesn't.
- Truly novel ops (FlashAttention variants, MoE routing). You're back to writing custom kernels.
- Quantized matmul beyond a small set of formats. The INT4 LLM matmul that MLX and llama.cpp use is *not* in MPS; both libraries roll their own.

The split with MLX (next two lessons): MLX uses MPS for some primitives (matmul on certain shapes) and uses custom Metal kernels for others (quantized matmul, optimized attention). The choice is invisible to the user.

## MPSGraph

MPSGraph is Apple's tensor-graph framework, built on top of MPS. It's the closest thing Apple has to TensorFlow's old `tf.Graph` or PyTorch's `torch.fx`. You build a graph of operations, compile it, then execute the compiled graph.

The graph API:

```python
import metal_performance_shaders_graph as mpsg
import metal
device = metal.MetalDevice.default_device
graph = mpsg.MPSGraph()
a = graph.placeholder([1, 1024])
b = graph.constant(weight_data)
c = graph.matmul(a, b)
result = graph.add(c, graph.constant(bias_data))
# Compile.
exe = graph.compile(device=device, feeds={a: shape, ...}, target_tensors=[result])
# Execute many times.
out = exe.run(feeds={a: actual_data})
```

(That's pseudo-Python; the real API is Objective-C / Swift; Python bindings exist via PyObjC.)

What you get from MPSGraph:

- **Operator fusion**: MPSGraph's compiler will fuse matmul + bias + activation into a single kernel where it can.
- **Auto-differentiation**: pass a `gradients` argument; MPSGraph will produce backward graphs.
- **Inference of MPS specializations**: chooses the right MPS primitives for each op.

What you give up:

- Some flexibility. The graph has to be defined up front; you can't easily do dynamic control flow.
- Direct kernel control. If you have a custom MSL kernel, you can call it through MPSGraph (there's an `MPSGraphExecutable.encodeCustomKernel` path), but it's clunkier than calling Metal directly.

MPSGraph is the layer that Apple's Core ML compiler often targets internally. It's also what some PyTorch operators ultimately dispatch to (via `MPSGraph` calls in the PyTorch MPS backend's C++ code).

For Module 4, MPSGraph mostly matters as a thing you'll see referenced in profiling output. You rarely need to call it directly from your own code; either you're using PyTorch or MLX (which call it for you) or you're writing custom Metal (which is more direct).

## How the layers compose in real frameworks

PyTorch MPS backend (Lesson 5): a PyTorch op like `torch.matmul` dispatches to C++ that, for the MPS device, calls into MPSGraph (or MPS directly for simple ops), which compiles down to Metal kernels.

MLX (Lessons 6–7): MLX has its own dispatch layer. For matmul, it picks between MPS's matrix-multiply path, a hand-written MSL kernel, and (for quantized formats) a custom dequant+matmul kernel. The choice depends on shape, dtype, and quantization format.

Core ML (Lesson 11): models compiled to Core ML are run by the Core ML runtime, which schedules across CPU, GPU (via Metal/MPS/MPSGraph), and ANE. The compilation step bakes in which ops go to which unit.

llama.cpp: Apple-specific Metal kernels written in MSL, dispatched via Metal directly (no MPS). The kernels are specifically designed for the GGUF quantized format. The author (Georgi Gerganov) wrote them from scratch rather than using MPS, because MPS's quantized-matmul coverage was inadequate when the project started.

The picture: every ML library on Apple Silicon ultimately produces Metal kernels and dispatches them via the Metal API. The differences are in how high a level of abstraction the library presents and which optimizations it bakes in.

## When to reach for which layer

A decision matrix for the case "I'm writing custom code on Apple Silicon and need to pick the layer":

| Goal | Use |
| ---- | --- |
| Drop-in NumPy replacement that uses the GPU | MLX |
| Train a model from scratch on M-series | MLX (or PyTorch MPS if you must match a CUDA codebase) |
| Inference for a pre-trained LLM | MLX (`mlx-lm`) or llama.cpp |
| Inference for a vision/audio model | Core ML (best ANE path) or MLX |
| One-off custom kernel for a research idea | Custom MSL kernel, called from MLX or directly via Metal |
| Building a new framework | Direct Metal + MPSGraph as the autodiff backbone |
| Production iOS app | Core ML (the deployment story is built around it) |

The bottom row — production iOS app — is the case where Core ML wins decisively. Everywhere else, MLX is the default for new code in 2026.

## What you should believe after this lesson

Three sentences:

**1. The Apple GPU stack has three layers** — Metal (low-level API + MSL shaders), MPS (pre-built optimized neural-net ops), MPSGraph (graph + autodiff on top of MPS). Real frameworks (MLX, PyTorch MPS, Core ML) compose these layers; understanding which layer is which lets you read profiler output and reason about performance.

**2. MPS is the Apple equivalent of "cuBLAS + cuDNN bundled,"** but with a smaller op surface and weaker coverage of recent ML primitives (quantized matmul, custom attention). The MLX and llama.cpp projects work around MPS gaps by writing custom Metal kernels.

**3. MPSGraph is the autodiff/graph-compilation layer** and matters mostly as the substrate that PyTorch MPS and Core ML build on. You rarely call it directly from new code; you let MLX or PyTorch do the dispatch.

## Hands-on (at home)

A minimal Metal kernel dispatched from Swift, then through a higher-level Python path.

The Swift version (file `kernel.metal` + a tiny Swift driver):

```metal
// kernel.metal
#include <metal_stdlib>
using namespace metal;

kernel void scale_add(
    device const float* x [[buffer(0)]],
    device const float* y [[buffer(1)]],
    device float* out      [[buffer(2)]],
    constant float& alpha  [[buffer(3)]],
    uint idx               [[thread_position_in_grid]]
)
{
    out[idx] = alpha * x[idx] + y[idx];
}
```

```swift
// run_kernel.swift
import Metal
import Foundation

let device = MTLCreateSystemDefaultDevice()!
let queue = device.makeCommandQueue()!
let source = try String(contentsOfFile: "kernel.metal", encoding: .utf8)
let library = try device.makeLibrary(source: source, options: nil)
let fn = library.makeFunction(name: "scale_add")!
let pipeline = try device.makeComputePipelineState(function: fn)

let n = 1024 * 1024
let x = device.makeBuffer(length: n * MemoryLayout<Float>.size, options: .storageModeShared)!
let y = device.makeBuffer(length: n * MemoryLayout<Float>.size, options: .storageModeShared)!
let out = device.makeBuffer(length: n * MemoryLayout<Float>.size, options: .storageModeShared)!
let xp = x.contents().bindMemory(to: Float.self, capacity: n)
let yp = y.contents().bindMemory(to: Float.self, capacity: n)
for i in 0..<n { xp[i] = Float(i); yp[i] = Float(i*2) }

var alpha: Float = 0.5
let cmd = queue.makeCommandBuffer()!
let enc = cmd.makeComputeCommandEncoder()!
enc.setComputePipelineState(pipeline)
enc.setBuffer(x, offset: 0, index: 0)
enc.setBuffer(y, offset: 0, index: 1)
enc.setBuffer(out, offset: 0, index: 2)
enc.setBytes(&alpha, length: 4, index: 3)
enc.dispatchThreadgroups(MTLSize(width: n / 256, height: 1, depth: 1),
                          threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
enc.endEncoding()
cmd.commit()
cmd.waitUntilCompleted()
let op = out.contents().bindMemory(to: Float.self, capacity: n)
print("out[0..4]:", op[0], op[1], op[2], op[3])
```

Build & run: `swiftc -framework Metal run_kernel.swift -o run_kernel && ./run_kernel`.

You'll see `out[0..4]: 0.0 2.5 5.0 7.5` (which is `0.5*i + 2*i` for i=0,1,2,3).

The equivalent in MLX (Python, no MSL needed because the op is built-in):

```python
import mlx.core as mx
n = 1024 * 1024
x = mx.arange(n).astype(mx.float32)
y = (mx.arange(n) * 2).astype(mx.float32)
out = 0.5 * x + y
mx.eval(out)
print("out[0..4]:", out[:4].tolist())
```

Same result, 5 lines of Python, no MSL, no buffer management. This is the abstraction layer MLX gives you over Metal — and it's why MLX is the default for ML code on Apple Silicon, while Metal is reserved for the cases MLX can't reach.

## Further reading

- Apple's "Metal Programming Guide" — the authoritative tutorial for the low layer.
- Apple's "Metal Shading Language Specification" — the MSL language reference.
- Metal Performance Shaders framework documentation (developer.apple.com) — for the MPS op catalog.
- "MPSGraph Programming Guide" — the graph-compilation layer.
- Performance Shaders by Halide author — Halide's MSL backend gives some insight into the layer.

Next lesson: **PyTorch's MPS backend.** What's implemented, what's not, the operator coverage problem and the CPU fallback path, and the practical implications for someone running a PyTorch model on Apple Silicon — including the cases where MPS surprises you with a 5× speedup over CPU and the cases where it surprises you with a 5× *slowdown*.
