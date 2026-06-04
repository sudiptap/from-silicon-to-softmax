---
title: "Lesson 7 — MLX Internals"
date: "2026-06-04"
module: "mlx-apple-silicon"
order: 7
tags: ["mlx", "lazy-evaluation", "compute-graph", "streams", "compile", "fusion"]
author: "Sudipta Pathak"
prerequisites: ["06-mlx-intro"]
---

# Lesson 7 — MLX Internals

## Why this lesson exists

The previous lesson introduced MLX from the user's perspective. This one looks at how MLX actually executes the code you write. Three concepts do almost all the work: the computation graph (the lazy DAG built as you write expressions), the streams model (the queue layer that turns the graph into ordered Metal/CPU dispatches), and `mx.compile` (the optional whole-region compilation that produces fused kernels beyond what the eager fusion can do).

If you only use MLX at the surface — define a model, train or infer — you can mostly ignore these. But when something is mysteriously slow, when a model uses way more memory than the parameter count suggests, or when you want to write a custom kernel that integrates cleanly, you need the internals picture. This lesson is that picture.

The lesson is reading. The Hands-on inspects the computation graph of a small expression and measures the wins of `mx.compile` on a real example.

## The computation graph

When you write `c = a @ b + mx.sin(a)` in MLX:

1. `a` and `b` are leaf nodes in the graph (existing arrays).
2. `a @ b` constructs a new node of operation type `matmul` with `a` and `b` as inputs. The node has a known shape and dtype (deduced from the inputs). No computation happens.
3. `mx.sin(a)` constructs a `sin` node with `a` as input. No computation.
4. The `+` constructs an `add` node with the two intermediates as inputs.
5. `c` is the resulting `add` node.

The graph is a DAG: nodes are operations, edges are tensor dependencies, leaves are existing arrays. Every operation in MLX is a node type; the framework has a fixed set (~150 op types in 2026).

When you call `mx.eval(c)`, the runtime:

1. Walks the graph backward from `c` to find all nodes needed.
2. Topologically sorts them.
3. For each node, dispatches the appropriate kernel — on GPU (Metal) or CPU (Accelerate / native) depending on where MLX thinks the op should run.
4. Discards intermediates whose results are no longer needed by any live node.

Between step 2 and step 3, MLX runs *eager fusion*: adjacent element-wise ops that share inputs collapse into a single Metal kernel. `mx.sin(a) + a * 2` becomes one kernel that loads `a` once and produces the result; without fusion it would be three loads of `a`, three Metal kernel launches, and two intermediate writes.

The fusion machinery is constrained: it fuses element-wise ops with each other, fuses element-wise ops onto the output of certain primitives (matmul + add + activation, the classic fusion pattern), and doesn't fuse across reductions or shape-changing ops. The next subsection covers `mx.compile` which extends fusion beyond what eager can do.

## Materialization: when does work actually happen

The list of things that force materialization:

- Explicit: `mx.eval(x)`, `mx.async_eval(x)`.
- Implicit from inspection: `x.tolist()`, `x.item()`, `print(x)`, `x.shape` (no, this is metadata), `bool(x)` (yes, scalar).
- Implicit from interop: passing to NumPy via `np.array(x)`, writing to disk via `mx.savez`.
- Implicit from training: optimizer state updates require parameters to be materialized.

If you never trigger materialization, the graph keeps growing. This is rarely an issue in inference (you `mx.eval` to get the output) but can be a memory leak in training loops if you forget to `mx.eval(net.parameters(), opt.state)` per step. The graph accumulates one iteration's worth of history, then two, then ten.

The diagnostic: if MLX memory usage grows monotonically across training iterations even when batch size is fixed, you're probably not materializing.

## The streams model

MLX's streams are the analog of CUDA streams: ordered queues of operations that the framework dispatches asynchronously. By default there's one stream per device:

- `mx.cpu`: the CPU stream.
- `mx.gpu`: the GPU stream.

Operations on the same stream execute in order. Operations on different streams can overlap.

You can place an op on a specific stream with `mx.eval(..., stream=mx.cpu)` or by using `mx.stream(mx.cpu)` as a context manager. Most code doesn't need to think about this — MLX picks the right stream for each op — but for advanced patterns (overlap CPU preprocessing with GPU compute, or run two independent GPU kernels concurrently) you can be explicit.

The async dispatch is what makes lazy evaluation pay off. When you write a model's forward pass, MLX builds up the graph. When you `mx.eval` the output, the runtime dispatches kernels to the GPU stream and returns *immediately* — the CPU continues to run the next Python statement while the GPU is still working on the previous one. The wall-clock cost of the Python overhead drops out of the inner loop.

This is one of the practical reasons MLX feels faster than PyTorch MPS even when the underlying kernels are similar: PyTorch MPS dispatches synchronously by default; MLX's lazy + async pattern overlaps Python execution with GPU work.

## `mx.compile` for whole-function fusion

`mx.compile` is MLX's analog to `torch.compile`. You wrap a function:

```python
@mx.compile
def step(x, w):
    return mx.sin(x @ w) + x

y = step(some_x, some_w)
mx.eval(y)
```

On first call, MLX traces the function (running it once with the actual input shapes), captures the resulting graph, and compiles a fused version. Subsequent calls with same-shaped inputs reuse the compiled version.

What you get:

- **Wider fusion windows.** The compiler can fuse across operations the eager fuser doesn't (some reductions + element-wise, certain shape changes that are no-ops).
- **Lower per-call overhead.** No graph re-building each call.
- **Better dead-code elimination.** Intermediates that the eager runtime would have produced get removed.

What you give up:

- Dynamic control flow (if/else based on tensor values) breaks compilation; the function must be statically traceable.
- The first call is slower (compilation overhead). Subsequent calls are faster.

For a typical transformer forward pass, `mx.compile` is worth 5–15% throughput. For element-wise-heavy code (custom activations, normalizations chained together), it can be 30%+.

## Memory management

MLX's memory model is built on Metal's `MTLHeap` allocator. The framework maintains a pool of pre-allocated heaps and serves new array allocations from them; freed arrays return memory to the heap pool, not directly to the OS.

This means:

- Peak memory usage is what matters for fitting a workload, not the high-water mark across history.
- A model that uses 8 GB peak memory will show up as "MLX is using 8 GB" in Activity Monitor, then often stay there even after you release the arrays (the heap doesn't shrink).
- `mx.metal.clear_cache()` releases the heap back to the OS.

For inference, this is fine — peak is the constraint. For training, where peak can be 2–4× the model size (because of activations and gradients), this is the relevant memory pressure.

You can query the runtime: `mx.metal.get_active_memory()` returns the current allocation size; `mx.metal.get_peak_memory()` returns peak since the process started.

## How a typical LLM inference call composes

Walking through `model.generate(prompt, max_tokens=128)` with `mlx-lm` on Apple Silicon:

1. **Prefill** (run the whole prompt through the model once to fill the KV cache):
   - `mlx-lm` builds a graph for the whole prompt's forward pass.
   - The graph includes matmuls (Q, K, V projections; attention output; FFN), the attention computation itself, layer norms, RoPE.
   - MLX dispatches the kernels to the GPU stream.
   - The KV cache is materialized into pre-allocated buffers.
2. **Decode loop** (one token at a time):
   - For each token, a new tiny graph: project the most recent token's hidden state to Q/K/V, append K/V to the cache, compute attention against the full cache, run the FFN, sample.
   - The graph is small (~50 ops) but executes many times (128 times for the 128-token generation).
   - `mx.compile` is applied here: the per-token graph is compiled once and reused.
   - Each decode step is one or two GPU dispatches at most.
3. **Detokenize**: the integer token IDs come back to Python; the tokenizer (CPU) converts them to text.

The overall pattern: one big graph for prefill, many tiny compiled graphs for decode. Memory peaks during prefill (large activation tensors); the decode phase is steady-state and bandwidth-bound.

## What you should believe after this lesson

Three sentences:

**1. MLX is a lazy-graph framework: operations build a DAG; nothing executes until you call `mx.eval` or trigger an implicit materialization** like printing or `tolist()`. The graph enables eager fusion of adjacent element-wise ops.

**2. The async dispatch model overlaps Python overhead with GPU compute**, which is one of the practical reasons MLX feels faster than PyTorch MPS even when the underlying kernels are similar. Streams are MLX's CUDA-stream analog and rarely need explicit management.

**3. `mx.compile` extends fusion beyond what eager can do** and is worth 5–15% on transformer inference. The MLX memory model uses pool allocators on top of Metal heaps; peak memory is what matters and `mx.metal.clear_cache()` releases pool memory back to the OS.

## Hands-on (at home)

Inspect a small MLX graph and measure the `mx.compile` win.

```python
# mlx_graph_inspect.py
import mlx.core as mx

a = mx.random.normal((1024, 1024))
b = mx.random.normal((1024, 1024))
c = mx.sin(a @ b) + a * 2  # build a tiny graph
print("c is lazy:", c.shape, c.dtype)  # shape and dtype known, but no eval yet
# Force eval to actually see the value.
mx.eval(c)
print("c[0,0]:", c[0, 0].item())
```

Now measure `mx.compile` on a function with a deeper fusion opportunity.

```python
# mlx_compile_bench.py
import mlx.core as mx
import time

def step(x, w1, w2):
    h = nn_silu(x @ w1)
    return h @ w2 + x  # residual

def nn_silu(x):
    return x * mx.sigmoid(x)

step_compiled = mx.compile(step)

x = mx.random.normal((128, 1024), dtype=mx.float16)
w1 = mx.random.normal((1024, 4096), dtype=mx.float16)
w2 = mx.random.normal((4096, 1024), dtype=mx.float16)
mx.eval(x, w1, w2)

# Warm.
for _ in range(5): mx.eval(step(x, w1, w2))
for _ in range(5): mx.eval(step_compiled(x, w1, w2))

# Bench.
def bench(fn, n=200):
    t0 = time.time()
    for _ in range(n):
        y = fn(x, w1, w2)
        mx.eval(y)
    return (time.time() - t0) * 1000 / n

print(f"eager:    {bench(step):.3f} ms/iter")
print(f"compiled: {bench(step_compiled):.3f} ms/iter")
```

You should see a 5–20% improvement from `mx.compile`, depending on the SoC. The win is bigger on smaller batch sizes (where Python overhead is a larger fraction) and on element-wise-heavy patterns.

Part 3 — verify memory accounting.

```python
# mlx_memory.py
import mlx.core as mx

print(f"initial active: {mx.metal.get_active_memory()/1e6:.1f} MB")

big = mx.random.normal((4096, 4096), dtype=mx.float16)
mx.eval(big)
print(f"after big array: {mx.metal.get_active_memory()/1e6:.1f} MB")
print(f"peak so far:     {mx.metal.get_peak_memory()/1e6:.1f} MB")

del big
print(f"after del:       {mx.metal.get_active_memory()/1e6:.1f} MB")
# Note: del doesn't necessarily release immediately — depends on the pool.
mx.metal.clear_cache()
print(f"after clear:     {mx.metal.get_active_memory()/1e6:.1f} MB")
```

You'll see the heap doesn't shrink on `del`; it shrinks on `clear_cache()`. For long-running processes this is mostly fine (peak is what matters); for memory-constrained scenarios you may need to call `clear_cache` periodically.

## Further reading

- MLX source (github.com/ml-explore/mlx) — `mlx/core/graph_utils.h`, `mlx/core/eval.cpp`, and `mlx/core/compile.cpp` are the most informative files for the internals.
- MLX docs: "Lazy Evaluation," "Compilation," "Unified Memory" — short, well-written pages on each topic.
- Awni Hannun's "MLX Design Choices" talks — direct from the lead author; the most authoritative source on design rationale.

Next lesson: **Custom Metal kernels from MLX.** When MLX's built-in primitives aren't enough, you can write a custom MSL kernel and call it from MLX with reasonable ergonomics. This is the path for writing the kinds of fused INT4 matmul or FlashAttention variants that ship in `mlx-lm` and its ecosystem.
