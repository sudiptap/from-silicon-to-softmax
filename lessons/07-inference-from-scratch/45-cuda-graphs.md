---
title: "Lesson 45 — CUDA Graphs for Decode"
date: "2026-06-04"
module: "inference-from-scratch"
order: 45
tags: ["cuda-graphs", "kernel-launch", "decode", "overhead", "optimization"]
author: "Sudipta Pathak"
prerequisites: ["44-radix-attention"]
---

# Lesson 45 — CUDA Graphs for Decode

## Why this lesson exists

Each decode step launches dozens of small CUDA kernels: matmul kernels, attention kernel, layer-norm kernel, activation kernel, etc. Each launch has a fixed overhead — typically 5-15 microseconds on modern GPUs. For a 32-layer model with ~10 kernels per layer, that's 32 × 10 × 10 μs ≈ 3.2 ms per decode step just in launch overhead.

For a fast decode workload (e.g., 7B model at 100 tok/s = 10 ms per token), kernel-launch overhead can be 30-50% of the per-token time. Eliminating it is a real throughput win.

CUDA Graphs (NVIDIA, 2018; matured in 2022-2024 for ML use cases) captures the kernel-launch sequence once into a graph and replays the graph without per-launch overhead. After capture, launching the entire decode step is one API call instead of dozens.

This lesson covers CUDA Graphs as applied to LLM decode and the production deployments that use them.

The lesson is reading. The Hands-on captures and replays a simple computation graph.

## The kernel-launch overhead problem

Each `cudaLaunchKernel` (or PyTorch's `op.cuda.launch`) involves:
- The host (CPU) thread submitting the launch request.
- The CUDA driver validating arguments and packaging them.
- The driver placing the work on the GPU's command queue.
- The GPU pulling and dispatching the kernel.

Each step takes microseconds. For latency-critical workloads with many small kernels, the overhead dominates.

For LLM decode at batch 1:
- Many kernels are very small (a `[1, D] @ [D, D]` matmul takes microseconds of actual compute).
- The launch overhead is proportionally large.
- The GPU spends much of its time idle, waiting for the next kernel to arrive.

For larger batches, each kernel does more work, so the overhead is a smaller fraction. CUDA Graphs is most impactful at batch 1.

## What CUDA Graphs do

The CUDA Graphs API:

1. **Capture**: run the decode step under a "capture" mode. Each kernel launch is recorded into a graph rather than executed.
2. **Instantiate**: compile the graph into a launch-ready form.
3. **Launch**: replay the entire graph in one API call. The GPU executes all the kernels back-to-back with no per-launch overhead.

The cost is paid once during capture; subsequent launches are fast.

The graph captures both the kernels and their dependencies. The GPU executes them in the correct order with optimal scheduling.

## What CUDA Graphs save

Empirically:
- For LLM decode at batch 1, CUDA Graphs eliminates ~50-80% of the kernel-launch overhead.
- Net throughput improvement: 10-30% on bandwidth-bound decode workloads where the launch overhead was a significant fraction.

For larger batches: smaller improvement (overhead is already amortized).

## Constraints

CUDA Graphs has constraints:
- **Tensor shapes must be fixed** during capture. If shapes change (e.g., different sequence lengths each step), you need a separate graph per shape combination.
- **No dynamic control flow** (if/else based on tensor values; loops with data-dependent bounds).
- **No in-place mutation of captured tensors** between launches.

For LLM decode:
- Sequence length (and hence KV cache size) grows each step. Either capture multiple graphs (one per padded sequence length) or use careful workarounds.
- Different batch sizes need different graphs.

vLLM and TensorRT-LLM handle this by capturing graphs for a *padded* set of common shapes and routing each request to its matching graph.

## Production deployments

- **vLLM**: enabled by default for decode steps. Specifically captures graphs for power-of-2 batch sizes; falls back to eager mode for unusual shapes.
- **TensorRT-LLM**: integrates CUDA Graphs deeply; nearly all kernels are graph-captured.
- **SGLang**: similar to vLLM.
- **llama.cpp**: has CUDA Graphs support for decode (recent addition).

In 2026 the technique is standard. If your runtime doesn't use CUDA Graphs, you're leaving 10-30% throughput on the table.

## What you should believe after this lesson

Three sentences:

**1. Kernel-launch overhead is a meaningful fraction of LLM decode time at batch 1** — ~3 ms per step for a typical 32-layer model. Each of the dozens of small kernels per step has ~10 μs of launch overhead; it adds up.

**2. CUDA Graphs capture the kernel-launch sequence once and replay it without per-launch overhead.** The graph executes all kernels back-to-back; launch cost is a single API call instead of dozens. Throughput improvement: 10-30% on bandwidth-bound decode.

**3. Production runtimes (vLLM, TensorRT-LLM, SGLang) use CUDA Graphs by default** for decode steps. The constraints (fixed shapes, no data-dependent control flow) are managed by capturing multiple graphs for different shape configurations.

## Hands-on (at home)

A toy CUDA Graphs capture and replay (requires NVIDIA GPU).

```python
# cuda_graphs_demo.py
import torch
import time

device = 'cuda'
N = 1024
x = torch.randn(N, N, device=device)
w = torch.randn(N, N, device=device)

# Without CUDA Graphs.
torch.cuda.synchronize()
t0 = time.time()
for _ in range(1000):
    y = (x @ w).relu()
    y = y @ w
torch.cuda.synchronize()
t_naive = time.time() - t0
print(f"Naive 1000 iter: {t_naive*1000:.2f} ms")

# With CUDA Graphs.
stream = torch.cuda.Stream()
stream.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(stream):
    # Warmup.
    for _ in range(3):
        y = (x @ w).relu()
        y = y @ w
torch.cuda.current_stream().wait_stream(stream)

# Capture.
graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph):
    y = (x @ w).relu()
    y = y @ w

# Replay.
torch.cuda.synchronize()
t0 = time.time()
for _ in range(1000):
    graph.replay()
torch.cuda.synchronize()
t_graph = time.time() - t0
print(f"CUDA Graph 1000 iter: {t_graph*1000:.2f} ms")
print(f"Speedup: {t_naive / t_graph:.2f}x")
```

For small kernels and many iterations, you should see meaningful speedup (1.5-3×). For large kernels (compute-bound), the speedup shrinks.

For real LLM decode benchmarks, vLLM's `--enforce-eager` flag disables CUDA Graphs; compare with vs without to see the production impact.

## Further reading

- NVIDIA's CUDA Graphs documentation.
- "Accelerating PyTorch with CUDA Graphs" (NVIDIA blog).
- vLLM source: `vllm/worker/model_runner.py` for the CUDA Graphs capture logic.
- TensorRT-LLM's documentation on graph capture.

Next lesson: **Tensor parallelism for inference.** TP for training was developed by Megatron-LM; TP for inference has different constraints and tradeoffs. We close Part 7 with the inference-specific TP story.
