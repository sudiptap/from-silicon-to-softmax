---
title: "Lesson 12 — Profiling GPU Kernels"
date: "2026-06-03"
module: "gpu-computing"
order: 12
tags: ["profiling", "nsight", "xcode", "metal-system-trace", "roofline"]
author: "Sudipta Pathak"
prerequisites: ["11-flashattention-apple"]
---

# Lesson 12 — Profiling GPU Kernels

## Why this matters

You have kernels. You have benchmarks. What you don't yet have is the ability to look at a kernel and say, with precision, "this is at 78% of peak memory bandwidth, register pressure is fine, occupancy is 50%, and the bottleneck is L2 cache hit rate." That's what GPU profilers give you, and without them, every optimization is guesswork.

On NVIDIA, the modern profiler is **Nsight Compute** (for per-kernel deep-dive) plus **Nsight Systems** (for end-to-end timeline). On Apple Silicon, it's **Xcode Instruments → Metal System Trace** plus the **GPU Frame Capture** debugger. Both ecosystems give you the same questions — *what's the bottleneck, where are the cycles, where is memory* — answered with their own tooling.

This lesson covers the questions you ask, the metrics that answer them, and the tools that produce the metrics. After it, the GFLOPs numbers from Lessons 4–11 turn into structured diagnoses.

## Concept

### The questions

Three questions cover 90% of GPU performance diagnosis:

1. **Am I memory-bound or compute-bound?** If memory-bound, optimize memory access. If compute-bound, optimize the inner loop.
2. **What fraction of the relevant peak am I hitting?** A "memory-bound" kernel at 90% of peak bandwidth is done; one at 30% has work to do.
3. **What's the time breakdown by kernel?** End-to-end, how much wall time is each kernel costing me, and what's the launch overhead?

The first two are per-kernel; the third is whole-program. You need both kinds of tooling.

### The Roofline model

A useful framework for question 1: plot every kernel as a dot on a 2D graph where the X-axis is arithmetic intensity (FLOPs/byte) and the Y-axis is achieved throughput (TFLOPs). The "roofline" is two lines meeting in a corner — the bandwidth ceiling (slope = memory bandwidth) for low arithmetic intensity, the FLOP ceiling (flat at peak TFLOPs) for high arithmetic intensity.

Kernels that are far below their roofline have headroom. Kernels at the roofline are saturated and can't go faster without algorithmic change. The roofline tells you *which* ceiling you're hitting and *how close* you are.

Nsight Compute generates this plot automatically for each kernel — the "Speed of Light" report. It's the single most useful metric a beginner can use to triage kernels.

### NVIDIA tooling

**Nsight Compute (`ncu`)** is the per-kernel profiler. Launch it with your binary:

```bash
ncu --set full -o profile ./my_kernel
```

It captures a per-kernel report including:
- Speed of Light: % of bandwidth and % of compute used.
- Occupancy: warps active / warps max.
- Memory throughput: by hierarchy level (HBM, L2, L1, shared).
- Instruction throughput: per-pipeline (FP32, FP64, INT, special).
- Roofline plot.
- Source-line attribution if compiled with `-lineinfo`.

For a quick scan during development:

```bash
ncu --set basic ./my_kernel    # short report
ncu --set detailed ./my_kernel # full breakdown
```

The GUI version (`ncu-ui`) opens the same profile in a navigable interface — vastly better for digging into details.

**Nsight Systems (`nsys`)** is the whole-system / timeline view:

```bash
nsys profile -o timeline ./my_program
nsys-ui timeline.nsys-rep
```

Shows you:
- CUDA API calls (launches, memcpys) on a host timeline.
- Kernel executions on each GPU stream.
- CPU/GPU overlap or its absence.
- NVTX ranges if you've annotated your code.

This is where you find that your kernel is fast but you launched it 1000 times when you should have launched it 10 times, or that there's a 5 ms PCIe transfer between two 0.5 ms kernels.

### Apple tooling

**Instruments → Metal System Trace.** Apple's equivalent of Nsight Systems. Open Xcode, then Xcode → Open Developer Tool → Instruments. Choose the "Metal System Trace" template. Either attach to a running process or launch one. You see:
- CPU and GPU activity on a unified timeline.
- Metal commands (encoders, buffers, kernels).
- Per-kernel duration and shape.
- Memory bandwidth usage (recent macOS versions).

**Xcode GPU Frame Capture.** The deep-dive equivalent of Nsight Compute. While your program is running, click Xcode's Capture GPU Frame button. You get:
- A frame's worth of GPU work as a navigable tree.
- Per-encoder timing.
- Per-kernel breakdown.
- Shader source view (with the compiled AIR / MSL).
- GPU counters (memory throughput, ALU utilization on supported macOS).

The depth on Apple is genuinely less than NVIDIA's tooling — fewer counters exposed, less detail on bank conflicts, no roofline plot built in. But the basics are well covered.

**`sudo powermetrics --samplers gpu_power -i 100`** gives a low-overhead view of GPU utilization during a benchmark. Useful for "is the GPU pegged or idling" sanity checks.

### Key metrics, side by side

| Question | NVIDIA (ncu) | Apple (Xcode) |
| -------- | ------------ | ------------- |
| % memory bandwidth used | `dram__throughput.avg.pct_of_peak_sustained_elapsed` | "Memory bandwidth" in GPU Frame Capture (varies by macOS) |
| % compute used | `sm__pipe_tensor_op_fma_cycles_active.avg.pct_of_peak_sustained_elapsed` | "ALU utilization" |
| Occupancy | `sm__warps_active.avg.pct_of_peak_sustained_elapsed` | "Threadgroup occupancy" |
| L2 hit rate | `lts__t_sector_hit_rate.pct` | Not directly exposed |
| Shared memory bank conflicts | `l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum`, etc. | Not directly exposed |
| Kernel duration | `gpu__time_duration.sum` | Per-kernel timing in Frame Capture |

Apple exposes less than NVIDIA. For deep kernel-level optimization, NVIDIA's tooling is still ahead.

## Code walkthrough

A practical workflow for diagnosing a Triton kernel.

### 1. Time it first

```python
import time, torch, triton
# ... kernel definition ...

a = torch.randn(4096, 4096, device='cuda', dtype=torch.float16)
b = torch.randn(4096, 4096, device='cuda', dtype=torch.float16)
matmul(a, b)  # warm
torch.cuda.synchronize()

t0 = time.time()
for _ in range(20):
    matmul(a, b)
torch.cuda.synchronize()
print((time.time() - t0) / 20 * 1000, "ms")
```

This gives you the number. If the number is 1× cuBLAS, you're done. If it's 2× cuBLAS, you have work.

### 2. Roofline diagnosis with ncu

```bash
ncu --set roofline --target-processes all -o matmul_roof python bench.py
ncu-ui matmul_roof.ncu-rep
```

Open in GUI. Look at the Roofline page. Your kernel is a dot. Is it:

- **Below the bandwidth roof?** Bandwidth-bound. Optimize access patterns, increase reuse, increase block size to amortize loads.
- **Below the compute roof but above the bandwidth roof?** Compute-bound, but not at peak. Probably an underused instruction unit; check tensor-core utilization.
- **At the corner (roughly compute-bound and bandwidth-bound at the same time)?** Well-balanced kernel. Marginal gains require both sides moving.

### 3. Drill in with `--set full`

```bash
ncu --set full --target-processes all -o matmul_full python bench.py
```

Open in GUI. Notable pages:

- **Speed of Light**: Shows % of peak for each major metric.
- **Memory Workload Analysis**: Per-hierarchy bandwidth usage.
- **Scheduler Statistics**: Whether warps are stalling.
- **Instruction Statistics**: What types of instructions dominate.
- **Source Counters**: If `-lineinfo` was used at compile, source-line attribution.

### 4. Apple equivalent

For a Triton-class kernel via MLX:

1. Run your benchmark in a small Mac app or script.
2. Open Xcode → Capture GPU Frame (icon in the debug area, or `Cmd+Opt+G`).
3. Navigate the frame: encoder → command buffer → compute pipeline.
4. Per-kernel: see duration, dispatched threadgroups, shader.

For end-to-end timeline:
1. Instruments → Metal System Trace.
2. Record a few seconds of the benchmark.
3. Look at the GPU lane — see kernel sequences, gaps (CPU bottlenecks), overlap.

## Mental model & pitfalls

Single sentence: **A profiler turns "this kernel feels slow" into "this kernel is at 30% of peak HBM bandwidth, with 80% L2 hit rate and high warp stall on memory dependency" — actionable, instead of guesswork.**

Pitfalls:

- **Profiling at wrong granularity.** Per-kernel deep-dive (`ncu`) on a 100-kernel program produces 100 reports and no insight. Use a system trace (`nsys`) first to find the top 3 kernels, then drill in.
- **Not warming up.** The first kernel invocation includes JIT compilation, cache cold-fills, and one-time setup. Always warm.
- **Measuring with `time.time()` and trusting the number to 5%.** Use CUDA events or `torch.cuda.synchronize` carefully; small numbers are noisy.
- **Targeting metrics in isolation.** Maximizing occupancy at the cost of register-tile size can hurt overall throughput. Always check end-to-end timing alongside per-metric optimization.
- **Misreading the roofline.** A kernel "at the roof" is at the *current* roof, not the theoretical peak of the chip — which can be much higher if a different precision or instruction mix is used. Knowing which precision your roofline reflects matters.
- **Apple's lower-fidelity counters.** Some questions you'd ask Nsight aren't answerable on Apple. For experimental work, develop and tune on NVIDIA, then port and verify on Apple.

## Hands-on (at home)

1. **Profile your tiled CUDA matmul** from Lesson 5 with `ncu --set basic`. Note:
   - DRAM throughput as % of peak.
   - SM throughput as % of peak.
   - Achieved occupancy.

2. **Compare to the naive matmul** from Lesson 4. The naive should show:
   - Higher DRAM throughput (bandwidth saturated wastefully).
   - Lower SM throughput.
   - Similar occupancy.

3. **Profile a Triton matmul** under the same conditions:

```bash
ncu --set roofline --import-source yes python bench_triton.py
```

The roofline plot should show Triton's kernel closer to the corner (well-balanced).

4. **Run `nsys profile` on a small Transformer** (a single attention + MLP layer in PyTorch):

```bash
nsys profile -o transformer python transformer_step.py
nsys-ui transformer.nsys-rep
```

Identify the kernels by name. Note total time per kernel. Often >50% of time is in one or two kernels; those are the targets for hand-optimization or fusion.

5. **(Mac) Run Xcode Metal System Trace** on an MLX inference loop:

```python
import mlx.core as mx
# ... load a model, run generation ...
```

Click Capture GPU Frame mid-run. Navigate the frame. Find the attention kernel; check its duration. Compare to what you measured wall-clock.

6. **(Optional) Annotate with NVTX** for cleaner system traces:

```python
import torch.cuda.nvtx as nvtx

with nvtx.range("attention_layer"):
    out = attention(q, k, v)
```

These ranges show up in `nsys` timelines as named blocks, making the trace much easier to read.

## Further reading

- *Nsight Compute User Guide* (NVIDIA docs) — exhaustive, slow read but the authoritative source.
- *Nsight Systems User Guide* — same, for system-level tracing.
- Apple's Metal Debugging Tools documentation — Apple's equivalent.
- "Roofline: An Insightful Visual Performance Model for Multicore Architectures" (Williams et al.) — the foundational paper.
- "How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance: a Worklog" (Simon Boehm) — applies the profiler workflow throughout. Highly recommended.

Next lesson: the module wrap. We assemble the matmul + attention + tooling story into a coherent picture, identify the remaining gap to peak, and set up Module 3 (ML Internals & Optimization).
