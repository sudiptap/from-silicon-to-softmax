---
title: "Lesson 12 — Module Wrap: From 1 GFLOPs to 50+ GFLOPs"
date: "2026-06-03"
module: "bare-metal"
order: 12
tags: ["matmul", "wrap", "perf", "summary"]
author: "Sudipta Pathak"
prerequisites: ["11-memory-numa"]
---

# Lesson 12 — Module Wrap: From 1 GFLOPs to 50+ GFLOPs

## Why this lesson exists

A 12-lesson arc deserves a summing-up. We need to look at the whole journey, see exactly how many multipliers we stacked, identify the remaining gap to peak hardware throughput, and frame what comes next in Module 2 (GPU & Parallelism).

This lesson is short on new technical content and long on synthesis. Read it on a phone over coffee. The "Hands-on" at the end consolidates the entire module project into a single benchmark script you can run for a final number.

## The journey, replayed

We started with the dumbest possible matmul: three nested loops, row-major data, no thought given to anything. At a 512-cube problem on a single P-core, that gets ~1 to 5 GFLOPs depending on machine. The hardware can do 50 to 100+ GFLOPs per core on FP32. We were at 1–5% of peak.

Across 11 lessons we applied, in order, the following multipliers:

| Lesson | Technique | Typical speedup over prior |
| -- | --------- | -------------------------- |
| 4 | Loop reorder (`i, k, j`) | ~3–5× |
| 5 | Cache blocking (~64-tile) | ~3–4× |
| 6 | SIMD AVX2 (or implicit auto-vectorization) | ~2–4× |
| 7 | NEON on Apple Silicon | (sideways — same or better) |
| 8 | Multi-threading (8 P-cores) | ~6–8× |
| 11 | Huge pages, allocator hygiene | ~1.1–1.3× |

Multiply those together: 4 × 3.5 × 3 × 7 × 1.2 ≈ **350×**. Starting from ~1.5 GFLOPs, that's ~500 GFLOPs aggregate across threads on a typical modern multi-core CPU. Single-core: 50–100 GFLOPs, which is in the right ballpark for "as fast as a well-tuned single-threaded matmul on a CPU."

Compared to vendor BLAS:

- On x86 with MKL or OpenBLAS: vendor BLAS typically hits 600–900+ GFLOPs aggregate on a modern 8-core desktop. Our hand-tuned version lands at 60–80% of that. Gaps are: more aggressive packing, better edge-tile handling, multi-level (3-level) blocking, kernel-specific tuning per microarchitecture.

- On Apple Silicon with Accelerate: vendor BLAS uses AMX, which we don't have access to. Single-core Accelerate can hit 200+ GFLOPs (NEON-only is bounded around 100). Our hand-NEON lands at ~40–50% of single-core Accelerate. The AMX gap is structural — not closeable from user-mode without unofficial instructions.

These are the honest numbers. We are not *as fast as* the best library, but we *understand why*. That understanding is the deliverable of this module.

## What we did, what we skipped

We covered:

- The CPU's mental model — pipelines, ILP, branch prediction, OoO.
- The memory hierarchy — caches, lines, prefetchers, TLB, the latency table.
- Rust enough to write systems code without footguns.
- The naive matmul as the canonical bad-baseline.
- Cache blocking — the biggest single multiplier.
- SIMD on x86 (AVX2) and ARM (NEON), the inner kernel structure.
- Multi-threading with Rayon, work-stealing, false sharing, NUMA.
- Profiling with `perf` (Linux) and Instruments / `samply` (macOS).
- Memory profiling, page faults, huge pages, NUMA placement.

We deliberately skipped:

- **Packing details.** We hand-waved A and B packing. A serious matmul library has 100–300 lines just on packing kernels and panel layout. The performance difference is meaningful (10–30%) but the conceptual content is small. The BLIS papers cover it well.
- **Multi-level (3-level) blocking.** We did one block level (~64). Production code blocks for register file → L1 → L2 → L3, four levels. The structure repeats; we'd add 100 lines of code for 10–20% more performance.
- **Mixed-precision (bf16, fp16, int8).** All our work was f32. Reducing precision unlocks more SIMD width (twice as many lanes per register), which is exactly how vendor BLAS implementations get the headline "fp16 hits 2 TFLOPs" numbers on bigger CPUs. We'll come back to this in Module 3 (ML Internals & Quantization).
- **The Rust GPU frontier.** We compared single-core CPU performance only; the next module is the GPU's turn.
- **Apple AMX.** Deliberately — see Lesson 7. The AMX gap is the answer to "why are we 2.5× behind Accelerate," and we acknowledged it rather than pretended to close it.

## Mental models to carry forward

Three sentences worth memorizing.

**The CPU is fast; memory is slow; caches make memory look fast when accessed predictably.** Module 2 reframes this for GPUs (where the memory hierarchy is different but the principle is the same: keep hot data close).

**Optimizing numeric code is mostly about reducing memory pressure, not reducing flops.** This is true for matmul, convolution, attention, MoE — every important ML kernel. FlashAttention is exactly this insight, applied to attention.

**SIMD and threading multiply; cache discipline multiplies them again.** When the GPU lesson talks about thousands of threads, the same compositional logic applies — except the GPU's memory hierarchy and ILP work differently, with different costs.

## What's next

**Module 2: GPU & Parallelism.** We rebuild a fast matmul on the GPU. The setup is similar — naive kernel, blocked kernel, tiled-with-shared-memory kernel — but the language is CUDA (for NVIDIA), Triton (for cross-vendor ease), or Metal Shading Language (for Apple Silicon). The performance ceiling jumps by 1–3 orders of magnitude. The diagnostic tools change (Nsight on NVIDIA, Xcode Metal debugger on Apple).

**The matmul project carries forward.** The same `Mat` / `MatMut` types, the same benchmark harness, the same correctness check. We add GPU implementations alongside the CPU ones and watch the speedups.

**On-device work picks up in Module 4.** The CPU work from this module remains directly useful — many on-device LLM inference engines (llama.cpp, MLX-LM) split work between the CPU (for fast pre/post-processing, small problem sizes, edge tiles) and the GPU/ANE (for the matmul-heavy core). Knowing both halves is the point.

## Hands-on (at home)

The final benchmark of the module. Write one program that runs every variant we've built, prints a comparison table, and writes it to a CSV for posterity.

```rust
use std::time::Instant;
mod matmul;
use matmul::*;

fn bench<F: FnMut()>(name: &str, gflops_factor: f64, mut f: F, runs: usize) -> (f64, f64) {
    f(); // warm
    let t0 = Instant::now();
    for _ in 0..runs { f(); }
    let dt = t0.elapsed().as_secs_f64() / runs as f64;
    let gflops = gflops_factor / dt / 1e9;
    println!("{name:>22}: {:>7.2} ms   {gflops:>7.2} GFLOPs", dt * 1e3, gflops);
    (dt * 1e3, gflops)
}

fn main() {
    let (m, k, n) = (1024usize, 1024, 1024);
    let a_buf = make_matrix(m, k, 1);
    let b_buf = make_matrix(k, n, 2);
    let mut c_buf = vec![0.0f32; m * n];

    let a = Mat { rows: m, cols: k, data: &a_buf };
    let b = Mat { rows: k, cols: n, data: &b_buf };
    let mut c = MatMut { rows: m, cols: n, data: &mut c_buf };

    let flops = 2.0 * m as f64 * n as f64 * k as f64;

    println!("\n=== Matmul ({m}x{k}) * ({k}x{n}) — single-core unless noted ===\n");

    bench("naive (ijk)", flops, || {
        c.data.iter_mut().for_each(|x| *x = 0.0);
        matmul_naive(&a, &b, &mut c);
    }, 3);

    bench("loop-reorder (ikj)", flops, || {
        c.data.iter_mut().for_each(|x| *x = 0.0);
        matmul_ikj(&a, &b, &mut c);
    }, 3);

    bench("blocked (BLOCK=64)", flops, || {
        c.data.iter_mut().for_each(|x| *x = 0.0);
        matmul_blocked(&a, &b, &mut c);
    }, 5);

    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        bench("AVX2+FMA", flops, || {
            c.data.iter_mut().for_each(|x| *x = 0.0);
            unsafe { matmul_avx2(&a, &b, &mut c); }
        }, 10);
    }

    #[cfg(target_arch = "aarch64")]
    {
        bench("NEON", flops, || {
            c.data.iter_mut().for_each(|x| *x = 0.0);
            unsafe { matmul_neon(&a, &b, &mut c); }
        }, 10);
    }

    bench("threaded blocked", flops, || {
        c.data.iter_mut().for_each(|x| *x = 0.0);
        matmul_threaded(&a, &b, &mut c);
    }, 10);

    // Best-of-all: threaded + SIMD inner kernel.
    bench("threaded SIMD", flops, || {
        c.data.iter_mut().for_each(|x| *x = 0.0);
        matmul_threaded_simd(&a, &b, &mut c);
    }, 10);

    // Vendor BLAS for ground truth.
    #[cfg(target_os = "macos")]
    bench("Accelerate (AMX)", flops, || {
        c.data.iter_mut().for_each(|x| *x = 0.0);
        matmul_accelerate(&a, &b, &mut c);
    }, 10);

    println!("\nDone.");
}
```

Run on your machine. You should see a clean ladder of speedups, ending at the vendor BLAS line.

**Save the output.** Commit it to your local notes. When you start Module 2 and your GPU version eventually beats `Accelerate (AMX)` by 5×, you'll want this number to compare against.

**(Optional)** Write a CSV with rows for each variant and columns for problem size — sweep 256, 512, 1024, 2048, 4096. Plot GFLOPs vs problem size. Each variant should hit its own roof at a different problem size: naive starts dying around 1024, blocked extends to 4096 cleanly, the SIMD+threaded version should sustain peak across the range. This plot tells you something the single-point numbers don't: where each technique stops mattering.

## Further reading

- BLIS framework paper: "BLIS: A Framework for Rapidly Instantiating BLAS Functionality" — the modular structure of a production matmul.
- "Anatomy of High-Performance Matrix Multiplication" (Goto & van de Geijn) — the foundational design.
- Robert van de Geijn's *How To Optimize GEMM* (his GitHub repo) — the most readable hands-on walkthrough; we built a tiny version of his arc.
- "Per-CPU optimization through differential analysis" (various) — the CPU-tuning literature you'd dig into for the last 20%.

End of Module 1.

Module 2 — **GPU & Parallelism** — picks up next.
