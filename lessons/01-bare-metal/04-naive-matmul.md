---
title: "Lesson 4 — Naive Matrix Multiply"
date: "2026-06-03"
module: "bare-metal"
order: 4
tags: ["matmul", "naive", "baseline", "rust", "perf"]
author: "Sudipta Pathak"
prerequisites: ["03-rust-for-systems"]
---

# Lesson 4 — Naive Matrix Multiply

## Why this matters

The triple-loop matmul is the *Hello, World* of numerical computing — and the most instructive example of how far code can be from the hardware it's running on. The version every textbook prints is also the slowest version you will ever write that returns the right answer. Across the next four lessons we will beat it by approximately 100× without changing the algorithm. The first step is to stand still and look at it: measure it, and explain in cache-and-CPU terms why it is bad.

This lesson does not introduce new ideas — it consolidates the two we have. Lesson 1 said "the CPU runs many operations in parallel when dependencies allow." Lesson 2 said "cache misses are 100× more expensive than hits." We will use both to predict, before we measure, how the naive matmul should perform — and then check.

## Concept

### What a matmul does

Given matrices `A` (m × k), `B` (k × n), the product `C = A · B` is `m × n`, with each entry defined as

```
C[i, j] = Σ_{kk=0..k}  A[i, kk] * B[kk, j]
```

For each of the `m × n` output cells, we do `k` multiply-accumulates. Total floating-point ops: `2 · m · n · k` (each multiply and each add counts as one).

For a 512×512×512 problem, that's `2 · 512³ ≈ 268 million` flops. Tiny by modern standards — a single CPU core's peak throughput is ~50 billion flops per second, so this should take a few milliseconds. We saw last lesson that it actually takes ~85 ms. The gap is real and it is entirely about memory.

### The naive triple loop

```rust
for i in 0..m {
    for j in 0..n {
        let mut acc = 0.0f32;
        for kk in 0..k {
            acc += A[i*k + kk] * B[kk*n + j];   // <-- the inner loop
        }
        C[i*n + j] = acc;
    }
}
```

The order of the loops is `i, j, kk` — for each output cell `(i, j)`, walk the entire reduction dimension `kk`.

Now look at the memory access pattern in the inner loop:

- `A[i*k + kk]` for `kk = 0, 1, 2, ...` — we walk **row `i` of A** from left to right. Stride 1. Cache-friendly. Hardware prefetcher loves it.
- `B[kk*n + j]` for `kk = 0, 1, 2, ...` — we walk **column `j` of B** from top to bottom. Stride `n` (in elements) = stride `4n` bytes. For `n = 512`, that's 2048-byte stride. **Every single access is a different cache line.**

So out of `k` inner-loop iterations, the `A` accesses get ~16 cache hits for every miss (one miss per 16 floats per 64-byte line), while the `B` accesses miss every time. The inner loop spends most of its time waiting on memory.

### Predicting the actual cost

Let's do the calculation. For a 512×512×512 matmul:

- Inner loop runs `k = 512` times per (i, j). It does 1 multiply + 1 add (one FMA = 2 flops).
- Of those 512 iterations, all 512 hit a fresh `B` cache line. So we get `m · n · k = 134 million` cache misses on `B` alone.
- A cache miss to DRAM is ~80 ns. `134M × 80 ns = 10.7 seconds` if every miss went all the way to DRAM. The fact that we see 85 ms instead means **most** of those `B` accesses hit L2 or L3 (since `B` is only 1 MB and fits in L3), not DRAM. But many hit L2 (~5 ns) rather than L1 (~1 ns), and that's still ~5× slower than L1.

A more useful frame: the working set for the inner loop is "all of `B`" — 1 MB. That fits in L3 on every modern CPU, doesn't fit in L1 (32 KB), and may or may not fit in L2 depending on the CPU. So we are essentially L2-bound. The CPU's matrix multiplier *would like* to do an FMA every cycle (50+ GFLOPs achievable); it's gated by waiting for `B` from L2/L3 at maybe 1/10th that rate. Hence the ~1–5 GFLOPs we measured.

This is the right way to think about it. We're not "slow" because the loop is bad. We're slow because we are spending most of our cycles waiting for L2/L3 to return cache lines for `B`.

### Why this is also instructive

The naive matmul exposes three patterns that recur across all numeric optimization:

1. **The hot data structure is bigger than your fast cache.** Fix: process it in tiles that do fit.
2. **One of your operands has a hostile access pattern.** Fix: transpose, or reorder loops, or block.
3. **Arithmetic is cheap, memory is expensive.** Fix: reuse each loaded value as many times as possible before evicting it.

The next lesson (cache blocking) applies all three.

## Code walkthrough

Last lesson's `matmul_naive` is what we're measuring. Let's add a slightly subtler variant — one that demonstrates how loop order changes everything — and a small instrumentation pass that helps us reason about the numbers.

### Loop-order variants

```rust
// I-J-KK (the one we wrote): innermost reads B by column. Bad.
pub fn matmul_ijk(a: &Mat, b: &Mat, c: &mut MatMut) {
    for i in 0..a.rows {
        for j in 0..b.cols {
            let mut acc = 0.0f32;
            for kk in 0..a.cols {
                acc += a.at(i, kk) * b.at(kk, j);
            }
            *c.at_mut(i, j) += acc;
        }
    }
}

// I-KK-J: innermost reads B by row! Same flops, very different cache behavior.
pub fn matmul_ikj(a: &Mat, b: &Mat, c: &mut MatMut) {
    for i in 0..a.rows {
        for kk in 0..a.cols {
            let a_ik = a.at(i, kk);
            for j in 0..b.cols {
                *c.at_mut(i, j) += a_ik * b.at(kk, j);
            }
        }
    }
}
```

`matmul_ikj` hoists the `A` access out of the innermost loop (one load per `kk`, not per `j`) and walks both `B` and `C` by row in the innermost loop. Both inner accesses are stride 1. The prefetcher loves it. **This single reordering, without any algorithmic change, often gives a 3–5× speedup on the naive 512-cube problem.**

### Allocation discipline

Notice what's *not* in the inner loop: any `Vec::new`, any `.collect`, any `format!`, any conversion that allocates. The accumulator is on the stack. The matrices are pre-allocated. This is the systems discipline — every byte of allocation per inner-loop iteration is a chunk of the budget gone. Most "this is slow in Python" results are dominated by allocations that the equivalent Rust simply doesn't do.

### A note on FMA

`acc += a * b` is a *fused multiply-add*. Modern CPUs (x86 FMA3, ARM NEON FMA) can do this as a single instruction, single cycle, with one rounding step instead of two. The compiler emits this for us automatically when targets allow. We get the "2 flops per FMA" benefit just by counting; the hardware does the heavy lifting.

You can verify FMA is being emitted by looking at the assembly: you'll see `vfmadd231ss` (x86) or `fmla` (ARM) where the inner loop lives. If you see separate multiply and add (`vmulss` then `vaddss`), the compiler decided FMA wasn't safe — usually a target-feature issue. We'll come back to this in the SIMD lessons.

## Mental model & pitfalls

One sentence: **The naive matmul is slow because the inner loop reads `B` along the cache-hostile dimension, and that one fact dominates everything else.**

Pitfalls when measuring:

- **Forgetting to clear `C` between runs.** If you accumulate, you're measuring something different each iteration; you'll see misleading slowdowns or speedups.
- **Forgetting to warm caches.** First call into a fresh function with cold caches reads everything from DRAM. Average across multiple runs after a warm-up pass.
- **Letting the compiler optimize the call away.** If `main` doesn't use the result of `C`, the compiler may delete the whole call. Read the assembly the first time, or use `std::hint::black_box`.
- **Measuring with debug builds.** `cargo run` without `--release` is 10–50× slower; the bottleneck shifts away from cache to interpreter-like overhead. Always `--release` for performance numbers.
- **Cherry-picking matrix sizes.** A 1024 cube can be much slower per flop than a 1023 cube because 1024 is a power of 2 and cache-aliases brutally. We'll see this in the next lesson.

## Hands-on (at home)

Extend the project from Lesson 3.

1. **Add `matmul_ikj`** alongside `matmul_naive` in `src/matmul.rs`.
2. **Bench both** in `main.rs`, on the 512-cube problem.

```rust
bench("matmul_ijk", flops, || {
    for x in c.data.iter_mut() { *x = 0.0; }
    matmul_naive(&a, &b, &mut c);
});

bench("matmul_ikj", flops, || {
    for x in c.data.iter_mut() { *x = 0.0; }
    matmul_ikj(&a, &b, &mut c);
});
```

3. **Run and compare.** Expected: `ikj` is somewhere between 2× and 6× faster than `ijk`. Both are the "same algorithm" by your textbook. The only difference is access pattern.

4. **Verify correctness.** Add a check that the two functions produce the same `C` (allowing for floating-point order-of-operations differences):

```rust
fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0_f32, f32::max)
}
```

You should get a difference under `1e-3` for random matrices of this size. If you get exactly zero, you may have asked the compiler to elide one of the calls — re-check.

5. **Try non-square sizes.** Run both on (256, 1024, 256), (1024, 256, 1024), (768, 768, 768). The ratio of `ikj` to `ijk` should vary — and so should the absolute GFLOPs. Try to predict, *before* you run, which size will be fastest in GFLOPs and why.

**Optional**: try a few sizes that are *almost* powers of two — 511, 512, 513, 1023, 1024, 1025. On most CPUs you'll see a noticeable performance dip at the power-of-two sizes. That's cache aliasing — addresses that are exactly the same modulo a cache set count collide and evict each other. Next lesson begins with this curiosity.

## Further reading

- "Anatomy of High-Performance Matrix Multiplication" (Goto & van de Geijn) — the foundational paper on the structure of fast matmul kernels. Dense; worth a careful read after the next two lessons.
- Eigen's matmul implementation source — readable C++ that maps to ideas we're building toward.
- BLIS framework (github.com/flame/blis) — modern, modular take on BLAS internals. Their writeups are excellent.
- *Hennessy & Patterson, Computer Architecture* — the cache and memory chapters, for the formal model behind what we just hand-waved through.

Next lesson: cache blocking. We make `B` look small to the cache, and the matmul speeds up by another 4–8×.
