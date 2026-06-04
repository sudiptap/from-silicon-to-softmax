---
title: "Lesson 5 — Cache-Blocked Matrix Multiply"
date: "2026-06-03"
module: "bare-metal"
order: 5
tags: ["matmul", "tiling", "blocking", "cache", "rust"]
author: "Sudipta Pathak"
prerequisites: ["04-naive-matmul"]
---

# Lesson 5 — Cache-Blocked Matrix Multiply

## Why this matters

This is the lesson where the matmul gets fast. The technique is *cache blocking* (also called *tiling*), and it is the single highest-leverage optimization in numerical computing. Every fast matmul, every fast convolution, every fast attention kernel is a tiled algorithm. FlashAttention is, at its core, a tiled attention.

The mental shift is small: instead of computing one big multiplication, you decompose it into many small multiplications, each one sized to fit nicely in a particular cache level. The arithmetic count doesn't change. The memory access pattern changes utterly.

After this lesson, our 512-cube matmul moves from ~5 GFLOPs to ~20+ GFLOPs — still a fraction of peak, but with the right shape now. The next lessons (SIMD, threading) push toward peak by filling in vectorization and parallelism.

## Concept

### Why blocking works

Recall the inner loop of the naive matmul: for each output `C[i, j]`, we walk all `k` of `A[i, *]` and all `k` of `B[*, j]`. Across the full matrix we touch `B` a total of `m × n × k` times, in a column-walk pattern that misses cache lines.

Now consider this: most CPUs can hold a few hundred kilobytes in L2. A 256×256 float32 sub-block of `B` is 256 KB — fits. If we **load that sub-block once** and reuse it many times before evicting it, we pay the load cost amortized over many output computations.

Cache blocking is exactly this. Decompose the matrices into tiles. For each tile pair `(A_tile, B_tile)`, do a small matmul whose footprint fits in the cache, then move on. The total flops don't change. The total memory traffic *drops dramatically* because each loaded byte is reused many more times.

### The structure

The blocked matmul has more loops, but each loop has a clear purpose:

```rust
for i_outer in (0..m).step_by(MR) {
    for j_outer in (0..n).step_by(NR) {
        for k_outer in (0..k).step_by(KR) {
            // Inner micro-kernel: a small matmul of size MR × NR
            // accumulating over a block of size KR.
            // The micro-kernel data fits in L1.
            inner_matmul(
                A_block(i_outer, k_outer, MR, KR),
                B_block(k_outer, j_outer, KR, NR),
                C_block(i_outer, j_outer, MR, NR),
            );
        }
    }
}
```

`MR`, `NR`, `KR` are the **block sizes** — knobs you tune to the cache hierarchy. A typical first-cut for a modern CPU:

- `MR × NR ≈ 64 × 64` for L1 (16 KB of floats; well inside 32 KB L1d).
- An outer level of blocking sized to L2 — e.g., 256 × 256 — wrapping the L1 blocks.

Most serious matmul libraries (BLIS, Eigen, OpenBLAS) use 3 levels of blocking: register, L1, L2. We'll do 2 levels here (L1 + register-ish) for clarity; the structure scales.

### A worked example: 64×64 inner kernel

Take MR = NR = KR = 64. Each tile is 64 × 64 = 4096 floats = 16 KB. Three tiles (a block of A, a block of B, a block of C) is 48 KB — close to L1, fits in L2 cleanly. Inside the inner kernel, the CPU reuses each loaded A and B value 64 times before moving on (each A element contributes to 64 output cells in the C block; each B element similarly). Memory pressure drops by ~64×.

The inner-kernel flop count is `2 · 64³ ≈ 524K flops`. Done correctly with the right SIMD width, a modern CPU executes this in tens of microseconds. The outer loops repeat this kernel ~512 times (8 × 8 × 8 outer iterations for a 512-cube). Total: ~10 ms, at 25+ GFLOPs.

### The right block size

Block sizes that "just barely fit" are good. Block sizes that don't fit are catastrophic — you've reverted to a naive algorithm with more bookkeeping. Block sizes that fit too easily (e.g., 8 × 8 in L1) underutilize the cache and don't amortize the load cost enough.

A useful rule of thumb: take your L1d size, divide by 3 (for A, B, C tiles), divide by 4 (bytes per float), take square root. For 32 KB L1d: `√(32 × 1024 / 3 / 4) ≈ 51`. Round to a SIMD-friendly multiple (e.g., 48 or 64). That's your inner block size. The exact optimum is found empirically — sweep across a few sizes around the rule-of-thumb value and pick the winner.

## Code walkthrough

A straightforward 1-level cache-blocked matmul. We'll use the `i, kk, j` loop order from Lesson 4 because it's the better inner pattern, then wrap blocking around it.

```rust
// Block size. 64 is a reasonable start on most CPUs with 32KB L1d.
const BLOCK: usize = 64;

pub fn matmul_blocked(a: &Mat, b: &Mat, c: &mut MatMut) {
    assert_eq!(a.cols, b.rows);
    assert_eq!(a.rows, c.rows);
    assert_eq!(b.cols, c.cols);

    let m = a.rows;
    let k = a.cols;
    let n = b.cols;

    let mut i0 = 0;
    while i0 < m {
        let i_end = (i0 + BLOCK).min(m);
        let mut k0 = 0;
        while k0 < k {
            let k_end = (k0 + BLOCK).min(k);
            let mut j0 = 0;
            while j0 < n {
                let j_end = (j0 + BLOCK).min(n);

                // Inner kernel on a [i0..i_end, j0..j_end] x [k0..k_end] block.
                for i in i0..i_end {
                    for kk in k0..k_end {
                        let a_ik = a.data[i * k + kk];
                        // Hot inner loop: walk j stride 1 in both B and C.
                        for j in j0..j_end {
                            c.data[i * n + j] += a_ik * b.data[kk * n + j];
                        }
                    }
                }

                j0 = j_end;
            }
            k0 = k_end;
        }
        i0 = i_end;
    }
}
```

**Reading the structure.** The three outer `while` loops decompose the problem into 64×64×64 sub-matmuls. Each sub-matmul reuses three blocks worth of memory — 64×64 of A, 64×64 of B, 64×64 of C — totaling 48 KB, which lives happily in L1 for an `i, kk, j` walk. The innermost three loops are exactly the `ikj` matmul from Lesson 4, just over sub-ranges.

**`a_ik` hoist.** Same trick as Lesson 4's `ikj`. We load `A[i, kk]` once and reuse it across the entire inner `j` range. That's 64 multiplies per A load, compared to 1 per A load in the naive version.

**Boundary handling.** `i_end = (i0 + BLOCK).min(m)` lets the loop work for any `m` that isn't a multiple of 64. The edge tiles will be slightly smaller, but the code stays uniform.

### Why the inner loop is now SIMD-ready

Look at the innermost loop body: `c.data[i*n + j] += a_ik * b.data[kk*n + j]`. As `j` increments, both `c` and `b` indices step by 1. We're reading a sequential range of `B`, accumulating into a sequential range of `C`. Both are stride-1, both are cache-resident. The compiler can — and on `--release` with the right targets, *will* — automatically unroll this and vectorize it with SIMD. We get a chunk of next lesson's benefit for free, today, by writing the right loop shape.

You can verify this by running on `--release` and inspecting the assembly: the inner loop body should show vector loads (`movaps`, `movups` on x86; `ld1.4s` on ARM) and vector FMAs (`vfmadd231ps`; `fmla v0.4s, v1.4s, v2.4s`). If you see scalar FMAs instead, the compiler decided your loop wasn't vectorizable — usually a sign of dependency or alias confusion. Adding `#[inline]` hints to the kernel function, or using slice references explicitly typed as non-overlapping, often nudges it.

### The aliasing footnote

You might wonder: how does the compiler know that `b.data` and `c.data` don't overlap? In C, it doesn't, and has to assume the worst case — emitting one extra load after each store, ruining vectorization. In Rust, `&mut [f32]` and `&[f32]` are statically guaranteed not to alias by the borrow checker. The compiler emits vectorized code freely. This is one of the unsung benefits of Rust for numeric work.

## Mental model & pitfalls

The single sentence: **Blocking trades the cost of "every load goes to L3" for the cost of "every block is loaded once into L1 and reused."**

Pitfalls:

- **Wrong block size.** Too small and you fail to amortize. Too big and you spill to L2 or L3 and lose the benefit. Sweep empirically — it's 30 lines of test code.
- **Wrong loop nesting.** Putting block loops inside element loops is no improvement. Block loops have to be outside; element loops are the innermost.
- **Block boundaries that don't divide cleanly.** Edge tiles smaller than the block. Handle them correctly (the `min(m)` trick above) or you'll either skip work or read out of bounds.
- **Forgetting accumulation semantics.** The inner kernel does `C += A · B`, not `C = A · B`. The outer block loop over `k` accumulates partial products. If you initialize `C` to nonzero before calling, the result is offset. Initialize to zero (or document the addition behavior clearly).
- **Power-of-two cache aliasing.** Worth knowing about: a block size that's a power of two can hit cache aliasing pathologies. Many BLAS libraries use sizes like 63 or 65 to avoid this. We'll usually be fine with 64 in this module's experiments.

## Hands-on (at home)

Extend the project.

1. **Add `matmul_blocked`** with `BLOCK = 64` to `src/matmul.rs`.

2. **Bench it alongside the others.** Expected on a 512-cube:

```
        matmul_ijk:   85.20 ms     1.57 GFLOPs    (naive, kk-innermost)
        matmul_ikj:   18.40 ms     7.30 GFLOPs    (loop reorder)
    matmul_blocked:    6.20 ms    21.65 GFLOPs    (this lesson)
```

Numbers will differ across machines — Apple M-series tends to be a bit lower-clocked but with better cache behavior; high-clock x86 tends to be the opposite. The pattern of speedups is what we care about.

3. **Sweep block size.** Run with `BLOCK ∈ {16, 32, 48, 64, 80, 96, 128, 256}`. Plot or print the GFLOPs for each. You should see a clear peak somewhere in 48–96 on most CPUs, with a slow falloff above and a sharp falloff below.

4. **Sweep problem size.** Run blocked-matmul on (256, 512, 768, 1024, 1535, 2048). You'll see:
   - GFLOPs are reasonably stable across non-pathological sizes.
   - 1024 (or whatever power of two your CPU dislikes) may dip — this is the aliasing curiosity from Lesson 4's optional exercise. Tile sizes like 63 or 65 sometimes help; we'll address it more thoroughly when we add SIMD.

5. **Correctness check.** Verify against `matmul_ikj` for the same inputs. `max_abs_diff` should be small (`< 1e-3`); not zero because the addition order in the blocked version is different and floating-point addition isn't associative.

**Optional**: read the assembly of `matmul_blocked`'s inner loop. Specifically look for:
   - Vector instructions (`vfmadd231ps`, `fmla`, or similar).
   - Unrolling — multiple FMAs in sequence with different register names.
   - Absence of branch instructions inside the innermost loop body.

If you see scalar instructions in the inner loop, try moving `BLOCK` to a constant and re-compiling. Compile-time constants help the compiler unroll aggressively.

## Further reading

- BLIS' "Anatomy of high-performance many-threaded matrix multiplication" — the canonical paper on multi-level blocking.
- OpenBLAS source for any `gemm_kernel` — production-grade blocked matmul, dense with macros but readable once you know what blocking looks like.
- Eigen's `GeneralBlockPanelKernel` — C++ but expressive about block structure.
- "What I learned writing matrix multiplications" (various blog posts in the BLIS / FlashAttention orbit) — practical writeups of the same trade-offs.

Next lesson: SIMD. We take the vectorized loop the compiler made us, then write it ourselves to understand exactly what's happening — and to handle the cases where the compiler can't quite see how.
