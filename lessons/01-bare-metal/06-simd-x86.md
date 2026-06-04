---
title: "Lesson 6 — SIMD on x86 (SSE, AVX2, AVX-512)"
date: "2026-06-03"
module: "bare-metal"
order: 6
tags: ["simd", "avx", "sse", "intrinsics", "rust", "matmul"]
author: "Sudipta Pathak"
prerequisites: ["05-cache-blocked-matmul"]
---

# Lesson 6 — SIMD on x86

## Why this matters

When the compiler auto-vectorizes well, you don't *need* to write SIMD by hand. When it doesn't, you need to know what it failed at and how to write it yourself. More importantly, you need to know what the CPU *can* do — what data-parallel instructions exist, how wide they are, what they cost — so that even when you stay at a higher level (Triton, MLX, MPSGraph) you can evaluate whether your code is using the available width.

Modern CPUs have wide vector registers and vector instructions that do 4, 8, or 16 floating-point operations at once. A matmul that uses one float at a time is leaving 4–16× of throughput on the table. Even a 30-line by-hand SIMD inner kernel often beats whatever the compiler decided to do, and seeing that gap by yourself is the cheapest way to develop intuition.

This lesson focuses on x86 because the AVX family is the de-facto vector ISA on desktops and servers. Next lesson covers ARM NEON for Apple Silicon and mobile. Both follow the same mental pattern; the syntax differs.

## Concept

### Vector registers and widths

Each generation of x86 SIMD doubled the register width and added new instructions:

| ISA | Vector width | f32 lanes | Year | Notes |
| --- | -----------: | --------: | ---- | ----- |
| SSE / SSE2 | 128-bit | 4 | 1999 / 2001 | Baseline; available on every x86-64 CPU. |
| AVX | 256-bit | 8 | 2011 | Doubled width; FP-only on first parts. |
| AVX2 | 256-bit | 8 | 2013 | Integer ops; gather; FMA on most chips. |
| AVX-512 | 512-bit | 16 | 2016 | Wider; predicate ("mask") registers. Subset of CPUs. |

When you say "this code uses AVX2," you mean it uses the 256-bit `__m256` registers and FMA instructions (`vfmadd231ps` etc.). Every Intel CPU from Haswell (2013) onward and every AMD Zen (2017+) has AVX2. It's a reasonable baseline for desktop/laptop code. AVX-512 is best treated as opt-in.

The relevant Rust intrinsics live in `std::arch::x86_64`. They map 1:1 to the underlying assembly. Using them safely requires `#[target_feature]` annotations to tell the compiler to assume the CPU has those features.

### What a single FMA does

`_mm256_fmadd_ps(a, b, c)` computes `a * b + c` lane-wise across 8 `f32` lanes simultaneously. That's 16 floating-point operations in one instruction, one cycle (on the right CPU). For matmul, this is the workhorse.

### The micro-kernel pattern

A SIMD matmul inner kernel typically processes a small fixed-size tile of `C` per call. The canonical shape for AVX2 is `8 × 4`:

- 8 output rows × 4 output columns = 32 output cells.
- Each output column held in one `__m256` register (8 f32 = 8 rows of one column).
- 4 such registers = 4 columns. Total: 4 accumulator registers.
- Inner loop iterates over the `k` dimension, broadcasting `A[i:i+8, kk]` across one register and `B[kk, j:j+4]` as 4 broadcast values, doing 4 FMAs per `kk`.

The point of this layout: each `A` load is reused 4 times (across 4 output columns); each `B` load is reused 8 times (across 8 output rows). Arithmetic intensity per byte loaded skyrockets.

This is also why SIMD and cache blocking compose perfectly. Blocking ensures the data is in L1; SIMD ensures the inner kernel uses that data with maximum throughput.

### Aligned vs unaligned loads

The instructions come in aligned (`_mm256_load_ps`, requires 32-byte alignment) and unaligned (`_mm256_loadu_ps`, no alignment requirement) forms. On modern CPUs the *cost* difference is negligible — unaligned loads are essentially as fast as aligned, provided they don't cross a cache line. The *correctness* difference is real: aligned-load on misaligned data faults. Default to unaligned (`loadu`/`storeu`) unless you have a specific reason and a guarantee.

For matmul buffers we created with `Vec<f32>`, the allocator gives 32-byte-aligned memory by default for sizes that need it. We're fine.

### `#[target_feature]` and safety

Intrinsics functions are marked `unsafe` because they will execute illegal-instruction faults on CPUs without the feature. The fix is to annotate the calling function so the compiler knows the feature is required, *and* gate the call on a runtime check:

```rust
#[target_feature(enable = "avx2,fma")]
unsafe fn matmul_inner_avx2(...) { ... }

if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
    unsafe { matmul_inner_avx2(...); }
} else {
    matmul_fallback(...);
}
```

This is the safe pattern. Every production library does it. We'll do it too.

## Code walkthrough

A by-hand AVX2 micro-kernel for an 8×4 output tile of `C`, accumulating over a `KR` block of `A` and `B`.

```rust
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn micro_kernel_8x4(
    a: *const f32,   // pointer to A[i:i+8, k_start..k_start+kr], column-stride = a_stride (in elements)
    a_stride: usize,
    b: *const f32,   // pointer to B[k_start:k_start+kr, j:j+4], row-stride = b_stride
    b_stride: usize,
    c: *mut f32,     // pointer to C[i:i+8, j:j+4], row-stride = c_stride
    c_stride: usize,
    kr: usize,
) {
    // Load existing C tile into 4 accumulators.
    let mut c0 = _mm256_loadu_ps(c.add(0 * c_stride));
    let mut c1 = _mm256_loadu_ps(c.add(1 * c_stride));
    let mut c2 = _mm256_loadu_ps(c.add(2 * c_stride));
    let mut c3 = _mm256_loadu_ps(c.add(3 * c_stride));

    // Wait — c0..c3 are per-ROW above, but we want per-COLUMN accumulators.
    // Reload as per-column: each register holds 8 rows of one column.
    c0 = _mm256_setr_ps(
        *c.add(0 * c_stride + 0), *c.add(1 * c_stride + 0),
        *c.add(2 * c_stride + 0), *c.add(3 * c_stride + 0),
        *c.add(4 * c_stride + 0), *c.add(5 * c_stride + 0),
        *c.add(6 * c_stride + 0), *c.add(7 * c_stride + 0),
    );
    c1 = _mm256_setr_ps(
        *c.add(0 * c_stride + 1), *c.add(1 * c_stride + 1),
        *c.add(2 * c_stride + 1), *c.add(3 * c_stride + 1),
        *c.add(4 * c_stride + 1), *c.add(5 * c_stride + 1),
        *c.add(6 * c_stride + 1), *c.add(7 * c_stride + 1),
    );
    c2 = _mm256_setr_ps(
        *c.add(0 * c_stride + 2), *c.add(1 * c_stride + 2),
        *c.add(2 * c_stride + 2), *c.add(3 * c_stride + 2),
        *c.add(4 * c_stride + 2), *c.add(5 * c_stride + 2),
        *c.add(6 * c_stride + 2), *c.add(7 * c_stride + 2),
    );
    c3 = _mm256_setr_ps(
        *c.add(0 * c_stride + 3), *c.add(1 * c_stride + 3),
        *c.add(2 * c_stride + 3), *c.add(3 * c_stride + 3),
        *c.add(4 * c_stride + 3), *c.add(5 * c_stride + 3),
        *c.add(6 * c_stride + 3), *c.add(7 * c_stride + 3),
    );

    for kk in 0..kr {
        // Broadcast B[kk, 0..4] into 4 vector registers.
        let b0 = _mm256_set1_ps(*b.add(kk * b_stride + 0));
        let b1 = _mm256_set1_ps(*b.add(kk * b_stride + 1));
        let b2 = _mm256_set1_ps(*b.add(kk * b_stride + 2));
        let b3 = _mm256_set1_ps(*b.add(kk * b_stride + 3));

        // Load A[i:i+8, kk] — 8 rows of one column.
        // This requires A to be in a column-friendly layout for this tile,
        // i.e. we've pre-packed A so this is a contiguous read.
        let a_col = _mm256_loadu_ps(a.add(kk * a_stride));

        c0 = _mm256_fmadd_ps(a_col, b0, c0);
        c1 = _mm256_fmadd_ps(a_col, b1, c1);
        c2 = _mm256_fmadd_ps(a_col, b2, c2);
        c3 = _mm256_fmadd_ps(a_col, b3, c3);
    }

    // Write back per-column accumulators into the row-major C tile.
    let mut tmp: [f32; 8] = std::mem::zeroed();

    _mm256_storeu_ps(tmp.as_mut_ptr(), c0);
    for row in 0..8 { *c.add(row * c_stride + 0) = tmp[row]; }

    _mm256_storeu_ps(tmp.as_mut_ptr(), c1);
    for row in 0..8 { *c.add(row * c_stride + 1) = tmp[row]; }

    _mm256_storeu_ps(tmp.as_mut_ptr(), c2);
    for row in 0..8 { *c.add(row * c_stride + 2) = tmp[row]; }

    _mm256_storeu_ps(tmp.as_mut_ptr(), c3);
    for row in 0..8 { *c.add(row * c_stride + 3) = tmp[row]; }
}
```

A lot to unpack. Five things to focus on.

**1. The accumulator layout (per-column).** Each `__m256` holds 8 rows of *one* column of the output tile. This is the right shape because the inner loop multiplies an 8-row column-of-A by a single B scalar (broadcast) — producing 8 row-aligned output contributions to one column of C. Four columns → 4 accumulators.

**2. Packing A.** The `_mm256_loadu_ps(a.add(kk * a_stride))` line assumes that successive elements at offset `kk * a_stride` are the 8 rows of A for one fixed `kk`. In the original row-major A, this would be a strided read — exactly what we're trying to avoid. The standard answer is to **pre-pack A** into a column-major scratch buffer, where for the current 8-row tile, all `kr` columns are stored contiguously. That packing costs O(MR · KR) memory traffic per tile, but the inner kernel runs many FMAs per loaded value afterward.

In production kernels (BLIS, OpenBLAS), packing both A and B into well-laid-out scratch buffers is the dominant memory cost; the inner kernel is essentially L1-resident. We're glossing over packing details here for word count — the next lesson's hands-on or a deeper-dive lesson covers it explicitly.

**3. The FMA chain.** `c_k = fma(a_col, b_k, c_k)` is the per-cycle workhorse: 8-wide multiply-add into 4 accumulators per inner iteration → 32 FLOPs per cycle in steady state, on a CPU that can issue 2 FMAs per cycle (Skylake, Zen 4+). That's near peak for one core.

**4. Why broadcast B and load A as a vector** (and not the other way around). Either direction is workable; the chosen direction is what matches the row-major data layout of B (one B value at a time, naturally) and the packed column-major A (a contiguous chunk of 8 rows). Reversing them works for column-major B and row-major A — i.e., the conventional FORTRAN layout. The structure mirrors.

**5. The store-back ugliness.** The transpose-back at the end exists because our internal accumulators are per-column but the output C is row-major. In production code, you'd often store C in panel-packed format too, or perform the transpose with a SIMD transpose instruction sequence (8 `unpcklps`/`unpckhps`/`shuffles`) instead of scalar copies. We've left it scalar for clarity.

### Compiler-vectorized version vs hand-written

A small confession: with `--release` and the right `RUSTFLAGS`, the cache-blocked version from Lesson 5 will often hit 80–90% of what this hand-written kernel does, because LLVM is genuinely good at this loop shape. The reason to write the by-hand kernel anyway:

- You see what the compiler is doing (or is failing to do).
- You can force a layout (8×4 vs 8×8 vs 16×4) the compiler wouldn't pick.
- For tighter precision control or unusual reductions (mixed-precision accumulation), the by-hand version is the only way.
- The skill transfers: writing CUDA kernels feels exactly like this once you've done it on the CPU.

## Mental model & pitfalls

The single sentence: **SIMD turns one floating-point operation into 4, 8, or 16 — and the right matmul kernel structure makes the cache and the SIMD lanes both happy at the same time.**

Pitfalls:

- **Loading the wrong dimension into a SIMD register.** If you load 8 floats that aren't co-located in memory (strided read), you do 8 scalar loads under the hood and get nothing. Either pre-pack, or change which axis you vectorize.
- **Hoping aliasing works out.** If the compiler thinks two pointers might overlap (rare in Rust due to borrow rules), it serializes. In `unsafe` raw-pointer kernels, you lose this guarantee — the `core::ptr::might_alias` story is real. Be careful with raw pointers.
- **Mixing precisions accidentally.** `_mm256_fmadd_ps` is f32. `_mm256_fmadd_pd` is f64. The lane count halves with precision. Mixing them in the same accumulator chain is a footgun.
- **Forgetting target-feature gates.** Code compiled without `-C target-feature=+avx2,+fma` may use the slower SSE path or refuse to emit FMA. Either pass the flag or use `#[target_feature]` annotations.
- **AVX-512 frequency throttling.** Heavy AVX-512 use on older Intel parts (Skylake-X especially) downclocks the entire core. The "free" 2× width came with a 10–25% clock penalty. On newer parts and AMD this is mostly gone. Worth measuring the wall-clock effect, not just the per-cycle throughput.

## Hands-on (at home)

Extend the matmul project.

1. **Detect features at startup** and print which path will be used:

```rust
fn main() {
    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        println!("Using AVX2 + FMA path");
    } else {
        println!("Falling back to scalar / SSE");
    }
    // ... rest as before
}
```

2. **Add `matmul_avx2`** as a wrapper that, for each (i0, j0, k0) outer tile, calls the `micro_kernel_8x4` inner. Implement A-packing inline if you want to keep the file self-contained; the simpler approach is to constrain `m % 8 == 0`, `n % 4 == 0`, `k % KR == 0` for now and skip the irregular tiles.

3. **Bench it.** Expected on a 512-cube on a modern x86 desktop:

```
        matmul_ijk:    85 ms     1.5 GFLOPs
        matmul_ikj:    18 ms     7.3 GFLOPs
    matmul_blocked:     6 ms    21.6 GFLOPs
      matmul_avx2:     2.4 ms    55.0 GFLOPs
```

If you don't see the jump from blocked to AVX2, the compiler was already auto-vectorizing the blocked version aggressively — verify by inspecting the assembly of both.

4. **Correctness.** Compare AVX2 result against `matmul_ikj`. `max_abs_diff < 1e-2` is acceptable; SIMD reductions sum in a different order from scalar.

5. **(Optional) AVX-512 variant.** If your CPU supports it (`is_x86_feature_detected!("avx512f")`), try a 16-lane kernel (16×4 = 64 outputs per tile). Compare wall-clock to AVX2 — the speedup is often less than 2× because of clock throttling on Skylake-X, but closer to 2× on Sapphire Rapids and Zen 4+.

6. **(Optional) Read the assembly.** `cargo show-asm matmul_avx2 --release`. Confirm:
   - 4 FMAs in the inner loop body, none of them spilling to memory.
   - One `vbroadcastss` per B load.
   - The store-back transpose lives outside the inner loop.

## Further reading

- Intel Intrinsics Guide (intel.com/content/.../intrinsics-guide.html) — the searchable reference for every x86 SIMD intrinsic. Indispensable.
- Agner Fog's instruction tables (agner.org/optimize) — latency and throughput for every SIMD instruction across CPU families.
- "The AVX-512 Performance Story" (cloudflare blog, various) — sober look at when AVX-512 wins and loses.
- BLIS micro-kernel sources — multiple SIMD micro-kernels (AVX2, AVX-512, NEON) side by side. Excellent reading once you've written your own.

Next lesson: NEON on Apple Silicon and ARM. The same patterns; different syntax. Then you'll be able to write fast inner kernels on whatever CPU is in front of you.
