---
title: "Lesson 7 — SIMD on ARM and Apple Silicon (NEON, AMX)"
date: "2026-06-03"
module: "bare-metal"
order: 7
tags: ["simd", "neon", "arm", "apple-silicon", "amx", "intrinsics", "rust"]
author: "Sudipta Pathak"
prerequisites: ["06-simd-x86"]
---

# Lesson 7 — SIMD on ARM and Apple Silicon

## Why this matters

Most on-device AI runs on ARM cores — phones, tablets, Apple laptops, embedded boards. On Apple Silicon specifically, the CPU has competitive performance with desktop x86 *and* sits next to an integrated GPU and Neural Engine that share unified memory. Knowing how to write a fast CPU kernel on this hardware is the entry point to the whole rest of this curriculum's depth track.

The SIMD instruction set on ARM is **NEON**. It's 128 bits wide (vs AVX2's 256, AVX-512's 512). At first glance you'd expect ARM SIMD to be half as fast. In practice, ARM cores compensate with higher IPC and FMA throughput, and Apple Silicon adds a **separate matrix coprocessor (AMX)** accessible from the CPU that does small dense matmuls at outrageous bandwidth. The headline number: an M-series P-core can sustain GEMM at 100+ GFLOPs *per core* using only NEON, and much higher with AMX.

This lesson covers NEON intrinsics in Rust, the mental shift from AVX2, the layout details specific to Apple Silicon, and a brief look at AMX (which you can't call from intrinsics — it's coprocessor-routed and currently reachable only via Apple's libraries like Accelerate).

## Concept

### NEON registers

ARM64 (AArch64) has 32 vector registers of 128 bits each. Each holds:

- 4 × f32 (the case we care about most for matmul)
- 2 × f64
- 8 × f16 / bf16 (since ARMv8.2)
- 16 × i8
- ... and more integer variants

128-bit vectors mean 4 f32 FMAs per instruction — half the width of AVX2. But you have 32 registers (vs AVX2's 16), so you can hold more accumulators in flight simultaneously, which makes the micro-kernel typically wider in C tiles. A common NEON matmul micro-kernel is **8 × 12** (96 outputs per kernel call) using 24 accumulator registers, whereas a typical AVX2 kernel is 8 × 4 (32 outputs).

### Apple Silicon specifics

The Firestorm / Avalanche / Everest cores (the P-cores in M1/M2/M3+) have:

- 4-wide FMA dispatch per cycle on the P-core (so 4 NEON FMAs × 4 lanes × 2 flops = 32 FLOPs/cycle/core peak).
- A separate FP unit cluster that can do parallel adds while FMAs are in flight.
- High L1d (192 KB on M-series P-cores — bigger than typical x86 cores).
- A very large shared System-Level Cache (SLC) on the SoC, often 16–32 MB.

The practical implication: NEON on an M-class P-core is *not* half the speed of AVX2 on x86. It's often within 10–30%, and the larger L1d means cache-blocked matmuls can use bigger block sizes — closer to 96 or 128 — without spilling.

### The AMX coprocessor

Apple Silicon contains an undocumented (officially) but well-explored matrix accelerator called **AMX** (not to be confused with Intel's AMX, which is unrelated). It's a per-cluster coprocessor that does small dense outer-products and accumulates them. Used well, it gives Apple Silicon ~3–5× the per-core matmul throughput over NEON alone.

You don't access AMX through Rust intrinsics — Apple doesn't publish them. The supported way is via `Accelerate.framework`'s BLAS (`cblas_sgemm`), which lights up AMX automatically. There are also community headers (e.g., Dougall Johnson's reverse-engineering work, the Asahi Linux project's open work) that expose AMX directly through hand-encoded instruction sequences, but these are CPU-generation-specific and unstable.

**For this module's matmul:** we'll write fast NEON-only kernels and treat AMX as the "official BLAS does this; we don't" frontier. We compare our hand-NEON to Accelerate's `sgemm` and acknowledge the gap.

### SVE and SVE2 (forward-looking)

ARM also defines **SVE** and **SVE2**, vector ISAs with variable register widths (128 to 2048 bits, decided per implementation). Servers and some embedded systems use SVE; iPhones and Macs don't currently expose it. We'll mention it for completeness, but for Apple Silicon you target NEON.

### Rust intrinsics for NEON

Available in `std::arch::aarch64`. The mapping is direct: `vfmaq_f32(c, a, b)` is `c = a * b + c` across 4 f32 lanes. The naming convention follows the C `arm_neon.h` header. Some functions:

| Intrinsic | Operation |
| --- | --- |
| `vld1q_f32(ptr)` | Load 4 f32 from contiguous memory |
| `vst1q_f32(ptr, v)` | Store 4 f32 to contiguous memory |
| `vdupq_n_f32(x)` | Broadcast scalar `x` to all 4 lanes |
| `vmulq_f32(a, b)` | Lane-wise multiply |
| `vfmaq_f32(c, a, b)` | Lane-wise FMA: c = a*b + c |
| `vaddq_f32(a, b)` | Lane-wise add |
| `vfmaq_laneq_f32(c, a, b, lane)` | FMA where `b`'s `lane` is broadcast first |

The last one — `vfmaq_laneq_f32` — is key for matmul. It lets us multiply a vector of A by *one lane* of a B vector (effectively a broadcast-and-FMA), without the explicit `vdupq_n_f32` step. This saves a register and a uop.

## Code walkthrough

A NEON micro-kernel for an 8×4 output tile. Same idea as the AVX2 kernel from Lesson 6, but with 4-wide registers.

```rust
#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn micro_kernel_8x4_neon(
    a: *const f32,   // packed: 8 rows × kr columns, contiguous as 8 floats per kk
    a_stride: usize, // = 8 in elements for the packed layout
    b: *const f32,   // packed: kr rows × 4 columns
    b_stride: usize, // = 4 in elements for the packed layout
    c: *mut f32,
    c_stride: usize, // in elements (= n for our row-major C)
    kr: usize,
) {
    // 8 accumulators: 2 × NEON registers per column × 4 columns.
    // Each NEON reg holds 4 floats, so 8-row tall column = 2 NEON registers.
    let mut c00 = vdupq_n_f32(0.0); // rows 0..4, col 0
    let mut c01 = vdupq_n_f32(0.0); // rows 4..8, col 0
    let mut c10 = vdupq_n_f32(0.0); // rows 0..4, col 1
    let mut c11 = vdupq_n_f32(0.0);
    let mut c20 = vdupq_n_f32(0.0);
    let mut c21 = vdupq_n_f32(0.0);
    let mut c30 = vdupq_n_f32(0.0);
    let mut c31 = vdupq_n_f32(0.0);

    // Pre-load existing C into accumulators.
    // (Loop unrolled here for brevity — assume initial C is zero or pre-loaded.)

    for kk in 0..kr {
        // 8 rows of A for this kk. Two NEON loads of 4 lanes each.
        let a_lo = vld1q_f32(a.add(kk * a_stride + 0));
        let a_hi = vld1q_f32(a.add(kk * a_stride + 4));

        // 4 columns of B for this kk, all in one NEON register.
        let b_row = vld1q_f32(b.add(kk * b_stride));

        // FMA: each lane of b_row multiplies the entire a_lo/a_hi and accumulates.
        c00 = vfmaq_laneq_f32(c00, a_lo, b_row, 0);
        c01 = vfmaq_laneq_f32(c01, a_hi, b_row, 0);
        c10 = vfmaq_laneq_f32(c10, a_lo, b_row, 1);
        c11 = vfmaq_laneq_f32(c11, a_hi, b_row, 1);
        c20 = vfmaq_laneq_f32(c20, a_lo, b_row, 2);
        c21 = vfmaq_laneq_f32(c21, a_hi, b_row, 2);
        c30 = vfmaq_laneq_f32(c30, a_lo, b_row, 3);
        c31 = vfmaq_laneq_f32(c31, a_hi, b_row, 3);
    }

    // Store back. C is row-major; we have per-column accumulators.
    // For brevity, scalar transpose-store, like the AVX2 version.
    let mut tmp: [f32; 8] = [0.0; 8];
    macro_rules! store_col {
        ($cc_lo:expr, $cc_hi:expr, $col:expr) => {
            vst1q_f32(tmp.as_mut_ptr().add(0), $cc_lo);
            vst1q_f32(tmp.as_mut_ptr().add(4), $cc_hi);
            for row in 0..8 {
                *c.add(row * c_stride + $col) += tmp[row];
            }
        };
    }
    store_col!(c00, c01, 0);
    store_col!(c10, c11, 1);
    store_col!(c20, c21, 2);
    store_col!(c30, c31, 3);
}
```

Things to notice that differ from the AVX2 version.

**Two registers per column.** Because NEON is 128-bit, 8 rows = 2 vector registers per column. So 4 columns × 2 = 8 accumulators. Still inside the 32-register file; plenty of headroom.

**`vfmaq_laneq_f32` with a lane index.** We load 4 B values into one register and use the per-lane FMA to do `a * b[lane]` without a broadcast. This is a NEON-specific efficiency that AVX2 doesn't have as directly (you'd use `_mm256_broadcastss_ps` or precomputed broadcasts).

**Packed layouts.** Both `a_stride = 8` and `b_stride = 4` indicate that we expect A and B to be **pre-packed** into scratch buffers with these strides. A's pack: take the 8-row, KR-column tile and store as 8 contiguous floats per `kk` (so `vld1q_f32` twice loads them). B's pack: take the KR-row, 4-column tile and store as 4 contiguous floats per `kk`. Pack-once, kernel-many-times.

**No explicit accumulator transpose at the end.** We accumulate per column, store as a scalar transpose. Same as AVX2; we could use NEON's `zip1` / `zip2` / `uzp1` / `uzp2` to do a SIMD transpose more efficiently if this part became a bottleneck (it usually doesn't).

### Calling Accelerate to see what's possible

Apple's framework includes a high-performance BLAS that uses AMX behind the scenes. From Rust:

```toml
# Cargo.toml
[target.'cfg(target_os = "macos")'.dependencies]
accelerate-src = "0.3"

[target.'cfg(target_os = "macos")'.build-dependencies]
accelerate-src = "0.3"
```

```rust
// In main.rs or a benchmarking module.
extern "C" {
    fn cblas_sgemm(
        order: i32,      // 101 = row-major
        trans_a: i32,    // 111 = no transpose
        trans_b: i32,    // 111
        m: i32, n: i32, k: i32,
        alpha: f32,
        a: *const f32, lda: i32,
        b: *const f32, ldb: i32,
        beta: f32,
        c: *mut f32, ldc: i32,
    );
}

fn matmul_accelerate(a: &Mat, b: &Mat, c: &mut MatMut) {
    let m = a.rows as i32;
    let n = b.cols as i32;
    let k = a.cols as i32;
    unsafe {
        cblas_sgemm(
            101, 111, 111, m, n, k,
            1.0, a.data.as_ptr(), k,
            b.data.as_ptr(), n,
            0.0, c.data.as_mut_ptr(), n,
        );
    }
}
```

Bench this. On an M-series Mac for a 512-cube, you'll likely see **150–250 GFLOPs** — single-core. That's the AMX number. Our hand-NEON kernel will land at perhaps **60–100 GFLOPs** single-core. The gap is what AMX is doing for us.

This is the right framing: we're not trying to beat Accelerate. We're trying to understand what's between us and Accelerate, and what fraction of it we can recover with code we wrote ourselves.

## Mental model & pitfalls

The single sentence: **NEON is 4-wide (vs AVX2's 8-wide), but with more registers and higher per-cycle FMA throughput; Apple Silicon's AMX is the rest of the gap to "this is what a vendor BLAS does."**

Pitfalls:

- **Assuming 4-wide means 2× slower than AVX2.** It doesn't, on Apple Silicon. The register file is bigger, the FMA dispatch is wider, and the cache is bigger. Wall-clock is often within 10–30%.
- **Using `vdupq_n_f32` when `vfmaq_laneq_f32` would suffice.** Both work; the per-lane form is one instruction shorter.
- **Forgetting to pack A.** Loading A's rows directly with stride-K reads is the same disaster as on x86: a single load per row, lots of cache lines. Pack once, kernel many times.
- **Trying to access AMX from Rust intrinsics.** You can't — the instructions are unofficial. Use Accelerate, or accept the gap.
- **Misreading which P-cores you got.** M-series chips have P-cores and E-cores; the OS scheduler will sometimes put a workload on E-cores under thermal pressure. For benchmarks, pin to a P-core (or at least be aware of the variance).

## Hands-on (at home)

You need an ARM64 machine for this — Apple Silicon Mac, or an ARM Linux box, or a Raspberry Pi if you want a slower demonstration. The code is the same; performance will differ.

1. **Add `#[cfg(target_arch = "aarch64")]` blocks** to `src/matmul.rs` for the NEON kernel.

2. **Implement the packing** for A and B. The simplest version copies the tile data into a scratch `Vec<f32>` per-tile; the optimized version reuses a single packing buffer per outer iteration.

3. **Wire it up.** A `matmul_neon` function that loops over outer tiles and calls `micro_kernel_8x4_neon`.

4. **Bench.** Expected on M-series for a 512-cube:

```
        matmul_ikj:     ~20 ms     7 GFLOPs
    matmul_blocked:      ~6 ms    22 GFLOPs
      matmul_neon:      ~3 ms    50 GFLOPs
matmul_accelerate:     ~1.5 ms   180 GFLOPs   (AMX through Accelerate)
```

5. **Run with `taskpolicy`** (macOS) to keep on P-cores for reproducible numbers:

```bash
taskpolicy -c utility cargo run --release      # prefer E-cores (don't use)
taskpolicy -c background cargo run --release   # prefer E-cores (don't use)
taskpolicy -c maintenance cargo run --release  # closer to P-cores
# Or use:
sudo nice -n -20 cargo run --release
```

The clean way is to set the QoS class to user-interactive or higher, which keeps you on P-cores. The exact mechanism varies by macOS version; experiment.

6. **Read the assembly.** `cargo show-asm matmul_neon --release`. Look for:
   - `fmla` instructions in the inner loop (4-wide FMA).
   - `ld1` instructions for loads.
   - No spills to memory in the inner loop.

7. **(Optional) Compare large vs small block sizes** on M-series. With its 192 KB L1d, block sizes up to ~128 should work cleanly. On a Raspberry Pi or older ARM chip with smaller L1d, you'll want to stay at 64 or smaller.

## Further reading

- ARM NEON Programmer's Guide — the official reference. Especially the chapter on matrix and vector operations.
- Apple's "Optimizing for Apple silicon CPUs" WWDC session videos (multiple years) — Apple-internal perspective on the P-core microarchitecture, including SIMD throughput.
- Dougall Johnson's "Apple AMX" reverse-engineering writeups — the public documentation for AMX. Use cautiously; the ISA is unsupported.
- Asahi Linux's GPU & CPU documentation — community-built deep dives into M-series internals.
- Accelerate framework documentation — for `cblas_sgemm` and friends.

Next lesson: threading. We have one P-core hitting ~100 GFLOPs. M-series chips have 8–12 P-cores. Threading is how we get to 600+.
