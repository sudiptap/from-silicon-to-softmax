---
title: "Lesson 8 — Kernel Fusion"
date: "2026-06-03"
module: "gpu-computing"
order: 8
tags: ["fusion", "memory-bound", "operator-fusion", "triton", "torch-compile"]
author: "Sudipta Pathak"
prerequisites: ["07-metal-shading-language"]
---

# Lesson 8 — Kernel Fusion

## Why this matters

Most ML kernels — element-wise ops, layer norms, activations, even small matmuls at low batch size — are not bounded by how fast the GPU can compute. They're bounded by how fast the GPU can read and write memory. A ReLU on a tensor of 100 million floats is 100 million reads, 100 million writes, 100 million negligible compares. The compute is essentially free; the memory traffic is everything.

This is the structural reality behind the headline numbers. Modern GPUs have arithmetic intensity ceilings of 100–300 FLOPs per byte before they become compute-bound. A ReLU is 1 FLOP per 4 bytes (FP32) — 0.25 FLOPs per byte. ReLU lives 1000× below the compute ceiling. Most "simple" ML ops are like this.

The optimization that matters for these ops is not "make the ReLU faster" — it's "stop reading and writing memory in between ops." If your network does `Linear → ReLU → Dropout → LayerNorm` as four separate kernels, you've made *four* round-trips to global memory. If you fuse them into one kernel, you make *one*. Same flops; quarter the memory traffic; usually 3× the throughput.

This lesson is about that compositional view of GPU performance. Once you have it, every kernel-level performance story — torch.compile, FlashAttention, FlashMLA, custom CUDA — reads as a different answer to the same question: *how do we stop wasting bandwidth between operations.*

## Concept

### The arithmetic intensity ceiling

For each kernel, define:

```
arithmetic_intensity = FLOPs / bytes_of_memory_traffic
```

Each GPU has a peak `intensity_ceiling = compute_TFLOPs / memory_bandwidth_TB_per_s`. Kernels below this ceiling are memory-bound; kernels above are compute-bound. Examples:

- **H100**: 989 TFLOPs FP8 / 3 TB/s ≈ 330 FLOPs/byte.
- **A100**: 312 TFLOPs FP16 / 2 TB/s ≈ 156 FLOPs/byte.
- **M3 Max**: ~17 TFLOPs FP16 / 0.4 TB/s ≈ 42 FLOPs/byte.

For each kernel, the *actual* arithmetic intensity:

- **Dense matmul (FP16) at moderate batch**: thousands of FLOPs/byte. Far above ceiling. Compute-bound. Tensor cores fed.
- **Element-wise op (ReLU, Dropout, even GELU)**: 1–10 FLOPs/byte. Memory-bound by 100×.
- **LayerNorm**: ~5–10 FLOPs/byte. Memory-bound.
- **Attention (full, naive)**: ~10 FLOPs/byte for small head dim. Memory-bound at long context.
- **Reductions (sum, max)**: 1 FLOP per byte. Memory-bound.

The implication: most of the ops that aren't matmul are memory-bound. They contribute to total step time in proportion to their bytes touched, not their FLOPs done. Fusion is the standard answer.

### Vertical fusion (the common kind)

The classic case: a sequence of element-wise ops applied to the same tensor.

```python
# Three kernels, three round-trips through memory:
x = mat_a @ mat_b              # matmul, writes x
x = x + bias                   # element-wise add, reads x, writes x
x = torch.relu(x)              # element-wise relu, reads x, writes x
```

Total: 3 reads of `x` + 3 writes of `x` (plus the matmul's reads of `A`, `B`).

Fused, the matmul kernel computes `relu(x + bias)` directly before writing `x` for the first and only time. Total: 1 write of `x`. A 3× reduction in memory pressure for the post-matmul portion.

This is **vertical fusion**: fusing operations that share a producer-consumer relationship through the same tensor.

### Horizontal fusion (the trickier kind)

Less common but valuable: fusing operations that share *inputs* but produce *different outputs*. Example from attention:

```python
q = x @ Wq  # produces Q
k = x @ Wk  # produces K
v = x @ Wv  # produces V
```

Three separate matmuls read `x` three times. Horizontally fused: one matmul reads `x` once, produces all three outputs (stacking the weight matrices). This is the "fused QKV projection" you'll see in any production LLM kernel.

### Why this matters more on GPU than CPU

CPUs have deep cache hierarchies; intermediate results often live in L1/L2 across operations naturally. GPUs have very fast on-chip memory but a tiny per-block budget. Intermediates that live in registers across an op-boundary are free. Intermediates that spill to global memory and come back are full HBM round-trips.

So GPU fusion has stronger ROI than CPU fusion. A multi-op pipeline that PyTorch CPU does in 10 ms might take 5 ms even unfused on GPU (raw bandwidth), 2 ms fused. The unfused version is leaving 60% on the table; the CPU version, less so.

### What fuses easily, what doesn't

Easily fused:

- Element-wise ops (add, mul, relu, gelu, dropout, mask).
- Bias add after matmul.
- LayerNorm with optional bias and weight scaling (the affine transformation).
- Activation immediately after a matmul.
- Reduction immediately after an element-wise op.

Hard to fuse:

- Ops with different output shapes (a reduction following an element-wise op, where the next op needs the full pre-reduction tensor).
- Ops that need synchronization between blocks (most cross-block reductions).
- Heterogeneous data types where one op needs FP32 precision and the next is happy with FP16.
- Ops separated by data-dependent control flow (if branches that pick different next ops).

### How fusion happens in practice

Three paths, in increasing power and complexity:

1. **Hand-written fused kernels** — you write one CUDA / Triton / MSL kernel that does the full chain. Maximum control, most code. The FlashAttention way.

2. **Pattern-matching compilers** — `torch.compile`, JAX's XLA, TensorRT, Apple's MPSGraph. The framework sees a graph of ops, recognizes fusable subgraphs, and emits a fused kernel automatically. Works well for common patterns; falls back to unfused for unrecognized ones.

3. **General-purpose ML compilers** — Triton, IREE, MLIR-based compilers. Take an arbitrary computation, decompose it into fusable tiles, generate a fused kernel. This is the frontier.

For ML engineers in 2026, the pragmatic order is: use `torch.compile` first (automatic, free). For hot paths that don't fuse the way you want, write a Triton kernel. Drop to raw CUDA only when Triton can't express what you need.

### Case study: fused attention

Standard attention does:

```
S = Q @ K^T              # matmul; intermediate (B, H, N, N)
P = softmax(S)           # element-wise; reads S, writes P
O = P @ V                # matmul; reads P
```

The intermediate `S` and `P` are **N × N** — quadratic in sequence length. For a 32K-token sequence, `S` is 32K × 32K × 4 bytes per head = ~4 GB per head. *Per head*. With 64 heads on a batch this exceeds any GPU's memory.

FlashAttention's insight: never materialize `S` or `P`. Tile the computation along the rows of `Q`. For each tile of `Q`, walk through tiles of `K` and `V`, computing partial outputs and tracking the running softmax in registers. The full softmax is computed *implicitly* through the "online softmax" trick (Lesson 10). At the end, store only `O`. Memory traffic: O(N · head_dim), not O(N²). For long sequences, this isn't a small speedup — it's an order of magnitude.

This is fusion taken to the extreme: not just merging element-wise ops, but merging two matmuls and a softmax into a single, custom, tiling-aware kernel.

## Code walkthrough

A small but illustrative example: fused matmul + bias + ReLU in Triton.

```python
@triton.jit
def fused_matmul_bias_relu_kernel(
    a_ptr, b_ptr, bias_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        a_mask = (offs_m[:, None] < M) & ((k + offs_k)[None, :] < K)
        b_mask = ((k + offs_k)[:, None] < K) & (offs_n[None, :] < N)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # Fuse: load bias, add, ReLU — all before we store.
    bias_ptrs = bias_ptr + offs_n
    bias_mask = offs_n < N
    bias = tl.load(bias_ptrs, mask=bias_mask, other=0.0)
    acc = acc + bias[None, :]              # broadcast bias across rows
    acc = tl.maximum(acc, 0.0)             # ReLU

    # Store once, fused result.
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)
```

The only changes from the Lesson 6 matmul are the four lines after the inner loop: load bias, add, ReLU, then the store. The output write happens once, with the fully-computed `relu(x + bias)` already in registers.

Compare wall-clock vs. unfused PyTorch:

```python
def unfused(a, b, bias):
    x = a @ b
    x = x + bias
    return torch.relu(x)
```

For 4096×4096 FP16, the unfused version takes roughly 1.5× the time of the fused version on a 4090 — the bias-add and ReLU each contribute their own memory pass.

### Where torch.compile fits

```python
@torch.compile
def fused(a, b, bias):
    x = a @ b
    x = x + bias
    return torch.relu(x)
```

`torch.compile` will recognize this pattern and generate a fused kernel automatically. For the matmul + element-wise pattern it does an excellent job — usually matching hand-written Triton. For more exotic patterns (attention, MoE routing, custom reductions), hand-written usually still wins, but the gap is narrowing.

The pragmatic workflow: write your model in plain PyTorch. Add `@torch.compile` (or use `torch.compile(model)`). Profile. If a hotspot remains, write a Triton kernel. Replace the hot path.

## Mental model & pitfalls

Single sentence: **Most ML ops are memory-bound; fusing them so intermediate results stay in registers (not global memory) is the single biggest performance lever after using tensor cores; FlashAttention is what happens when you take fusion to its logical extreme.**

Pitfalls:

- **Fusing too aggressively.** A single kernel that does five things uses more registers and more shared memory than five small kernels. Past a point, the per-block resource budget breaks and occupancy collapses. Profile.
- **Fusion that breaks numerical stability.** Some sequences need FP32 precision in the middle even if inputs and outputs are FP16. Naive fusion loses precision. Be deliberate about accumulator types.
- **Trusting `torch.compile` blindly.** It does the right thing 90% of the time. The 10% it doesn't is exactly where you need to look. Inspect generated code (`TORCH_LOGS=output_code python ...`) when you have a hot path that isn't speeding up.
- **Fusing across non-fusable boundaries.** A reduction in the middle of a chain (e.g., LayerNorm requires mean and variance over a sequence dim) typically forces a kernel boundary. Trying to push past it produces wrong code.
- **Excessive specialization.** A fused kernel that handles 20 cases is slower than 20 small kernels each specialized for one case. The specialization overhead (constants in the right place, masked-out instructions) is real.

## Hands-on (at home)

1. **Write the fused Triton kernel** above. Verify correctness against the unfused version.

2. **Bench.** A 4096×4096 matmul + bias + ReLU. Compare the fused Triton to the unfused PyTorch (each op separate). Expected: 1.3–1.8× speedup.

3. **`torch.compile` the unfused version.** Time it. You should see it close most of the gap automatically.

4. **Inspect what `torch.compile` produced:**

```bash
TORCH_LOGS=output_code python your_script.py
```

This dumps the generated Triton code. Read it. You'll see a kernel very similar to the one you wrote by hand.

5. **Try a chain that doesn't fuse.** Add a `torch.sum(x, dim=-1)` in the middle of the chain (forcing a reduction kernel boundary). Re-time. See where compile's fusion stops.

6. **Run a small Transformer block** (multi-head attention + MLP) under `torch.compile`. Use `TORCH_LOGS=output_code`. Count how many fused kernels were generated. For a typical 6-op chain (linear → bias → activation → linear → bias → residual add), you should see 2–3 fused kernels.

7. **(Optional) Profile the fused vs unfused versions** with Nsight Compute. Look at memory bandwidth utilization — the unfused version should show higher absolute bandwidth use (more bytes moved) for less compute done.

## Further reading

- "Halide: A Language and Compiler for Optimizing Parallelism, Locality, and Recomputation" (Ragan-Kelley et al.) — the original paper that introduced the schedule-vs-algorithm distinction central to modern fusion compilers.
- *PyTorch 2.x torch.compile documentation* — what it does, how to debug it.
- "Roofline Performance Model" — formalization of the arithmetic-intensity ceiling discussed here.
- TVM and IREE design docs — alternative approaches to ML compilation with strong fusion stories.
- The FlashAttention paper and code (next lesson) — the canonical extreme case of fusion as memory optimization.

Next lesson: reductions. A small but important topic — most fused kernels need a fast reduction primitive — and the warp shuffle pattern is the GPU programmer's swiss-army knife.
