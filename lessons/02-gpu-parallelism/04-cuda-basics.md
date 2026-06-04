---
title: "Lesson 4 — CUDA Basics: Your First Kernel"
date: "2026-06-03"
module: "gpu-computing"
order: 4
tags: ["cuda", "kernel", "launch", "nvcc", "matmul"]
author: "Sudipta Pathak"
prerequisites: ["03-apple-silicon-gpu-architecture"]
---

# Lesson 4 — CUDA Basics: Your First Kernel

## Why this matters

CUDA is the programming surface for NVIDIA GPUs. Even if you ultimately ship on Apple Silicon or write everything in Triton, CUDA is the substrate every other framework and DSL sits on. PyTorch's GPU backend is mostly CUDA kernels. Triton compiles down to CUDA. cuBLAS, cuDNN, NCCL — all CUDA. Knowing the language directly means you can read what the framework you're using is actually doing, debug when it goes wrong, and write the specialty kernel that no framework gives you.

This lesson teaches the smallest set of CUDA that gets you to a working matmul. We deliberately keep the kernel naive — there's a faster version coming in Lesson 5 — because the goal here is to build the toolchain and the mental model, not to optimize.

## Concept

### The host/device split

A CUDA program is two programs that share source. The **host** code runs on the CPU (your `main` function, allocations, kernel launches, file I/O). The **device** code runs on the GPU (your kernels, marked `__global__`). The nvcc compiler reads the source, splits it, and produces a binary that contains both.

Memory has two homes too: host RAM (allocated with `malloc` / `new`) and device memory (allocated with `cudaMalloc`). Pointers from one cannot be dereferenced by the other without an error or undefined behavior. (Unified Memory APIs blur this, but conceptually the split is real.)

A typical CUDA program flow:

```
1. Allocate host buffers, fill with data.
2. Allocate device buffers.
3. Copy host data → device.
4. Launch kernel.
5. Copy device result → host.
6. Free device buffers.
```

Steps 3 and 5 are where you pay PCIe transfer cost. Steps 1, 2, 6 are housekeeping. Step 4 is where the actual computation happens.

### Kernel launch syntax

```cuda
my_kernel<<<grid_dim, block_dim>>>(args...);
```

The triple-bracket `<<<...>>>` is the launch syntax (CUDA-specific; nvcc-specific). It tells the runtime how many threads to launch. `grid_dim` is the number of blocks. `block_dim` is the number of threads per block. Total threads = `grid_dim × block_dim`.

Both are `dim3` types, supporting up to 3 dimensions. For 1D problems: `grid_dim = (n_blocks)`, `block_dim = (threads_per_block)`. For 2D problems (like matmul over an MxN output): `grid_dim = (n_blocks_y, n_blocks_x)`, `block_dim = (block_y, block_x)`. Convention varies; we'll use 1D where possible and 2D where natural.

### Inside the kernel: figuring out which thread you are

A kernel sees three built-in variables:

- `threadIdx` — `dim3` giving the thread's position within its block. `threadIdx.x` is 0 to `blockDim.x - 1`.
- `blockIdx` — `dim3` giving the block's position within the grid.
- `blockDim` — `dim3` giving the size of each block (the values passed to `<<<...>>>`).

From these, each thread computes its global ID:

```cuda
int i = blockIdx.x * blockDim.x + threadIdx.x;
```

For a 2D problem:

```cuda
int row = blockIdx.y * blockDim.y + threadIdx.y;
int col = blockIdx.x * blockDim.x + threadIdx.x;
```

Each thread uses its global ID to decide which output element to compute. This is the whole "each thread does one piece of work" pattern.

### Memory management

```cuda
float* d_a;
cudaMalloc(&d_a, n * sizeof(float));      // allocate on device
cudaMemcpy(d_a, h_a, n * sizeof(float),
           cudaMemcpyHostToDevice);        // copy host → device
// ... launch kernels ...
cudaMemcpy(h_result, d_result, n * sizeof(float),
           cudaMemcpyDeviceToHost);        // copy device → host
cudaFree(d_a);                              // free device memory
```

Naming conventions you'll see across CUDA code: `d_` prefix for device pointers, `h_` prefix for host pointers. Not enforced; helpful.

### Error handling

CUDA calls can fail silently if you don't check. The idiom:

```cuda
cudaError_t err = cudaMalloc(&d_a, n * sizeof(float));
if (err != cudaSuccess) {
    fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
    exit(1);
}
```

Most production code wraps this in a macro:

```cuda
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while (0)
```

After every kernel launch, also check for errors:

```cuda
my_kernel<<<grid, block>>>(args);
CUDA_CHECK(cudaGetLastError());            // catch launch errors
CUDA_CHECK(cudaDeviceSynchronize());       // catch runtime errors
```

`cudaGetLastError` returns the result of the kernel launch itself (config errors). `cudaDeviceSynchronize` waits for the kernel to complete and returns any runtime error (out-of-bounds access in some cases, divide-by-zero on some configs, etc.).

### Synchronous vs asynchronous

Kernel launches are **asynchronous** by default — the call returns immediately, the kernel runs in the background. `cudaMemcpy` is synchronous by default (blocks until done). `cudaMemcpyAsync` is the asynchronous version, used with streams.

For our first kernel, `cudaDeviceSynchronize()` before reading the result is the simplest correctness-first pattern. Once you're more comfortable, async launches and streams enable kernel/memcpy overlap — the GPU equivalent of double-buffering. We'll see those in Lesson 8.

## Code walkthrough

A naive matmul kernel. Each thread computes one output element by running the inner-loop reduction itself.

```cuda
// matmul.cu

__global__ void matmul_naive_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int K, int N
) {
    // Which output element does this thread compute?
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Out-of-bounds guard for non-multiple sizes.
    if (row >= M || col >= N) return;

    // The inner-loop reduction.
    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
        acc += A[row * K + k] * B[k * N + col];
    }
    C[row * N + col] = acc;
}
```

A few things worth noticing.

**`__global__`** marks the function as a CUDA kernel callable from host. (Compare: `__device__` = callable from device only; `__host__` = host-callable; default is `__host__`.)

**`__restrict__`** is a hint that the three pointers don't alias each other. Same as Lesson 6 (SIMD) — it lets the compiler emit more aggressive code, since it doesn't need to assume a store to `C` might invalidate a future load from `A` or `B`.

**The bounds check.** When M or N isn't a multiple of the block dim, we launch slightly more threads than we have work for. The check makes the extra threads early-return harmlessly.

**The inner loop runs scalar.** Each thread does its own `K` iterations of FMA. This is "one CUDA core doing the work of one output cell." It's not using tensor cores. It's not using shared memory. It's the absolute baseline.

### Host code to launch the kernel

```cuda
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err)); \
        exit(1); \
    } \
} while (0)

int main() {
    const int M = 1024, K = 1024, N = 1024;

    // Allocate and initialize host data.
    float *h_A = (float*)malloc(M * K * sizeof(float));
    float *h_B = (float*)malloc(K * N * sizeof(float));
    float *h_C = (float*)malloc(M * N * sizeof(float));
    for (int i = 0; i < M * K; ++i) h_A[i] = ((float)rand() / RAND_MAX) - 0.5f;
    for (int i = 0; i < K * N; ++i) h_B[i] = ((float)rand() / RAND_MAX) - 0.5f;

    // Allocate device data.
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, M * K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, K * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C, M * N * sizeof(float)));

    // Copy A and B to device.
    CUDA_CHECK(cudaMemcpy(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice));

    // Launch configuration. Each block computes a 16x16 sub-grid of C.
    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);

    // Warm up.
    matmul_naive_kernel<<<grid, block>>>(d_A, d_B, d_C, M, K, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Timed runs.
    auto t0 = std::chrono::high_resolution_clock::now();
    const int runs = 10;
    for (int i = 0; i < runs; ++i) {
        matmul_naive_kernel<<<grid, block>>>(d_A, d_B, d_C, M, K, N);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    double dt = std::chrono::duration<double>(
        std::chrono::high_resolution_clock::now() - t0).count() / runs;

    double flops = 2.0 * M * N * K;
    printf("matmul_naive: %.3f ms  %.2f GFLOPs\n", dt * 1000, flops / dt / 1e9);

    // Read result back.
    CUDA_CHECK(cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    // Cleanup.
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    free(h_A); free(h_B); free(h_C);

    return 0;
}
```

Compile and run:

```bash
nvcc -O3 -arch=sm_80 -o matmul matmul.cu
./matmul
```

(`sm_80` targets Ampere; use `sm_90` for Hopper, `sm_89` for Ada/4090. The compiler defaults work but explicit is better.)

Expected performance for a 1024-cube on a 4090: maybe **2000–5000 GFLOPs** (2–5 TFLOPs). On an H100: maybe 5000–10000 GFLOPs. **Already 100× faster than our best CPU number from Module 1.** Note that this is the naive kernel; cuBLAS / cuDNN will be 10–20× faster than this on the same hardware. We're at maybe 5–10% of peak. Same situation as the naive CPU matmul in Module 1 lesson 4.

## Mental model & pitfalls

Single sentence: **A CUDA kernel is a function many threads run in parallel; each thread figures out which piece of work it owns from its grid coordinates; the host orchestrates allocations, transfers, and launches.**

Pitfalls:

- **Forgetting the bounds check.** With non-multiple sizes, the extra threads write out of bounds and corrupt memory. The `if (row >= M || col >= N) return` is mandatory unless you're guaranteed multiples.
- **Allocating per kernel launch.** `cudaMalloc` is slow (~50 µs). Allocate once at startup, reuse.
- **Copying per kernel launch.** PCIe is slow (~12 GB/s typical, vs HBM at 1–3 TB/s on the device). Stage data on the device and keep it there.
- **Launching too few threads.** A 16×16 block size on a 32×32 grid is 256 × 1024 = 256K threads. Plenty. A 16×16 block on a 4×4 grid is 4K threads — barely fills the chip.
- **Ignoring kernel launch errors.** `cudaGetLastError` after launch. Always.
- **Confusing thread count with block count.** Total threads = grid × block. Easy to misread `<<<grid, block>>>` and launch the wrong number.
- **Forgetting `cudaDeviceSynchronize` for timing.** Without it you're timing the launch, not the execution.

## Hands-on (at home)

1. **Write and run the naive matmul** as shown. Confirm performance numbers in the expected range for your GPU.

2. **Verify correctness against CPU.** Use the Module 1 CPU matmul or a numpy comparison:

```python
# verify.py
import numpy as np
a = np.fromfile('a.bin', dtype=np.float32).reshape(1024, 1024)
b = np.fromfile('b.bin', dtype=np.float32).reshape(1024, 1024)
c_gpu = np.fromfile('c.bin', dtype=np.float32).reshape(1024, 1024)
c_cpu = a @ b
print("max abs diff:", np.max(np.abs(c_gpu - c_cpu)))
```

You should see `max abs diff < 1e-2` for FP32 matmul; the difference is reduction-order, not bugs.

3. **Sweep block sizes.** Try `dim3 block(8,8)`, `(16,16)`, `(32,8)`, `(32,32)`. Performance varies. Why? Block size affects how many warps per block (block size 256 = 8 warps; the SM scheduler likes 8+ warps active).

4. **Sweep problem sizes.** 512, 1024, 2048, 4096 cubed. GFLOPs should *increase* with problem size — bigger problems amortize launch overhead. At very large sizes (8192+) you may run out of memory; that's expected.

5. **Compare to cuBLAS.**

```cuda
#include <cublas_v2.h>
cublasHandle_t handle;
cublasCreate(&handle);
float alpha = 1.0f, beta = 0.0f;
cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
            N, M, K,
            &alpha, d_B, N, d_A, K, &beta, d_C, N);
cublasDestroy(handle);
```

(Note: cuBLAS is column-major; the argument order above computes `C = A @ B` row-major by swapping operand order.) Time and compare. cuBLAS will be 5–20× faster. That's the gap we close over the next few lessons.

6. **(Optional) Profile with `nvprof` (older) or Nsight Compute (newer).** A taste of what's coming in Lesson 12:

```bash
ncu --set basic ./matmul
```

Look for the "Achieved Occupancy" and "Memory Throughput" metrics. The naive kernel will show high occupancy but moderate memory throughput — bandwidth-bound, not because of access pattern (it's coalesced) but because there's no data reuse.

## Further reading

- *CUDA C Programming Guide* — NVIDIA's official reference. The first 5 chapters are the right "intro" reading.
- "An Even Easier Introduction to CUDA" — Mark Harris, NVIDIA developer blog.
- *CUDA by Example* (Sanders & Kandrot) — the friendly introductory book. Dated in some specifics but the foundations are timeless.
- `cuda-samples` repo on GitHub — every CUDA primitive with a working example. Read 10 of them.

Next lesson: the memory hierarchy. We use shared memory to cut the bandwidth pressure on our matmul — and the speedup will be the same kind of jump as cache blocking gave us on the CPU.
