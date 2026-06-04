---
title: "Lesson 7 — Memory-Mapped Weights and Weight Streaming"
date: "2026-06-04"
module: "on-device-llm-inference"
order: 7
tags: ["mmap", "weight-streaming", "page-fault", "llama-cpp", "model-loading"]
author: "Sudipta Pathak"
prerequisites: ["06-kv-cache-tiny-budget"]
---

# Lesson 7 — Memory-Mapped Weights and Weight Streaming

## Why this lesson exists

A naive model load reads the weight file from disk and copies it into a buffer in process memory. For a 2 GB quantized model on a phone, that means:
1. Allocate 2 GB of RAM.
2. Read 2 GB from flash storage (takes 4–8 seconds at typical mobile NAND speeds).
3. Free the file handle, the model is "loaded."

For a desktop with fast NVMe, this is fast enough; on a phone the 4–8 seconds of cold start is user-visible and painful. There's also the memory pressure question: a 2 GB resident weight buffer leaves little room for KV cache and other state.

The alternative — `mmap` (memory-map) the file — sidesteps both problems. The OS maps the file's pages into virtual memory; no actual reads happen until the application touches a page; the OS pages in on demand and pages out under memory pressure. Cold start is instant. Memory use is bounded by what's currently touched. This is what llama.cpp does and the technique is one of the under-appreciated reasons it's the production on-device LLM runtime.

This lesson is the practical view of mmap'd weights: what mmap actually does at the OS level, the page-fault behavior during inference, when this strategy wins, and when explicit loading wins.

The lesson is reading. The Hands-on profiles the page-fault behavior of a mmap'd model.

## What mmap does

`mmap(file, length)` is a POSIX system call that asks the OS: "make this file's contents accessible in my virtual address space starting at some address, with length `length`." After the call, you have a pointer; reading through that pointer behaves as if the file were already in memory.

What actually happens:
- The OS sets up page table entries that say "this virtual page corresponds to this offset in this file."
- No actual disk I/O happens at mmap time. The pages aren't loaded; they're just *mapped*.
- When you first access (read or write) a page, the OS catches the page fault, reads the page from disk, and resumes your code. This is a *demand-paged* load.
- The OS keeps recently-accessed pages in its file cache. Subsequent reads are from RAM, not from disk.
- Under memory pressure, the OS can evict cached pages (they're clean — they're backed by the file on disk — so eviction is free).

Two flavors:
- **`MAP_PRIVATE`**: writes to the mapped region are private to your process; the file isn't modified. Used for read-only access plus copy-on-write modifications.
- **`MAP_SHARED`**: writes to the mapped region get written back to the file. Used for memory-mapped files used as IPC.

For model weights you use `MAP_PRIVATE` — the model file is read-only.

## Why mmap wins for model loading

The benefits of mmap for LLM weights:

**1. Instant "load."** `mmap` itself is microseconds. The model file is "available" after that call returns. The first inference triggers the actual reads. Cold start time drops from seconds to ~100 ms.

**2. Memory use is what you touch, not what you allocated.** A 2 GB model mmap'd only consumes RAM for the pages you've actually accessed. During the first inference, you access most pages (the forward pass touches every layer); after that the file cache holds them. But if the OS needs that RAM, it evicts the pages — they'll fault back in on next access.

**3. Multiple processes sharing the same model.** If two processes mmap the same model file, the OS maps the same physical pages into both processes' address spaces. Memory is shared. (For a chat app with multiple conversation windows backed by separate processes, this matters.)

**4. Page-cache friendliness.** The OS's file cache is well-tuned. Hot pages stay; cold pages get evicted. You get the OS's smart-eviction-policy for free.

**5. Power efficiency.** Pages not touched aren't loaded. For a model that's loaded but inference hasn't started yet, almost zero RAM consumed.

## The page-fault profile during inference

What does "page fault on first access" look like in practice? For a Llama 3.2 3B Q4_K_M model (~1.8 GB), mmap'd from flash:

**First prefill** (process the input prompt through the model):
- Every layer's weights are touched. Most pages fault in.
- Cold prefill on a phone is slower than warm prefill — maybe 2–3× slower for the first run, because the page faults serialize with the kernel reads.
- Some pages stay cold longer than others (e.g., the high-numbered layers if the prompt is short and decode begins quickly).

**Steady-state decode** (generating tokens after prefill):
- Every layer's weights are accessed once per token.
- Pages stay warm in the OS file cache; no faults.
- Throughput is what you'd see from a fully-loaded model.

**Idle period** (no inference for several minutes):
- The OS may evict the cached pages if it needs the RAM for something else (browser, etc.).
- The next inference faults pages back in, paying the cold-start cost again.

This profile means mmap'd LLMs have *latency stability* issues. The first inference after a cold start is slow; the next several are fast; if the model gets paged out (because the user opens a memory-heavy app), the next inference is slow again. For a chat where the user types occasionally, the cache stays warm. For an inference triggered minutes apart (e.g., a notification that triggers a generation), each one may pay the cold cost.

Mitigation: explicit `mlock` to keep specific pages in RAM (POSIX). llama.cpp has `--mlock` for this. The cost: the memory is now permanently locked, removing the "elastic memory use" benefit of mmap.

## When mmap loses

There are cases where explicit-load beats mmap:

**1. The model fits comfortably in RAM and stays loaded.** If you have plenty of RAM and the model is small, explicit load + keep resident has no downsides. The page-fault overhead is real (a few microseconds per fault, multiplied by millions of faults during prefill). With explicit load you skip them.

**2. Network or remote filesystem.** mmap'ing a file from a network filesystem is technically possible but the page-fault behavior becomes a network round-trip, which is catastrophic. Always explicit-load if the file is remote.

**3. Strict latency requirements with cold-start tolerance.** Real-time systems where the cold-start cost matters and the application can do explicit preloading. Set up the model once during app launch (perhaps with a progress bar), then run inference without any chance of page faults.

The 90% case on consumer hardware: mmap wins. The 10% cases are server-side or specialized.

## How llama.cpp uses mmap

llama.cpp's GGUF loader uses mmap by default. The model file is mapped; the runtime accesses weights through the mapped region. No explicit "load weights into RAM" step.

The flags relevant to mmap behavior:

- **`--mmap` / `--no-mmap`**: enable / disable mmap. Default is enabled.
- **`--mlock`**: lock the mapped pages in RAM so they can't be evicted. Useful for predictable latency at the cost of giving up the elasticity.
- **`-ngl N`**: offload N layers to the GPU. The offloaded layers' weights are copied from the mapped region into GPU-accessible memory (which, on Apple Silicon's unified memory, doesn't actually move physically — see Module 4 Lesson 2). On NVIDIA with discrete VRAM, this is a real copy that happens at load time, eliminating the page-fault benefit for GPU-offloaded layers.

On Apple Silicon, the combination is particularly nice: mmap'd weights + unified memory means the GPU reads pages from the OS cache without any copy. Pages fault in as the GPU touches them.

## Streaming weights from disk

A related but distinct technique: instead of mmap'ing the whole file, *stream* weights from disk on demand. The model deliberately doesn't hold all weights in memory simultaneously; instead, layer N's weights are loaded right before layer N executes, then evicted.

When this is interesting:
- The model is too large to fit even with mmap. Streaming lets you run a 70B model on a 16 GB Mac by holding ~2 GB in memory at any time (the current few layers).
- Generation throughput is acceptable at the disk-read rate (typically tens of MB/s on flash; 1 GB/s on NVMe).

The catch: throughput is now disk-bound. A 7B model that runs at 30 tok/s when fully resident might run at 5 tok/s when streamed. Acceptable for batch inference (overnight summarization of documents); painful for interactive use.

For most on-device deployments, this isn't the right answer; pick a smaller model that fits. But it's a known fallback for cases where you must run a model larger than the device's RAM.

Apple's "Speculative Decoding for Streaming Inference" research (2024) sketches a more sophisticated streaming pattern that overlaps disk reads with computation. As of 2026 it's not in production runtimes; the practical recipe is "fit the model or pick a smaller one."

## What you should believe after this lesson

Three sentences:

**1. mmap is the default model-loading strategy for on-device LLMs in 2026** — instant cold start, demand-paged memory use, OS-managed cache. llama.cpp uses it; explicit-load wins only in specific cases (small model + plenty of RAM, network filesystems, real-time strict-latency apps with preloading).

**2. The page-fault profile creates latency-stability issues** — the first inference after a cold start is slow because pages must fault in; idle periods can evict pages and re-create the cold-start cost. `mlock` mitigates this at the cost of giving up the elastic memory benefit.

**3. Streaming weights from disk lets you run models larger than RAM** but throughput drops to disk-bound rates. It's a known technique with limited practical use; the on-device recipe in 2026 is usually "fit the model or pick a smaller one."

## Hands-on (at home)

Profile mmap behavior on your machine.

```bash
# Cold start vs warm start.
# (Use `purge` on macOS to clear file cache before first run.)
# On Linux: `sync; echo 3 > /proc/sys/vm/drop_caches`.

# Run 1 (cold): time the prefill of a long prompt.
time ./llama-cli -m gguf/Llama-3.2-3B-Instruct-Q4_K_M.gguf \
    --no-conversation -p "$(cat 4k_prompt.txt)" -n 1 -ngl 0
# Note the "prompt eval time" line in the output.

# Run 2 (warm): immediately after.
time ./llama-cli -m gguf/Llama-3.2-3B-Instruct-Q4_K_M.gguf \
    --no-conversation -p "$(cat 4k_prompt.txt)" -n 1 -ngl 0
# Should be 2-4x faster on prefill thanks to OS file cache.

# Run 3 (--no-mmap): the explicit-load comparison.
time ./llama-cli -m gguf/Llama-3.2-3B-Instruct-Q4_K_M.gguf \
    --no-mmap --no-conversation -p "$(cat 4k_prompt.txt)" -n 1 -ngl 0
# Cold start now includes the explicit read (slower), but no page faults during inference.
```

The pattern you'll see:
- Cold mmap'd: ~100 ms startup + slower first prefill due to page faults.
- Warm mmap'd: ~100 ms startup + fast prefill (everything cached).
- `--no-mmap`: longer startup (file is read entirely) but steady prefill throughput.

For an interactive workload where multiple inferences happen close in time, mmap with warm cache is the win. For a one-shot inference where you load and immediately exit, explicit load is sometimes competitive.

For the `mlock` experiment:

```bash
# Run with mlock; the model can't be evicted from RAM.
./llama-cli -m gguf/Llama-3.2-3B-Instruct-Q4_K_M.gguf \
    --mlock --no-conversation -p "..." -n 100 -ngl 0
```

On macOS this may require running with sudo or adjusting ulimits because mlock memory is generally restricted.

## Further reading

- POSIX `mmap` documentation — the foundational system call.
- "What Every Programmer Should Know About Memory" (Drepper, 2007) — has a thorough mmap section; old but still mostly accurate.
- llama.cpp source: `llama-model-loader.cpp` and the GGUF loader for the production mmap-based loader.
- "Memory-Mapped I/O" (kernel docs across Linux / macOS) — for OS-side details on page-cache management.
- "Practical Vulnerability Demonstration of mmap'd LLM Weights" — there have been a few interesting writeups about the security implications of mmap'd weights (process can see weights of any model another process mmap'd if permissions allow); worth knowing about for security-sensitive deployments.

Next lesson: **Streaming generation patterns.** Now that the model loads fast and runs efficiently, we look at the UX layer: how to stream tokens to the user as they're generated, handle cancellation cleanly, and roll back partial output when the user interrupts.
