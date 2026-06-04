---
title: "Lesson 2 — NVLink, NVSwitch, and the Bandwidth Wall"
date: "2026-06-04"
module: "distributed-systems"
order: 2
tags: ["nvlink", "nvswitch", "intra-node", "bandwidth", "topology"]
author: "Sudipta Pathak"
prerequisites: ["01-rdma-infiniband-roce"]
---

# Lesson 2 — NVLink, NVSwitch, and the Bandwidth Wall

## Why this lesson exists

NVLink is NVIDIA's intra-node GPU-to-GPU interconnect. Each generation has doubled bandwidth: NVLink 1 (160 GB/s aggregate) → NVLink 4 (900 GB/s on H100) → NVLink 5 (1.8 TB/s on Blackwell). NVSwitch — the chip that connects multiple GPUs in a fully-connected topology — enables 8-GPU and larger configurations within one node.

For distributed ML, the key fact: intra-node NVLink delivers 5-20× the bandwidth of inter-node IB/RoCE. The "node" is the unit within which tight communication (tensor parallelism, expert parallelism with all-to-all) lives; cross-node is reserved for looser communication (data parallel all-reduce on gradients).

This lesson covers NVLink's architecture, NVSwitch's topology role, and the bandwidth wall that shapes ML system design.

The lesson is reading. The Hands-on inspects NVLink connectivity on a multi-GPU machine.

## NVLink architecture

A single NVLink "lane" delivers ~25 GB/s in each direction (50 GB/s bidirectional). Each generation increases per-lane bandwidth and per-GPU lane count:

| Generation | Per-lane | Lanes per GPU | Aggregate per GPU |
| ---------- | -------- | ------------- | ----------------- |
| NVLink 1 (P100) | 20 GB/s | 4 | 80 GB/s bidir |
| NVLink 2 (V100) | 25 GB/s | 6 | 150 GB/s bidir |
| NVLink 3 (A100) | 25 GB/s | 12 | 300 GB/s bidir |
| NVLink 4 (H100) | 25 GB/s | 18 | 450 GB/s bidir (900 GB/s aggregate) |
| NVLink 5 (B100) | 50 GB/s | 18 | 900 GB/s bidir (1.8 TB/s aggregate) |

The "aggregate per GPU" is what determines how much data the GPU can send/receive concurrently. At 900 GB/s, an H100 can transfer 1 GB in ~1.1 ms.

The wire-level: NVLink uses NVIDIA's proprietary high-speed serial (similar to PCIe but customized). Cable lengths are short (sub-meter); NVLink is strictly within a node.

## NVSwitch

With many GPUs in a node (8 in DGX, 16 in some configurations), connecting them all peer-to-peer requires N×(N-1)/2 NVLink connections. For 8 GPUs that's 28 connections — physically impractical at the per-GPU lane budget.

NVSwitch solves this: an NVLink switch chip that connects multiple GPUs in a *fully-connected* topology with the same per-pair bandwidth as direct NVLink.

DGX H100:
- 8 GPUs.
- 4 NVSwitch chips.
- Every GPU pair has full NVLink-4 bandwidth (450 GB/s bidir).
- Total cross-section bandwidth: massive.

DGX B100 / GB200 NVL72:
- Up to 72 GPUs in one "NVLink domain" (across multiple physical machines linked by NVSwitch).
- All 72 GPUs have NVLink-5 bandwidth to each other.
- This is the largest single tight-coupling group as of 2026.

The NVLink domain is the unit of "tight" communication. Within it, TP and EP work efficiently. Across NVLink domains, you fall back to IB/RoCE.

## The bandwidth wall

Why does this matter so much?

A 70B model in BF16 with 8-way tensor parallel:
- Per-step all-reduce per layer: ~2 GB (rough; depends on hidden dim).
- 80 layers: 160 GB of all-reduce per step.
- Over 900 GB/s NVLink: 180 ms per step.
- Over 100 GB/s RoCE (cross-node TP): 1600 ms per step. 9× slower.

This is why TP is kept within a node. If you tried 8-way TP across nodes over RoCE, the communication would dwarf the compute; you'd train at a tiny fraction of the GPU's potential FLOPs.

The "bandwidth wall": each step, you have to all-reduce or all-to-all some amount of data. The faster the link, the less time spent on this; the higher the effective throughput. Beyond a certain link speed, communication becomes negligible; below it, communication dominates.

NVLink puts the bandwidth wall high enough that intra-node parallelism is mostly free. IB/RoCE put it lower; you have to design around it.

## Topology awareness

NCCL (Lesson 3) is topology-aware: it inspects the GPU-to-GPU connectivity at startup and picks all-reduce algorithms appropriate for the topology. Within an NVLink domain, NCCL uses fast paths; across NVLink boundaries, slower paths.

A typical 2026 large-scale topology:
- **NVLink domain** (8 GPUs in a DGX, or 72 in a GB200): tight coupling. NVSwitch full-bandwidth all-pairs.
- **Rack**: 4-8 nodes connected by top-of-rack InfiniBand switches.
- **Pod**: many racks connected by spine switches.
- **Cluster**: pods connected at the fabric core.

Communication bandwidth and latency degrade as you cross levels: NVLink → IB local → IB regional → IB fabric. NCCL routes through the optimal path for each collective.

## The Grace Hopper / Grace Blackwell story

NVIDIA's Grace CPUs (ARM-based) and the Grace Hopper / Grace Blackwell superchips link CPU memory to GPU memory at NVLink-class bandwidths (900+ GB/s). The CPU memory becomes a fast extension of GPU memory.

For ML: the Grace Hopper allows loading models from CPU memory at much higher bandwidth than PCIe (the traditional CPU-GPU link is ~64 GB/s on PCIe 5). For large-model training, this is meaningful — you can stage weights or KV cache in CPU memory and fetch on demand.

The GB200 NVL72 takes this further: 72 Grace+Blackwell pairs in a single rack, all coupled by NVLink and NVSwitch. Effectively a "single GPU" with 13.5 TB of unified memory. The largest training jobs run on these.

## What you should believe after this lesson

Three sentences:

**1. NVLink is intra-node GPU-to-GPU interconnect with 5-20× the bandwidth of inter-node IB/RoCE** — currently 900 GB/s on H100, 1.8 TB/s on Blackwell. NVSwitch enables all-pairs fully-connected topology within a node.

**2. The "bandwidth wall" is real**: communication-heavy parallelism (TP, EP) must live within the NVLink domain to avoid being dominated by communication cost. Cross-NVLink-domain communication falls to IB/RoCE which is 10× slower.

**3. The NVLink domain is the unit of tight coupling** — 8 GPUs in current DGX, 72 in GB200 NVL72. The largest training runs are designed around the NVLink domain boundaries; algorithms partition tight-communication work to fit.

## Hands-on (at home)

Inspect NVLink connectivity on a multi-GPU machine.

```bash
# Show the GPU topology.
nvidia-smi topo -m
```

Output example for an 8-GPU H100 DGX:

```
        GPU0    GPU1    GPU2    GPU3    GPU4    GPU5    GPU6    GPU7
GPU0     X      NV18    NV18    NV18    NV18    NV18    NV18    NV18
GPU1    NV18     X      NV18    NV18    NV18    NV18    NV18    NV18
...
```

Each "NV18" means 18 NVLink lanes (full H100 NVLink-4) between that pair. For consumer cards, you'll see "PIX" (PCIe) or "PXB" (PCIe bridge).

Run NCCL bandwidth test:

```bash
# Single-node 8-GPU test.
mpirun -np 8 ./build/all_reduce_perf -b 1M -e 1G -f 2 -g 1
```

The output shows bandwidth at increasing message sizes. On NVLink-4 you should see ~400-450 GB/s for large messages; on PCIe you'll see ~25 GB/s.

## Further reading

- NVIDIA NVLink documentation.
- "DGX H100 White Paper" and "DGX B100 White Paper" — architectural references.
- "GB200 NVL72 Reference Architecture" — the 72-GPU NVLink domain.
- Module 4 Lesson 1 — Apple Silicon's unified memory as a different solution to the same bandwidth problem.

Next lesson: **NCCL: collectives and topology awareness.** NVIDIA's collective communication library, the abstraction layer above NVLink + IB. How the algorithms pick the right path through the topology automatically.
