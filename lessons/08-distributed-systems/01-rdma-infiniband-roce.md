---
title: "Lesson 1 — RDMA, InfiniBand, RoCE"
date: "2026-06-04"
module: "distributed-systems"
order: 1
tags: ["rdma", "infiniband", "roce", "networking", "latency", "bandwidth"]
author: "Sudipta Pathak"
prerequisites: ["00-overview"]
---

# Lesson 1 — RDMA, InfiniBand, RoCE

## Why this lesson exists

Distributed ML training and inference involve transferring large tensors between GPUs in different machines. The network technology that makes this fast — RDMA (Remote Direct Memory Access) — is the unifying abstraction; the wire-level transports are InfiniBand (the established standard) and RoCE (RDMA over Converged Ethernet, the Ethernet-based alternative).

Without RDMA, multi-GPU training would be unviable at scale. Standard TCP/IP networking adds latency (microseconds per operation) and CPU overhead (the kernel handles every packet) that dominate the per-step communication cost.

This lesson is the foundation: what RDMA is, what InfiniBand and RoCE deliver, and why ML system performance depends on these technologies even though the average ML engineer never thinks about them.

The lesson is reading. The Hands-on inspects a node's network configuration.

## What RDMA does

Standard networking (TCP/IP):
1. Application calls `send()` with a buffer.
2. Kernel copies the buffer into kernel memory.
3. Kernel TCP stack packetizes, adds headers, queues for the NIC.
4. NIC sends packets over the wire.
5. On the receiver, NIC receives packets; kernel reassembles; kernel notifies application; application reads.

Latency: ~10-100 microseconds per call. CPU overhead per transferred byte. The kernel is involved at both ends.

RDMA (Remote Direct Memory Access):
1. Application pre-registers a memory region with the NIC.
2. Application calls "RDMA write" specifying source buffer + remote address.
3. NIC reads the buffer from memory directly (no kernel involvement).
4. Wire transmission.
5. On the receiver, NIC writes directly into the remote application's pre-registered memory (no kernel, no notification needed if it's a one-sided operation).

Latency: ~1-2 microseconds per call. Zero CPU overhead for the data transfer itself. The application's CPU is free to do other work while the transfer happens.

The "direct" in Remote Direct Memory Access means the NIC reads from and writes to application memory directly, bypassing the kernel. This is the central optimization.

## InfiniBand

InfiniBand (IB) is a network architecture purpose-built for high-performance computing. It supports RDMA natively from day one (early 2000s).

Properties:
- **Bandwidth**: 100 Gbps (FDR), 200 Gbps (EDR), 400 Gbps (NDR), 800 Gbps (XDR — emerging). Per link.
- **Latency**: ~1 microsecond switch-to-switch.
- **Hardware**: dedicated IB switches and host channel adapters (HCAs).
- **Cost**: expensive. IB equipment is a fraction of the total cluster cost.

InfiniBand is the dominant interconnect in supercomputers and large ML training clusters. NVIDIA's DGX systems use IB; most large cloud GPU clusters use IB internally.

## RoCE

RoCE (RDMA over Converged Ethernet) runs RDMA over standard Ethernet hardware. The advantage: leverages existing data-center Ethernet infrastructure; cheaper than dedicated IB.

Two versions:
- **RoCE v1**: RDMA over Layer 2 Ethernet. Limited to a single broadcast domain.
- **RoCE v2** (the one that matters): RDMA over UDP/IP. Routable, can span subnets.

RoCE v2 is what most ML training clusters that don't use IB use. Properties:
- Bandwidth: 100, 200, 400 Gbps depending on the Ethernet hardware.
- Latency: 1-3 microseconds (slightly higher than IB).
- Requires careful network configuration (lossless Ethernet via PFC and ECN) — packets dropped on the wire are catastrophic for RDMA.

The lossless-Ethernet requirement is the catch: configuring it correctly across many switches is non-trivial. Some clouds (AWS) historically struggled here.

## NVLink (preview, full coverage in Lesson 2)

NVLink is *intra-node* GPU-to-GPU interconnect — distinct from IB/RoCE which are *inter-node*. NVLink delivers 600-900 GB/s between GPUs in the same node; IB delivers 50-100 GB/s between nodes. The order-of-magnitude difference is why distributed parallelism strategies are typology-aware (Lesson 9).

## What this means for ML

For data-parallel training across multiple nodes:
- All-reduce of gradients each step is the dominant communication.
- For Llama 3.1 70B in BF16: ~140 GB of gradients per step.
- At 400 Gbps RoCE: ~3 seconds per step just for the all-reduce.
- At 800 Gbps NDR IB: ~1.5 seconds.
- The all-reduce algorithm and topology determine effective throughput; Lesson 4 covers this.

For TP across nodes:
- Multiple all-reduces per layer, each ~100-1000 MB.
- Per-layer all-reduce latency: ~10-100 microseconds.
- With 32 layers and 100 microseconds per layer all-reduce: 3.2 ms per step purely from cross-node communication.
- This is why TP is usually kept within a node (over NVLink).

The general principle: cross-node communication is 10-100× slower than intra-node. Design the parallelism to minimize cross-node communication.

## Production ML clusters in 2026

Typical configurations:
- **Hyperscale (Meta, Google, Microsoft, OpenAI)**: NDR InfiniBand (800 Gbps) within "scale unit" subdomains; RoCE or IB across the larger fabric.
- **Mid-scale (universities, startups, cloud rentals)**: 100-200 Gbps RoCE typically; some IB for premium configurations.
- **Small (researchers, hobbyists)**: single-node 8-GPU systems with NVLink; no cross-node concern.

For most ML engineers, RDMA exists silently. PyTorch + NCCL handles it; you set up the cluster correctly once and it works. The lesson here is to know enough to debug when it doesn't (e.g., NCCL failures often trace to network configuration problems).

## What you should believe after this lesson

Three sentences:

**1. RDMA enables zero-CPU-overhead, low-latency network transfers** by having the NIC read/write application memory directly. Standard TCP/IP would dominate ML communication cost; RDMA makes distributed training feasible.

**2. InfiniBand and RoCE v2 are the two RDMA transports**: IB is the established standard with dedicated hardware; RoCE runs over Ethernet with careful lossless configuration. Both deliver 100-800 Gbps per link with ~1-3 microsecond latency.

**3. Cross-node communication is 10-100× slower than intra-node (NVLink)** — the design principle for distributed parallelism is to minimize cross-node traffic. TP usually stays within a node; DP can span nodes; PP and sequence parallelism are flexible.

## Hands-on (at home)

Inspect your network configuration (Linux).

```bash
# Check for InfiniBand devices.
ibstat 2>/dev/null || echo "No InfiniBand"

# Check Ethernet links and speed.
for iface in $(ls /sys/class/net | grep -v lo); do
    speed=$(cat /sys/class/net/$iface/speed 2>/dev/null)
    echo "$iface: ${speed:-unknown} Mbps"
done

# Check for RDMA capabilities.
ls /sys/class/infiniband 2>/dev/null

# Check NVLink topology on a multi-GPU machine.
nvidia-smi topo -m 2>/dev/null
```

The `nvidia-smi topo -m` output is particularly useful: it shows the connectivity matrix between GPUs (NVLink vs PCIe vs Cross-NUMA). For a DGX H100, you'll see "NV18" between every GPU pair (18 NVLink lanes); for a consumer multi-GPU system, you'll see "PIX" or "PXB" (PCIe-based connectivity).

For a real cluster bandwidth test:

```bash
# Use NCCL's all_reduce benchmark.
git clone https://github.com/NVIDIA/nccl-tests
cd nccl-tests
make
# Run on 2 GPUs.
mpirun -np 2 ./build/all_reduce_perf -b 8 -e 128M -f 2 -g 1
```

The output shows bandwidth at various message sizes; you can identify the saturation point.

## Further reading

- "RDMA Aware Programming Manual" (NVIDIA / Mellanox) — the canonical reference.
- "InfiniBand Architecture" specification.
- "RoCE v2 Considerations for Loss-Sensitive Workloads" — practical RoCE configuration.
- "Data Center TCP" — for why TCP doesn't suffice without enhancements.

Next lesson: **NVLink, NVSwitch, and the bandwidth wall.** Intra-node GPU-to-GPU networking. The hardware that makes 8-GPU TP feasible and shapes the topology of every large training cluster.
