---
title: "Lesson 3 — NCCL: Collectives and Topology Awareness"
date: "2026-06-04"
module: "distributed-systems"
order: 3
tags: ["nccl", "collectives", "all-reduce", "all-gather", "topology"]
author: "Sudipta Pathak"
prerequisites: ["02-nvlink-nvswitch"]
---

# Lesson 3 — NCCL: Collectives and Topology Awareness

## Why this lesson exists

NCCL (NVIDIA Collective Communications Library) is the standard library for GPU-to-GPU collective communication. Every PyTorch / DeepSpeed / Megatron distributed training run goes through NCCL. The library handles the topology-aware algorithm selection (Lesson 4), the chunking of large transfers into pipelined messages, and the integration with both NVLink and IB.

This lesson covers what NCCL provides at the API level, how it manages topology, and the small set of collectives that account for nearly all ML distributed traffic.

The lesson is reading. The Hands-on calls NCCL collectives via PyTorch.

## The NCCL collective set

The core operations:

- **AllReduce**: every process contributes a tensor; the result is the elementwise sum/max/min of all contributions, replicated on every process. Used for DDP gradient sync.
- **AllGather**: every process contributes a tensor; the result is the concatenation of all contributions, replicated on every process. Used for FSDP parameter gathering.
- **ReduceScatter**: every process contributes a tensor (split into chunks); each process gets the reduction of its corresponding chunk. The inverse of AllGather + AllReduce in some sense.
- **Broadcast**: one process sends a tensor; all others receive a copy. Used for initial weight broadcasting.
- **Reduce**: like AllReduce, but the result lands on only one process (not replicated to all).
- **AllToAll**: every process sends a different tensor chunk to every other process. Used for expert parallelism in MoE.
- **Send/Recv**: point-to-point. Used by pipeline parallelism for stage-to-stage transfers.

These cover ~95% of ML distributed communication.

## The topology awareness

When NCCL initializes, it inspects the GPU topology:
- Direct NVLink connections (which GPUs can talk to which over NVLink).
- NVSwitch presence.
- PCIe topology (which GPUs share a PCIe switch).
- Inter-node interconnect (which GPUs are on the same node vs different nodes).
- IB / RoCE configuration.

From this, NCCL builds a "topology graph" and picks algorithms appropriate for the graph. Within an NVLink domain, it uses high-bandwidth ring or tree algorithms (Lesson 4); across nodes, it uses bandwidth-conscious algorithms that minimize cross-node traffic.

This is automatic. You don't typically configure it; NCCL gets it right. The cases where you intervene:
- Cluster has non-standard topology (NCCL's automatic detection misses something).
- Specific performance debugging (NCCL_DEBUG=INFO shows the algorithm and topology choices).
- Tuning for a particular collective + size (NCCL_ALGO and NCCL_PROTO env vars).

## NCCL versions and feature evolution

NCCL has evolved through several major versions:
- **NCCL 2.0** (2017): foundational; supports AllReduce on NVLink + IB.
- **NCCL 2.7+** (2020): tree-based AllReduce for large clusters.
- **NCCL 2.10** (2021): double-binary-tree AllReduce (Lesson 4).
- **NCCL 2.18+** (2023-2024): low-latency protocols (LL128, LL); better small-message performance.
- **NCCL 2.20+** (2024+): user-buffer API; per-stream prioritization; better integration with MPI.

The version matters for performance. Recent NCCL has algorithmic improvements that older versions don't. PyTorch typically bundles a specific NCCL version; check `torch.cuda.nccl.version()` to see what you have.

## Communicators and groups

NCCL operations happen within a *communicator* — a group of processes that participate in collectives together. The communicator has:
- A unique ID.
- A list of participating ranks (process IDs).
- A topology that NCCL discovered.

For typical PyTorch DDP, there's one communicator across all training processes. For more complex setups (TP within a node + DP across nodes), you create multiple communicators:
- TP communicator: groups of 8 (one per node's tensor-parallel group).
- DP communicator: groups across nodes (one per data-parallel rank).

The PyTorch API exposes this via `torch.distributed.new_group(ranks=[...])` which creates a sub-communicator.

## The streams model

NCCL collectives are *asynchronous* by default. Calling `dist.all_reduce(tensor)` doesn't wait; it queues the work on a CUDA stream and returns. The work happens in the background.

For correctness, you wait on the operation before using the result:
- `tensor.wait()` — wait for this specific op.
- `torch.cuda.synchronize()` — wait for everything on the current stream.
- Or, structure your code so the next op that uses the result is on the same stream (it'll wait automatically).

The async model is what lets you overlap communication with compute. While the all-reduce is in flight, the GPU can be doing other compute work. This overlap is critical for performance; without it, communication time adds directly to step time.

## The protocols

NCCL has several "protocols" for moving data:
- **Simple** (default for large messages): direct DMA transfers.
- **LL** (Low Latency): for small messages; optimized for latency.
- **LL128**: a refined variant of LL.

The protocol choice is part of NCCL's automatic algorithm selection. You can override via `NCCL_PROTO=LL128` etc. for debugging or specific tuning.

## NCCL in production

In a normal training run, you never call NCCL directly. PyTorch's distributed module (`torch.distributed`) wraps it. DeepSpeed, Megatron, FSDP — all sit on PyTorch's wrapper.

What you do see:
- Configuration via environment variables (`NCCL_DEBUG`, `NCCL_SOCKET_IFNAME`, etc.).
- NCCL error messages when things go wrong.
- Profiling output showing per-collective times (via PyTorch profiler or NCCL's own logs).

The most common NCCL issue: network configuration. RoCE clusters with bad PFC/ECN configuration cause NCCL hangs and reduced throughput. Diagnosing requires understanding both NCCL and the network.

## What you should believe after this lesson

Three sentences:

**1. NCCL is the standard GPU-to-GPU collective communication library** — provides AllReduce, AllGather, ReduceScatter, Broadcast, AllToAll, and point-to-point operations. Every distributed PyTorch / DeepSpeed / Megatron training run uses it.

**2. NCCL is topology-aware**: it auto-discovers the GPU connectivity (NVLink, NVSwitch, IB, RoCE) at startup and picks algorithms appropriate for the topology. You rarely need to manually tune this.

**3. NCCL collectives are async by default**, queued on CUDA streams; this is what enables communication/compute overlap. The async model is central to distributed performance; without overlap, communication time adds directly to step time.

## Hands-on (at home)

Call NCCL collectives via PyTorch (requires 2+ GPUs, or simulate via `torchrun --nproc_per_node=2` with `device_id=0` for everyone).

```python
# nccl_demo.py
import torch
import torch.distributed as dist
import os

def main():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    
    # Each rank starts with a different tensor.
    x = torch.tensor([float(rank)] * 8, device=device)
    print(f"rank {rank} before: {x.tolist()}")
    
    # AllReduce: every rank gets the sum.
    dist.all_reduce(x, op=dist.ReduceOp.SUM)
    print(f"rank {rank} after AllReduce SUM: {x.tolist()}")
    # Each rank's tensor is now the sum of all ranks' original values.
    
    # AllGather: every rank collects all ranks' tensors.
    y = [torch.zeros(8, device=device) for _ in range(world_size)]
    dist.all_gather(y, torch.tensor([float(rank)] * 8, device=device))
    print(f"rank {rank} after AllGather: {[t.tolist() for t in y]}")
    
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
```

Run with `torchrun --nproc_per_node=2 nccl_demo.py` (or however many GPUs you have).

For a real distributed-bandwidth measurement, use the nccl-tests suite:

```bash
git clone https://github.com/NVIDIA/nccl-tests
cd nccl-tests && make
mpirun -np 4 ./build/all_reduce_perf -b 8 -e 1G -f 2 -g 1
```

You'll see effective bandwidth at various message sizes; the algorithm crossovers (LL → tree → ring) appear as bandwidth jumps.

## Further reading

- NCCL documentation (docs.nvidia.com/deeplearning/nccl/).
- "NCCL Algorithms and Protocols" (NVIDIA internal docs).
- PyTorch distributed documentation.
- "MPI and NCCL interactions" — for hybrid MPI+NCCL workloads.

Next lesson: **AllReduce algorithms — ring, tree, double binary tree.** The three families of all-reduce implementations and why each wins at different scales. The most important algorithm in distributed ML, derived from first principles.
