---
title: "Lesson 11 — Topology-Aware Scheduling"
date: "2026-06-04"
module: "cluster-orchestration"
order: 11
tags: ["topology", "scheduling", "nvlink", "numa", "switch-aware"]
author: "Sudipta Pathak"
prerequisites: ["10-kuberay"]
---

# Lesson 11 — Topology-Aware Scheduling

## Why this lesson exists

Two 32-GPU jobs requesting the same resources from a cluster can perform dramatically differently based on which 32 GPUs they're allocated:

- 32 GPUs across 4 NVLink-connected nodes (one IB rack): NCCL all-reduce ~5 ms.
- 32 GPUs across 16 nodes spanning 2 racks: same all-reduce ~25 ms.

The 5× difference is just topology. The scheduler that picks the GPUs determines this; naive scheduling spreads jobs across the cluster's available nodes regardless of topology, halving throughput compared to a topology-aware scheduler.

This lesson covers what topology-aware scheduling does, how Slurm and Kubernetes implement it (or fail to), and the practical impact on multi-node training throughput.

The lesson is reading. The Hands-on inspects topology on a real cluster.

## What topology to be aware of

ML clusters have multiple network levels:

1. **NVLink within a node**: 900 GB/s; 100 ns latency. Free for tight communication.
2. **NUMA within a node**: GPUs may be on different CPU sockets, with PCIe paths through different NUMA domains. Affects CPU-GPU traffic.
3. **Top-of-rack (ToR) switch**: nodes within a rack share a switch. NVLink-class IB for the latest gear (NDR), ~1 μs latency.
4. **Cluster spine**: racks connect to a spine layer. Additional hops; higher latency.
5. **Multi-cluster/region**: cross-data-center. Don't go here for ML training.

The "topology-aware" scheduler considers these levels when allocating multi-node jobs.

## What naive scheduling does wrong

A naive scheduler (default K8s, plain Slurm without topology config) assigns nodes based on availability alone:

- 32 GPUs requested → find 32 GPUs across any 4 nodes.
- If the cluster has 100 nodes, the chosen 4 nodes might be scattered across racks.

Inter-rack communication is ~5× slower than intra-rack. The 4-node job that lands on 4 different racks pays this cost every all-reduce.

For a typical training run with 30% communication time, the topology mismatch can mean the job runs at 60-70% of the topology-aware throughput.

## Slurm's topology awareness

Slurm has had topology support since 2008+ via `topology.conf`:

```
SwitchName=rack1 Nodes=node[001-008]
SwitchName=rack2 Nodes=node[009-016]
SwitchName=spine Switches=rack1,rack2
```

This tells Slurm:
- Nodes 001-008 are in rack1 (shared switch).
- Nodes 009-016 are in rack2.
- rack1 and rack2 connect via the spine.

When scheduling a multi-node job, Slurm prefers nodes "close" in the topology — same rack first; spine-spanning only if necessary.

The `--switches` flag on `srun` / `sbatch` enforces this:

```bash
sbatch --switches=1 --nodes=8 train.sh
```

Asks for 8 nodes that all share one switch (one rack). Slurm waits until such a contiguous allocation is available.

## Kubernetes' topology story

Kubernetes' topology-aware scheduling is younger and less mature.

**Built-in features**:
- **Pod affinity/anti-affinity**: hint to schedule pods "near" or "away from" others. Coarse-grained.
- **Topology spread constraints**: distribute pods across failure zones evenly. Helpful for HA, not for tight coupling.
- **Node labels**: label nodes with rack info; use nodeSelector to constrain. Manual.

**With operators**:
- **Volcano's task-topology plugin**: schedules pods of a single job on topology-close nodes.
- **Kueue's topology-aware admission** (recent): admits jobs only when topology-suitable nodes are available.
- **Custom schedulers**: many production K8s clusters write custom schedulers for ML topology.

Cloud providers help here:
- **GKE**: GPU clusters with topology-aware scheduling for H100 / A100 pods.
- **AWS Capacity Blocks**: pre-defined GPU placement groups; you get topology guarantees by booking blocks.
- **CoreWeave / Lambda**: ML-specialized clouds where topology is baked into the cluster design.

In 2026 K8s topology-aware scheduling for ML is improving rapidly but isn't yet as mature as Slurm's. Production K8s ML clusters often combine Volcano + custom annotations + manual rack labeling.

## NUMA awareness

Within a node, NUMA topology matters for CPU-GPU traffic:
- Each GPU is connected to a specific PCIe root complex.
- Each PCIe root complex is closer to one CPU socket (NUMA node).
- Data loading or other CPU-side work should run on the CPU socket "close" to its GPU.

NUMA-aware scheduling pins data-loader threads to the right CPUs. Without it, the cross-socket traffic slows things down by 10-20% on data-loading-heavy workloads.

Tools:
- **numactl**: manual CPU pinning (`numactl --cpunodebind=0 --membind=0 ./trainer`).
- **PyTorch DataLoader's `num_workers`**: implicit; the OS handles distribution. Not optimal.
- **NVIDIA's Magnum IO**: handles NUMA-aware scheduling internally.

For most production ML, the framework handles NUMA reasonably. For squeezing the last 10-20%, explicit pinning helps.

## How much does it actually matter

Empirical numbers for a 64-GPU training run:

| Topology | Step time | Communication % |
| -------- | --------- | --------------- |
| 8 nodes, all in one rack (NDR IB) | 100 ms | 10% |
| 8 nodes, spanning 2 racks | 130 ms | 30% |
| 8 nodes, spanning 4 racks | 160 ms | 45% |
| 8 nodes, random across cluster | 180-200 ms | 55%+ |

A topology-aware allocation can be 2× faster than a worst-case naive allocation. Over a multi-day training run, the difference is enormous.

## What to do

For production ML clusters:

1. **Label your nodes with topology metadata** (rack, switch, NVLink group). Automate this if possible (e.g., from CMDB or cluster spec).
2. **Use a topology-aware scheduler**: Slurm with `topology.conf`; K8s with Volcano + appropriate config.
3. **Request specific topology in job submissions**: `--switches=1` in Slurm; Volcano task-topology in K8s.
4. **Monitor**: dashboard that shows per-job NCCL bandwidth; outliers indicate topology problems.

For cloud deployments:
- AWS: use Capacity Blocks for ML; specify placement groups.
- GCP: use GKE's compact placement groups; A3 / A3-mega instances with H100 / H200 are NVLink-optimized.
- Azure: similar; reserved Ndv5 instances for tight coupling.

## What you should believe after this lesson

Three sentences:

**1. Topology-aware scheduling is the difference between 60% and 100% throughput** on multi-node ML training. The same job on the same hardware runs ~2× faster with the right node allocation.

**2. Slurm has mature topology awareness** via `topology.conf` and `--switches`; Kubernetes is catching up via Volcano's task-topology and custom schedulers. Cloud providers (GKE, AWS Capacity Blocks) offer topology guarantees baked into instance types.

**3. NUMA awareness within nodes** is a 10-20% optimization for data-loading-heavy workloads. Most frameworks handle it implicitly; explicit pinning via numactl matters when squeezing the last bit.

## Hands-on (at home)

Inspect topology on a multi-GPU machine.

```bash
# Show GPU-to-GPU connectivity.
nvidia-smi topo -m

# Show NUMA topology.
numactl -H

# For a GPU pod's CPU affinity recommendations.
nvidia-smi topo -c <GPU_INDEX>
```

For a multi-node cluster, inspect the Slurm topology configuration:

```bash
scontrol show topology
```

For a Kubernetes cluster, check node labels for topology metadata:

```bash
kubectl get nodes --show-labels | grep -E "topology|rack|zone"
```

If your cluster lacks topology labels, you're at the mercy of the default scheduler. Add them for measurable training-time improvement.

## Further reading

- Slurm `topology.conf` documentation.
- Volcano task-topology plugin documentation.
- "Topology-Aware Scheduling for HPC on Kubernetes" — various papers.
- AWS Capacity Blocks for ML announcement.

Next lesson: **Multi-tenancy.** Quotas, priorities, preemption, fair-share. How to share a $50M cluster across ten teams without anyone starving.
