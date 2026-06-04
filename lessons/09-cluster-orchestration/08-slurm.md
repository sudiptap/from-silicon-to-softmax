---
title: "Lesson 8 — Slurm"
date: "2026-06-04"
module: "cluster-orchestration"
order: 8
tags: ["slurm", "hpc", "batch", "scheduler", "supercomputer"]
author: "Sudipta Pathak"
prerequisites: ["07-training-operator"]
---

# Lesson 8 — Slurm

## Why this lesson exists

Slurm (Simple Linux Utility for Resource Management) is the dominant scheduler for HPC clusters and a significant share of ML training clusters. While the Kubernetes ecosystem has grown rapidly, Slurm remains the choice at many frontier labs — OpenAI's training clusters, Anthropic's, large national-lab GPU systems, most academic HPC.

The reasons: Slurm has 20+ years of operational maturity for tight-coupled batch jobs; it's built around the things ML training needs (gang scheduling, topology, fair-share); its overhead is much lower than Kubernetes for the batch case.

This lesson covers Slurm's design, its API, and the practical workflow for ML on Slurm.

The lesson is reading. The Hands-on installs a single-node Slurm via Docker.

## What Slurm is

Slurm is a workload manager / scheduler:
- Runs on a head node (the controller, slurmctld) and on every compute node (slurmd).
- Users submit batch jobs (`sbatch`), interactive jobs (`salloc`/`srun`), or single commands (`srun`).
- The scheduler decides when each job runs based on priority, fair-share, resource availability, and topology.
- Jobs run until completion, failure, or time limit.

A typical Slurm cluster:
- Login nodes (where users submit jobs).
- Controller node (runs slurmctld).
- Compute nodes (run slurmd, execute jobs).
- A shared filesystem (NFS, Lustre, WekaFS — Lesson 14).

## The Slurm API

Users interact via command-line tools:

**sbatch**: submit a batch job script.
```bash
sbatch --gres=gpu:8 --nodes=4 --time=12:00:00 train.sh
```

**srun**: run a command (with allocation). Used both standalone and inside sbatch scripts.
```bash
srun --gres=gpu:1 --nodes=1 nvidia-smi
```

**squeue**: list jobs.
```bash
squeue -u $USER
```

**scancel**: cancel a job.
**sacct**: historical accounting (job runtime, exit code, etc.).
**sinfo**: cluster node status.

The mental model: jobs are submitted to *partitions* (queues); the scheduler runs them as resources are available.

## A typical training script

```bash
#!/bin/bash
#SBATCH --job-name=llama-train
#SBATCH --nodes=4
#SBATCH --gres=gpu:8       # 8 GPUs per node
#SBATCH --ntasks-per-node=8  # 8 tasks per node
#SBATCH --cpus-per-task=8
#SBATCH --mem=512G
#SBATCH --time=24:00:00
#SBATCH --output=/path/to/logs/%x-%j.out

module load cuda/12.4 openmpi/4.1.5

# srun launches the training on all allocated GPUs.
srun python train.py
```

`#SBATCH` directives configure the job; `srun python train.py` is what actually runs. Inside `srun`, environment variables tell each process its `SLURM_PROCID` (rank), `SLURM_NTASKS` (world size), etc.

For PyTorch distributed:
- Use `SLURM_PROCID` as `RANK`.
- Use `SLURM_NTASKS` as `WORLD_SIZE`.
- Use the first node's hostname as `MASTER_ADDR`.

```python
import os
import torch.distributed as dist

os.environ['RANK'] = os.environ['SLURM_PROCID']
os.environ['WORLD_SIZE'] = os.environ['SLURM_NTASKS']
os.environ['MASTER_ADDR'] = os.environ['SLURM_NODELIST'].split(',')[0]
os.environ['MASTER_PORT'] = '29500'
dist.init_process_group(backend='nccl')
```

This is the standard Slurm + PyTorch integration. Many wrapper scripts handle it.

## Scheduling features

Slurm has rich scheduling:

**Fair-share**: each user/account has a "share"; the scheduler accounts for past usage to determine priority. Heavy users get deprioritized; light users get boosted. This keeps the cluster fair across long timescales.

**Backfill**: when a long high-priority job is waiting for resources, Slurm checks if any low-priority short job can fit in the gap without delaying the high-priority job. Improves utilization.

**Topology awareness**: Slurm knows the network topology (defined in `topology.conf`) — racks, switches, nodes. When allocating multi-node jobs, it prefers nodes that are "close" on the network.

**Gang scheduling and preemption**: jobs can preempt lower-priority ones; with checkpointing, the preempted job resumes later.

**Reservations**: admins can reserve nodes for specific users, times, or jobs. Useful for guaranteed-capacity SLAs.

These features make Slurm well-suited for shared HPC-style ML clusters.

## Slurm performance profile

Slurm's overhead for batch jobs is low:
- Job submission: <100 ms.
- Job start: <1 second.
- Per-job overhead during execution: negligible.

Compare to Kubernetes: a PyTorchJob's setup (pod creation, scheduling, image pull, SSH setup, etc.) can take 30-60 seconds before training starts.

For long-running jobs (hours+), this overhead doesn't matter. For short jobs (interactive shells, quick benchmarks), Slurm's low overhead is a real win.

## Slurm's limitations

The flip side:

**Less flexible than Kubernetes.** Slurm is a batch scheduler; running a long-lived service (e.g., a serving endpoint) is awkward. Possible, but you're fighting the model.

**Mostly Linux/bare-metal.** Slurm doesn't have a great cloud story. AWS ParallelCluster wraps Slurm for AWS but isn't as fluid as native EKS.

**Less containerized.** Slurm uses Conda/module/spack environments natively; containers via Singularity (later renamed Apptainer) work but feel bolted-on. K8s is container-native.

**Smaller community.** K8s has orders of magnitude more contributors. Slurm community is HPC-focused; smaller but deeply expert.

**Single point of failure** at the controller. K8s has more inherent HA. Slurm has HA controllers but it's more involved.

For pure ML training at scale on dedicated hardware, Slurm wins. For mixed workloads, multi-cloud, or service-style deployments, K8s wins. Lesson 9 covers the comparison in depth.

## Production Slurm patterns

A few practical Slurm patterns for ML:

**Job arrays**: submit many similar jobs at once.
```bash
sbatch --array=1-100 train.sh
```
Each array index runs the script with `SLURM_ARRAY_TASK_ID` set. Used for hyperparameter sweeps.

**Dependencies**: jobs can wait for others.
```bash
sbatch --dependency=afterok:JOBID_OF_PRETRAIN train_finetune.sh
```

**Checkpointing**: handled by your training script. Slurm doesn't automatically checkpoint. The standard is to checkpoint every N steps; on Slurm's signal (default SIGTERM at time limit minus a few minutes), save and exit cleanly.

**Requeueing**: `sbatch --requeue` causes the job to be re-queued automatically on certain failures (preemption, node failures). Combined with checkpointing, this gives Slurm's version of fault tolerance.

## What you should believe after this lesson

Three sentences:

**1. Slurm is the dominant HPC scheduler** and a strong choice for dedicated ML training clusters. Mature, low-overhead, built around the things ML training needs (gang scheduling, topology, fair-share).

**2. The Slurm API is sbatch / srun / squeue / scancel / sacct** — submit batch jobs with `#SBATCH` directives, launch processes with srun. PyTorch distributed integrates via SLURM_PROCID / SLURM_NTASKS environment variables.

**3. Slurm wins for pure ML training on dedicated hardware**; K8s wins for mixed workloads, multi-cloud, or service-style deployments. The choice depends on what else lives on the cluster.

## Hands-on (at home)

Run a single-node Slurm via Docker.

```bash
# Pull a Slurm Docker image.
docker run --rm -it -d --name slurm \
    -h slurmctl \
    schedmd/slurm-deploy:24.05

# Open a shell in the Slurm container.
docker exec -it slurm bash

# Inside the container:
sinfo  # show cluster status
srun hostname  # run a simple command
sbatch <<'EOF'
#!/bin/bash
#SBATCH --job-name=hello
#SBATCH --output=/tmp/hello.out
echo "Hello from Slurm job $SLURM_JOB_ID"
sleep 10
EOF
squeue  # see the job
cat /tmp/hello.out  # see the output
```

For a real ML deployment, you'd have a multi-node Slurm cluster with GPUs; the principles are the same.

## Further reading

- Slurm documentation (slurm.schedmd.com).
- "Slurm Workload Manager Quick Start" guide.
- "Running PyTorch on Slurm" tutorials.
- AWS ParallelCluster / Azure CycleCloud — wrappers for Slurm in the cloud.

Next lesson: **Slurm vs Kubernetes.** The genuine tradeoffs. Not "Slurm bad" or "K8s ate the world" but a honest comparison for ML workloads.
