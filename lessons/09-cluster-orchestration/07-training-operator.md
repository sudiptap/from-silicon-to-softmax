---
title: "Lesson 7 — Training Operator (Kubeflow)"
date: "2026-06-04"
module: "cluster-orchestration"
order: 7
tags: ["kubeflow", "pytorchjob", "training-operator", "tfjob", "operator-pattern"]
author: "Sudipta Pathak"
prerequisites: ["06-mpi-operator"]
---

# Lesson 7 — Training Operator (Kubeflow)

## Why this lesson exists

The Kubeflow Training Operator is a unified operator that manages multiple ML training frameworks: PyTorchJob, TFJob, MPIJob, XGBoostJob, PaddleJob. One operator, many CRDs, each tailored to a specific framework's launching pattern.

For PyTorch distributed training on K8s, PyTorchJob (provided by Training Operator) is the standard. It handles the MASTER_ADDR/RANK/WORLD_SIZE env vars, the headless service, the optional gang scheduling, the retries — everything the cluster-side of Lesson 3 needs.

This lesson covers what the Training Operator does, the PyTorchJob CRD specifically, and the operator pattern that generalizes.

The lesson is reading. The Hands-on submits a PyTorchJob.

## The operator pattern

A Kubernetes "operator" is a control loop that watches custom resources (CRDs) and reconciles cluster state to match. For ML training:

- The CRD is `PyTorchJob` (or `TFJob`, etc.).
- The operator's reconciler watches PyTorchJob objects.
- For each, it creates a set of pods (Master + N Workers), a headless Service, env vars, and tracks the job's progress.
- When pods complete, it updates the PyTorchJob's status.

This is the same pattern as the MPI Operator (Lesson 6) but generalized for multiple frameworks.

## PyTorchJob structure

```yaml
apiVersion: kubeflow.org/v1
kind: PyTorchJob
metadata:
  name: my-training
spec:
  pytorchReplicaSpecs:
    Master:
      replicas: 1
      template:
        spec:
          containers:
          - name: pytorch
            image: my-pytorch:latest
            command: ["python", "train.py"]
            resources:
              limits: {nvidia.com/gpu: 8}
    Worker:
      replicas: 3
      template:
        spec:
          containers:
          - name: pytorch
            image: my-pytorch:latest
            command: ["python", "train.py"]
            resources:
              limits: {nvidia.com/gpu: 8}
```

This creates:
- 1 Master pod (rank 0) with 8 GPUs.
- 3 Worker pods (ranks 1-3) with 8 GPUs each.
- Total: 4 pods × 8 GPUs = 32 GPUs.

The operator injects environment variables into each pod:
- `MASTER_ADDR`: hostname of the Master pod's service.
- `MASTER_PORT`: chosen by the operator.
- `WORLD_SIZE`: total ranks (4 in this example).
- `RANK`: the pod's rank.

Your training script reads these env vars (typically via `torch.distributed.init_process_group()`) and connects to the master.

## RestartPolicy and BackoffLimit

For ML training, what happens when a worker fails?

The PyTorchJob spec has `restartPolicy`:
- `Always`: restart on any exit. Rare for training.
- `OnFailure`: restart only on failure. Useful for transient failures.
- `Never`: never restart. The job fails on first failure.
- `ExitCode`: restart based on the exit code.

And `backoffLimit`: max restart attempts before failing the job permanently.

For training that should checkpoint and resume, `OnFailure` with a high backoff (or none) is common. The script reads the latest checkpoint on startup.

## Elastic training (PyTorch elastic / TorchElastic)

For workloads that can handle dynamic worker counts, PyTorch Elastic (now integrated as `torchrun`) lets workers come and go without crashing the job. PyTorchJob supports the elastic mode:

```yaml
spec:
  elasticPolicy:
    rdzvBackend: c10d
    minReplicas: 4
    maxReplicas: 16
```

This says: start with at least 4 workers; allow up to 16; reconfigure the world size as workers join or leave.

Elastic training is good for spot/preemptible environments — losing one worker doesn't kill the job. The training script needs to support dynamic re-sharding (which adds complexity).

In practice, elastic training is more common for batch fine-tuning than for from-scratch pretraining (which usually demands a fixed world size for reproducibility).

## Other CRDs in the Training Operator

Beyond PyTorchJob:

**TFJob**: TensorFlow distributed training. The Master/Worker/PS (parameter server) split.

**MPIJob v1** (deprecated; v2 is in the standalone MPI Operator now): MPI training jobs.

**XGBoostJob**: distributed XGBoost training.

**PaddleJob**: PaddlePaddle.

**MXJob**: MXNet (mostly historical now).

Each CRD has framework-specific defaults. The PyTorchJob's defaults reflect PyTorch's distributed conventions; TFJob's reflect TF's.

The CRD set has been consolidating; some older ones are deprecated. PyTorchJob is the well-supported, modern flagship.

## Status and observability

PyTorchJob has a status section showing:
- Current state: Created / Running / Succeeded / Failed.
- Pod conditions: which replicas are running.
- Start time, end time, last reconcile time.

`kubectl describe pytorchjob my-training` shows all of this. Combined with `kubectl logs` on the pods, this is the diagnostic surface.

## Integration with batch schedulers

PyTorchJob integrates with both Kueue and Volcano:

For Kueue:
```yaml
metadata:
  labels:
    kueue.x-k8s.io/queue-name: research-queue
```

For Volcano:
```yaml
spec:
  schedulerName: volcano
```

The operator creates the pods; Kueue gates admission OR Volcano schedules with gang semantics. Use one or the other.

## What you should believe after this lesson

Three sentences:

**1. The Kubeflow Training Operator manages framework-specific CRDs (PyTorchJob, TFJob, MPIJob, etc.)** — one operator handles them all. PyTorchJob is the modern flagship for PyTorch distributed training on K8s.

**2. PyTorchJob injects MASTER_ADDR, WORLD_SIZE, RANK environment variables** into each pod and manages the headless service for stable hostnames. Your training script reads these env vars and connects via `torch.distributed.init_process_group()`.

**3. The CRD integrates with Kueue (admission control) and Volcano (gang scheduling)** via labels and `schedulerName`. For multi-node training, you need one of these for proper queue management.

## Hands-on (at home)

Submit a PyTorchJob (requires Kubeflow Training Operator installed).

```bash
# Install Training Operator.
kubectl apply --server-side -k "github.com/kubeflow/training-operator/manifests/overlays/standalone?ref=v1.8.1"

# A simple CPU-only PyTorchJob.
cat > pytorchjob-demo.yaml <<'EOF'
apiVersion: kubeflow.org/v1
kind: PyTorchJob
metadata:
  name: pt-demo
spec:
  pytorchReplicaSpecs:
    Master:
      replicas: 1
      template:
        spec:
          containers:
          - name: pytorch
            image: pytorch/pytorch:2.4.0-cpu
            command:
            - python
            - -c
            - |
              import os
              import torch.distributed as dist
              dist.init_process_group(backend='gloo')
              print(f"rank={dist.get_rank()}, world_size={dist.get_world_size()}, MASTER_ADDR={os.environ.get('MASTER_ADDR')}")
              dist.destroy_process_group()
    Worker:
      replicas: 2
      template:
        spec:
          containers:
          - name: pytorch
            image: pytorch/pytorch:2.4.0-cpu
            command: ["python", "-c", "<same as Master>"]
EOF
kubectl apply -f pytorchjob-demo.yaml

# Watch.
kubectl get pytorchjob pt-demo
kubectl logs -l job-name=pt-demo
```

You'll see each pod print its rank, world_size, and master_addr — the operator-injected env vars.

For a real GPU training job, replace the image with `pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime`, add `nvidia.com/gpu: N` to the container resources, and replace `gloo` with `nccl` for the backend.

## Further reading

- Kubeflow Training Operator GitHub.
- "PyTorchJob" documentation in Kubeflow.
- "TorchElastic" documentation for elastic training.

Next lesson: **Slurm.** The HPC standard, why frontier labs still use it, and how it compares to the K8s ecosystem we've built up.
