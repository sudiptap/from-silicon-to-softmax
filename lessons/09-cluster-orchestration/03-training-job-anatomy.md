---
title: "Lesson 3 — The Cluster-Side View of a Training Job"
date: "2026-06-04"
module: "cluster-orchestration"
order: 3
tags: ["pods", "containers", "init-containers", "sidecars", "anatomy"]
author: "Sudipta Pathak"
prerequisites: ["02-nvidia-gpu-operator"]
---

# Lesson 3 — The Cluster-Side View of a Training Job

## Why this lesson exists

A real ML training job on a cluster isn't just one container running `torchrun`. It's a coordinated set of containers, init containers, sidecars, environment variables, volume mounts, and network configurations. Understanding the anatomy is what lets you debug "why is my job's first epoch slow" or "why does process 3 of 8 hang."

This lesson walks through the components of a typical distributed training pod, what each does, and the common failure modes.

The lesson is reading. The Hands-on inspects a real PyTorchJob pod's anatomy.

## The components

A typical distributed PyTorch training job's pod looks like this:

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: training-worker-0
spec:
  initContainers:
  - name: wait-for-master
    image: busybox
    command: ["sh", "-c", "until nslookup master; do sleep 2; done"]
  
  containers:
  - name: trainer
    image: my-pytorch:latest
    command: ["torchrun", ...]
    env:
    - name: MASTER_ADDR
      value: "training-worker-0.training-svc"
    - name: WORLD_SIZE
      value: "32"
    - name: RANK
      value: "0"  # this worker's rank
    resources:
      limits:
        nvidia.com/gpu: 8
        memory: 256Gi
    volumeMounts:
    - name: checkpoint-dir
      mountPath: /checkpoints
    - name: shared-data
      mountPath: /data
  
  - name: log-shipper
    image: fluentd:latest
    volumeMounts:
    - name: log-dir
      mountPath: /var/log/training
  
  volumes:
  - name: checkpoint-dir
    persistentVolumeClaim:
      claimName: training-checkpoints
  - name: shared-data
    persistentVolumeClaim:
      claimName: training-data
  - name: log-dir
    emptyDir: {}
```

Each piece does something specific. Let's walk through.

## Init containers

`initContainers` run sequentially *before* the main containers. Use cases:
- Wait for dependencies (e.g., the master node to be ready before workers start).
- Download or pre-process data (e.g., extract a dataset from object storage).
- Run schema migrations or any setup that must complete before the main work starts.

In ML, init containers are most useful for the "wait for master" pattern in distributed training. Workers shouldn't start `torchrun` until the master is reachable; the init container blocks the main container until then.

## Main containers (typically one for ML)

The container with the actual training process. For ML, this is usually a single container running `torchrun` (or `mpirun`, or the framework's entrypoint).

Multiple containers in the same pod share network and (optionally) storage but are otherwise independent. Multi-container pods are useful for:
- Sidecars (next).
- Specialized roles (a "data loader" container feeding a "trainer" container via shared memory).

For most ML, a single training container per pod is the right structure.

## Sidecars

A *sidecar* is a secondary container that runs alongside the main container, sharing its lifecycle. Common ML sidecars:

- **Log shipper** (fluentd, fluent-bit): collects training logs and ships them to a central system.
- **Metrics exporter** (custom): exposes training metrics for Prometheus scraping.
- **Auth proxy**: handles authentication to external services on behalf of the trainer.
- **GPU monitor**: collects per-GPU metrics during the job (DCGM-style).

Sidecars don't have GPU access; they're CPU/network only. They typically use much less resource than the main container.

The sidecar pattern was popular for many use cases but is increasingly being replaced by node-level agents (DaemonSets that handle logging cluster-wide, not per-pod). For ML, the typical pattern is: DaemonSet for cluster-wide concerns; sidecar for job-specific concerns.

## Environment variables

The trainer container needs environment variables to know its role in the distributed group:

- `MASTER_ADDR`: hostname or IP of the master.
- `MASTER_PORT`: port the master listens on.
- `WORLD_SIZE`: total number of workers.
- `RANK`: this worker's rank (0 to WORLD_SIZE - 1).
- `LOCAL_RANK`: this worker's rank within its node (0 to gpus_per_node - 1).

For PyTorch distributed, these are the standard. `torchrun` sets `LOCAL_RANK` for each spawned process; the operator sets `RANK`, `WORLD_SIZE`, etc. before launching the container.

When you use PyTorchJob (Lesson 7), the operator handles the env var injection. When you launch pods manually, you set them yourself.

## Volume mounts

The trainer needs:
- **Checkpoint directory**: persistent storage for saving and resuming checkpoints. Survives pod restarts.
- **Data directory**: shared training data, typically a network filesystem.
- **(Sometimes) shared memory**: tmpfs or shared volume for inter-process communication on the same node.
- **(Sometimes) cache directories**: for HuggingFace model cache, conda packages, etc.

Storage choices (covered in detail in Lesson 14):
- `emptyDir`: temporary, deleted when pod ends. Fine for scratch space.
- `hostPath`: a path on the node. Useful for caches; not portable across nodes.
- `PersistentVolumeClaim`: backed by a CSI driver (Lustre, WekaFS, EFS, GCS Fuse). The standard for shared training data and checkpoints.

## Networking

For distributed training, each pod needs to know how to reach other pods. The patterns:

**StatefulSet + Headless Service**: pods get stable hostnames (`worker-0.svc-name.namespace.svc.cluster.local`). The headless service doesn't load-balance; it returns DNS records for each pod. PyTorch uses these for `MASTER_ADDR`.

**Pod IPs**: each pod has an IP; the operator can inject these into env vars. Less stable (IPs change on pod restart) but works.

**Host networking**: pods share the node's network namespace. Useful for high-performance networking but breaks K8s's port isolation. Used in some HPC-style ML deployments.

For most ML clusters, StatefulSet + headless service is the standard pattern.

## Resource requests vs limits

The pod's resource spec has two parts:
- **Requests**: what the scheduler reserves for the pod (for scheduling decisions).
- **Limits**: what the kernel enforces as a maximum.

For GPUs, requests = limits (you can't have a fractional GPU unless you use MIG). For CPU and memory, the values can differ; the scheduler reserves the request, but the pod may burst up to the limit.

For ML training, set request = limit for memory (avoid OOM kills mid-training) and for CPU often as well (avoid throttling). For GPUs, the limit is the count; request must match.

## The failure modes

Common failures of this pattern:

**Init container fails**: the main container never starts. Logs are in the init container.

**Volume mount fails**: pod stuck in `ContainerCreating`. Check the events: `kubectl describe pod`.

**Env var missing**: `torchrun` fails with confusing errors. Usually `MASTER_ADDR` unreachable.

**Sidecar dies**: depending on `restartPolicy`, may restart or crash the whole pod.

**OOM kill**: limits exceeded. Check `kubectl describe pod` for `OOMKilled`. Increase memory limit or fix the leak.

**GPU not visible**: container runs but `torch.cuda.is_available()` returns False. Check the device plugin and the pod's resource request.

## What you should believe after this lesson

Three sentences:

**1. A real ML training pod has multiple parts**: an init container for setup/wait, a main container with the trainer (env vars, volume mounts, resource limits), often a sidecar for logging or monitoring. The operator (Kubeflow's PyTorchJob etc.) generates this YAML for you.

**2. Environment variables (MASTER_ADDR, WORLD_SIZE, RANK, LOCAL_RANK)** are how the trainer knows its role; the operator injects them based on the pod's position in the StatefulSet.

**3. Persistent volumes (PVCs) for checkpoints and data** are the storage layer; volume mounts attach them to the container. Without proper PVCs, your training can't survive pod restarts.

## Hands-on (at home)

Inspect a real PyTorchJob pod's anatomy.

```bash
# Apply a PyTorchJob (requires Kubeflow Training Operator installed).
cat > pytorchjob.yaml <<'EOF'
apiVersion: kubeflow.org/v1
kind: PyTorchJob
metadata:
  name: simple-train
spec:
  pytorchReplicaSpecs:
    Master:
      replicas: 1
      template:
        spec:
          containers:
          - name: pytorch
            image: pytorch/pytorch:2.4.0-cpu
            command: ["python", "-c", "import torch.distributed as dist; dist.init_process_group('gloo'); print(f'rank={dist.get_rank()}, world_size={dist.get_world_size()}')"]
    Worker:
      replicas: 2
      template:
        spec:
          containers:
          - name: pytorch
            image: pytorch/pytorch:2.4.0-cpu
            command: ["python", "-c", "import torch.distributed as dist; dist.init_process_group('gloo'); print(f'rank={dist.get_rank()}, world_size={dist.get_world_size()}')"]
EOF
kubectl apply -f pytorchjob.yaml

# Inspect the generated pods.
kubectl get pods -l job-name=simple-train -o yaml | head -100
```

You'll see the pod has env vars `MASTER_ADDR`, `WORLD_SIZE`, `RANK` automatically injected by the operator.

For a real GPU job, replace the image with `pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime` and add `nvidia.com/gpu: 1` to the resource limits.

## Further reading

- Kubernetes documentation on Pods, Containers, InitContainers, and Sidecars.
- "Designing Distributed Systems" (Burns) — chapters on the multi-container pod patterns.
- Kubeflow Training Operator documentation.

Next lesson: **Kueue.** Kubernetes-native job queueing. The basic batch scheduler for ML on K8s.
