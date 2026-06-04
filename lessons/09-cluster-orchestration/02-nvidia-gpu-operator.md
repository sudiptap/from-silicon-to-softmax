---
title: "Lesson 2 — NVIDIA GPU Operator + Device Plugin"
date: "2026-06-04"
module: "cluster-orchestration"
order: 2
tags: ["nvidia", "gpu-operator", "device-plugin", "kubernetes", "drivers"]
author: "Sudipta Pathak"
prerequisites: ["01-kubernetes-for-ml"]
---

# Lesson 2 — NVIDIA GPU Operator + Device Plugin

## Why this lesson exists

To use GPUs on a Kubernetes cluster, three things need to be true:

1. The OS / kernel has NVIDIA drivers installed.
2. The container runtime can pass GPUs into containers.
3. The Kubernetes scheduler knows about GPUs as a resource.

Doing this manually on every node is tedious and error-prone. The NVIDIA GPU Operator automates it: install one CRD, get a working GPU stack across the cluster.

This lesson covers what the GPU Operator does, the underlying components (Device Plugin, driver, container toolkit, MIG manager), and what breaks if you skip the operator.

The lesson is reading. The Hands-on installs the GPU Operator on a cluster.

## What the GPU Operator does

The GPU Operator is a Helm-installable suite that deploys (as DaemonSets, one pod per GPU node):

1. **NVIDIA driver**: kernel modules. Either installed on the host or via a containerized driver that runs in a privileged pod.
2. **NVIDIA container toolkit**: extends the container runtime so containers can request and use GPUs.
3. **Device plugin**: a Kubernetes DaemonSet that registers each node's GPUs as schedulable resources (`nvidia.com/gpu: 8`).
4. **DCGM exporter**: GPU metrics for Prometheus scraping.
5. **GPU Feature Discovery**: labels nodes with GPU type, NVLink topology, etc.
6. **MIG manager** (optional): partitions H100 / A100 GPUs into Multi-Instance GPU slices.
7. **Validator**: a sanity-check pod that runs after install and verifies the stack works.

Without the operator, you'd install each piece manually on every node. With it, the operator manages versions, restarts components on changes, and keeps the cluster in sync.

## What the Device Plugin specifically does

The Device Plugin is a Kubernetes API: each node runs a plugin that *advertises* available resources to the kubelet. NVIDIA's device plugin tells kubelet "this node has 8 `nvidia.com/gpu` resources."

When a pod requests `nvidia.com/gpu: 4`, the scheduler:
1. Finds nodes with at least 4 unused `nvidia.com/gpu`.
2. Schedules the pod to one such node.
3. The Device Plugin allocates 4 specific GPUs on that node and tells the container runtime which to expose.

The pod sees its 4 GPUs as devices in `/dev/nvidia*`. The Device Plugin handles the mapping.

This is what makes "request 4 GPUs" work at the pod level.

## What happens without the GPU Operator

If you skip the operator and try to use GPUs:

- Without drivers: containers can't see the GPU at all. CUDA initialization fails.
- Without container toolkit: the runtime doesn't pass GPU devices into containers. `nvidia-smi` returns nothing inside the container.
- Without device plugin: K8s doesn't know GPUs exist. Pods requesting `nvidia.com/gpu` are unschedulable; pods that don't request them run but can't use the GPUs (no device passed in).

The "scheduler doesn't know" failure is particularly annoying — your pods sit in `Pending` state with no clear error message. Diagnosing requires `kubectl describe node` to check for the `nvidia.com/gpu` capacity entry.

## MIG (Multi-Instance GPU)

H100 and A100 GPUs can be partitioned into smaller "instances" via MIG:
- H100: up to 7 instances per GPU.
- A100: up to 7 instances per GPU.
- Each instance has its own memory partition and compute slice.

MIG is useful when you want to share a GPU across multiple small jobs (inference, dev work) without one job hogging the whole card.

The GPU Operator's MIG manager handles the partitioning:
- You declare a MIG configuration ("split this GPU into 4 × 1g.10gb instances").
- The manager applies it via `nvidia-smi mig` commands.
- The Device Plugin advertises the instances as separate resources (`nvidia.com/mig-1g.10gb: 4`).

Pods request MIG instances by their specific name. The scheduler routes them to nodes with available instances.

## The cluster-level view

After GPU Operator installation, `kubectl get nodes -o wide` shows GPU nodes with labels:
```
nvidia.com/gpu.count=8
nvidia.com/gpu.product=NVIDIA-H100-SXM5-80GB
nvidia.com/gpu.memory=81559
nvidia.com/gpu.machine=DGX-H100
```

These labels let you target specific GPU types in pod specs:
```yaml
spec:
  nodeSelector:
    nvidia.com/gpu.product: NVIDIA-H100-SXM5-80GB
  containers:
  - resources:
      limits:
        nvidia.com/gpu: 8
```

For mixed-GPU clusters (some A100, some H100), this targeting is essential.

## Driver versioning

The GPU Operator can install drivers via:
- **Host-installed driver**: the cluster admin installs the driver on the host OS (via package manager); the operator skips driver installation. Used in tightly-controlled bare-metal clusters.
- **Containerized driver**: the operator runs a privileged pod that loads driver modules. Used in clouds where you don't control the host OS.

Both work. The containerized driver is more "self-contained" (you can upgrade drivers by updating the operator's image); the host driver is more "stable" (less likely to have version-mismatch issues with the kernel).

## Production considerations

A few things to know:

- **Driver upgrade requires node drain**: you can't swap the driver on a running node without draining its pods first. The operator handles this if you let it.
- **CUDA library version vs driver version**: the container can use any CUDA version that's compatible with the host's driver. The driver is forward-compatible (host driver N supports CUDA versions ≤ N).
- **MIG configuration is node-wide**: you can't have one GPU on a node in MIG mode and another not. Plan your MIG layouts carefully.
- **Monitoring**: DCGM exporter exposes per-GPU metrics. Combined with Prometheus, you get utilization, memory, temperature dashboards per GPU.

## What you should believe after this lesson

Three sentences:

**1. The NVIDIA GPU Operator installs and manages the entire GPU stack on a Kubernetes cluster** — drivers, container toolkit, device plugin, metrics. Without it, you'd install each piece manually on every node.

**2. The Device Plugin is the K8s component that registers GPUs as schedulable resources** (`nvidia.com/gpu: 8`). Without it, the scheduler doesn't know GPUs exist; pods requesting them sit `Pending` forever.

**3. MIG lets you partition H100/A100 GPUs into smaller instances**; useful for sharing GPUs across many small workloads. The GPU Operator's MIG manager handles the partitioning; instances are scheduled as their own resource type.

## Hands-on (at home)

Install the GPU Operator on a cluster (requires a GPU node; not feasible on Minikube or kind).

```bash
# Using Helm on an existing cluster with GPU nodes.
helm repo add nvidia https://helm.ngc.nvidia.com/nvidia
helm repo update
helm install --wait --generate-name \
    -n gpu-operator --create-namespace \
    nvidia/gpu-operator

# Verify.
kubectl get pods -n gpu-operator
kubectl get nodes -o yaml | grep nvidia.com
```

You should see the operator's pods (driver, device-plugin, dcgm-exporter, etc.) in Running state on each GPU node; the nodes' labels include `nvidia.com/gpu.count`, etc.

Test with a GPU pod:

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: cuda-check
spec:
  containers:
  - name: check
    image: nvidia/cuda:12.4.0-runtime-ubuntu22.04
    command: ["nvidia-smi"]
    resources:
      limits:
        nvidia.com/gpu: 1
  restartPolicy: Never
```

`kubectl logs cuda-check` should show the standard nvidia-smi output.

## Further reading

- NVIDIA GPU Operator documentation.
- "Device Plugin Framework" in Kubernetes documentation.
- DCGM (Data Center GPU Manager) documentation.
- "Sharing GPUs in Kubernetes with MIG" (NVIDIA blog).

Next lesson: **The cluster-side view of a training job.** Pods, containers, devices, init containers, sidecars — how the components compose for a real ML job.
