---
title: "Lesson 6 — MPI Operator"
date: "2026-06-04"
module: "cluster-orchestration"
order: 6
tags: ["mpi", "operator", "kubernetes", "mpirun", "nccl"]
author: "Sudipta Pathak"
prerequisites: ["05-volcano"]
---

# Lesson 6 — MPI Operator

## Why this lesson exists

MPI (Message Passing Interface) is the traditional HPC programming model: a "launcher" starts processes on a set of "workers," gives each one a rank, and they communicate via MPI primitives. `mpirun -np 32` is the canonical command — and it implicitly assumes you have SSH-reachable workers, shared filesystem, the right environment, etc.

The MPI Operator translates this model to Kubernetes. You submit an MPIJob; the operator creates a launcher pod plus worker pods, sets up SSH, ensures the launcher can `mpirun` against the workers, and reports back when the job completes.

For ML, MPI is most relevant for two things:
1. NCCL — the GPU collective library — supports MPI as its launcher and bootstrap.
2. Horovod — a Uber-developed ML training framework — uses MPI as its primary launch mechanism.

This lesson covers the MPI Operator and the MPI launch pattern.

The lesson is reading. The Hands-on installs the MPI Operator and runs an MPIJob.

## What the MPI Operator does

The MPI Operator (kubeflow/mpi-operator) is a Kubernetes operator that:
1. Watches for `MPIJob` resource creation.
2. Creates a *launcher* pod (the one that runs `mpirun`) and N *worker* pods.
3. Sets up SSH between launcher and workers (generates SSH keys, distributes them, configures sshd).
4. Generates a hostfile listing the worker pods' addresses.
5. Launches the launcher pod, which runs `mpirun -hostfile ...` against the workers.
6. Workers exit when their MPI processes exit; launcher exits when `mpirun` returns.

The user-facing API:

```yaml
apiVersion: kubeflow.org/v2beta1
kind: MPIJob
metadata:
  name: nccl-test
spec:
  slotsPerWorker: 8  # GPUs per worker
  runPolicy:
    cleanPodPolicy: Running
  mpiReplicaSpecs:
    Launcher:
      replicas: 1
      template:
        spec:
          containers:
          - name: mpi-launcher
            image: nvcr.io/nvidia/k8s/cuda-sample:nbody
            command:
            - mpirun
            args:
            - -np
            - "16"
            - --allow-run-as-root
            - -bind-to
            - none
            - -map-by
            - slot
            - -x
            - LD_LIBRARY_PATH
            - -x
            - PATH
            - -x
            - NCCL_DEBUG=INFO
            - /workspace/nccl-tests/build/all_reduce_perf
            - -b
            - "8"
            - -e
            - "128M"
            - -f
            - "2"
            - -g
            - "1"
    Worker:
      replicas: 2
      template:
        spec:
          containers:
          - name: mpi-worker
            image: nvcr.io/nvidia/k8s/cuda-sample:nbody
            resources:
              limits:
                nvidia.com/gpu: 8
```

This launches an `all_reduce_perf` benchmark across 16 GPUs (2 workers × 8 GPUs).

## What the operator handles for you

Without the operator:
1. You'd manually create launcher and worker pods.
2. You'd set up SSH keys somehow.
3. You'd construct a hostfile.
4. You'd worry about port conflicts.
5. You'd write logic to detect when workers are ready.

With the operator:
- It generates SSH keys, mounts them into pods.
- It maintains a hostfile via a ConfigMap.
- It uses a headless Service for stable worker hostnames.
- It waits for workers to be ready before starting the launcher.

The operator handles the "boring but tricky" parts. You write the MPIJob spec and the operator does the rest.

## MPI vs torchrun

`torchrun` (PyTorch's distributed launcher) is more recent and Kubernetes-friendly than MPI. For PyTorch-specific workloads, PyTorchJob (Lesson 7) is usually the better choice.

MPI is still relevant for:
- **NCCL benchmarks**: the standard nccl-tests use MPI to launch.
- **Horovod**: still used in some production systems.
- **Multi-framework workloads**: MPI is framework-agnostic; you can launch TF + PyTorch + custom code with one `mpirun`.
- **HPC-origin codebases**: scientific computing apps that already use MPI.

For new PyTorch-only training: PyTorchJob > MPI. For NCCL benchmarks and Horovod: MPI is still the right tool.

## The launcher-worker pattern

A subtlety: the MPI Operator's launcher is a *separate pod* from the workers. The launcher doesn't usually need GPU; it just runs `mpirun`. The workers have the GPUs and run the actual MPI processes.

If you set GPU on the launcher, you're wasting a GPU (it's idle while mpirun coordinates). Don't.

The headless service `nccl-test-worker` lets the launcher refer to workers as `nccl-test-worker-0`, `nccl-test-worker-1`, etc. — stable hostnames for the hostfile.

## SSH between pods

MPI uses SSH for launcher → worker communication. The operator:
1. Generates an SSH keypair on operator startup.
2. Stores the keypair in a Secret.
3. Mounts the Secret into both launcher and workers.
4. Configures sshd in the workers (the worker image must include sshd; nvcr.io's images usually do).

For images that don't include sshd, you need to add it via init container or rebuild the image.

The SSH-based pattern is showing its age in 2026; cleaner alternatives exist (`mpirun --bootstrap=plm/slurm` or rsh-style), but SSH remains the most portable.

## Combining with Volcano

The MPI Operator integrates with Volcano (Lesson 5) for gang scheduling:

```yaml
apiVersion: kubeflow.org/v2beta1
kind: MPIJob
metadata:
  name: nccl-test
spec:
  schedulerName: volcano
  # ...
```

With `schedulerName: volcano`, the MPI Operator creates a Volcano PodGroup; Volcano gang-schedules the launcher and workers together.

For multi-node ML training, this combination is standard: MPI Operator for the launch pattern, Volcano for gang scheduling.

## What you should believe after this lesson

Three sentences:

**1. The MPI Operator translates the MPI programming model (`mpirun -np N`) into Kubernetes** by creating a launcher pod + N worker pods, setting up SSH, and managing the hostfile. Standard for NCCL benchmarks, Horovod, and other MPI-based ML workloads.

**2. For PyTorch-specific workloads, PyTorchJob (Lesson 7) is the better choice** — more Kubernetes-native. MPI is still the right tool for cross-framework or HPC-origin workloads.

**3. MPI Operator integrates with Volcano for gang scheduling** — `schedulerName: volcano` in the MPIJob spec creates a PodGroup; Volcano ensures launcher and workers start atomically.

## Hands-on (at home)

Install the MPI Operator and run an MPIJob.

```bash
# Install MPI Operator.
kubectl apply --server-side -f \
    https://raw.githubusercontent.com/kubeflow/mpi-operator/master/deploy/v2beta1/mpi-operator.yaml

# Run a simple MPI hello-world (CPU only).
cat > mpi-hello.yaml <<'EOF'
apiVersion: kubeflow.org/v2beta1
kind: MPIJob
metadata:
  name: mpi-hello
spec:
  slotsPerWorker: 1
  runPolicy:
    cleanPodPolicy: Running
  mpiReplicaSpecs:
    Launcher:
      replicas: 1
      template:
        spec:
          containers:
          - name: mpi-launcher
            image: mpioperator/mpi-pi:openmpi
            command:
            - mpirun
            args: ["-np", "2", "--allow-run-as-root", "/home/mpiuser/pi"]
    Worker:
      replicas: 2
      template:
        spec:
          containers:
          - name: mpi-worker
            image: mpioperator/mpi-pi:openmpi
EOF

kubectl apply -f mpi-hello.yaml
kubectl logs -l job-name=mpi-hello,replica-type=launcher
```

You should see the MPI pi-calculator output. For GPU NCCL tests, swap to the nccl-tests image and add GPU resource requests.

## Further reading

- MPI Operator GitHub (kubeflow/mpi-operator).
- "Running MPI Jobs on Kubernetes with Volcano" tutorials.
- nccl-tests GitHub (NVIDIA/nccl-tests).

Next lesson: **Training Operator (Kubeflow).** PyTorchJob, MPIJob, TFJob, and friends — the framework-specific operators that simplify training on K8s.
