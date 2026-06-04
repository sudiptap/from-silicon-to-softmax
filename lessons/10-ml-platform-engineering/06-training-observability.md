---
title: "Lesson 6 — Training Observability"
date: "2026-06-04"
module: "ml-platforms"
order: 6
tags: ["observability", "prometheus", "grafana", "dcgm", "nan-canary", "anomaly-detection"]
author: "Sudipta Pathak"
prerequisites: ["05-checkpoint-storage"]
---

# Lesson 6 — Training Observability

## Why this lesson exists

A training run that's silently going wrong for hours is the worst case: you've burned compute, you don't have a usable model, and you'd have caught it earlier with the right monitoring. Common failure modes: loss NaN, gradient explosion, GPU thermal throttling, memory leaks, dead processes, divergent ranks.

This lesson covers training observability: GPU metrics via Prometheus + DCGM + Grafana, model-level metrics (loss, gradient norms), NaN canaries, anomaly detection. The goal is to catch problems early — minutes, not hours.

The lesson is reading. The Hands-on sets up a DCGM exporter for GPU metrics.

## The metric layers

ML training has several metric layers:

**Hardware**: GPU utilization, GPU memory, temperature, power, ECC errors. Per-GPU; sampled at 1-10 Hz.

**Framework**: forward/backward time per step, optimizer step time, allreduce time, batch processing rate. Per-rank; sampled per step.

**Model**: loss, gradient norms, parameter norms, learning rate, eval metrics. Per-step or per-evaluation-interval.

**System**: CPU usage, memory usage, network bandwidth, disk I/O. Per-node; sampled at 1-10 Hz.

A complete observability setup captures all four.

## DCGM and Prometheus

**DCGM (Data Center GPU Manager)**: NVIDIA's GPU monitoring agent. Runs as a daemon on each GPU node; exposes per-GPU metrics via a metrics endpoint.

**DCGM Exporter**: a sidecar that scrapes DCGM and exposes the metrics in Prometheus format (HTTP endpoint with key-value metrics).

**Prometheus**: time-series database that scrapes metrics endpoints, stores the data, runs queries.

**Grafana**: visualization layer on top of Prometheus. Dashboards.

The data flow:
```
GPU → DCGM (host daemon) → DCGM Exporter (sidecar) → Prometheus (scrapes) → Grafana (visualizes)
```

For a Kubernetes cluster, the GPU Operator (Lesson 2 of Module 9) installs DCGM Exporter automatically as a DaemonSet.

## What to alert on

The high-value alerts:

**GPU not in use (when it should be)**: utilization < 20% for 10+ minutes on an active training pod. Indicates a hang or a data-loading bottleneck.

**GPU memory > 95%**: about to OOM. Alert.

**GPU temperature > 85°C**: thermal throttling imminent. Alert.

**Loss NaN or Inf**: training has diverged. Alert immediately.

**Gradient norm > threshold**: usually 100-1000 for typical models. Above that, divergence imminent.

**ECC errors**: bad GPU, may need replacement.

**Stale job**: no metrics reported in N minutes. The training process died.

The right thresholds depend on the workload; tune over time.

## Loss curves and divergence detection

The simplest divergence check: is the loss decreasing (or stable)?

```python
# Pseudo-code in the training loop.
loss_history = []
for step in range(N):
    loss = train_step()
    loss_history.append(loss)
    
    if step % 100 == 0:
        recent = loss_history[-100:]
        if any(np.isnan(l) or np.isinf(l) for l in recent):
            alert("Loss is NaN!")
            checkpoint_and_exit()
        avg_recent = np.mean(recent)
        avg_prev = np.mean(loss_history[-200:-100])
        if avg_recent > 1.5 * avg_prev:
            alert("Loss spiking!")
```

These checks belong in the training script; alerting integrates with the observability platform.

## NaN canaries

A NaN canary: a small computation that's expected to produce a stable, non-NaN value. If it ever produces NaN, something upstream is wrong.

Common canary: a fixed sample's forward pass output. Log its mean / norm each step. If it suddenly NaNs, the model's weights are corrupted.

```python
canary_input = torch.tensor([[1.0, 2.0, 3.0]])  # fixed
canary_output = model(canary_input)
mlflow.log_metric("canary_norm", canary_output.norm().item(), step=step)
```

For sharded models, the canary needs to be computed on a coordinating rank.

## The Grafana dashboard

A useful training dashboard typically has:

- **Per-GPU utilization** (line chart, one line per GPU). Drops indicate compute waste.
- **Per-GPU memory** (line chart). Trend up indicates leak; spike indicates OOM imminent.
- **Per-GPU temperature** (heatmap). Hot spots indicate thermal issues.
- **Loss curve** (line chart). The headline metric.
- **Gradient norm** (line chart, log scale). Outliers indicate instability.
- **Throughput (tokens/sec or samples/sec)** (line chart). Drops indicate slowdown.
- **Step time** (line chart). Spikes indicate stragglers.
- **Active workers** (gauge). Drops indicate worker deaths.

Templates for these dashboards are available in the Grafana marketplace (search for "GPU monitoring" or "DCGM").

## Anomaly detection

Beyond manual threshold-based alerts, anomaly detection (machine-learning-based or statistical) can flag unexpected behavior.

Examples:
- **Per-rank slow detector**: if rank N's step time is consistently 2× the median across ranks, rank N is a straggler.
- **Sudden distribution shift**: the activations look different than usual; might indicate data corruption.
- **Loss-curve deviation**: a hidden Markov model on past loss curves; flag when the current run deviates.

These are more sophisticated; many orgs use simpler thresholds and only add anomaly detection if needed.

## Distributed training observability

For multi-node training, per-rank metrics need to be collected centrally:
- Each rank reports its metrics to a central tracker (W&B, MLflow, or a custom Prometheus push gateway).
- The dashboard shows per-rank views and aggregate views.

The "straggler" problem (one rank consistently slower than others) is the most common multi-node anomaly. The dashboard should make it easy to spot.

## What you should believe after this lesson

Three sentences:

**1. ML training observability has four metric layers**: hardware (DCGM via Prometheus), framework (per-step timings), model (loss, gradients), system. A complete setup captures all four; the Grafana dashboard is the operational view.

**2. The high-value alerts catch silent failures early** — GPU utilization drops (hangs), memory spikes (OOM), loss NaN (divergence), stale jobs (process death). Catching these in minutes vs hours saves substantial compute.

**3. NaN canaries** (fixed inputs whose stable outputs you monitor) detect corruption that aggregate metrics miss. Combined with gradient norm and loss tracking, they catch most divergence scenarios early.

## Hands-on (at home)

Set up DCGM Exporter on a GPU machine.

```bash
# Run DCGM exporter as a Docker container.
docker run --rm -d --gpus all \
    --name dcgm-exporter \
    -p 9400:9400 \
    nvcr.io/nvidia/k8s/dcgm-exporter:3.3.5-3.4.0-ubuntu22.04

# Query metrics.
curl http://localhost:9400/metrics

# You'll see metrics like:
# DCGM_FI_DEV_GPU_UTIL{gpu="0"} 25
# DCGM_FI_DEV_MEM_COPY_UTIL{gpu="0"} 18
# DCGM_FI_DEV_FB_USED{gpu="0"} 1024
```

To complete the loop, run Prometheus pointed at this endpoint and Grafana pointed at Prometheus. The community has Docker-Compose stacks for the full setup.

For a custom metric in your training:

```python
from prometheus_client import Gauge, start_http_server

loss_gauge = Gauge('training_loss', 'Current loss')
start_http_server(8000)

for step in range(N):
    loss = train_step()
    loss_gauge.set(loss)
```

Prometheus scrapes :8000/metrics; the loss is queryable and alertable.

## Further reading

- DCGM documentation.
- Prometheus documentation.
- Grafana documentation.
- NVIDIA's "GPU monitoring for Kubernetes" blog posts.
- "Observability Engineering" book (Honeycomb / Charity Majors).

Next lesson: **Cost monitoring.** Per-job and per-experiment cost attribution. "How much did this paper cost us" should have a one-click answer.
