---
title: "Lesson 11 — Module Wrap: Building a 3D-Parallel Training Run"
date: "2026-06-04"
module: "distributed-systems"
order: 11
tags: ["wrap", "3d-parallelism", "training-run", "worked-example", "module-summary"]
author: "Sudipta Pathak"
prerequisites: ["10-sequence-parallelism-ring"]
---

# Lesson 11 — Module Wrap: Building a 3D-Parallel Training Run

## Why this lesson exists

Eleven lessons of network primitives and parallelism strategies collapse into one practical question: given a model and a cluster, how do you configure the training run? This lesson walks through that decision end-to-end for a worked example, then wraps Module 8.

The lesson is reading. The Hands-on uses the walked-through configuration to estimate training time.

## The journey, replayed

Module 8 went networking → collectives → parallelism → composition. The arc:

- Lessons 1-2 (networking): RDMA, InfiniBand, RoCE; NVLink, NVSwitch. The two-tier networking that defines what's tight (intra-node NVLink at 900 GB/s) and what's loose (inter-node IB at 100-400 Gbps).
- Lessons 3-4 (collectives): NCCL as the standard abstraction; AllReduce algorithms (ring for bandwidth, tree for latency, double binary tree for big multi-node).
- Lesson 5 (DDP): the baseline parallelism. Replicate model, split batch, AllReduce gradients.
- Lesson 6 (ZeRO/FSDP): shard parameters/gradients/optimizer state. Makes DDP-style training memory-feasible at much larger model sizes.
- Lesson 7 (TP): shard individual matmuls. Within-node parallelism via NVLink.
- Lesson 8 (PP): shard layers. The "bubble" problem and 1F1B scheduling.
- Lesson 9 (3D parallelism): TP × PP × DP composition. The grid for the largest training runs.
- Lesson 10 (sequence parallelism + Ring Attention): the additional dimensions for long context.

The synthesis: distributed ML training is a multi-axis optimization problem. The right configuration depends on model size, sequence length, cluster topology, and target throughput.

## Worked example: Llama 405B on 256 H100s

The setup:
- **Model**: Llama 405B (405 billion parameters, 126 layers, hidden_dim=16384, n_heads=128, GQA=8).
- **Sequence length**: 8192 tokens (typical pretraining context).
- **Global batch size**: 1024 sequences × 8K tokens = 8.4M tokens per step.
- **Cluster**: 32 nodes × 8 H100 = 256 GPUs total.
- **Interconnect**: NVLink within nodes (900 GB/s), InfiniBand NDR between nodes (400 Gbps).

### Step 1: Pick TP

TP within a node. With 8 GPUs per node, TP=8 is the natural choice.

Per-GPU parameter share after TP=8: 405B / 8 ≈ 50.6B params. Still too large.

### Step 2: Pick PP

We need to further shard the model across nodes. PP partitions the 126 layers.

PP=8: each pipeline stage holds 126/8 ≈ 16 layers. Each TP=8 group within a node holds 16 layers' worth of TP'd parameters.

Per-GPU parameter share after TP=8 × PP=8: 405B / 64 ≈ 6.3B params. At BF16: ~12.6 GB. Fits.

8 nodes for one pipeline group; we have 32 nodes total → 4 pipeline groups.

### Step 3: Pick DP

The remaining 4 pipeline groups become 4 DP replicas. DP=4.

Total: TP=8 × PP=8 × DP=4 = 256 GPUs. Matches the cluster size.

### Step 4: Estimate memory

Per-GPU (within a TP+PP group, with DP sharding):
- Parameters (BF16, TP+PP sharded): 12.6 GB.
- Gradients (BF16, TP+PP sharded, DP-sharded via ZeRO-2 = FSDP-2): 12.6 / 4 = 3.15 GB.
- Optimizer state (Adam 12 bytes, DP-sharded): 6.3B × 12 / 4 = 19 GB.
- Activations (heavily sharded by TP+SP, ~5-10 GB at this scale): say 8 GB.
- Misc: ~5 GB buffer.

Total per-GPU: ~48 GB. Fits in H100's 80 GB with room.

### Step 5: Estimate step time

Compute per step: roughly the model's TFLOPs divided by the effective TFLOPs of the system. For Llama 405B at 8K context, 8.4M tokens per step: ~10^18 FLOPs.

At 256 H100s × 989 TFLOPs/H100 (BF16) × ~40% MFU = ~100 PFLOPs effective.

Step compute time: 10^18 / 10^17 = 10 seconds.

Communication per step:
- TP all-reduce (intra-node, NVLink): ~5-10% of step time. Hidden under compute.
- PP point-to-point (inter-node, IB): ~few percent. Mostly hidden.
- DP all-reduce of gradients (cross-DP-replica, IB): more substantial. ~1-2 seconds.

Step time: ~10-12 seconds.

### Step 6: Total training time

Llama 405B trained on ~15T tokens.
Per step: 8.4M tokens.
Total steps: 15 × 10^12 / 8.4 × 10^6 ≈ 1.8 million steps.
At 12 seconds per step: 1.8M × 12 = 21.6M seconds ≈ 250 days.

This matches roughly published training durations for frontier models.

## Mental models to carry forward

Five sentences:

**1. The network defines the parallelism**: NVLink (intra-node, 5-20× faster) for tight coupling (TP, EP); IB/RoCE (inter-node, slower) for loose coupling (PP, DP, gradient AllReduce). Design topology-aware.

**2. AllReduce is the central collective in ML**; its algorithm choice (ring vs tree vs double-binary-tree) matters for performance. NCCL picks automatically based on message size and topology; the right choice for typical ML messages is double-binary-tree.

**3. FSDP/ZeRO are the memory savers**: shard parameters/gradients/optimizer state across DP. Modern PyTorch large-model training defaults to FSDP. The communication cost is paid via all-gather and reduce-scatter; well-overlapped, this fits within the compute time.

**4. The 3D parallelism grid (TP × PP × DP)** is the standard for 100B+ models. The decision is: how big is the model (TP); how deep is it (PP); how much throughput do you need (DP). Trillion-parameter training adds EP for MoE and sequence parallelism for long context.

**5. Long context adds Ring Attention** (or equivalent) for sequence sharding across nodes. Frontier training increasingly uses 4D-5D parallelism. The complexity is real; design carefully.

## What we covered, what we skipped

Covered: the eleven topics in the roadmap.

Skipped:
- **Specific cluster vendor architectures** (Slurm, Kubernetes specifics). Module 9.
- **Heterogeneous-GPU training** (mixed A100 + H100 in one job).
- **Federated learning** across user devices.
- **Reinforcement learning training infrastructure** in depth (similar to standard distributed training with extras).
- **Specific framework internals** beyond what each lesson touched (DeepSpeed's full feature set, Megatron's many configurations).

## What's next

**Module 9: Cluster Orchestration.** The infrastructure layer that schedules training jobs onto the network we just learned about. Kubernetes for ML, Slurm, the scheduler, gang scheduling, GPU fractional allocation.

**Module 10: ML Platform Engineering.** The operational layer above orchestration: experiment tracking, model registry, CI/CD for models, monitoring.

**Module 11: Agents from Scratch.** The application layer: tool use, planning, agent architectures, the multi-step reasoning patterns that consume LLM inference.

## End of Module 8

Module 1 made the CPU fast. Module 2 made the GPU fast. Module 3 made the model small. Module 4 made on-Apple-Silicon fast. Module 5 made on-every-platform fast. Module 6 made on-device deployments real. Module 7 built the inference engine. Module 8 built the network and parallelism layer that makes multi-GPU and multi-node training and serving possible.

Module 9 picks up at the cluster level — the scheduler, the resource management, the orchestration. The layer that turns raw GPUs into a usable cluster.

## Hands-on (at home)

Compute the configuration for a model and cluster of your choice.

```python
# config_planner.py

def plan_training_run(
    model_params_b,
    n_layers,
    hidden_dim,
    seq_len,
    global_batch_tokens,
    cluster_n_nodes,
    gpus_per_node,
    gpu_mem_gb=80,
    gpu_tflops=989,  # H100 BF16
    mfu=0.4,
    nvlink_gbps=900,
    ib_gbps=400,
):
    total_gpus = cluster_n_nodes * gpus_per_node
    
    # Start with TP=gpus_per_node (within a node).
    tp = gpus_per_node
    
    # Pick PP to fit memory. Each GPU needs to hold params + grads + optim state.
    # With FSDP/ZeRO-2 across DP, sharded by DP.
    # Approximate per-GPU param mem = 16 bytes/param × (params / (TP*PP)) / DP.
    
    # Try different PP, DP combinations.
    for pp in [1, 2, 4, 8, 16]:
        if total_gpus % (tp * pp) != 0:
            continue
        dp = total_gpus // (tp * pp)
        params_per_gpu = model_params_b * 1e9 / (tp * pp)
        # 2 (BF16 params) + 2/dp (grads) + 12/dp (Adam state)
        mem_per_gpu = params_per_gpu * (2 + 14/dp) / 1e9
        # Plus activations (rough).
        act_mem = seq_len * (global_batch_tokens // seq_len // dp) * hidden_dim * n_layers / pp * 8 / tp / 1e9
        total_mem = mem_per_gpu + act_mem
        
        if total_mem < gpu_mem_gb * 0.85:  # leave 15% headroom
            # Estimate step time.
            total_tflops = total_gpus * gpu_tflops * mfu
            # Compute FLOPs per step: roughly 6 × params × tokens.
            flops_per_step = 6 * model_params_b * 1e9 * global_batch_tokens
            step_time = flops_per_step / (total_tflops * 1e12)
            
            print(f"TP={tp}, PP={pp}, DP={dp}: {total_mem:.1f} GB/GPU, step time ≈ {step_time:.1f}s")
            return tp, pp, dp, step_time
    
    print("No feasible configuration found")
    return None

plan_training_run(
    model_params_b=70, n_layers=80, hidden_dim=8192,
    seq_len=8192, global_batch_tokens=4*1024*1024,
    cluster_n_nodes=8, gpus_per_node=8
)

plan_training_run(
    model_params_b=405, n_layers=126, hidden_dim=16384,
    seq_len=8192, global_batch_tokens=8*1024*1024,
    cluster_n_nodes=32, gpus_per_node=8
)
```

You'll see the feasible TP/PP/DP combinations for each model+cluster. Pick the one with the best memory utilization and lowest step time.

For real training, the planning is more sophisticated (NVIDIA's NCCL-tests + MFU benchmarks + DeepSpeed's tutorial configurations), but the framework is the same.
