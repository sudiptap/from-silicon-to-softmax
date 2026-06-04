---
title: "Lesson 6 — MLX Intro"
date: "2026-06-04"
module: "mlx-apple-silicon"
order: 6
tags: ["mlx", "lazy-evaluation", "numpy-api", "autodiff", "apple-silicon"]
author: "Sudipta Pathak"
prerequisites: ["05-pytorch-mps"]
---

# Lesson 6 — MLX Intro

## Why this lesson exists

MLX is Apple's ML framework, released in late 2023, designed from scratch for Apple Silicon's unified-memory architecture. It has the API surface of NumPy + PyTorch combined: array operations, automatic differentiation, neural network primitives, optimizers. It runs natively on the Apple GPU via Metal, falls back cleanly to the Apple CPU via Accelerate (which hits the AMX matrix coprocessor), and supports the small set of Apple-specific operations (custom Metal kernel injection) that matter for ML systems work.

What sets MLX apart from a "PyTorch but for Mac" is a handful of fundamental design choices: lazy evaluation (operations build a computation graph; nothing executes until you materialize), unified-memory-native arrays (no device placement to manage), automatic kernel fusion via the lazy graph, and a NumPy-leaning API that's easier to compose than PyTorch's stateful module classes.

This lesson is the "what is MLX, why does it look like this" introduction. The next lesson goes deep on the internals (graphs, streams, materialization, compilation). The two together prepare you to use MLX effectively for the rest of Module 4.

The lesson is reading. The Hands-on installs MLX and runs through the canonical "build a small transformer" example.

## What MLX looks like

A snippet of MLX code that anyone with NumPy or PyTorch experience can read:

```python
import mlx.core as mx
import mlx.nn as nn

# Create some arrays. Lives in unified memory; no .to(device).
a = mx.random.normal((1024, 1024), dtype=mx.float16)
b = mx.random.normal((1024, 1024), dtype=mx.float16)
c = a @ b + mx.sin(a)
# Nothing has run yet. c is a lazy graph node.

mx.eval(c)  # Force materialization. Now the GPU actually does the work.
print(c.shape, c.dtype)
```

The lazy-evaluation contract is the headline difference from PyTorch: operations don't execute when you write them. They build a graph. The graph executes when you call `mx.eval` or implicitly when you call `.tolist()` / `mx.savez` / similar.

Why this matters: the graph builder can fuse adjacent operations into a single Metal kernel. `a @ b + mx.sin(a)` doesn't execute as "matmul → save intermediate → sin → save → add" with three GPU dispatches. The fusion machinery merges what it can into one dispatch, similar to what `torch.compile` does for PyTorch (but built into MLX from day one, not as a separate compilation step).

Define a small neural network:

```python
import mlx.core as mx
import mlx.nn as nn

class TinyMLP(nn.Module):
    def __init__(self, in_dim, hidden, out_dim):
        super().__init__()
        self.l1 = nn.Linear(in_dim, hidden)
        self.l2 = nn.Linear(hidden, out_dim)

    def __call__(self, x):
        return self.l2(nn.silu(self.l1(x)))

net = TinyMLP(784, 128, 10)
x = mx.random.normal((32, 784))
y = net(x)
mx.eval(y)
print(y.shape)  # (32, 10)
```

The API is NumPy/PyTorch-flavored. `nn.Module` is the structuring unit; `nn.Linear`, `nn.silu`, etc. are the primitives. The `__call__` (not `forward`) is what gets invoked. There's no `.to('mps')` step.

## The four design choices that shape MLX

**1. Lazy evaluation.** As shown above, operations build a graph; evaluation happens on demand. This enables fusion, gives the framework license to reorder operations for performance, and matches the streaming-data idioms that ML inference naturally fits.

**2. Unified-memory-native arrays.** An `mx.array` lives in unified memory and is accessible by both CPU and GPU backends. There is no device argument. The framework picks the right backend per-op based on shape, dtype, and what's already cached.

**3. NumPy-leaning API for arrays + PyTorch-leaning API for modules.** Array operations look like NumPy (no in-place mutation, functional style, broadcasting rules match). Neural network modules look like PyTorch (subclass `nn.Module`, define a callable, use `nn.Linear` etc.). The two APIs sit cleanly side-by-side.

**4. Functional autodiff.** Gradients are computed via `mx.grad(fn)` returning a new function rather than via `tensor.backward()` setting attributes. This is closer to JAX's style than PyTorch's. For training loops it requires a small shift; for inference (no gradients) it makes no difference.

A complete training-loop sketch:

```python
import mlx.optimizers as optim

net = TinyMLP(784, 128, 10)
opt = optim.Adam(learning_rate=1e-3)

def loss_fn(model, x, y):
    logits = model(x)
    return nn.losses.cross_entropy(logits, y, reduction='mean')

loss_and_grad = nn.value_and_grad(net, loss_fn)

for x, y in dataloader:
    loss, grads = loss_and_grad(net, x, y)
    opt.update(net, grads)
    mx.eval(net.parameters(), opt.state)
```

The `mx.eval` on `net.parameters()` and `opt.state` is the explicit materialization point that says "force everything to actually run now." Without it, the lazy graph would keep growing across iterations and consume memory.

## When MLX is the right choice

MLX is the right choice on Apple Silicon when:

- You're starting a new project and have no PyTorch lock-in.
- You're doing LLM inference and want the best of what's available on Mac.
- You need quantized formats that MLX supports natively (INT4, INT8 with per-group scales) without the bitsandbytes-on-Mac hassle.
- You want to write custom Metal kernels and call them from Python with reasonable ergonomics (Lesson 8 covers the custom-kernel path).

MLX is *not* the right choice when:

- You have a large existing PyTorch codebase. Porting is not free.
- You need an ecosystem feature MLX doesn't have yet (some less-common optimizers, some niche neural-net primitives, distributed-training abstractions). The MLX ecosystem is young and growing fast but doesn't match PyTorch's breadth.
- You're targeting deployment to non-Apple platforms. MLX is Apple-only. (You can train in MLX and export to ONNX for cross-platform inference, but it's not a smooth path.)
- You're doing classical ML (gradient boosting, tabular methods). MLX is for deep learning; sklearn / xgboost / catboost on CPU is the appropriate stack for non-deep workloads.

## The performance picture

On the same model, same input, same machine:

- MLX is typically 10–30% faster than PyTorch MPS for inference.
- MLX is typically comparable or slightly faster than PyTorch MPS for training.
- MLX is typically 5–15× faster than PyTorch CPU.
- MLX vs llama.cpp: comparable for quantized LLM inference; the workload dominates.

The performance gap with PyTorch MPS isn't huge, but it's consistent. The bigger MLX win is in the things that aren't reflected in the headline number: better quantization support, fewer silent CPU fallbacks, custom-kernel ergonomics, and a more Apple-Silicon-native programming model that produces fewer surprises.

## The ecosystem in 2026

What's in the MLX ecosystem:

- **`mlx-core`**: the framework itself (arrays, ops, autodiff).
- **`mlx-nn`**: neural network modules.
- **`mlx-data`**: data loading and preprocessing pipelines.
- **`mlx-lm`**: LLM-focused convenience layer (model loading, quantization, generation).
- **`mlx-vlm`**: vision-language models.
- **`mlx-examples`** (GitHub repo): worked examples for many model families.
- **Community ports**: Whisper, Stable Diffusion, RAG/embedding models, fine-tuning tooling.

What isn't yet:

- A reinforcement learning library on par with `stable-baselines3` or RLHF tooling on par with TRL.
- A unified profiler that matches PyTorch's `torch.profiler` ergonomics.
- A distributed-training story for multi-Mac setups (this is a 2026 frontier).

For the topics this module covers — LLM inference on Apple Silicon — the MLX ecosystem is fully sufficient.

## What you should believe after this lesson

Three sentences:

**1. MLX is Apple's native ML framework with a lazy-evaluation graph, unified-memory-native arrays, and a NumPy + PyTorch hybrid API** — designed from scratch for Apple Silicon rather than retrofitted onto an existing CUDA-shaped framework.

**2. The performance edge over PyTorch MPS is modest (10–30%)** but the broader UX wins (no device placement, no silent CPU fallback, native quantization, easier custom kernels) make MLX the default for new ML work on Apple Silicon.

**3. The lazy evaluation model is the single biggest design departure from PyTorch** — operations build a graph and don't run until you call `mx.eval`. This requires a small mental shift but enables the fusion and optimization that makes MLX competitive.

## Hands-on (at home)

Install MLX and run the canonical training example.

```bash
pip install mlx mlx-lm
```

Train a small MLP on MNIST:

```python
# mlx_mnist.py
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import gzip, struct, urllib.request, os
from array import array
import numpy as np

def load_mnist():
    base = "https://ossci-datasets.s3.amazonaws.com/mnist/"
    files = {
        "train_x": "train-images-idx3-ubyte.gz",
        "train_y": "train-labels-idx1-ubyte.gz",
        "test_x":  "t10k-images-idx3-ubyte.gz",
        "test_y":  "t10k-labels-idx1-ubyte.gz",
    }
    os.makedirs("./mnist_data", exist_ok=True)
    paths = {}
    for k, f in files.items():
        p = f"./mnist_data/{f}"
        if not os.path.exists(p):
            urllib.request.urlretrieve(base + f, p)
        paths[k] = p
    def load_imgs(p):
        with gzip.open(p, 'rb') as f:
            _, n, r, c = struct.unpack(">IIII", f.read(16))
            data = np.frombuffer(f.read(), dtype=np.uint8).reshape(n, r*c).astype(np.float32) / 255
        return data
    def load_labels(p):
        with gzip.open(p, 'rb') as f:
            _, n = struct.unpack(">II", f.read(8))
            return np.frombuffer(f.read(), dtype=np.uint8).astype(np.int32)
    return load_imgs(paths['train_x']), load_labels(paths['train_y']), \
           load_imgs(paths['test_x']),  load_labels(paths['test_y'])

tr_x, tr_y, te_x, te_y = load_mnist()
tr_x = mx.array(tr_x); tr_y = mx.array(tr_y)
te_x = mx.array(te_x); te_y = mx.array(te_y)

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(784, 256)
        self.l2 = nn.Linear(256, 10)
    def __call__(self, x):
        return self.l2(nn.relu(self.l1(x)))

net = MLP()
opt = optim.Adam(learning_rate=1e-3)

def loss_fn(model, x, y):
    return nn.losses.cross_entropy(model(x), y, reduction='mean')

loss_and_grad = nn.value_and_grad(net, loss_fn)

batch_size = 128
n_epochs = 3
for epoch in range(n_epochs):
    perm = np.random.permutation(len(tr_x))
    for i in range(0, len(perm), batch_size):
        idx = mx.array(perm[i:i+batch_size])
        x, y = tr_x[idx], tr_y[idx]
        loss, grads = loss_and_grad(net, x, y)
        opt.update(net, grads)
        mx.eval(net.parameters(), opt.state)
    # Eval.
    preds = mx.argmax(net(te_x), axis=-1)
    acc = (preds == te_y).mean().item()
    print(f"epoch {epoch+1}: test accuracy = {acc:.4f}")
```

This should hit ~97% test accuracy in 3 epochs, in <30 seconds on M3 Pro. The whole thing — model definition, training loop, GPU acceleration — is ~50 lines and feels NumPy-natural.

Now run an LLM inference example via `mlx-lm`:

```bash
mlx_lm.generate --model mlx-community/Qwen2.5-1.5B-Instruct-4bit \
    --prompt "Explain transformers in one sentence." \
    --max-tokens 96
```

You'll see ~80–150 tok/s on M3 Pro for the 4-bit quantized 1.5B model. The first run downloads the model; subsequent runs are immediate.

## Further reading

- MLX documentation (ml-explore.github.io/mlx) — the official tutorial and API reference.
- `mlx-examples` repository on GitHub — a wide tour of worked examples.
- "MLX: An Array Framework for Apple Silicon" — the Apple ML team's blog announcement.
- Awni Hannun's talks on MLX at conferences and on YouTube — for design-rationale discussion from the lead author.

Next lesson: **MLX internals.** We go under the hood: how the lazy computation graph is built, what `mx.eval` actually does, the streams model that allows concurrent execution, `mx.compile` for additional fusion, and the memory-management story.
