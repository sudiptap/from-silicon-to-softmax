---
title: "Lesson 5 — ExecuTorch"
date: "2026-06-04"
module: "mobile-edge-runtimes"
order: 5
tags: ["executorch", "pte", "pytorch", "edge", "xnnpack", "backend-delegate"]
author: "Sudipta Pathak"
prerequisites: ["04-ort-apple"]
---

# Lesson 5 — ExecuTorch

## Why this lesson exists

ExecuTorch is Meta's edge-inference framework, designed as PyTorch's native deployment story. It went stable in early 2024 and is the runtime Meta itself uses to ship ML to WhatsApp, Instagram, Quest, and other apps with billion-user-scale deployments. The pitch: stay in PyTorch from training through deployment; no ONNX intermediate; no separate framework to learn.

The architectural innovation is the **backend delegate** pattern. A single `.pte` file can have regions of the graph targeting different backends — XNNPACK for CPU, Core ML for Apple ANE/GPU, QNN for Qualcomm Hexagon, Vulkan for cross-platform GPU. The ExecuTorch runtime orchestrates these; you ship one `.pte` and it runs on whichever backend is available, optimally.

This lesson is the practical ExecuTorch workflow, the delegate pattern, and an honest assessment of what's production-ready vs still rough in 2026. ExecuTorch is the clearest "future-of-edge-PyTorch" bet; whether it's mature enough for your use case today depends on the specifics.

The lesson is reading. The Hands-on builds a `.pte` from a small PyTorch model targeting XNNPACK, then exercises it via the ExecuTorch runtime.

## The shape of the toolchain

The conversion pipeline:

```
PyTorch nn.Module
       │
       │  torch.export
       ▼
ExportedProgram (graph IR)
       │
       │  to_edge()
       ▼
Edge IR (lowered, with edge-specific ops)
       │
       │  to_backend(backend_name) (optional, per-region)
       ▼
Lowered Edge IR with backend-delegated subgraphs
       │
       │  to_executorch()
       ▼
.pte file (binary)
```

Each step is a PyTorch / ExecuTorch API call; the whole pipeline is Python.

The Edge IR is a constrained subset of PyTorch's IR — fixed dtypes, no dynamic shapes (unless explicitly declared), no Python-only ops. This constraint is what makes the model deployable; it also means some training-time idioms (e.g., `assert` statements, `print`, `if torch.cuda.is_available()`) need to be removed before export.

A complete conversion example for a small model targeting XNNPACK:

```python
import torch
from torch.export import export
from executorch.exir import to_edge
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner

class SmallNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 32, 3, padding=1)
        self.fc = torch.nn.Linear(32 * 224 * 224, 10)
    def forward(self, x):
        x = torch.relu(self.conv(x))
        return self.fc(x.flatten(1))

model = SmallNet().eval()
example = torch.randn(1, 3, 224, 224)

# 1. Export.
exported = export(model, (example,))

# 2. Lower to edge IR.
edge = to_edge(exported)

# 3. Delegate to XNNPACK where possible.
edge = edge.to_backend(XnnpackPartitioner())

# 4. Compile to .pte.
pte_program = edge.to_executorch()

# 5. Save.
with open("small_net.pte", "wb") as f:
    f.write(pte_program.buffer)
```

The output is a single binary file that the ExecuTorch runtime loads.

## The backend-delegate pattern

The interesting layer: `to_backend(partitioner)` walks the graph, identifies subgraphs the named backend can handle, and *delegates* them — the backend produces its own optimized binary blob for the subgraph, and the `.pte` contains both the blob and the un-delegated remainder of the graph.

At run time, the runtime:
1. Executes the un-delegated parts using its own kernel set.
2. For each delegated subgraph, calls into the backend (XNNPACK / Core ML / QNN / Vulkan) to execute the blob.

The backends available in 2026:

- **XNNPACK**: CPU kernels for x86 and ARM. Mature; the default fallback.
- **Core ML**: delegates to Apple's Core ML runtime, which can route to ANE/GPU/CPU. Apple deployment path.
- **QNN**: Qualcomm Hexagon NPU. Android-Snapdragon path.
- **Vulkan**: cross-platform GPU. Works on Android Mali/Adreno, desktop Linux, Windows.
- **MPS**: Apple GPU via Metal Performance Shaders. Less commonly used (Core ML is usually better).
- **CoreMLPartitioner with Apple Neural Engine**: Apple platforms specifically targeting ANE.

A `.pte` can be built with multiple delegates: one part of the graph targets Core ML (for Apple-resident execution), another targets XNNPACK (for the fallback). The runtime composes them at execution time.

This is the unique architectural value vs ONNX Runtime: ORT has execution providers, but they're chosen at runtime based on availability. ExecuTorch bakes the delegation into the `.pte` at compile time, which lets the backends do deeper specialization.

## The PTE binary

A `.pte` file is a FlatBuffer-serialized binary containing:

- The Edge IR program (lowered graph).
- Constant data (weights, biases).
- Backend-specific binary blobs for delegated regions.
- Metadata (input/output spec, dtypes, expected shapes).

Sizes are competitive: a typical mobile classifier is 1–10 MB; a small LLM (1B params, INT4) is 500 MB-1 GB.

## ExecuTorch on the runtime side

The runtime is C++, with bindings in Python (for testing), Java/Kotlin (Android), Swift/Objective-C (iOS).

A minimal runtime call in C++:

```cpp
#include <executorch/runtime/executor/program.h>
#include <executorch/runtime/executor/method.h>

using namespace torch::executor;

// Load the .pte file.
auto file = DataLoader::load("small_net.pte");
auto program = Program::load(file.get());

// Create an executor for the "forward" method.
auto method = program->load_method("forward");

// Set inputs.
EValue input = TensorPtr(...);  // wrap your input data
method->set_input(input, 0);

// Execute.
method->execute();

// Get outputs.
auto output = method->get_output(0);
```

The Python binding wraps this as `executorch.runtime.Runtime`, similar shape.

The runtime itself is small (~5 MB for a stripped build with XNNPACK only; more with additional backends bundled).

## What's production-ready in 2026

Production-grade:

- **XNNPACK backend** for CPU on x86 and ARM. Mature; what Meta uses for the majority of its ExecuTorch deployments.
- **Core ML backend** for Apple platforms. Good support for vision models; growing for LLMs.
- **QNN backend** for Snapdragon Hexagon. Production for Meta's Quest devices and parts of WhatsApp.
- **The export pipeline** from PyTorch (with `torch.export`). Works for standard architectures; some quirks for models with complex control flow.
- **Quantization tools** (`torchao` integration). INT8 PTQ via XNNPACK works well; INT4 paths are newer.

Still rough:

- **Vulkan backend**. Functional but not the throughput of native CUDA / Metal paths.
- **LLM deployments at scale**. Possible but more involved than llama.cpp; the LLM serving path through ExecuTorch is improving but llama.cpp remains the simpler choice.
- **Custom ops**. Adding a new op to ExecuTorch is harder than in PyTorch eager; the export pipeline has to know about it.
- **Dynamic shapes**. Supported but with constraints; the static-shape path is more reliable.

## When ExecuTorch is the right answer

The strong cases:

- **PyTorch model targeting iOS + Android with shared deployment code.** ExecuTorch's backend-delegate pattern means one `.pte` + the appropriate backend per platform; no separate conversion to Core ML on one side and LiteRT on the other.
- **Meta-scale apps where the deployment infrastructure is already built around ExecuTorch.** Meta's own apps; companies adopting their toolchain.
- **Vision/audio model deployment where backend-delegated execution is meaningful.** A model with conv-heavy parts (delegated to Core ML/QNN) and Python-y parts (handled by the runtime) can compose cleanly.

The weak cases:

- **LLM deployment.** llama.cpp / MLX win in 2026; ExecuTorch's LLM story is catching up.
- **Models from non-PyTorch sources.** TF / JAX models have to go through PyTorch first, which is awkward.
- **Tiny binary requirements.** ExecuTorch runtime is small but not tiny; LiteRT or hand-written runtimes are smaller.

## Comparison to alternatives

| Aspect | ExecuTorch | Core ML | LiteRT | ORT |
| ------ | ---------- | ------- | ------ | --- |
| PyTorch native? | Yes | No (convert) | No (convert) | No (convert) |
| Cross-platform? | Yes | No (Apple only) | Yes | Yes |
| Backend delegation? | Yes (compile-time) | No | Delegates (run-time) | EPs (run-time) |
| Maturity | Newer | Mature | Mature | Mature |
| LLM story | Improving | Limited | Limited | Decent |
| Binary size | Medium | OS-provided | Small | Medium-large |

The pattern: ExecuTorch is the newest entrant with the strongest PyTorch story and the most flexible backend system. Its maturity is improving; the other runtimes are more battle-tested but less PyTorch-idiomatic.

## What you should believe after this lesson

Three sentences:

**1. ExecuTorch is PyTorch's native edge deployment story** — `torch.export` → `to_edge` → `to_backend` → `to_executorch` produces a `.pte` file that runs on iOS, Android, Linux, embedded targets with no separate conversion pipeline. It's the cleanest path for a PyTorch model that needs to ship cross-platform.

**2. The backend-delegate pattern is its key architectural feature**: a single `.pte` can have regions targeting different accelerators (Core ML, QNN, Vulkan, XNNPACK), with the delegation baked in at compile time rather than chosen at run time. This is more powerful than ORT's runtime EP selection.

**3. ExecuTorch in 2026 is production-grade for vision/audio with XNNPACK or Core ML backends** but its LLM story still trails llama.cpp / MLX. Choose ExecuTorch when PyTorch integration and cross-platform deployment matter more than peak LLM throughput.

## Hands-on (at home)

Convert a small PyTorch model to a `.pte` file and run it through the ExecuTorch runtime.

```bash
pip install executorch torch torchvision
```

```python
# executorch_demo.py
import torch
import torch.nn as nn
from torch.export import export
from executorch.exir import to_edge
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.runtime import Runtime

class TinyClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, 10)
    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))

model = TinyClassifier().eval()
example = torch.randn(1, 64)

# Convert.
exported = export(model, (example,))
edge = to_edge(exported)
edge = edge.to_backend(XnnpackPartitioner())
pte = edge.to_executorch()
with open("tiny.pte", "wb") as f:
    f.write(pte.buffer)
print(f"saved tiny.pte ({len(pte.buffer)} bytes)")

# Run via ExecuTorch runtime.
rt = Runtime.get()
program = rt.load_program("tiny.pte")
method = program.load_method("forward")
out = method.execute([example])
print("output:", out[0])

# Compare to eager PyTorch.
eager_out = model(example)
print("eager:", eager_out)
print("max abs error:", (out[0] - eager_out).abs().max().item())
```

The outputs should match within floating-point noise. The `.pte` file is your deployment artifact — drop it into an iOS/Android app, load via the ExecuTorch runtime SDK, and you have inference.

For a richer experiment, change `XnnpackPartitioner()` to `CoreMLPartitioner()` (on Apple) and re-export. The same model is now delegated to Core ML; the runtime path goes through Apple's compiled subgraph instead of XNNPACK's kernels.

## Further reading

- ExecuTorch documentation (pytorch.org/executorch) — the official intro, with worked examples for each backend.
- "Introducing ExecuTorch" (Meta blog, 2024) — the announcement and rationale.
- ExecuTorch GitHub examples (`executorch/examples/`) — for end-to-end deployments of LLMs, vision models, audio models.
- "ExecuTorch at Meta" talks — for the production deployment story at scale.
- `torch.export` documentation — for the underlying PyTorch export mechanism, which is also useful outside ExecuTorch.

Next lesson: **LiteRT (formerly TFLite).** Google's mobile model format and runtime. The dominant ML runtime on Android, with mature delegate support for GPU and NPU acceleration. We look at conversion, the delegate model, and MediaPipe Tasks as the high-level wrapper around LiteRT for common ML tasks.
