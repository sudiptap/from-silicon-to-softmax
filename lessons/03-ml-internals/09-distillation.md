---
title: "Lesson 9 — Distillation: From Logits to Step-by-Step"
date: "2026-06-04"
module: "ml-internals"
order: 9
tags: ["distillation", "hinton", "soft-labels", "minillm", "rationale", "step-by-step"]
author: "Sudipta Pathak"
prerequisites: ["08-pruning"]
---

# Lesson 9 — Distillation: From Logits to Step-by-Step

## Why this lesson exists

Quantization and pruning shrink an existing model. Distillation does something different: it trains a *new, smaller* model to imitate the larger one. This sounds expensive — you're paying for additional training — but the math says it's a uniquely powerful lever for one specific situation: when the smaller model's architecture is fundamentally well-matched to the task, but training it from scratch on raw data wastes the larger model's already-paid-for knowledge.

In 2026 this lever is doing more work than ever. The wave of small reasoning models (Phi, Gemma-class, Qwen 0.5B, MobileLLM) trained via distillation from frontier-scale models has demonstrated that 1B-class models can match 7B-class performance on many tasks when trained correctly. The distillation isn't just from logits anymore — it's from full chains of thought, from intermediate states, from preference rankings. This lesson surveys the technique from its 2014 Hinton roots to the 2025–2026 step-by-step methods.

The lesson is reading. The Hands-on does a basic teacher-student distillation on a small classification task to demonstrate the soft-label mechanism.

## Hinton-style soft-label distillation

The 2014 Hinton, Vinyals & Dean paper formalized what's now called *response* distillation. Setup:

- Teacher model `T` (large, trained, frozen).
- Student model `S` (smaller, to be trained).
- Training data `x_i` (the inputs; ground-truth labels `y_i` may or may not be available).

For each input `x`, the teacher produces logits `z_T(x)`. Softmax with temperature `τ > 1` gives a "soft" probability distribution:

```
p_T(x; τ) = softmax(z_T(x) / τ)
```

The student is trained to minimize the KL divergence between its softened output and the teacher's:

```
L_distill = τ² × KL(p_T(x; τ) || p_S(x; τ))
```

Often combined with a standard cross-entropy loss against true labels (`L_CE(S(x), y)`) when labels are available:

```
L = α × L_distill + (1 - α) × L_CE
```

The `τ²` factor cancels out the gradient scaling that the temperature introduces, keeping the two loss terms comparable.

The key insight from the original paper: the teacher's soft labels carry more information than hard one-hot labels. The hard label "this is a cat" tells the student that the answer is class 7. The soft label "this is 60% cat, 25% dog, 10% lynx, 5% everything-else" tells the student about the *similarity structure* between classes — that cats and lynxes are visually related, that dogs are sort of close, that birds are far away. This is the "dark knowledge" Hinton's paper named.

Soft-label distillation is the foundation; everything since has been refinements or generalizations of this idea.

## Feature distillation

Soft-label distillation only matches the *output* of the teacher. *Feature* distillation matches intermediate activations as well.

For each intermediate layer (or a chosen subset), add a regression loss between the student's hidden state at some layer and the teacher's hidden state at a matched (possibly differently-numbered) layer:

```
L_feat = sum_l ||h_S^l - h_T^{f(l)}||²
```

The matching `f(l)` between student layers and teacher layers is an architectural choice. If the student has half the depth, you might match every other teacher layer.

Feature distillation typically requires a small linear projection on the student's hidden state to bring it into the teacher's dimensionality. The projection is part of the student's trainable parameters.

This was the dominant distillation technique for BERT-class models (DistilBERT, MobileBERT) where matching attention patterns or hidden states layer-by-layer was the headline metric. For LLMs the technique still applies but is less prominent than the newer step-by-step methods.

## Relation distillation

Distill the *relations* between examples rather than the examples themselves. For a batch of inputs `x_1, ..., x_n`, compute pairwise similarities of their teacher outputs (or hidden states), then train the student so its pairwise similarities match.

This is theoretically attractive (it removes the requirement that the student space resemble the teacher space at all — only the geometry of the input space matters) but has not seen as much practical adoption as the response and feature variants.

## Distillation in the LLM era

The classical formulation hits a problem at LLM scale: the teacher's output distribution over a 100K+ token vocabulary is enormous. Storing `p_T(x; τ)` for every training example is impractical. Re-computing it on the fly during student training requires running the (huge) teacher in parallel, which is expensive.

Modern recipes work around this in three ways:

**1. Distill from a single sample of the teacher's distribution (top-k or temperature-sampled).** Lose some "dark knowledge" but become tractable. This is what Distil-style recipes do.

**2. Distill on-policy: run the student to generate, then have the teacher score those generations.** The student learns to produce sequences the teacher rates highly. This is the MiniLLM approach (Gu et al, 2023) and is closely related to RLHF / preference distillation.

**3. Rationale (step-by-step) distillation: have the teacher produce chains of thought, train the student to reproduce both the answer and the reasoning.** This is the Phi family of techniques (Microsoft) and the broader category of "reasoning distillation" that dominates 2024–2026 small-model training.

The third approach — rationale distillation — has been the most impactful. The intuition: a teacher that says "the answer is 42 because (1) we identified the operation as addition, (2) we computed 17 + 25, (3) we verified..." is providing structured supervision that a smaller student can learn from far more efficiently than from the answer alone. The student becomes capable of producing similar reasoning chains and, by extension, answering similar questions correctly.

This is how Microsoft Phi, Google Gemma 2, and the small Qwen models reach surprising performance at 1–3B parameters: their pretraining corpus was synthesized in significant part by larger teachers producing high-quality reasoning chains over diverse domains.

## When distillation beats quantization, when it doesn't

The decision is not "either-or" — production small models often use both — but the relative leverage varies by setting.

**Distillation wins when:**

- The target architecture is fundamentally different from the teacher (e.g., distilling an LLM into a small CNN for some specific task).
- The target is dramatically smaller (e.g., from 70B to 1B). At that compression ratio, quantization alone can't get you there — the 70B model at INT2 would be 17 GB, still too big for many deployments — and the 1B distilled model at FP16 is 2 GB.
- The training cost can be amortized over many deployments. Distillation pays its training cost once and serves billions of inference requests.

**Quantization wins when:**

- The compression ratio is modest (2–8×).
- You need the deployed model to retain the full teacher's capabilities, not a subset.
- You have a tight engineering schedule. Quantizing a model takes hours; distilling a model takes days-to-weeks.

**Both together:** common in production. The big teacher distills into a smaller student via reasoning distillation; the student is then quantized to INT4 for deployment. The Phi-3 mini → on-device pipeline looks roughly like this.

## A subtle point: distillation as data generation

The most surprising thing about modern distillation isn't the algorithm — it's how much of it is now framed as *data generation*. Instead of "train student to match teacher's logits on dataset X," it's "have teacher generate dataset X', then train student on X' with standard cross-entropy."

X' here is the teacher's outputs (or curated subset of them) treated as ground truth for the student's training. Phi, MiniLLM, and many post-2023 small-model training recipes are predominantly this: a careful pipeline that uses the teacher to *manufacture training data* of higher quality than what's freely available on the web, then trains the student on the manufactured data with conventional supervised learning.

This blurs the line between "distillation" and "synthetic data training." Functionally, the wins look the same to the engineer deploying the small model. Conceptually it's worth knowing: the lever is "use the teacher to produce better training signal," not specifically "match the teacher's logits."

## What you should believe after this lesson

Three sentences:

**1. Hinton soft-label distillation is the foundational technique** — match the teacher's softened output distribution to transfer dark knowledge about class similarity. Feature and relation variants extend this to intermediate states and inter-example geometry, respectively.

**2. The dominant distillation paradigm in 2026 is rationale / step-by-step distillation**, where the teacher produces full reasoning chains and the student is trained to reproduce them. This is how 1–3B parameter models match 7B+ models on many reasoning tasks.

**3. Distillation and quantization are complementary, not competitive** — distillation gets you a smaller architecture with the teacher's capabilities transferred; quantization then compresses the bytes of that smaller architecture. Production pipelines combine both.

## Hands-on (at home)

A minimal teacher-student distillation on MNIST to demonstrate the mechanism without LLM-scale infrastructure.

```python
# distill_mnist.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

torch.manual_seed(0)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Teacher(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 512), nn.ReLU(),
            nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, 10),
        )
    def forward(self, x): return self.net(x)

class Student(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 64), nn.ReLU(),  # 8x smaller hidden
            nn.Linear(64, 10),
        )
    def forward(self, x): return self.net(x)

train = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=256, shuffle=True)
test = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=False, transform=transforms.ToTensor()),
    batch_size=512)

def acc(m):
    m.eval()
    n, correct = 0, 0
    with torch.no_grad():
        for x, y in test:
            x, y = x.to(device), y.to(device)
            p = m(x).argmax(-1)
            correct += (p == y).sum().item()
            n += y.size(0)
    return correct / n

def train_one(m, epochs=3, distill_from=None, tau=4.0, alpha=0.5):
    m.to(device).train()
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    for _ in range(epochs):
        for x, y in train:
            x, y = x.to(device), y.to(device)
            logits = m(x)
            loss_ce = F.cross_entropy(logits, y)
            if distill_from is not None:
                with torch.no_grad():
                    t_logits = distill_from(x)
                p_t = F.log_softmax(t_logits / tau, dim=-1)
                p_s = F.log_softmax(logits / tau, dim=-1)
                loss_kd = F.kl_div(p_s, p_t, reduction='batchmean', log_target=True) * tau * tau
                loss = alpha * loss_kd + (1 - alpha) * loss_ce
            else:
                loss = loss_ce
            opt.zero_grad(); loss.backward(); opt.step()
    return m

# 1) Train the teacher from scratch.
t = train_one(Teacher())
print(f"Teacher accuracy: {acc(t):.4f}")

# 2) Train the student from scratch (no distillation).
s_scratch = train_one(Student())
print(f"Student (no distill): {acc(s_scratch):.4f}")

# 3) Train the student with distillation from the teacher.
s_distill = train_one(Student(), distill_from=t.eval())
print(f"Student (distilled):  {acc(s_distill):.4f}")
```

You should see the distilled student outperform the from-scratch student by ~0.5–1.5% on test accuracy. The gap is small on MNIST because the task is too easy for the gap to matter much; on harder tasks (CIFAR-100, ImageNet) the distillation gap typically widens to 2–5%.

For a rationale-distillation experiment at LLM scale, the standard recipe is: take a strong teacher (Llama 3 70B or similar), generate chain-of-thought completions on a math dataset (GSM8K or MATH), train a small student (Llama 3.2 1B) on (problem, chain-of-thought, answer) triples with standard cross-entropy. The student gets a striking boost over training on (problem, answer) alone.

## Further reading

- "Distilling the Knowledge in a Neural Network" (Hinton, Vinyals & Dean, 2015) — the seminal paper.
- "DistilBERT: a distilled version of BERT" (Sanh et al, 2019) — the canonical feature-distillation example.
- "MiniLLM: Knowledge Distillation of Large Language Models" (Gu et al, 2023) — the on-policy LLM distillation paper.
- "Phi-3 Technical Report" (Microsoft, 2024) — the small-model-via-curated-synthetic-data approach.
- "Tinystories" (Eldan & Li, 2023) — for a clean small-data, small-model demonstration of how much you can do with the right training data and a smaller architecture.

Next lesson: **The Rust GPU Frontier — cubecl, rust-gpu, Burn.** Pivoting from algorithms to ecosystem: where the Rust-on-GPU story sits in 2026, why it might matter for the kinds of low-precision kernels this module has been discussing, and what's production-ready versus research-ware.
