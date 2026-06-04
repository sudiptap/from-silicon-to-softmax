---
title: "Lesson 11 — Model Serving Infrastructure"
date: "2026-06-04"
module: "ml-platforms"
order: 11
tags: ["serving", "infrastructure", "autoscaling", "request-routing", "triton", "kserve"]
author: "Sudipta Pathak"
prerequisites: ["10-feature-stores"]
---

# Lesson 11 — Model Serving Infrastructure

## Why this lesson exists

Module 7 covered inference engines (vLLM, TensorRT-LLM, llama.cpp). Module 5 covered runtime deployment. This lesson is the *platform* view: what does it take to operate model serving as a service? Autoscaling, request routing, observability, fleet management, multi-model serving.

The model is part of it; the infrastructure around the model is the bulk of the platform's serving concerns.

The lesson is reading. The Hands-on deploys a model behind KServe.

## What model serving requires

Beyond the inference engine:

- **Endpoint**: HTTP / gRPC / SDK access for clients.
- **Load balancing**: distribute requests across replicas.
- **Autoscaling**: add/remove replicas based on load.
- **Routing**: traffic splitting (canary), A/B (next lesson), per-tenant routing.
- **Authentication and rate limiting**.
- **Logging and tracing**.
- **Health checks**.
- **Deployment automation**: rolling updates, blue/green, canary.
- **Multi-model support**: many models on shared infrastructure.

Stock K8s Deployment + Service handles some; specialized ML serving platforms handle more.

## The platforms

The main options in 2026:

**KServe**: Kubernetes-native ML serving. Supports many backends (TF Serving, TorchServe, Triton, vLLM, custom). Provides autoscaling, traffic splitting, monitoring out of the box.

**Triton Inference Server**: NVIDIA's serving runtime. High-performance; supports many backends; integrates with TensorRT for optimization. Often used as the engine inside KServe.

**vLLM**: LLM-specific (Module 7). Increasingly deployed standalone with custom serving infrastructure, or behind KServe.

**Ray Serve**: Ray's serving framework (Module 9 Lesson 10). Strong for autoscaling and composition.

**SageMaker / Vertex AI Endpoints / Azure ML Endpoints**: cloud-managed equivalents.

The choice depends on:
- LLM vs traditional ML.
- Cloud vs self-hosted.
- Composition needs (single model vs pipeline).

## Autoscaling

ML serving has spiky load. Autoscaling adjusts replicas based on observed load.

**Metric-based autoscaling**:
- CPU/GPU utilization.
- Request rate.
- Latency.

**Predictive autoscaling**: forecast load; scale ahead of time. More sophisticated; less common.

For LLM serving, **request-rate-based autoscaling** is most common (model is GPU-bound; CPU isn't relevant; per-request resource isn't easy to predict).

Knative (the K8s autoscaler used by KServe) supports both reactive (request-rate) and predictive autoscaling.

The scale-from-zero challenge: cold start of a 70B model takes minutes. If you scale to zero replicas and need to handle a burst, the first request waits the cold-start time. Mitigations:
- **Keep min_replicas ≥ 1**: never scale to true zero.
- **Pre-warm**: keep a replica ready but not serving traffic.
- **Predictive scale-up**: scale up before the load arrives.

For LLMs the cold-start cost dominates the scale-to-zero benefit; most production keeps a minimum of 1-2 replicas always.

## Request routing

Beyond load balancing, request routing can:

- **Traffic split for canary**: 5% to v2, 95% to v1.
- **Per-tenant routing**: tenant A's requests go to dedicated replicas with their fine-tuned model.
- **Model variant routing**: route by request content (e.g., code requests to a code-specialized model, chat to a chat-specialized model).
- **Geographic routing**: low-latency by routing to the nearest data center.

KServe and Istio (service mesh) provide the primitives.

For LoRA hot-swap (Module 6 Lesson 10): the router selects which LoRA adapter to apply for each request; the underlying model is shared.

## Multi-model serving

A single replica can serve multiple models:
- Different model families (code completion + chat + classification all on the same fleet).
- Different model versions of the same family.
- Different LoRA adapters on the same base.

Multi-model serving is more memory-efficient than one-replica-per-model.

Triton has explicit multi-model support; KServe via the InferenceGraph CRD.

For LLM specifically, multi-LoRA serving (Module 6 Lesson 10) is the dominant pattern. vLLM has built-in multi-LoRA support; many production servings configure this.

## Observability for serving

The serving observability stack:

- **Request rate, latency, error rate**: the RED metrics (Rate, Errors, Duration). Standard service monitoring.
- **Per-model metrics**: same RED metrics, broken down by model version and route.
- **Token throughput**: for LLMs, tokens-per-second per replica.
- **GPU utilization**: per replica.
- **Queue depth**: requests waiting for a replica.
- **End-to-end latency**: from client to response, including all components.

For traceability, distributed tracing (OpenTelemetry, Jaeger) tracks a request through the system.

## Cold-start latency

For LLM serving:
- Image pull: 1-5 minutes for a large container image.
- Model load: 30 seconds to 5 minutes depending on size.
- Warm-up: first few requests are slower (JIT compilation, cache priming).

Total cold start: often 5-15 minutes for a 70B model. Plan capacity such that you don't routinely cold-start.

Optimizations:
- Pre-pulled images on every node.
- Locally-cached weights on NVMe.
- mmap-based loading (Module 6 Lesson 7).

## What you should believe after this lesson

Three sentences:

**1. Model serving infrastructure is more than the inference engine** — it's autoscaling, request routing, load balancing, observability, deployment automation, multi-model support. KServe, Triton, Ray Serve, and cloud-managed options provide these.

**2. Cold starts dominate LLM serving operations** — 5-15 minutes for large models. Most production keeps min_replicas ≥ 1 (never scale to true zero) and pre-warms via image pre-pull and weight caching.

**3. Multi-model and multi-LoRA serving (vLLM, Triton)** share a fleet across many models / adapters; more memory-efficient than one-replica-per-model. The router selects model/adapter per request.

## Hands-on (at home)

Deploy a small model with KServe.

```bash
# Install KServe (requires K8s with Istio or similar).
curl -s "https://raw.githubusercontent.com/kserve/kserve/release-0.13/hack/quick_install.sh" | bash

# Define an InferenceService for a scikit-learn model.
cat > inference-service.yaml <<'EOF'
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: sklearn-iris
spec:
  predictor:
    sklearn:
      storageUri: gs://kfserving-examples/models/sklearn/1.0/model
EOF
kubectl apply -f inference-service.yaml

# After it's ready:
INGRESS_HOST=$(kubectl get svc istio-ingressgateway -n istio-system -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
SERVICE_HOSTNAME=$(kubectl get inferenceservice sklearn-iris -o jsonpath='{.status.url}' | cut -d"/" -f3)

curl -v -H "Host: ${SERVICE_HOSTNAME}" \
    "http://${INGRESS_HOST}/v1/models/sklearn-iris:predict" \
    -d '{"instances": [[6.8,2.8,4.8,1.4]]}'
```

You'll get back a prediction. KServe handled the autoscaling (default min_replicas=1, max=infinity), routing, and the prediction endpoint.

For an LLM, swap the sklearn predictor for a custom one using vLLM or Triton.

## Further reading

- KServe documentation.
- Triton Inference Server documentation.
- Knative autoscaling documentation.
- "Production-grade ML Serving" — various blog posts (Anyscale, Vertex AI).

Next lesson: **A/B testing infrastructure.** Traffic splitting, statistical rigor, attribution — how to actually know a new model is better.
