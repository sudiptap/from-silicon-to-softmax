---
title: "Lesson 24 — Distributed Agent Scheduling + Module Wrap"
date: "2026-06-04"
module: "agents"
order: 24
tags: ["distributed", "scheduling", "queue", "ray", "durable", "wrap", "curriculum-end"]
author: "Sudipta Pathak"
prerequisites: ["23-tracing-evaluation"]
---

# Lesson 24 — Distributed Agent Scheduling + Module Wrap

## Why this lesson exists

A single-process agent works fine for prototypes. Production agents at scale need: many concurrent agents, durable state (survive process crashes), back-pressure / rate-limiting on LLM APIs, cost control across the fleet, multi-tenant routing.

This is distributed agent scheduling: take the per-agent patterns from Lessons 1-23 and scale them across processes / machines.

This lesson covers the scheduling architecture, then closes Module 11 and the entire curriculum.

The lesson is reading. The Hands-on sketches a queue-backed agent worker.

## The architecture

```
              ┌─────────────────┐
              │  Frontend / API │
              └────────┬────────┘
                       │ submit task
                       ▼
              ┌─────────────────┐
              │  Task Queue     │ (Redis, SQS, RabbitMQ)
              └────────┬────────┘
                       │ pull task
              ┌────────▼────────┐
              │  Agent Worker   │  ×N (pool of workers)
              └────────┬────────┘
                       │ run agent loop
              ┌────────▼────────┐
              │  LLM API        │
              │  Tool services  │
              │  State store    │ (durable: Postgres, Redis)
              └─────────────────┘
                       │
                       ▼
              ┌─────────────────┐
              │  Result store   │
              └────────┬────────┘
                       │ retrieve
                       ▼
              ┌─────────────────┐
              │  Frontend / API │
              └─────────────────┘
```

The frontend submits tasks to a queue; workers pull tasks; each worker runs the agent loop; results go back via another queue or DB.

## Why queue-backed

Direct synchronous (frontend → agent → response) has issues:
- Long-running agent (minutes / hours) holds the HTTP connection.
- If the agent process crashes, the task is lost.
- Hard to apply back-pressure when LLM APIs are rate-limited.
- Doesn't scale: each frontend request creates an agent process.

Queue-backed:
- Frontend submits; gets a task ID; polls or webhooks for the result.
- Workers operate at their pace; bounded concurrency.
- Failed tasks can be retried (move back to queue).
- Easy to scale workers up/down based on queue depth.

The pattern is standard for any long-running async task; agents fit it.

## Durable state

For multi-step agents, the state (message history, scratchpad, intermediate results) needs to survive worker restarts.

Pattern:
- Each task has a unique ID.
- State is persisted to a DB (Postgres, Redis) keyed by task ID.
- The worker reads state at the start of each step; writes after.
- If the worker dies mid-step, another worker picks up; reads state; continues.

For most agent steps (single LLM call + tool call), the state update is small. The overhead of persisting is modest.

For very-long-running tasks (Devin-style, hours-long), durable state is essential. The worker process can be recycled multiple times during a single task.

## Rate limiting and back-pressure

LLM APIs have rate limits (RPM, TPM). Hitting them causes errors; ignoring causes throttling.

Patterns:
- **Token bucket** per LLM provider; the worker waits if no tokens available.
- **Adaptive rate**: reduce request rate on 429 (rate-limit error).
- **Queue prioritization**: high-priority tasks bypass low-priority during throttle.

For multi-provider deployments:
- **Failover**: if OpenAI throttles, try Anthropic.
- **Routing**: pick the provider with the most capacity right now.

## Cost control

Each agent task has a cost; aggregate cost across many tasks can be large.

Controls:
- **Per-task budget**: kill the agent if it exceeds N tokens.
- **Per-user budget**: rate-limit users who consume too much.
- **Per-tenant budget**: organizational quotas.
- **Cheaper-model routing**: simple queries to cheaper models; complex to better ones.

Cost-aware scheduling: when budget is tight, prefer cheaper paths.

## Multi-tenant routing

For multi-tenant agent services:
- Each tenant has their own configuration (system prompt, tool set, model preference).
- The worker reads the tenant's config; configures the agent accordingly.
- Per-tenant rate limits and budgets.

This is how commercial agent products operate at scale.

## Ray and other distributed frameworks

Ray (Module 9 Lesson 10) provides distributed Python primitives. For agents:

```python
import ray

@ray.remote
class AgentWorker:
    def run_task(self, task):
        # Standard agent loop.
        return result

workers = [AgentWorker.remote() for _ in range(10)]
results = ray.get([w.run_task.remote(task) for w in workers])
```

Ray handles the worker pool, distribution, fault tolerance.

Alternatives:
- **Custom workers + Redis queue**: lighter than Ray; simpler to operate.
- **Celery**: traditional Python distributed-task queue; works for agents.
- **Temporal / Restate**: durable execution engines; agents as workflows.

Temporal and similar are increasingly popular for long-running stateful agents (Devin / Manus / OpenHands all use durable-execution patterns).

## Putting it together

A production agent service in 2026 has:

- **Frontend**: HTTP API that accepts tasks, returns task IDs, serves status / results.
- **Queue**: Redis or SQS for task dispatch.
- **Worker pool**: agent workers (Ray, Celery, or custom).
- **State store**: Postgres / Redis for durable agent state.
- **LLM gateway**: rate-limited, multi-provider, retry-aware proxy to LLM APIs.
- **Tool services**: MCP servers (Lesson 20) hosting tool implementations.
- **Observability**: Langfuse / OpenTelemetry traces; per-task metrics; cost tracking.
- **Eval pipeline**: continuous evaluation on a test set; alerts on regression.
- **Multi-tenancy**: per-tenant config, rate limits, budgets.

This is the production agent platform. Lots of moving parts; each is necessary at scale.

## Module 11 wrap

Twenty-four lessons across ten parts:

**Part 1 (1-3)**: agent fundamentals — definition, the loop, planner-vs-worker.

**Part 2 (4-6)**: tool use — function calling, parallel calls, tool design.

**Part 3 (7-8)**: memory hierarchy and compaction.

**Part 4 (9-12)**: RAG — pipeline, chunking, hybrid search, agentic.

**Part 5 (13-14)**: planning and self-reflection.

**Part 6 (15-16)**: context engineering, subagent isolation.

**Part 7 (17-18)**: multi-agent — orchestration, failure modes.

**Part 8 (19-20)**: build it from scratch (100 lines), MCP.

**Part 9 (21-22)**: vertical agents — coding, browser/computer-use.

**Part 10 (23-24)**: production — tracing/eval, distributed scheduling.

The arc: from "what is an agent" to "production agent platform" with all the engineering between.

## Mental models to carry forward

Five sentences:

**1. An agent = autonomy + tools + feedback loop.** ReAct is the foundational pattern; everything else is variations.

**2. Tools are the most consequential design decision.** Good tools (clear names, schemas, error messages) make agents reliable; bad tools cause confusing failures.

**3. RAG is the dominant production pattern.** Hybrid search + reranking + agentic retrieval is the 2026 standard for any agent that needs external knowledge.

**4. Context is a resource to budget**, not infinite. Compaction, prompt caching, subagent isolation are the strategies.

**5. Production = distributed + durable + observable + evaluated.** A working agent demo is a starting point; a production agent platform is months of engineering.

## End of Module 11

Eleven modules, 188 lessons, hundreds of thousands of words. From "the CPU mental model" to "production agent platform."

The journey:
1. Bare metal (CPU primitives, kernels).
2. GPU & parallelism.
3. ML internals & optimization.
4. MLX & Apple Silicon internals.
5. Mobile & edge runtimes.
6. On-device LLM inference.
7. Inference from scratch (54 lessons; the deep dive).
8. Distributed systems.
9. Cluster orchestration.
10. ML platform engineering.
11. Agents from scratch (this module).

Each layer built on the previous. The system view: silicon to softmax to agent.

## End of curriculum

For someone who's worked through all 188 lessons: you have the depth to design ML systems end-to-end, from kernels to agents. You can reason about decisions across the stack — "this agent's latency is high; is it the model? The inference engine? The kernel? The network? The orchestration?"

The breadth and depth together are the value. ML systems engineering is a multi-layer discipline; understanding only one layer is insufficient.

What's next:
- **Practice**: build things. Take the patterns and apply them to a real product.
- **Stay current**: the field moves fast. Read papers, blogs, listen to interviews.
- **Specialize**: pick a layer that interests you most; go deeper.
- **Contribute**: open-source projects, papers, talks, your own writing.

The curriculum is the foundation. Your career builds on top.

## Hands-on (at home)

Sketch a queue-backed agent worker.

```python
# worker_sketch.py
import redis
import json

r = redis.Redis()

def worker_loop():
    while True:
        # Pull a task from the queue.
        task_id, task_json = r.blpop("agent_tasks", timeout=0)
        task = json.loads(task_json)
        
        try:
            # Load state if exists (durable resume).
            state_key = f"agent_state:{task['id']}"
            state_json = r.get(state_key)
            state = json.loads(state_json) if state_json else None
            
            # Run the agent.
            result = run_agent(task, state)
            
            # Persist final result.
            r.set(f"agent_result:{task['id']}", json.dumps(result))
            r.delete(state_key)
        
        except Exception as e:
            # Save state for retry.
            r.rpush("agent_tasks", task_json)  # re-queue
            log_error(task["id"], str(e))

# Run multiple worker processes; they share the queue and state store.
```

For production, the worker would use proper async, would persist state at each step, would handle rate limits, would emit traces. This is the skeleton.

For a full production system, use Temporal or Ray for the orchestration; the simple sketch above is the conceptual frame.

## Further reading

- Temporal documentation (durable execution).
- "Building Production Agent Systems" — emerging blog topic.
- Anthropic, OpenAI, and other vendor's production-grade agent examples.

## End of From Silicon to Softmax

Modules 1-11 are complete. The vertical stack from CPU primitives to production agent platform is the depth-track foundation.

Thank you for reading.
