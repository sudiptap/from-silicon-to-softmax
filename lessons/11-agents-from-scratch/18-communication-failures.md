---
title: "Lesson 18 — Communication Failure Modes"
date: "2026-06-04"
module: "agents"
order: 18
tags: ["multi-agent", "failure-modes", "deadlock", "divergence", "role-drift"]
author: "Sudipta Pathak"
prerequisites: ["17-orchestrator-worker"]
---

# Lesson 18 — Communication Failure Modes

## Why this lesson exists

Multi-agent systems fail in distinctive ways. Single-agent failures (parse errors, infinite loops) are joined by communication failures: agents miscommunicate, end up in loops with each other, drift from their roles, deadlock.

This lesson catalogs the major communication failure modes, how to detect them, and how to mitigate them.

The lesson is reading. The Hands-on simulates each failure mode.

## Deadlock

Two or more agents wait for each other.

Example:
- Agent A: "Waiting for Agent B's input."
- Agent B: "Waiting for Agent A's clarification."

Both are stuck. No progress.

Causes:
- Hand-off protocols where each side expects the other to act first.
- Underspecified contracts ("you do something with this").
- Agents that won't act without being explicitly given a task.

Detection: no agent action for N steps.

Mitigation:
- Timeouts: if no action in N seconds, force one agent to act.
- Clear protocols: define which agent acts first in any handoff.
- Fail fast: if waiting, abort and restart.

## Divergence

Agents disagree and the disagreement doesn't resolve.

Example:
- Agent A proposes solution X.
- Agent B disagrees, proposes Y.
- Agent A insists on X.
- Agent B insists on Y.
- Iteration continues without convergence.

Causes:
- No defined arbitration mechanism.
- Each agent's confidence is invariant; the discussion doesn't update beliefs.
- The orchestrator/judge isn't authoritative.

Detection: repeated similar messages; no convergence after N iterations.

Mitigation:
- A judge agent with authority to decide.
- Iteration limit; pick the most-supported option.
- Force-pick based on initial confidence scores.

## Role drift

An agent starts behaving outside its assigned role.

Example:
- Setup: "researcher" agent should only gather facts.
- After a few turns: the researcher is writing summary opinions, not facts.
- Eventually it's hard to tell which agent is which.

Causes:
- Role definition not strict enough.
- Other agents' messages contaminate the role.
- Long contexts where the original role-setting is overshadowed by recent messages.

Detection: agent output doesn't match its role's expected pattern.

Mitigation:
- Re-inject the role into context regularly ("Remember: you are the researcher").
- Strict input filtering: the agent only sees role-relevant inputs.
- Periodic role checks (a separate critic verifies the role is preserved).

## Echo loops

Agents amplify each other's mistakes:
- Agent A makes an error.
- Agent B accepts A's output as ground truth; references it.
- Agent A now has B's reference to its error; treats it as confirmed.
- The error becomes "the truth" in the system.

Causes:
- No verification step.
- Trust between agents is implicit.

Mitigation:
- Independent verification (a verifier sees only the final claim, not the chain that produced it).
- Sources / citations: every claim must trace back to a verifiable source.

## Cascading errors

A small error in one agent's output causes large errors in downstream agents.

Example:
- Agent A misidentifies an entity.
- Agent B uses A's wrong entity; produces wrong analysis.
- Agent C synthesizes B's wrong analysis with other data; produces gibberish.

Causes:
- Lack of error checking between agents.
- Each agent trusts the previous one fully.

Mitigation:
- Each agent validates inputs.
- The orchestrator spots-checks intermediate results.
- Confidence scores propagate ("Agent A reports 60% confidence in entity X").

## Context pollution across agents

When agents share context, one agent's bad input contaminates others.

Example:
- Agent A's response contains a hallucinated fact.
- Agent B reads A's response as part of its input.
- Agent B incorporates the hallucination into its own reasoning.

Causes:
- Loose information flow between agents.
- No filtering at boundaries.

Mitigation:
- Structured outputs at boundaries (not free-text).
- Each agent's input is sanitized.
- Hallucinations caught earlier (Lesson 14 verifier agents).

## Coordination overhead dominating

When the orchestration cost exceeds the work cost:
- 10 agents each take 1 second; 9 seconds of orchestrator coordination overhead.
- Effective work: 1 second; total: 10 seconds.

Causes:
- Over-fragmenting tasks.
- Synchronous communication when async would work.
- Verbose protocols.

Mitigation:
- Fewer, more capable agents.
- Parallel agent execution where possible.
- Concise communication protocols.

## Tooling for diagnosis

To diagnose communication failures:

**Distributed tracing**: every message logged with timestamps; visualize the agent interaction as a sequence diagram.

**Per-agent observability**: monitor each agent's behavior (Lesson 23) — message count, role-adherence, output quality.

**Replay**: be able to re-run a failed interaction with debugging.

The tools: Langfuse, LangSmith, OpenTelemetry-based custom stacks. Lesson 23 covers this in detail.

## Design patterns to avoid the failures

**Use the orchestrator-worker pattern** (Lesson 17) rather than peer-to-peer for most cases — fewer interaction modes; clearer control.

**Define strict contracts** between agents — input schema, output schema, error format. Like API contracts in microservices.

**Limit iteration counts** — every multi-agent loop should have a max. If unreached, force termination.

**Independent verification** — the final output is checked by an agent (or human) outside the loop.

**Start simple** — single agent; add multi-agent only with clear benefit. Many "multi-agent" systems would work with one well-designed agent.

## What you should believe after this lesson

Three sentences:

**1. Multi-agent systems have distinctive failure modes**: deadlock, divergence, role drift, echo loops, cascading errors, context pollution, coordination overhead. Each requires specific mitigation; debugging requires tracing and per-agent observability.

**2. The mitigations are**: timeouts, iteration limits, strict contracts, independent verification, role re-injection, structured boundaries. Applied consistently, they make multi-agent reliable; skipped, the failures cascade.

**3. The strongest defense is "use fewer agents."** Many multi-agent designs are over-engineered for the actual problem; a well-prompted single agent with good tools often suffices. Multi-agent's overhead and failure modes only pay off when there's clear specialization or parallelism benefit.

## Hands-on (at home)

Simulate a deadlock and a divergence; observe; add mitigations.

```python
# failure_modes.py
class Agent:
    def __init__(self, name, response_fn):
        self.name = name
        self.response_fn = response_fn
    def respond(self, message):
        return self.response_fn(message)

# Deadlock setup: both agents say "your turn."
agent_a = Agent("A", lambda m: "Waiting for B's input.")
agent_b = Agent("B", lambda m: "Waiting for A's input.")

# Simulate.
messages = []
current_speaker = agent_a
last_message = "Begin task."
for step in range(10):
    response = current_speaker.respond(last_message)
    messages.append((current_speaker.name, response))
    print(f"{current_speaker.name}: {response}")
    if response == last_message:  # detect deadlock
        print("DEADLOCK DETECTED")
        break
    last_message = response
    current_speaker = agent_b if current_speaker == agent_a else agent_a

# Mitigation: force one to act after detection.
print("\nWith deadlock detection + force-action mitigation:")
# ... resume; force agent_a to act.
```

For divergence, use an iteration limit + a tiebreaker. For role drift, re-inject role at each turn.

The pattern: detect → mitigate → continue. Without explicit handling, the failures cascade.

## Further reading

- "Multi-Agent Systems" (Wooldridge) — classic textbook.
- "When Do LLM Agents Make Mistakes? A Study of Failures in Multi-Agent Settings" (various 2024 papers).
- LangSmith / Langfuse documentation for multi-agent debugging.

End of Part 7. Next: Part 8 begins with **A minimal agent in 100 lines** — the framework-agnostic foundation that every framework implements.
