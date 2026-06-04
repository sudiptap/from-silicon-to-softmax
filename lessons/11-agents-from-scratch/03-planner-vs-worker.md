---
title: "Lesson 3 — LLM-as-Planner vs LLM-as-Worker; State Machines vs Flexible Loops"
date: "2026-06-04"
module: "agents"
order: 3
tags: ["planner", "worker", "state-machine", "control-flow"]
author: "Sudipta Pathak"
prerequisites: ["02-minimal-agent-loop"]
---

# Lesson 3 — LLM-as-Planner vs LLM-as-Worker; State Machines vs Flexible Loops

## Why this lesson exists

The "where do we put the LLM" question shapes agent architecture. Two main choices:

**LLM-as-planner**: the LLM produces a plan (a sequence of steps); deterministic code executes the plan. The LLM's flexibility is constrained to the planning phase.

**LLM-as-worker**: the LLM is called inside the loop at each step. The LLM is the "doer," not just the planner.

A related axis: how rigid is the control flow?

**State machine**: predefined states and transitions. The agent moves between states; the LLM's choice is limited to "which transition." Predictable but inflexible.

**Flexible loop**: pure ReAct. The LLM decides every step. Adaptive but harder to debug.

This lesson covers the four quadrants and the right architectural choice for different tasks.

The lesson is reading. The Hands-on builds a state-machine agent for a fixed task.

## The four quadrants

|               | LLM-as-planner             | LLM-as-worker         |
| ------------- | -------------------------- | --------------------- |
| State machine | Plan + execute (clear states) | ReAct in a structured loop |
| Flexible loop | Plan + loose execution     | Pure ReAct (Lesson 1) |

Each has a sweet spot.

## LLM-as-planner + state machine

Best for: tasks with a known structure where you want predictability.

Example: a "research and summarize" agent with these states:
1. Receive query.
2. Generate sub-queries.
3. Search for each sub-query.
4. Read top results.
5. Synthesize.
6. Return.

The LLM picks the sub-queries (planning step) and synthesizes (final step). The middle steps (searching, fetching) are deterministic.

Pros: predictable behavior, easy debugging, easy testing.
Cons: not adaptive; if a step fails, the agent doesn't know how to recover beyond what's coded.

## LLM-as-worker + flexible loop

Best for: open-ended tasks where you can't predict the steps.

Example: "fix this bug in my codebase." The agent might need to read files, run tests, search docs, modify code, run again. The exact sequence isn't predictable.

Pros: adaptive; handles unexpected situations; matches open-ended tasks well.
Cons: harder to debug; can get stuck or wander; expensive (every step is an LLM call).

## LLM-as-planner + flexible loop

Plan first; then a flexible loop executes. The plan is a high-level outline; the loop fills in details.

Example: "write a research paper on X." The LLM outlines the paper (5 sections); a flexible loop fills each section using sub-agents.

This is the *hierarchical* pattern (Lesson 17). The top-level LLM is the planner; sub-agents (or the same LLM in a worker role) are the workers.

## LLM-as-worker + state machine

The middle ground: a structured loop where the LLM still does work at each state.

Example: a customer-support agent with states (greet, identify issue, look up account, propose solution, confirm, close). The LLM does the work at each state (drafts the response, looks up info) but the state machine controls flow.

Pros: predictable structure + LLM flexibility within each state.
Cons: more design effort upfront; less adaptive than pure flexible loop.

## When to use which

Practical recommendations:

**Task is well-defined and repeatable** (FAQs, customer support, specific data pipelines): state machine. The structure is right; the agent is just a smart filler.

**Task is open-ended exploration** (research, debugging, creative work): flexible loop. The structure can't be predicted in advance.

**Task has clear phases but uncertain execution within each** (e.g., plan a trip → book flights, hotels, activities): hierarchical. Plan once; each step is flexible.

**Task is short and one-shot**: just a prompt-and-response. Don't over-engineer.

The choice strongly affects engineering effort and operational characteristics. State machines are easier to deploy but require more design. Flexible loops are easier to start but harder to operationalize.

## The control plane

For complex agents, separating control from work helps:

- **Control plane**: the state machine, the loop, the routing.
- **Worker**: the LLM call(s) that do the actual reasoning at each step.

This separation makes the agent's behavior more testable. The control plane is deterministic (test with mocked LLM responses). The worker is the LLM (tested separately).

Frameworks like LangGraph make this separation explicit; the "graph" is the state machine.

## The "plan can be wrong" problem

A specific issue with LLM-as-planner: the plan might be bad. The LLM doesn't know what it'll find until it tries.

Mitigations:
- **Re-plan on failure**: if a step fails or returns unexpected results, ask the LLM to re-plan.
- **Plan + verify**: a second LLM call critiques the plan before execution.
- **Plan with reflection**: after each step, the LLM reflects on progress and decides whether to continue or re-plan.

These are mid-ground between rigid plan-then-execute and pure flexible loop.

## What you should believe after this lesson

Three sentences:

**1. The two main agent-architecture axes are LLM-as-planner vs LLM-as-worker, and state-machine vs flexible-loop.** Each quadrant has a sweet spot: state-machine + planner for well-defined tasks; flexible + worker (ReAct) for open-ended exploration; planner + flexible (hierarchical) for tasks with clear phases.

**2. Separating control plane from worker** makes agents more testable. The control plane (the state machine or loop) is deterministic; the worker (the LLM calls) is the variable part. This separation underpins frameworks like LangGraph.

**3. LLM-as-planner has the "plan can be wrong" problem** — the LLM doesn't know what it'll find until it tries. Mitigations: re-plan on failure, plan-and-verify, plan with reflection. Pure ReAct sidesteps this but loses planning benefits.

## Hands-on (at home)

Build a state-machine agent for a fixed task.

```python
# state_machine_agent.py — sketch.
from enum import Enum

class State(Enum):
    GREETING = 1
    IDENTIFY_ISSUE = 2
    LOOK_UP_ACCOUNT = 3
    PROPOSE_SOLUTION = 4
    CONFIRM = 5
    CLOSE = 6
    DONE = 7

class SupportAgent:
    def __init__(self):
        self.state = State.GREETING
        self.context = {}
    
    def step(self, user_input):
        if self.state == State.GREETING:
            response = "Hello! How can I help you today?"
            self.state = State.IDENTIFY_ISSUE
        elif self.state == State.IDENTIFY_ISSUE:
            issue = llm_call(f"Categorize this issue: {user_input}")
            self.context["issue"] = issue
            response = f"I understand you're having an issue with {issue}. Can I get your account ID?"
            self.state = State.LOOK_UP_ACCOUNT
        elif self.state == State.LOOK_UP_ACCOUNT:
            account = lookup_account(user_input)
            self.context["account"] = account
            response = llm_call(f"Propose a solution for {self.context['issue']} given account {account}")
            self.state = State.PROPOSE_SOLUTION
        # ... etc.
        return response

agent = SupportAgent()
while agent.state != State.DONE:
    user_input = input("> ")
    print(agent.step(user_input))
```

The control flow is explicit. Each state does specific work; the LLM is called where flexibility is needed. Easier to debug than pure ReAct; less adaptive.

## Further reading

- "ReAct" and "Plan-and-Execute" papers.
- LangGraph documentation — the canonical state-machine framework.
- "Designing Agentic Systems" — various 2024-2026 blog posts.

Next lesson: **Native function calling.** What's actually in the API payload; JSON-schema decoding; how the model knows which tool to call.
