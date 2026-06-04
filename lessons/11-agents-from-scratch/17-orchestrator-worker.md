---
title: "Lesson 17 — The Orchestrator-Worker Pattern"
date: "2026-06-04"
module: "agents"
order: 17
tags: ["orchestrator", "worker", "multi-agent", "debate", "crew", "handoff"]
author: "Sudipta Pathak"
prerequisites: ["16-subagent-isolation"]
---

# Lesson 17 — The Orchestrator-Worker Pattern

## Why this lesson exists

Multi-agent systems involve multiple LLM-powered agents coordinating to complete a task. The most common architecture: an *orchestrator* (or "manager," "coordinator") delegates subtasks to *workers* (or "specialists," "agents"); workers report back; orchestrator synthesizes.

This is the natural extension of subagent isolation (Lesson 16) to multiple, possibly specialized, agents.

This lesson covers the patterns: orchestrator-worker (the canonical), debate (multiple workers argue), role-based crews (CrewAI-style), handoff protocols (one agent passes control to another).

The lesson is reading. The Hands-on builds an orchestrator-worker for a research task.

## Orchestrator-worker

The structure:

```
                  ORCHESTRATOR
                  │
       ┌──────────┼──────────┐
       ▼          ▼          ▼
   Worker 1   Worker 2   Worker 3
   (research)  (analyze)  (write)
```

The orchestrator:
- Receives the user's task.
- Plans which workers to call.
- Sends subtasks to workers.
- Synthesizes worker outputs into the final response.

Workers:
- Receive a focused subtask.
- Have their own tools, system prompt, possibly fresh context.
- Return structured results.

The pattern is the multi-agent generalization of subagent isolation (Lesson 16). The differences:
- Multiple workers, possibly specialized.
- Workers may communicate with each other (or via the orchestrator).
- More elaborate planning is possible.

## Debate / adversarial agents

A variant: multiple agents argue from different perspectives; a "judge" picks the best.

```
TASK: "Is this code correct?"

  Optimist agent: "Yes, because..."
  Pessimist agent: "No, here are bugs..."
  Judge: "After considering both, my verdict is..."
```

Useful for:
- Tasks where multiple valid perspectives exist.
- Catching mistakes that a single agent would miss (the pessimist's role is to find problems).
- Improving robustness via diversity.

Cost: 3-5x a single agent. Worth it for high-stakes outputs.

## Role-based crews (CrewAI-style)

CrewAI popularized a different framing: each agent has a *role* (researcher, writer, editor, fact-checker); the crew works through a defined sequence.

```python
researcher = Agent(role="Researcher", goal="Find facts.", tools=[search])
writer = Agent(role="Writer", goal="Write a draft.", tools=[])
editor = Agent(role="Editor", goal="Polish the draft.", tools=[])

crew = Crew(agents=[researcher, writer, editor], tasks=[research_task, writing_task, editing_task])
result = crew.kickoff()
```

The crew structure makes the workflow explicit. Each agent's role shapes its prompts and tools.

Pros: declarative; readable; reusable agent definitions.
Cons: rigidity (the structure is fixed); not adaptive to unexpected situations.

## Handoff protocols (Swarm-style)

OpenAI's Swarm framework popularized *handoffs*: one agent transfers control to another.

```python
def transfer_to_specialist():
    return specialist_agent  # the returning agent runs next

triage_agent = Agent(
    instructions="You triage user requests. For technical issues, hand off to the specialist.",
    functions=[transfer_to_specialist],
)
```

The triage agent decides when another agent is better-suited; the handoff transfers the conversation.

This is a more flexible model than the strict orchestrator-worker hierarchy: any agent can pass to any other, based on the situation.

Pros: adaptive; mirrors real customer-service handoffs.
Cons: harder to reason about flow; can lead to handoff loops if not carefully designed.

## Communication patterns

The agents need to communicate. Three main patterns:

**Through the orchestrator**: agents talk to the orchestrator; orchestrator routes. Star topology.

**Direct messages**: agents talk to each other directly. Peer topology.

**Shared workspace**: agents read and write to a shared document / state. Like a Google Doc that all agents edit.

For most production agents, through-the-orchestrator is cleanest. Direct messages get tangled; shared workspace requires careful concurrency handling.

## Why multi-agent works (when it works)

The case for multi-agent:
- **Specialization**: a coding agent with code-specific tools beats a generalist for coding tasks.
- **Parallelism**: independent subtasks run in parallel.
- **Fresh context**: each agent's reasoning isn't polluted by others' irrelevant context.
- **Verifiability**: the orchestrator can verify each worker's output before integration.

The case against:
- **Coordination overhead**: orchestrating costs LLM calls.
- **Communication failures**: agents miscommunicate; results don't fit together.
- **Cost**: N agents at K calls each is K×N LLM calls per task.
- **Single agent might suffice**: many "multi-agent" systems would work with one well-prompted agent and tools.

A good rule: start with one agent. Add subagents only when there's a clear specialization or context-isolation benefit.

## A2A and interop

A2A (Agent-to-Agent) is an emerging interop standard for agents from different vendors talking to each other.

In 2026, the landscape:
- **MCP** (Model Context Protocol; Lesson 20): standard for agent-tool communication.
- **A2A**: emerging standard for agent-to-agent communication.

If your agent talks to another team's agent, A2A is what they'll use. Adoption is growing but not yet universal.

## What you should believe after this lesson

Three sentences:

**1. Orchestrator-worker is the canonical multi-agent pattern**: orchestrator plans and synthesizes; workers do specialized subtasks. Common variants: debate (multiple perspectives), role-based crews (declarative), handoff (Swarm-style adaptive routing).

**2. Multi-agent helps when there's specialization, parallelism, or context-isolation benefit.** It hurts via coordination overhead, communication failures, and cost. Start with one agent; add more only with clear benefit.

**3. MCP is the agent-tool standard; A2A is the emerging agent-agent standard.** Both are increasingly necessary for agents that need to interop across vendors.

## Hands-on (at home)

Build an orchestrator-worker for a research task.

```python
# orchestrator_worker.py
from openai import OpenAI
import json
client = OpenAI()

def llm(messages, tools=None):
    return client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        tools=tools or [],
    )

# Worker 1: research.
def research_worker(query):
    msgs = [
        {"role": "system", "content": "You are a research agent. Find information about the query and return a concise summary."},
        {"role": "user", "content": query}
    ]
    return llm(msgs).choices[0].message.content

# Worker 2: write.
def writing_worker(facts, style="formal"):
    msgs = [
        {"role": "system", "content": f"You are a writer. Write a paragraph in {style} style using the given facts."},
        {"role": "user", "content": f"Facts: {facts}"}
    ]
    return llm(msgs).choices[0].message.content

# Worker 3: edit.
def editing_worker(draft):
    msgs = [
        {"role": "system", "content": "You are an editor. Improve the draft for clarity and concision. Return the edited version."},
        {"role": "user", "content": draft}
    ]
    return llm(msgs).choices[0].message.content

# Orchestrator.
def orchestrate(task):
    # Plan: research → write → edit.
    print(f"Task: {task}\n")
    
    print("Worker 1 (research):")
    facts = research_worker(f"Research for: {task}")
    print(f"  {facts[:200]}...\n")
    
    print("Worker 2 (write):")
    draft = writing_worker(facts)
    print(f"  {draft[:200]}...\n")
    
    print("Worker 3 (edit):")
    final = editing_worker(draft)
    print(f"FINAL:\n{final}\n")

orchestrate("Write a paragraph about the history of the transistor.")
```

The orchestrator calls workers in sequence; each worker has its own fresh context. The final output is the synthesis.

For a real product, the orchestrator's planning would be LLM-driven (using `delegate` tools); workers would be agents themselves (not single LLM calls).

## Further reading

- "Building Effective Agents" (Anthropic, 2024) — has multi-agent patterns.
- CrewAI documentation.
- OpenAI Swarm documentation.
- "Multi-Agent Collaboration" (various academic papers, 2023-2026).

Next lesson: **Communication failure modes** — deadlock, divergence, role drift; the things that break multi-agent systems and how to detect / prevent them.
