---
title: "Lesson 16 — Subagent Isolation"
date: "2026-06-04"
module: "agents"
order: 16
tags: ["subagent", "isolation", "delegation", "fresh-context", "hierarchical"]
author: "Sudipta Pathak"
prerequisites: ["15-context-as-resource"]
---

# Lesson 16 — Subagent Isolation

## Why this lesson exists

An agent's context accumulates. After 50 turns, even with compaction, the context is full of distractions: old tool results, irrelevant decisions, side conversations. The agent's reasoning suffers.

Subagent isolation is a clean fix: when a complex subtask needs focused work, spin up a *fresh subagent* with its own minimal context, just for the subtask. The parent agent gets back the result; the subagent's context is discarded.

This lesson covers the pattern: when to use it, how to implement it, and the tradeoffs.

The lesson is reading. The Hands-on builds a parent+subagent system.

## The pattern

Parent agent receives a complex request. Instead of solving it in the same context (which is already cluttered), the parent:

1. Identifies the subtask.
2. Constructs a focused prompt for it.
3. Spawns a subagent (a fresh LLM call, or sub-loop) with just that subtask's context.
4. Receives the result.
5. Continues with its main task, incorporating the result.

Like calling a subroutine. The subroutine's local variables don't pollute the caller's scope.

## When subagent isolation helps

The strong cases:

**Complex subtask in a long conversation**: the main thread is full of unrelated context. The subtask needs focus.

**Parallel work**: spawn multiple subagents simultaneously for independent subtasks. (Combined with parallel tool calls; Lesson 5.)

**Specialized expertise**: a research subagent has access to different tools than a coding subagent. Use specialized agents for specialized work.

**Containment**: a risky operation (might call dangerous tools, might generate harmful content) is isolated in a subagent with restricted permissions.

## When NOT to use it

When the subtask is tightly coupled to the main thread's context:
- "Continue our discussion about X" — needs the main context.
- "Refer to what we decided earlier" — needs the history.

Subagent isolation loses this context (deliberately). For tight coupling, stay in the main thread.

## Implementation

The simplest version: the parent agent has a `delegate` tool that spawns a subagent.

```python
def delegate(task: str, system_prompt: str = "You are a focused assistant. Complete the task and return only the result.") -> str:
    """Delegate a focused subtask to a fresh subagent."""
    # Create a new agent with fresh context.
    sub_agent = Agent(system_prompt=system_prompt, tools=[...])
    return sub_agent.run(task)

# Used in the parent agent's tools.
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "delegate",
            "description": "Delegate a focused subtask to a subagent. Use when a subtask is complex enough to need its own focused context.",
            "parameters": {
                "type": "object",
                "properties": {
                    "task": {"type": "string", "description": "The subtask description."},
                },
                "required": ["task"]
            }
        }
    }
]
```

The parent decides when to delegate; the model often makes this call correctly when prompted to.

## The "research subagent" pattern

A common production pattern: the parent agent handles the user's conversation; for any non-trivial query, it delegates research to a subagent.

The research subagent:
- Has its own RAG tools.
- Has a fresh context (just the query, no prior conversation).
- Does iterative retrieval (Lesson 12).
- Returns a structured summary.

Why this pattern?
- The user's conversation history bloats the context if we tried to do RAG in-thread.
- The research subagent can use long context for retrieved chunks without affecting the parent.
- The research subagent's output (clean summary) is what enters the parent's context.

## Multiple subagent types

For complex products, you might have:
- **Research subagent**: information gathering.
- **Coding subagent**: writes / modifies code.
- **Browser subagent**: navigates web pages.
- **Critic subagent** (Lesson 14): reviews outputs.

Each has its own system prompt and tool set. The parent picks which to delegate to based on the task.

This is *hierarchical multi-agent* (Lesson 17 covers more). The parent is the orchestrator; the subagents are workers.

## Context-saving via isolation

A specific cost benefit: a 50K-token parent context isn't expanded by the subagent's work. The subagent has its own context (5K-20K typically); its 50 internal steps don't bloat the parent.

If you tried to do everything in the parent, after a few complex subtasks, the parent's context would be unmanageable. With isolation, the parent's context grows slowly (just the subagent results), while the work happens elsewhere.

This is a big win for long-running agents.

## Communication protocol

The subagent's interface should be clean:
- **Input**: focused task description.
- **Output**: structured result (not a long narration).

A bad subagent return value:

```
"I started by trying X. That didn't work because Y. So I tried Z. Z gave me result W, which I then refined to ..."
```

A good return value:

```
{
    "result": "The answer is 42.",
    "confidence": 0.9,
    "sources": ["doc-123", "doc-456"]
}
```

The first version dumps the subagent's internal monologue into the parent's context — defeats the isolation purpose. The second is concise.

Prompt the subagent to be terse: "Return only the final result, not your reasoning."

## Failure modes

Subagent failures:
- **Subagent gets stuck**: max-steps reached without result. Parent must handle.
- **Subagent returns garbage**: parent's reasoning gets confused.
- **Coordination issues**: parent's "complete the task" instruction is ambiguous; subagent does the wrong thing.

Mitigations: parent verifies the subagent's output makes sense; retries with more specific instructions if needed.

## What you should believe after this lesson

Three sentences:

**1. Subagent isolation gives complex subtasks their own fresh context** — the parent's accumulated history doesn't pollute the subtask's reasoning. Implemented as a `delegate` tool the parent calls.

**2. The pattern saves the parent's context** (subagent work happens elsewhere) and improves subtask quality (focused context). Common in production for research subagents, coding subagents, browser subagents.

**3. The subagent's interface should be clean** — focused input task; structured output result. Don't let the subagent's internal monologue bleed into the parent's context.

## Hands-on (at home)

Build a parent + research-subagent system.

```python
# subagent_demo.py
from openai import OpenAI
client = OpenAI()

def llm(messages):
    return client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages
    ).choices[0].message.content

def research_subagent(query):
    """A fresh-context subagent for research tasks."""
    messages = [
        {"role": "system", "content": "You are a focused research assistant. Answer the query concisely. Return only the answer, not your reasoning."},
        {"role": "user", "content": query}
    ]
    return llm(messages)

def parent_agent(user_input):
    # The parent has a tool to delegate.
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Use the delegate tool to spawn research subagents for any non-trivial research questions."},
    ]
    
    # Simplified: parent just makes one delegation call.
    # In reality, parent decides via function calling.
    research_query = user_input  # would normally be extracted by the LLM
    result = research_subagent(research_query)
    
    messages.append({"role": "user", "content": user_input})
    messages.append({"role": "user", "content": f"[Subagent result: {result}]"})
    messages.append({"role": "user", "content": "Now respond to the user using the subagent's result."})
    
    return llm(messages)

print(parent_agent("What's the population of Tokyo and how does it compare to Paris?"))
```

The parent's context stays small; the research subagent does the heavy lifting in isolation.

For a real production version, use the agent loop from Lesson 4 with `delegate` as one of the parent's tools.

## Further reading

- Anthropic's "Building Effective Agents" — mentions subagent patterns.
- Claude Code's architecture — heavy use of subagents for research and editing.
- Various blog posts on multi-agent design (Lesson 17 expands).

End of Part 6. Next: Part 7 begins with **The orchestrator-worker pattern** — the formal multi-agent architecture.
