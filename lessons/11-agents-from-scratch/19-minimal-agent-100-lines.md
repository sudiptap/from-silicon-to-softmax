---
title: "Lesson 19 — A Minimal Agent in 100 Lines"
date: "2026-06-04"
module: "agents"
order: 19
tags: ["minimal-agent", "framework-agnostic", "implementation", "patterns"]
author: "Sudipta Pathak"
prerequisites: ["18-communication-failures"]
---

# Lesson 19 — A Minimal Agent in 100 Lines

## Why this lesson exists

Frameworks (LangGraph, CrewAI, AutoGen, OpenAI Agents SDK) are useful abstractions. They're also opaque — when something goes wrong, you're debugging through layers of framework code.

This lesson builds the entire agent — loop, tool use, memory, multi-step — in ~100 lines of pure Python, no framework. You'll see that everything frameworks do is implementable directly. After this, framework code is more navigable; framework abstractions become a familiar pattern rather than magic.

The lesson is reading. The Hands-on is the 100-line implementation.

## The architecture

The components:
- **LLM call**: wrap the API.
- **Tool registry**: name → function.
- **Tool schemas**: for function calling.
- **Memory**: message list + optional scratchpad.
- **Loop**: run until done or max steps.
- **Logging**: print what's happening.

Each is small; the whole thing fits in ~100 lines.

## The code

```python
# minimal_agent.py
import json
import time
from typing import Callable, Any
from openai import OpenAI

client = OpenAI()

# -----------------------
# Tool registry.
# -----------------------
class Tool:
    def __init__(self, name: str, description: str, params_schema: dict, fn: Callable):
        self.name = name
        self.description = description
        self.params_schema = params_schema
        self.fn = fn
    
    def to_openai_schema(self):
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.params_schema,
            }
        }
    
    def __call__(self, **kwargs):
        return self.fn(**kwargs)


# -----------------------
# Agent.
# -----------------------
class Agent:
    def __init__(self, system_prompt: str, tools: list[Tool], model: str = "gpt-4o-mini", verbose: bool = True):
        self.system = system_prompt
        self.tools = {t.name: t for t in tools}
        self.model = model
        self.verbose = verbose
        self.history = []
    
    def _log(self, msg):
        if self.verbose:
            print(msg)
    
    def _llm(self, messages):
        return client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=[t.to_openai_schema() for t in self.tools.values()] if self.tools else None,
        )
    
    def reset(self):
        self.history = []
    
    def run(self, user_input: str, max_steps: int = 10) -> str:
        self.history.append({"role": "user", "content": user_input})
        messages = [{"role": "system", "content": self.system}] + self.history
        
        for step in range(max_steps):
            self._log(f"\n[Step {step}]")
            response = self._llm(messages)
            msg = response.choices[0].message
            messages.append(msg.model_dump(exclude_none=True))
            self.history.append(msg.model_dump(exclude_none=True))
            
            if msg.content and not msg.tool_calls:
                self._log(f"FINAL: {msg.content[:100]}")
                return msg.content
            
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    args = json.loads(tc.function.arguments)
                    tool_name = tc.function.name
                    self._log(f"  Calling {tool_name}({args})")
                    if tool_name not in self.tools:
                        result = f"Error: unknown tool {tool_name}"
                    else:
                        try:
                            result = self.tools[tool_name](**args)
                        except Exception as e:
                            result = f"Error: {e}"
                    self._log(f"    → {str(result)[:100]}")
                    tool_msg = {"role": "tool", "tool_call_id": tc.id, "content": str(result)}
                    messages.append(tool_msg)
                    self.history.append(tool_msg)
        
        self._log("MAX STEPS REACHED")
        return "(no answer)"


# -----------------------
# Example tools.
# -----------------------
def search_tool(query: str) -> str:
    return f"Search results for '{query}': [placeholder]"

def calculator_tool(expression: str) -> str:
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Error: {e}"

tools = [
    Tool(
        name="search",
        description="Search the web for information.",
        params_schema={"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
        fn=search_tool,
    ),
    Tool(
        name="calculator",
        description="Evaluate a Python math expression.",
        params_schema={"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]},
        fn=calculator_tool,
    ),
]


# -----------------------
# Run.
# -----------------------
if __name__ == "__main__":
    agent = Agent(
        system_prompt="You are a helpful assistant. Use tools when needed.",
        tools=tools,
    )
    answer = agent.run("What is 17 * 24?")
    print(f"\n>>> {answer}")
```

That's about 90 lines of meaningful code. The patterns:

- A `Tool` class wraps a function with its OpenAI-format schema.
- The `Agent` class has system prompt, tools, history, an `_llm` method, and a `run` method.
- `run` is a loop: call LLM, parse response, execute tool calls, append observations, repeat until final answer or max steps.

## What this implements

This minimal agent has:
- ✓ Tool use via native function calling.
- ✓ Multi-step reasoning (loops until done).
- ✓ Memory (message history persists across `run` calls).
- ✓ Logging.
- ✓ Error handling.
- ✓ Max-steps safety.

What it lacks:
- ✗ Async / parallel tool calls.
- ✗ Streaming.
- ✗ Multi-agent.
- ✗ Persistent state across processes.
- ✗ Observability beyond print statements.
- ✗ Retries / rate-limit handling.

For most prototypes, this is enough. For production, you add the missing pieces.

## What frameworks add

Comparing to LangGraph or similar:

| Feature | This impl | Framework |
| ------- | --------- | --------- |
| Tool use | Inline | Inline |
| Loop | Manual `for` | `Graph` with nodes / edges |
| Memory | List | Messages + state object |
| Async | No | Yes (often) |
| Multi-agent | DIY | First-class |
| Streaming | No | First-class |
| State persistence | No | Checkpointing |
| Observability | print | LangSmith / OpenTelemetry |
| Replay | No | First-class |

The framework's value is the things you'd otherwise build yourself: async, state persistence, observability, replay. For production agents, these matter.

For prototypes, learning, and understanding what frameworks do under the hood — build it yourself first.

## When to use a framework

Use a framework when:
- You need multi-agent quickly (CrewAI / Swarm).
- You need persistent state across processes (LangGraph's checkpointing).
- You need observability + tracing (LangSmith, Langfuse).
- You need streaming UIs (most frameworks).

Build from scratch when:
- Learning how agents work.
- Building something the framework abstractions don't fit.
- Small prototype where the framework's overhead isn't worth it.

The decision: framework for production / complex; from-scratch for understanding / simple.

## What you should believe after this lesson

Three sentences:

**1. An agent — loop, tool use, memory, multi-step — is implementable in ~100 lines of pure Python.** Frameworks add convenience for production concerns (async, state persistence, observability, streaming) but the core is small.

**2. Building from scratch first is the best way to understand frameworks.** After this lesson, LangGraph / CrewAI / AutoGen code is recognizable rather than magic.

**3. Use a framework for production agents with multi-process state, multi-agent, observability needs.** Use from-scratch for learning, prototypes, or when framework abstractions don't fit.

## Hands-on (at home)

Run the 100-line agent. Extend it:

1. Add a `memory` tool that lets the agent remember facts across calls.
2. Add a `delegate` tool that spawns a subagent (Lesson 16).
3. Add streaming output (use the OpenAI client's `stream=True`).
4. Add a `done` tool the agent can explicitly call to terminate.

Each extension is 10-20 lines. The result: a more featureful agent, still under 200 lines.

```python
# Example: a memory tool.
class MemoryStore:
    def __init__(self):
        self.facts = {}
    def remember(self, key, value):
        self.facts[key] = value
        return f"Remembered: {key} = {value}"
    def recall(self, key):
        return self.facts.get(key, "I don't know that.")

mem = MemoryStore()
memory_tools = [
    Tool("remember", "Remember a fact.", {"type": "object", "properties": {"key": {"type": "string"}, "value": {"type": "string"}}, "required": ["key", "value"]}, lambda key, value: mem.remember(key, value)),
    Tool("recall", "Recall a fact.", {"type": "object", "properties": {"key": {"type": "string"}}, "required": ["key"]}, lambda key: mem.recall(key)),
]

agent = Agent("You are an assistant with memory.", tools + memory_tools)
agent.run("Remember that my favorite color is blue.")
agent.reset()
agent.run("What's my favorite color?")  # should recall blue
```

## Further reading

- Anthropic's "Building Effective Agents" (2024) — supports the minimalist approach.
- LangGraph source code — compare to this implementation.
- "Building a ReAct Agent from Scratch" — various tutorials.

Next lesson: **MCP from scratch** — the Model Context Protocol. The 2024-2026 standard for agent-tool communication.
