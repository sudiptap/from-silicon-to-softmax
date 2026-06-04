---
title: "Lesson 1 — What Makes a System Agentic + ReAct from Scratch"
date: "2026-06-04"
module: "agents"
order: 1
tags: ["agents", "react", "tool-use", "agent-loop"]
author: "Sudipta Pathak"
prerequisites: ["00-overview"]
---

# Lesson 1 — What Makes a System Agentic + ReAct from Scratch

## Why this lesson exists

"Agent" is one of the most overloaded terms in 2026 ML. Some products call any LLM call an "agent;" some reserve the term for elaborate multi-step systems. Without a precise definition, the topic is impossible to discuss coherently.

This lesson sets the working definition (autonomy + tools + feedback loop) and builds the simplest meaningful agent: a ReAct loop in under 200 lines of Python. By the end you should be able to look at any "agent" claim and place it on a scale from "just a wrapped prompt" to "genuinely autonomous."

The lesson is reading. The Hands-on is the ReAct loop.

## The working definition

An *agent*, in this curriculum, is a system with three properties:

1. **Autonomy**: it decides what to do next, not the human.
2. **Tools**: it can take actions in the world (call APIs, modify files, query data).
3. **Feedback loop**: it observes the result of its action and uses that to decide what to do next.

A chat bot that answers questions is not an agent (no tools, no feedback loop). A workflow that runs `pip install && pytest` is not an agent (no autonomy; the steps are hardcoded). A system that decides "I should query the database, then summarize, then send an email, then check if the email bounced" is an agent.

Different products and papers define "agent" more or less loosely. The strict definition above is what we'll use; it's what makes the engineering challenges interesting.

## The ReAct pattern

ReAct (Reason + Act, Yao et al, 2022) is the seminal agent pattern. The loop:

```
While not done:
    1. THOUGHT: the LLM generates a thought ("I should look up X").
    2. ACTION: the LLM picks a tool and arguments ("search(X)").
    3. OBSERVATION: the tool runs; the result is appended to context.
    4. Repeat until the LLM emits "FINAL ANSWER: ...".
```

The LLM's output at each step interleaves THOUGHT and ACTION tokens. The agent loop parses these, executes the action, and appends the OBSERVATION.

The pattern is simple but powerful: the model has the full thread of reasoning + actions + observations in context; can plan and adapt as it learns from each observation.

## Why ReAct works

Two reasons:

**Explicit reasoning helps**: prompting the model to "think step by step" (the THOUGHT step) consistently improves multi-step task performance. The model uses the THOUGHT to plan and self-correct.

**Tool use grounds the model**: rather than hallucinating an answer, the model can look it up. Observations bring external facts into context.

ReAct generalizes: you can use any LLM, any tool set, any task. The pattern is the substrate; specifics vary.

## A ReAct loop in <200 lines

The reference implementation:

```python
# react_agent.py
import json
import re
from openai import OpenAI

client = OpenAI()

# Define tools.
def search(query: str) -> str:
    """Pretend search; in production this hits a real search API."""
    return f"Top result for '{query}': [...]"

def calculator(expression: str) -> str:
    return str(eval(expression))  # don't use eval in production

TOOLS = {"search": search, "calculator": calculator}

SYSTEM = """You are a ReAct agent. Use THOUGHT, ACTION, OBSERVATION cycles.

Format:
THOUGHT: <your reasoning>
ACTION: <tool_name>(<args>)

When you have the answer:
FINAL ANSWER: <answer>

Available tools: search(query), calculator(expression).
"""

def run_agent(question: str, max_steps: int = 10):
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": question},
    ]
    
    for step in range(max_steps):
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            stop=["OBSERVATION:"],  # stop generation before we add the observation
        )
        text = response.choices[0].message.content
        messages.append({"role": "assistant", "content": text})
        
        if "FINAL ANSWER:" in text:
            return text.split("FINAL ANSWER:")[1].strip()
        
        # Parse the action.
        action_match = re.search(r"ACTION:\s*(\w+)\((.*?)\)", text, re.DOTALL)
        if not action_match:
            return f"(parse error in step {step}: {text})"
        
        tool_name, args_str = action_match.groups()
        tool = TOOLS.get(tool_name)
        if not tool:
            obs = f"Error: unknown tool {tool_name}"
        else:
            try:
                # Simple arg parsing; production needs better.
                obs = tool(args_str.strip().strip('"').strip("'"))
            except Exception as e:
                obs = f"Error: {e}"
        
        messages.append({"role": "user", "content": f"OBSERVATION: {obs}"})
    
    return "(max steps reached)"

# Try it.
print(run_agent("What is 17 * 24?"))
print(run_agent("What is the capital of France?"))
```

Under 100 lines of meaningful code. The pieces:
- A tool registry (`TOOLS`).
- A system prompt that explains the format.
- A loop that calls the model, parses the action, runs the tool, appends the observation.
- A stop condition (`FINAL ANSWER`).

That's the entire ReAct agent. The rest of this module is variations and improvements on this base.

## What goes wrong

Failure modes you'll see immediately:

**Parse errors**: the model emits an action with weird formatting. Robust parsing is harder than it looks.

**Infinite loops**: the model never emits FINAL ANSWER; the max_steps guard kicks in.

**Bad tool choice**: the model tries calculator on "what is the capital of France." Tools need clear schemas.

**Hallucinated tool**: the model invents a tool that doesn't exist. The agent needs to handle this gracefully.

**Stuck on observation**: the model emits THOUGHT but no ACTION. Or repeats the same action.

The rest of Part 1 (Lessons 2-3) and Part 2 (Lessons 4-6) address these.

## Native function calling

Modern LLMs (GPT-4, Claude, Gemini) support *native function calling*: instead of parsing free-text actions, the API returns a structured `tool_call` object. The integration is cleaner:

```python
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages,
    tools=[
        {"type": "function", "function": {"name": "search", "parameters": {...}}},
        {"type": "function", "function": {"name": "calculator", "parameters": {...}}},
    ],
)

if response.choices[0].message.tool_calls:
    for tc in response.choices[0].message.tool_calls:
        tool_name = tc.function.name
        args = json.loads(tc.function.arguments)
        # ...
```

Lesson 4 covers native function calling in depth. The point here: ReAct's text-based parsing is the foundational pattern; native function calling is the production-grade version.

## What you should believe after this lesson

Three sentences:

**1. An agent is a system with autonomy + tools + feedback loop** — the LLM decides what to do next, executes via tools, observes the result, and continues. This definition rules in "the system that decides to query a DB, summarize, and email" and rules out "a chatbot that just answers questions."

**2. ReAct is the foundational agent pattern**: interleaved THOUGHT / ACTION / OBSERVATION cycles. Implementable in under 200 lines without any framework. The rest of the agent universe is variations on this base.

**3. The standard failure modes are parse errors, infinite loops, bad tool choice, hallucinated tools, and stuck states.** Lessons 2-6 address each.

## Hands-on (at home)

Run the ReAct loop above (needs an API key). Try:

- "What is 17 * 24?" → calculator should be called.
- "What is the capital of France?" → search or direct answer.
- "What is the population of Tokyo divided by the population of Paris?" → multiple tool calls.

Watch the agent's reasoning step-by-step. Note where it goes wrong; tune the system prompt; observe the improvement.

For a more robust parser, replace the regex with the model's native function-calling API (covered in Lesson 4).

## Further reading

- "ReAct: Synergizing Reasoning and Acting in Language Models" (Yao et al, 2022).
- LangChain ReAct documentation (for the framework view; we're going framework-free for this module).
- "Building LLM-powered agents" (various 2024-2026 blog posts).

Next lesson: **The minimal loop — perceive → reason → act → observe.** We formalize the agent loop, distinguish ReAct from variants, and look at when the four-step cycle matches the problem and when it doesn't.
