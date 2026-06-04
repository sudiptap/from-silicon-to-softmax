---
title: "Lesson 2 — The Minimal Loop: Perceive → Reason → Act → Observe"
date: "2026-06-04"
module: "agents"
order: 2
tags: ["agent-loop", "perceive", "reason", "act", "observe"]
author: "Sudipta Pathak"
prerequisites: ["01-react-from-scratch"]
---

# Lesson 2 — The Minimal Loop: Perceive → Reason → Act → Observe

## Why this lesson exists

ReAct (Lesson 1) is one instance of a more general pattern: the *agent loop*. Every agent — from a simple chatbot with a calculator tool to a fully autonomous research assistant — implements some version of perceive → reason → act → observe.

This lesson formalizes the loop, distinguishes the variants (ReAct, plan-then-act, observe-orient-decide-act / OODA), and gives you the vocabulary to describe any agent's control flow.

The lesson is reading. The Hands-on extends the Lesson 1 ReAct with explicit phase tracking.

## The canonical four phases

1. **Perceive**: gather inputs. For a chat agent, the user's message; for a browser agent, the screen state; for a data agent, the latest database row.

2. **Reason**: the LLM thinks. What does the input mean? What should I do?

3. **Act**: execute a tool / API call / action in the world.

4. **Observe**: see the result. Did it work? What new information do I have?

Repeat.

ReAct (Lesson 1) collapses these somewhat: the THOUGHT step is "reason"; ACTION is "act"; OBSERVATION is "observe." Perception is implicit (the input is already in context).

## Variations

The loop has many variations:

**Plan-then-act**: separate the plan from the execution. First call the LLM with "plan the steps;" then execute each step. Skipping the per-step reasoning makes execution faster but less adaptive.

**OODA (Observe-Orient-Decide-Act)**: from military strategy, applied to agents. Emphasizes the "orient" step — building/updating a mental model of the environment.

**Reactive vs deliberative**: reactive agents respond directly to inputs (perceive → act); deliberative agents reason explicitly (perceive → reason → act). Most LLM agents are deliberative.

**Hierarchical**: a high-level agent plans; low-level agents execute. The high-level loop is slow + reasoning; the low-level loops are fast + acting.

The choice of variant depends on:
- How predictable the environment is (more predictable → can plan ahead).
- How fast the loop needs to be (faster → less per-step reasoning).
- How autonomous the agent is (more autonomous → more reasoning needed).

## State

An agent has *state* — what it knows / has done so far.

State includes:
- The conversation / message history.
- Any working scratchpad (the agent's notes).
- Tool call history.
- Cached observations.

For a simple ReAct loop, state is just the message list. For more complex agents, state is a structured object that evolves over the loop.

In Python:

```python
class AgentState:
    messages: list[dict]  # the LLM context
    tool_history: list[dict]  # past tool calls
    scratchpad: str  # the agent's notes
    iteration: int  # how many loop steps so far
```

The loop's implementation reads state, makes the LLM call, parses the action, executes, updates state. Each iteration mutates state.

## Termination conditions

When does the loop end? Common conditions:

- **The agent says it's done** (FINAL ANSWER, or a structured "done" tool call).
- **Max iterations reached** (safety guard).
- **Max time elapsed** (latency budget).
- **External cancellation** (user clicked "stop").
- **An error occurred** that the agent can't recover from.

Termination logic lives in the loop driver, not the agent.

## A more structured loop

The Lesson 1 ReAct was tightly coupled to the LLM's text output. A more structured loop separates the phases:

```python
class Agent:
    def __init__(self, tools, llm_client):
        self.tools = tools
        self.llm = llm_client
    
    def perceive(self, input):
        """Update state with the new input."""
        self.state.messages.append({"role": "user", "content": input})
    
    def reason(self):
        """Call the LLM to get the next action."""
        response = self.llm.complete(self.state.messages, tools=self.tools.schemas())
        return response.tool_calls or response.content
    
    def act(self, action):
        """Execute the action via the tool registry."""
        if action.tool == "done":
            return None  # signal completion
        result = self.tools[action.tool](action.args)
        return result
    
    def observe(self, action, result):
        """Append the observation to state."""
        self.state.messages.append({
            "role": "tool",
            "tool_call_id": action.id,
            "content": result,
        })
    
    def step(self):
        """One iteration of the loop."""
        action = self.reason()
        if action is None or action.tool == "done":
            return False  # terminate
        result = self.act(action)
        self.observe(action, result)
        return True
    
    def run(self, input, max_steps=10):
        self.perceive(input)
        for _ in range(max_steps):
            if not self.step():
                break
        return self.state.messages[-1]["content"]
```

This is the same logic as Lesson 1's ReAct but with the phases explicit. Easier to extend (add a planner, swap reasoning strategy) and easier to test (mock the LLM; verify state evolution).

## The "thinking" inside reason

The "reason" step is where most agent design effort goes:

- **Single forward pass**: one LLM call → action. Cheapest.
- **Chain-of-thought**: explicit thinking before the action. Slightly more cost; better quality.
- **Plan-and-execute**: a plan upfront; execute steps. Cheaper per step than per-step reasoning.
- **Tree search**: explore multiple action options before picking one. Most expensive; sometimes worth it (Lesson 13-14).

The choice depends on the task's complexity. For simple ReAct, single forward pass is fine. For hard reasoning, tree search.

## What you should believe after this lesson

Three sentences:

**1. The canonical agent loop is perceive → reason → act → observe.** ReAct (Lesson 1) implements this with one forward pass per cycle; variations include plan-then-act, OODA, hierarchical, reactive vs deliberative. Pick the variant based on environment predictability and required autonomy.

**2. State is the agent's representation of "what I know / have done"** — typically the message list plus a scratchpad. The loop reads, calls the LLM, parses the action, executes, updates state. Each iteration mutates state.

**3. Termination conditions are**: agent says done, max iterations, max time, external cancellation, unrecoverable error. These live in the loop driver, not the agent's reasoning.

## Hands-on (at home)

Extend the Lesson 1 ReAct with explicit phase tracking.

```python
# minimal_loop.py — sketch.
class Agent:
    def __init__(self):
        self.state = {"messages": [], "tool_calls": []}
    
    def perceive(self, input_text):
        self.state["messages"].append({"role": "user", "content": input_text})
    
    def reason(self):
        # Call LLM; return text + parsed action.
        ...
    
    def act(self, action):
        # Execute tool.
        ...
    
    def observe(self, result):
        self.state["messages"].append({"role": "tool", "content": result})
    
    def run(self, input, max_steps=10):
        self.perceive(input)
        for i in range(max_steps):
            print(f"=== Step {i} ===")
            print(f"PERCEIVE: {self.state['messages'][-1]}")
            action = self.reason()
            print(f"REASON → {action}")
            if action.is_terminal: break
            result = self.act(action)
            print(f"ACT → {result}")
            self.observe(result)
            print(f"OBSERVE: appended")
        return self.state["messages"][-1]
```

Print-level instrumentation. For each step, you can see the four phases. This is the foundation for observability (Lesson 23).

## Further reading

- "ReAct" paper (Lesson 1's reference).
- "OODA Loop" — John Boyd's military strategy origin.
- Russell & Norvig "AI: A Modern Approach" — Chapter 2 on agents.

Next lesson: **LLM-as-planner vs LLM-as-worker; state machines vs flexible loops.** When to use the LLM for planning vs execution, and when to constrain the loop to a state machine vs let it be flexible.
