---
title: "Lesson 8 — Memory Compaction and Eviction"
date: "2026-06-04"
module: "agents"
order: 8
tags: ["memory-compaction", "summarization", "eviction", "sliding-window", "context-budget"]
author: "Sudipta Pathak"
prerequisites: ["07-memory-hierarchy"]
---

# Lesson 8 — Memory Compaction and Eviction

## Why this lesson exists

Multi-turn agents accumulate history. At turn 50 of a chat, the message list is huge; pasting it all into context is expensive and degrades the model. Beyond the model's context limit, you must drop something.

This lesson covers the strategies for fitting unbounded history into bounded context: summarization, sliding window with sinks, hybrid strategies, and the tradeoffs each makes.

The lesson is reading. The Hands-on builds an agent with summary-based compaction.

## The problem stated

A turn's context = system prompt + (all past messages) + current input.

After 50 turns, "all past messages" might be 100K tokens. At $5/M tokens, each turn costs ~$0.50 just for the historical context. And the model's "lost in the middle" effect kicks in.

We need to compress history. Several strategies.

## Strategy 1: sliding window (forget oldest)

Keep only the last N turns; drop older ones.

```python
def truncate_history(messages, max_messages=20):
    if len(messages) <= max_messages:
        return messages
    return [messages[0]] + messages[-max_messages+1:]  # keep system + last N-1
```

Pros: simple; cheap; predictable size.
Cons: loses information from older turns. For tasks where early context matters (project introduction, user preferences stated upfront), this is bad.

Mitigation: pin the first N "important" messages (system prompt, user introduction). Slide over the rest. This is the **attention sinks** pattern (Module 7 Lesson 21) applied to message history.

## Strategy 2: summarization

When history exceeds threshold, summarize old turns into a single summary message; keep the summary + recent turns.

```python
def compact_history(messages, threshold=20, keep_recent=5):
    if len(messages) <= threshold:
        return messages
    
    to_summarize = messages[1:-keep_recent]  # skip system; skip recent
    summary = llm_summarize(to_summarize)
    
    return [
        messages[0],  # system
        {"role": "user", "content": f"[Conversation summary so far: {summary}]"},
        *messages[-keep_recent:],
    ]
```

Pros: preserves key info; bounded size.
Cons: summarization itself costs an LLM call; quality of summary varies; some information always lost.

The summarization can be triggered:
- Every N turns (regular cadence).
- When context size exceeds threshold (reactive).
- When the agent explicitly asks (on-demand).

## Strategy 3: tiered summarization

Multiple levels of summary at different granularities:
- Recent 5 turns: full text.
- Last 50 turns: medium summary.
- Earlier: terse summary.

This is the **MemGPT** pattern: an agent that manages its own context like an OS manages memory, with tiers of varying detail.

For very long-running agents (research projects, multi-day tasks), tiered summarization is the right pattern.

## Strategy 4: scratchpad-based (Lesson 7)

Maintain a separate scratchpad with key facts; rotate message history aggressively (small window), rely on the scratchpad for important info.

This is what Claude Code does: long sessions don't keep the full bash command history; they keep the current task's context and an evolving scratchpad of what's been tried.

## Strategy 5: retrieval-augmented history

Store full history in vector DB; at each turn, retrieve relevant past turns into context.

Pros: never lose info.
Cons: retrieval might miss relevant turns; adds per-turn latency.

The "context window as cache" mental model: relevant items get pulled in; less-relevant stay in DB.

## The hybrid pattern

A typical production agent uses several:
- **Sliding window** over message history (last 20 turns).
- **Scratchpad** for current-task facts.
- **Vector memory** for persistent knowledge.
- **Summary** of pre-window history.

When the agent needs old info, the scratchpad or vector memory has it. The recent history provides flow.

## What triggers compaction

Compaction strategies:

**Token-based**: when context exceeds a threshold (e.g., 50K tokens), compact.

**Turn-based**: every N turns.

**Time-based**: every N minutes (for long-running agents).

**On-demand**: the agent explicitly calls "compact now" when it senses the context is getting cluttered.

The model-driven (on-demand) version requires the agent to know when to compact. Often a simple rule (every N turns) is cleaner.

## What to keep, what to drop

A heuristic for what to keep in summary:
- **Decisions made** (the agent decided to do X for reason Y).
- **Facts learned** (data discovered from tools).
- **Open questions** (things the agent still needs to address).
- **User-stated preferences** (always preserve).

What to drop:
- **Intermediate tool outputs** (already used).
- **Re-acknowledgments** ("OK, I'll do that").
- **Failed paths** (unless instructive).

## The Claude / GPT context-management features

Both Anthropic and OpenAI offer some context management:
- **Anthropic's prompt caching**: cached prefixes don't count against latency / cost. Stable parts (system prompt, large knowledge bases) can be cached.
- **OpenAI's predicted outputs / prompt caching**: similar.

These don't solve the "context is full" problem but reduce the cost of large repeated prefixes.

Lesson 15 covers cache-friendly prompts in depth.

## Anti-patterns

**No compaction at all**: agent runs OK until ~turn 30; then context blows up; costs and latency degrade.

**Aggressive compaction too early**: agent loses context from turn 2 because compaction triggered at turn 5. Tune the threshold.

**Summarization without preserving key facts**: the summary forgets the user's name / preferences. Always preserve high-information items.

**Dropping the system prompt**: rotating window evicts the system prompt; the agent loses its instructions. Always pin the system prompt.

## What you should believe after this lesson

Three sentences:

**1. Long agent histories need compaction**: sliding window, summarization, tiered summarization, scratchpad, retrieval-augmented. The strategies have different tradeoffs; production typically combines several.

**2. The sliding-window-with-sinks pattern (pin system prompt + last N turns)** is the simplest reliable baseline. Add summarization when N alone loses too much; add scratchpad for important per-task facts; add vector memory for cross-session persistence.

**3. Compaction triggers vary** — token-based, turn-based, time-based, on-demand. Token-based is most principled (the actual constraint is the context window); turn-based is simpler.

## Hands-on (at home)

Build an agent with summary-based compaction.

```python
# compacting_agent.py
from openai import OpenAI
client = OpenAI()

class CompactingAgent:
    def __init__(self, system_prompt, threshold=10):
        self.system = system_prompt
        self.messages = []
        self.summary = ""
        self.threshold = threshold
    
    def chat(self, user_msg):
        self.messages.append({"role": "user", "content": user_msg})
        
        # Compact if needed.
        if len(self.messages) > self.threshold:
            self._compact()
        
        # Build context.
        context = [{"role": "system", "content": self.system}]
        if self.summary:
            context.append({"role": "user", "content": f"[Previous summary: {self.summary}]"})
        context.extend(self.messages)
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=context,
        )
        reply = response.choices[0].message.content
        self.messages.append({"role": "assistant", "content": reply})
        return reply
    
    def _compact(self):
        # Summarize all but the last 5.
        to_summarize = self.messages[:-5]
        summary_text = "\n".join(f"{m['role']}: {m['content']}" for m in to_summarize)
        prompt = f"Summarize this conversation, preserving facts, decisions, and user preferences:\n{summary_text}"
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
        )
        self.summary = (self.summary + "\n" + response.choices[0].message.content)[:2000]
        self.messages = self.messages[-5:]
        print(f"[Compacted: kept summary of length {len(self.summary)}; recent: 5 turns]")

agent = CompactingAgent("You are a helpful assistant who remembers everything.")
for i in range(20):
    reply = agent.chat(f"My favorite number changes; this turn it's {i*7}.")
    print(f"Turn {i}: {reply[:60]}")

# At turn 20, ask about turn 3.
print(agent.chat("What was my favorite number on turn 3?"))
# It should remember (from the summary), even though that turn's exact message has been compacted.
```

Watch the agent compact over time; the summary preserves the right info if it's well-designed.

## Further reading

- "MemGPT" paper.
- "Lost in the Middle" (Liu et al, 2023) — for the context-degradation evidence.
- Various blog posts on context management for production agents.

End of Part 3. Next: Part 4 begins with **RAG: the core pipeline from scratch**. Retrieval-Augmented Generation gets four lessons because it's the most common production agent pattern.
