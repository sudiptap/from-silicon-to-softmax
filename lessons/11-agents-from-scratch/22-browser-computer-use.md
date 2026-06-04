---
title: "Lesson 22 — Browser and Computer-Use Agents"
date: "2026-06-04"
module: "agents"
order: 22
tags: ["browser-agent", "computer-use", "playwright", "osworld", "screenshot"]
author: "Sudipta Pathak"
prerequisites: ["21-coding-agents"]
---

# Lesson 22 — Browser and Computer-Use Agents

## Why this lesson exists

Browser and computer-use agents interact with software designed for humans: GUIs, web pages, desktop apps. The interface is visual (screenshots, DOM trees); the actions are clicks, keystrokes, scrolls.

This category includes:
- **Web browsing agents**: navigate web pages; fill forms; book flights; research.
- **Computer-use agents** (Claude's Computer Use, OpenAI's similar): screenshot + clicks + keyboard on a virtual desktop.
- **Automation agents**: scrape data; perform repetitive workflows.

The patterns are different from text-only or coding agents. The screen is high-dimensional; the action space is fine-grained; the feedback is visual.

This lesson covers the architecture and the open challenges.

The lesson is reading. The Hands-on builds a Playwright-based browser agent.

## The basic loop

For a web browsing agent:

```
While not done:
    1. Screenshot or DOM snapshot the page.
    2. LLM reasons about what to do next.
    3. Execute an action (click, type, navigate).
    4. Wait for the page to update.
    5. Repeat.
```

For computer-use:

```
While not done:
    1. Screenshot the screen.
    2. LLM reasons.
    3. Execute action (click at (x,y), type "...", scroll, etc.).
    4. Repeat.
```

The agent's "perception" is the screen; the "action" is OS-level input.

## Browser agents specifically

For web browsing, two main observation modalities:

**Screenshot**: the agent sees the rendered page as an image. The LLM is multimodal; it interprets the image.

**DOM tree**: the agent sees the page's HTML/DOM. Tools navigate by CSS selector / XPath.

Each has tradeoffs:

| Aspect | Screenshot | DOM |
| ------ | ---------- | --- |
| Robustness to layout changes | Worse (visual) | Better (semantic) |
| Handling of dynamic content | OK | Better (events, JS) |
| Cost per step (token-wise) | Higher (image tokens) | Lower (text) |
| Accessibility (ARIA labels) | Limited | Full |
| Captcha / dynamic challenges | Worse | Same |

Production browser agents often combine both: DOM for structured navigation; screenshot when the DOM doesn't reveal everything (e.g., visual layouts).

## Playwright

Playwright (Microsoft, open-source) is the standard browser-automation library:

```python
from playwright.async_api import async_playwright

async with async_playwright() as p:
    browser = await p.chromium.launch()
    page = await browser.new_page()
    await page.goto("https://example.com")
    await page.fill("input[name='q']", "agents")
    await page.click("button[type='submit']")
    text = await page.inner_text("body")
```

Playwright supports screenshot, DOM access, click, type, navigate, wait_for_selector, etc. It's the natural backend for a browser agent.

The agent's "tool" is essentially Playwright actions wrapped in a friendly interface.

## Computer-use agents

Computer-use is more general: the agent can interact with any application, not just browsers.

Anthropic's Computer Use API (2024): the model is given a screenshot; it responds with actions like:

```
{"action": "click", "coordinate": [500, 350]}
{"action": "type", "text": "hello"}
{"action": "key", "key": "Return"}
{"action": "screenshot"}
```

The agent runtime executes these on a virtual desktop (typically Docker container with Xvfb).

This is more general but also slower / less reliable than browser-specific agents. The screenshot model has to figure out what to click; precision matters.

OpenAI and others have similar offerings; the category is active in 2026.

## Action space challenges

The action space for browser/computer agents is enormous:
- Any pixel on the screen is a potential click target.
- Any text can be typed.
- Many timing-related actions (wait, scroll, key combos).

Even with a multimodal model, picking the right action precisely is hard:
- Off-by-a-few-pixels click misses the button.
- Wrong key combo doesn't have the expected effect.
- Dynamic content arrives mid-action.

Mitigations:
- **Bounded action space**: only standard widgets (button, input, link). Snap to nearby DOM elements.
- **Verification per action**: after each action, screenshot and verify state changed as expected.
- **Retries with adjustment**: if click missed, try slightly different coordinates.

## OSWorld and WebArena

The benchmarks for these agents:
- **OSWorld**: realistic desktop tasks (use Office apps, browse, manage files).
- **WebArena**: realistic web tasks (e-commerce, social media, GitHub).
- **VisualWebArena**: WebArena with required visual reasoning.
- **Mind2Web**: large-scale web-task dataset.

In 2026, top systems achieve 30-50% success on OSWorld; 50-70% on WebArena. Still far below human (90+%); the category is improving.

## Application: browser agent for booking

A complete browser agent for booking a flight:

```python
# pseudocode
async def book_flight(origin, destination, date):
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        await page.goto("https://flightsearch.com")
        
        # Agent loop with Playwright as the tool layer.
        agent = Agent(
            system_prompt="You are a flight-booking agent. Use the browser to find and book a flight.",
            tools=[
                Tool("screenshot", "Get a screenshot of the page.", {}, lambda: page.screenshot()),
                Tool("click", "Click an element.", {"selector": {"type": "string"}}, lambda selector: page.click(selector)),
                Tool("fill", "Fill an input.", {...}, lambda selector, value: page.fill(selector, value)),
                Tool("navigate", "Go to URL.", {...}, lambda url: page.goto(url)),
                Tool("get_dom", "Get the page DOM.", {}, lambda: page.content()),
            ]
        )
        
        task = f"Book a flight from {origin} to {destination} on {date}."
        return await agent.run(task)
```

The agent uses the browser tools to navigate, fill the form, select results, complete booking. The system prompt steers it.

For real production (a commercial flight-booking agent), much more engineering: edge cases (no flights, captchas, errors), error recovery, structured output of the booked itinerary.

## What you should believe after this lesson

Three sentences:

**1. Browser and computer-use agents interact with human-designed software** via screenshots and DOM (browser) or screenshots and OS actions (computer use). The perception is visual / structural; the actions are clicks, types, scrolls.

**2. Playwright is the dominant browser-automation backend**; the agent's tools wrap Playwright actions. Screenshot + DOM combined gives the best robustness.

**3. The category is active research in 2026** — top systems achieve 30-70% on benchmarks (vs 90+% human). Action-space precision and error recovery are the main challenges.

## Hands-on (at home)

Build a Playwright-based browser agent.

```python
# browser_agent.py
# pip install playwright openai
# playwright install chromium

import asyncio
from playwright.async_api import async_playwright
from openai import OpenAI

client = OpenAI()

async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)  # headless=False to watch
        page = await browser.new_page()
        await page.goto("https://example.com")
        
        # A simple agent that asks the LLM what to do.
        # For demo, just one step.
        screenshot = await page.screenshot()
        
        # The LLM would normally examine the screenshot; for this demo we just print.
        print(f"Page title: {await page.title()}")
        print(f"Page URL: {page.url}")
        
        # Execute an action (e.g., click a link).
        await page.click("a")
        await asyncio.sleep(2)  # wait for navigation
        print(f"After click, URL: {page.url}")
        
        await browser.close()

asyncio.run(main())
```

For a full agent, wrap the Playwright calls in tools (similar to Lesson 4); pass screenshots to a multimodal LLM (GPT-4o, Claude 3.5+); let the LLM decide actions.

For computer-use, use Anthropic's Computer Use API or OpenAI's Computer Use; both provide a managed runtime.

## Further reading

- Playwright documentation.
- "OSWorld" and "WebArena" papers / leaderboards.
- Anthropic's Computer Use documentation.
- OpenAI Operator / Computer Use docs.

End of Part 9. Next: Part 10 begins with **Tracing, evaluation, and observability** — the production-side concerns for shipping agents.
