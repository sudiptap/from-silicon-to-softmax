---
title: "Lesson 20 — MCP from Scratch"
date: "2026-06-04"
module: "agents"
order: 20
tags: ["mcp", "model-context-protocol", "interop", "tool-server"]
author: "Sudipta Pathak"
prerequisites: ["19-minimal-agent-100-lines"]
---

# Lesson 20 — MCP from Scratch

## Why this lesson exists

MCP (Model Context Protocol; Anthropic, 2024) is an open standard for connecting agents to tools. Before MCP, every agent vendor had its own way to declare tools (OpenAI's function-calling format, Anthropic's tool blocks, custom in-app shapes). After MCP, a tool can be written once and used by any compliant agent.

In 2026, MCP is the de-facto standard. Anthropic Claude, OpenAI, Google, and most agent frameworks support it. Tool servers (search, filesystem, database, browser) are increasingly published as MCP servers usable by any agent.

This lesson covers the MCP protocol, how to build a custom MCP server, and how to consume one from an agent.

The lesson is reading. The Hands-on builds a tiny MCP server and client.

## What MCP is

MCP defines:
- **A protocol** for client (agent) ↔ server (tool provider) communication.
- **A standard for declaring tools**: name, description, JSON schema.
- **Methods**: `list_tools()`, `call_tool(name, args)`, `list_resources()`, `read_resource(uri)`.
- **Transport**: typically WebSocket or stdio; JSON-RPC over the wire.

The agent (the "client") connects to an MCP server, lists available tools / resources, and calls them as needed.

## The vocabulary

- **Resource**: a piece of data the server exposes (a file, a database row, a webpage). Read-only access.
- **Tool**: a function the server lets the agent call. Side-effecting.
- **Prompt**: a structured prompt template the server provides (less commonly used).

A typical MCP server exposes both resources (data) and tools (actions).

## The wire format

MCP uses JSON-RPC 2.0. A client → server "list tools" call:

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "tools/list",
  "params": {}
}
```

Server responds:

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "tools": [
      {
        "name": "read_file",
        "description": "Read the contents of a file.",
        "inputSchema": {
          "type": "object",
          "properties": {"path": {"type": "string"}},
          "required": ["path"]
        }
      }
    ]
  }
}
```

Then a tool call:

```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "method": "tools/call",
  "params": {"name": "read_file", "arguments": {"path": "/tmp/test.txt"}}
}
```

And the response with the result.

## Why MCP matters

The pre-MCP world: every agent had to hand-code its tools. To add Slack integration to your agent, you wrote a Slack tool. To add Slack to my agent, I wrote my own.

Post-MCP: Slack publishes an MCP server. Any MCP-compliant agent uses it. Tools become reusable; the ecosystem benefits.

In 2026:
- Cloud providers offer MCP servers for their services (AWS, GCP, Azure).
- Productivity apps offer MCP (Notion, Linear, GitHub).
- Open-source tool collections exist (filesystem, browser automation, code execution).

For a new agent: don't write tools from scratch; use existing MCP servers when available.

## Building a custom MCP server

The official Python SDK is `mcp`:

```python
# mcp_server_demo.py
# pip install mcp
import asyncio
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("MyTools")

@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b

@mcp.tool()
def search_my_db(query: str) -> str:
    """Search the internal database."""
    return f"Internal results for '{query}': [...]"

@mcp.resource("config://settings")
def get_settings() -> str:
    """Read application settings."""
    return '{"theme": "dark", "version": "1.0"}'

if __name__ == "__main__":
    mcp.run()
```

`FastMCP` is a decorator-based DSL. Decorating a function with `@mcp.tool()` registers it; the schema is inferred from type hints + docstring.

Run with `python mcp_server_demo.py`. The server listens for client connections (via stdio by default, or WebSocket).

## Consuming MCP from an agent

The agent uses the MCP client library to connect to a server and call its tools:

```python
# mcp_client_demo.py
import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def main():
    server_params = StdioServerParameters(
        command="python",
        args=["mcp_server_demo.py"],
    )
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            
            # List tools.
            tools_response = await session.list_tools()
            print("Available tools:")
            for tool in tools_response.tools:
                print(f"  {tool.name}: {tool.description}")
            
            # Call a tool.
            result = await session.call_tool("add", {"a": 5, "b": 3})
            print(f"\nResult of add(5, 3): {result.content[0].text}")

asyncio.run(main())
```

The agent can now call `add` (or `search_my_db`) via the standardized protocol. No custom tool integration code; the MCP client handles the wire format.

## Integrating MCP with an agent loop

To use MCP tools in your agent (Lesson 19), convert MCP tool schemas to your agent framework's format:

```python
# Pseudo-code.
async def get_mcp_tools(session) -> list[Tool]:
    tools_response = await session.list_tools()
    return [
        Tool(
            name=t.name,
            description=t.description,
            params_schema=t.inputSchema,
            fn=lambda **kwargs: asyncio.run(session.call_tool(t.name, kwargs))
        )
        for t in tools_response.tools
    ]

# In the agent setup.
mcp_tools = await get_mcp_tools(session)
agent = Agent(system_prompt="...", tools=mcp_tools + my_local_tools)
```

The agent treats MCP tools like any other tools; the user doesn't see the distinction.

## MCP servers in the wild

In 2026, popular MCP servers:
- **Filesystem**: read / write local files.
- **Git**: query / modify git repos.
- **PostgreSQL / SQLite**: query databases.
- **Browser**: navigate / scrape web pages.
- **GitHub / GitLab**: code search, issues, PRs.
- **Linear, Notion**: productivity apps.
- **Custom cloud services**: AWS, GCP, Azure.

For a new agent, the workflow:
1. Identify which tools you need.
2. Find or build MCP servers for them.
3. Configure your agent to connect to these servers.
4. The tools are available; no custom integration code.

This is a big productivity win vs the pre-MCP world.

## What you should believe after this lesson

Three sentences:

**1. MCP (Model Context Protocol) is the 2024+ standard for agent-tool communication.** Defines a JSON-RPC protocol for listing and calling tools; allows tools to be written once and used by any compliant agent.

**2. MCP servers expose tools, resources, and prompts** via standardized methods. The Python SDK (`mcp`) makes server creation trivial (decorator-based DSL); the client library makes consumption equally easy.

**3. The MCP ecosystem in 2026** has servers for filesystem, git, databases, browsers, GitHub, productivity apps, and cloud services. For new agents, leverage existing MCP servers; write custom ones only for proprietary tools.

## Hands-on (at home)

Build a tiny MCP server and consume it.

```python
# my_mcp_server.py
from mcp.server.fastmcp import FastMCP
import datetime

mcp = FastMCP("DemoServer")

@mcp.tool()
def get_time() -> str:
    """Get the current time."""
    return datetime.datetime.now().isoformat()

@mcp.tool()
def echo(message: str) -> str:
    """Echo a message back."""
    return f"You said: {message}"

if __name__ == "__main__":
    mcp.run()
```

```python
# client.py
import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def main():
    server_params = StdioServerParameters(command="python", args=["my_mcp_server.py"])
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            
            # List.
            tools = (await session.list_tools()).tools
            for t in tools: print(f"  {t.name}")
            
            # Call.
            r = await session.call_tool("get_time", {})
            print(f"get_time → {r.content[0].text}")
            
            r = await session.call_tool("echo", {"message": "Hello!"})
            print(f"echo → {r.content[0].text}")

asyncio.run(main())
```

Run server in one terminal; client in another. You'll see the tool list + the tool calls work.

## Further reading

- Model Context Protocol specification.
- Anthropic MCP documentation and examples.
- Awesome MCP servers list (community-maintained).
- "Building Agent-to-Tool Interfaces" — emerging blog topic.

End of Part 8. Next: Part 9 begins with **Coding agents** — Claude Code, Cursor, Aider patterns. The vertical agents for code editing.
