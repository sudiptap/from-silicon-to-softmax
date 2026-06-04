# Agentic AI — Complete Curriculum

---

## Phase 1: Foundations (Weeks 1–2)

### 1.1 LLM Basics
- How transformers work — attention, tokenization, context windows
- Inference vs training — what happens when you call an API
- Temperature, top-p, top-k — sampling strategies and when they matter
- Token economics — pricing, context limits, input vs output tokens
- Models landscape — GPT-4, Claude, Gemini, Llama, Mistral, Command R

### 1.2 Prompt Engineering
- Zero-shot, few-shot, chain-of-thought prompting
- Structured outputs — JSON mode, function calling, schema enforcement
- System prompts — persona, constraints, formatting rules
- Prompt optimization — iterating on prompts systematically
- Common failure modes — hallucination, refusal, instruction drift

### 1.3 API Fundamentals
- OpenAI API, Anthropic API, Google Vertex AI — request/response patterns
- Streaming responses — SSE, chunked transfer
- Rate limiting, retries, exponential backoff
- Async API calls — batching, concurrent requests
- Cost tracking and token counting

**Build:** A CLI chatbot that maintains conversation history and supports multiple providers.

---

## Phase 2: Tool Use & Function Calling (Weeks 3–4)

### 2.1 Function Calling
- How function/tool calling works — schema definition, model decides when to call
- OpenAI function calling, Anthropic tool use, Gemini function declarations
- Parallel tool calls — handling multiple tool invocations in one turn
- Forced vs auto tool choice
- Error handling — what happens when a tool fails mid-conversation

### 2.2 Building Tools
- Web search tools — Tavily, SerpAPI, Brave Search
- Code execution tools — sandboxed Python, Docker-based execution
- File system tools — read, write, search
- Database tools — SQL query generation and execution
- API integration tools — wrapping external APIs as agent tools

### 2.3 MCP (Model Context Protocol)
- What MCP is — standardized protocol for tool/resource integration
- MCP servers and clients — architecture
- Building an MCP server — exposing tools, resources, prompts
- Connecting agents to MCP servers
- MCP vs native function calling — when to use which

**Build:** An agent that can search the web, read files, execute code, and query a database using tool calling.

---

## Phase 3: Agent Architectures (Weeks 5–7)

### 3.1 Single Agent Patterns
- ReAct (Reason + Act) — think, act, observe loop
- Plan-and-Execute — generate plan first, then execute steps
- Reflection — agent critiques its own output and improves
- Tool-augmented generation — agent decides when it needs a tool vs when to answer directly
- Iterative refinement — agent loops until quality threshold is met

### 3.2 Multi-Agent Systems
- Supervisor pattern — one agent delegates to specialist agents
- Debate pattern — multiple agents argue, one judges
- Pipeline pattern — agents in sequence, each handling a stage
- Swarm pattern — peer agents with handoff capabilities
- Hierarchical agents — managers, workers, reviewers

### 3.3 Agent Frameworks
- LangChain / LangGraph — chains, graphs, state machines for agents
- LlamaIndex — data agents, query engines, tool abstractions
- AutoGen — multi-agent conversations
- CrewAI — role-based multi-agent orchestration
- OpenAI Agents SDK — built-in agent primitives
- Anthropic Claude Agent SDK
- Build your own — when and why to skip frameworks

### 3.4 State Management
- Conversation memory — buffer, summary, vector-backed memory
- Agent state machines — tracking where an agent is in a workflow
- Checkpointing — saving and resuming agent runs
- Shared state between agents — blackboard pattern, message passing

**Build:** A research agent that takes a topic, searches multiple sources, synthesizes findings, and produces a report with citations. Use both single-agent and multi-agent approaches, compare results.

---

## Phase 4: RAG (Retrieval-Augmented Generation) (Weeks 8–10)

### 4.1 Core RAG Pipeline
- Document loading — PDFs, HTML, markdown, code, databases
- Chunking strategies — fixed size, semantic, recursive, document-aware
- Embedding models — OpenAI, Cohere, BGE, sentence-transformers
- Vector databases — Pinecone, Weaviate, Milvus, Chroma, pgvector
- Retrieval — similarity search, MMR, hybrid search (vector + keyword)
- Generation — context stuffing, map-reduce, refine chains

### 4.2 Advanced RAG
- Query transformation — HyDE, multi-query, step-back prompting
- Re-ranking — Cohere Rerank, cross-encoders, ColBERT
- Contextual retrieval — adding context to chunks before embedding
- Parent-child retrieval — retrieve small chunks, return larger context
- Agentic RAG — agent decides what to retrieve, when, and how
- Graph RAG — knowledge graphs + vector search combined
- Self-RAG — model decides when it needs retrieval

### 4.3 RAG Evaluation
- Metrics — faithfulness, relevance, context precision, context recall
- RAGAs framework — automated RAG evaluation
- Building eval datasets — synthetic and human-curated
- Debugging RAG — tracing retrieval failures, hallucination sources

**Build:** A RAG system over a large codebase or documentation set. Include hybrid search, re-ranking, and evaluation pipeline.

---

## Phase 5: Production Systems (Weeks 11–13)

### 5.1 Reliability & Error Handling
- Retry strategies — transient failures, rate limits, model errors
- Fallback chains — primary model fails, fall back to secondary
- Timeout management — tool calls, LLM calls, end-to-end
- Graceful degradation — what to do when parts of the system fail
- Idempotency — ensuring agent actions are safe to retry

### 5.2 Observability & Monitoring
- Tracing — LangSmith, Arize Phoenix, W&B Weave, OpenTelemetry
- Logging — structured logs for agent decisions, tool calls, outputs
- Dashboards — latency, token usage, error rates, success rates
- Alerting — SLOs for agent performance, drift detection
- Cost monitoring — per-request and per-agent cost tracking

### 5.3 Evaluation in Production
- Online evaluation — judge models scoring live outputs
- A/B testing agents — comparing agent versions on live traffic
- Human-in-the-loop — flagging low-confidence outputs for review
- Regression testing — golden datasets, snapshot testing for agents
- Continuous eval — automated pipelines that catch quality drops

### 5.4 Performance Optimization
- Caching — semantic caching, exact match caching, prompt caching
- Model routing — cheap model for easy tasks, expensive model for hard ones
- Parallel execution — running independent tool calls concurrently
- Streaming — delivering partial results as they're generated
- Batching — grouping requests for throughput
- Latency profiling — identifying bottlenecks across the agent pipeline

**Build:** Take your research agent from Phase 3, add tracing, monitoring, caching, fallbacks, and evaluation. Deploy it as an API.

---

## Phase 6: Infrastructure & Deployment (Weeks 14–15)

### 6.1 Serving & Scaling
- Containerization — Docker for agent services
- Orchestration — Kubernetes, scaling agent workers
- Queue-based architectures — Kafka, RabbitMQ, Redis Streams for async agent tasks
- Distributed execution — Ray for parallel agent workloads
- Serverless — Lambda/Cloud Functions for lightweight agent endpoints

### 6.2 LLM Serving (Self-Hosted)
- vLLM — high-throughput LLM serving
- SGLang — structured generation, constrained decoding
- TensorRT-LLM — NVIDIA-optimized serving
- Triton Inference Server — multi-model serving
- Quantization — GPTQ, AWQ, GGUF — trading quality for speed/cost

### 6.3 Security & Safety
- Prompt injection — detection and prevention
- Data leakage — PII filtering, output sanitization
- Guardrails — NeMo Guardrails, Guardrails AI, custom validators
- Rate limiting and abuse prevention
- Audit logging — tracking all agent actions for compliance

**Build:** Deploy a multi-agent system with a message queue, auto-scaling workers, and a monitoring dashboard.

---

## Phase 7: Advanced Topics (Weeks 16–18)

### 7.1 Agent Communication Protocols
- A2A (Agent-to-Agent) — Google's agent interop protocol
- MCP advanced — dynamic tool discovery, resource subscriptions
- Agent registries — discovering and connecting to agents at runtime
- Standardization landscape — where the industry is heading

### 7.2 Planning & Reasoning
- Tree-of-thought — branching reasoning paths
- Graph-of-thought — non-linear reasoning
- Monte Carlo Tree Search for agents — exploring action spaces
- Self-play and self-improvement loops
- Verification agents — checking other agents' work

### 7.3 Long-Running Agents
- Durable execution — surviving crashes, resuming from checkpoints
- Human-in-the-loop workflows — approval gates, escalation
- Scheduled agents — cron-like recurring agent tasks
- Event-driven agents — triggered by external events (webhooks, DB changes)
- Agent memory — long-term memory across sessions, memory consolidation

### 7.4 Fine-Tuning for Agents
- When to fine-tune vs prompt engineer
- Fine-tuning for tool calling — teaching models your custom tools
- RLHF/DPO for agent behavior — aligning agent actions with preferences
- Distillation — training smaller models to mimic larger agent behaviors
- LoRA/QLoRA — efficient fine-tuning on consumer hardware

**Build:** A long-running agent that monitors a data source, makes decisions over time, involves human approval for high-stakes actions, and improves from feedback.

---

## Recommended Resources

### Documentation
- LangChain docs — python.langchain.com
- LlamaIndex docs — docs.llamaindex.ai
- Anthropic docs — docs.anthropic.com
- OpenAI docs — platform.openai.com/docs
- MCP spec — modelcontextprotocol.io

### Papers
- "ReAct: Synergizing Reasoning and Acting in Language Models" (Yao et al.)
- "Toolformer: Language Models Can Teach Themselves to Use Tools" (Schick et al.)
- "Generative Agents: Interactive Simulacra of Human Behavior" (Park et al.)
- "Self-RAG: Learning to Retrieve, Generate, and Critique" (Asai et al.)
- "LATS: Language Agent Tree Search" (Zhou et al.)

### Courses & Tutorials
- DeepLearning.AI — "AI Agents in LangGraph", "Building Agentic RAG"
- Anthropic cookbook — github.com/anthropics/anthropic-cookbook
- LangChain Academy — academy.langchain.com

---

## Roles This Prepares You For
- NVIDIA — Lead Sr. SWE, Agentic AI Applications
- Google DeepMind — SWE, GeminiApp
- Any Sr. SWE / Staff SWE role in AI/ML platform, LLM infrastructure, or AI applications
