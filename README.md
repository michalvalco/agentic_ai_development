# Agentic AI Development

**Production-ready implementations of five core agentic capabilities. No hype. Just working patterns.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

---

## What This Is (And Isn't)

This isn't another theoretical framework for "autonomous agents" that breaks on the first edge case. It's a collection of battle-tested patterns for building AI systems that actually make decisions, write their own queries, and orchestrate tools reliably.

Built by synthesizing documentation from Anthropic, OpenAI, LangChain, LlamaIndex, and LangGraph into 12 comprehensive guides ([Personal Knowledge Base](docs/PKB/)), then implementing the patterns that survive contact with production.

**What you get:**
- Five distinct capabilities you can use independently or compose together
- Extensive code examples with failure modes documented
- Clear guidance on when each pattern works (and when it doesn't)
- Type-safe implementations with proper error handling
- Cost tracking built in, because LLM calls aren't free

**What you don't get:**
- Marketing promises about "fully autonomous" anything
- Black-box abstractions that hide complexity
- Patterns that only work in demos

---

## The Five Capabilities

### 1. Prompt Routing → Intent Detection & Dynamic Dispatch

**The Problem:** User says "check my sales." Do you query a database? Search documents? Hit an external API? The wrong choice wastes time and money.

**The Solution:** Classify intent, route to the right handler, fail gracefully if classification is uncertain.

**Real-world scenario:**
```python
user_input = "What were our Q3 sales in EMEA?"

# Routing classifies this as: database_query
# → Triggers SQL generation
# → Executes against sales DB
# → Returns structured results

user_input = "How do I improve sales conversion?"

# Routing classifies this as: knowledge_search
# → Searches internal docs / RAG system
# → Returns relevant guidance
# → Offers to connect to external resources if needed
```

**When it breaks:** Ambiguous prompts ("sales stuff") require clarification loops or multi-path execution.

---

### 2. Query Writing → Self-Constructing Database/API Queries

**The Problem:** You can't hardcode every possible query users might need. Writing SQL/API queries manually doesn't scale.

**The Solution:** LLM generates queries from natural language, constrained by schema, validated before execution.

**Real-world scenario:**
```python
prompt = "Show me customers who spent >$10k last quarter, sorted by total"

# System:
# 1. Retrieves schema for 'customers' and 'orders' tables
# 2. Generates: SELECT c.name, SUM(o.amount) as total 
#              FROM customers c JOIN orders o 
#              WHERE o.date >= '2024-07-01' 
#              GROUP BY c.id HAVING total > 10000 
#              ORDER BY total DESC
# 3. Validates against schema
# 4. Executes with safety limits
# 5. Returns results
```

**When it breaks:** Complex joins across >5 tables, ambiguous column names, schemas without clear documentation.

---

### 3. Data Processing → Transform, Validate, Enrich

**The Problem:** LLMs return unstructured text. Your application needs structured, validated data.

**The Solution:** Pydantic models enforce schemas, transformation pipelines clean/enrich, validation catches errors before they propagate.

**Real-world scenario:**
```python
# LLM returns: "The customer John Smith (john@example.com) 
#              wants to upgrade to Premium ($49/mo)"

# Processing pipeline:
# 1. Parse into structured fields (Pydantic model)
# 2. Validate email format
# 3. Enrich: lookup customer_id from email
# 4. Validate: Premium plan exists and customer eligible
# 5. Transform: Create upgrade request object
# 6. Return: Type-safe, validated, enriched data ready for business logic
```

**When it breaks:** Ambiguous entities, missing reference data, validation rules that conflict with real-world messiness.

---

### 4. Tool Orchestration → Chaining APIs with Fallback Handling

**The Problem:** Business processes require multiple API calls in sequence. One failure shouldn't crash everything.

**The Solution:** Tool registry, dependency-aware orchestration, graceful fallbacks, retry logic.

**Real-world scenario:**
```python
task = "Create a customer support ticket and notify the team"

# Orchestration:
# 1. Call Zendesk API → Create ticket #12345
# 2. Call Slack API → Post to #support channel
#    ↓ [Slack fails - rate limit]
# 3. Fallback: Send email to support@company.com
# 4. Log: Ticket created, Slack failed, email sent
# 5. Return: Success with fallback metadata
```

**When it breaks:** Circular dependencies, cascading failures in critical paths, state management across retries.

---

### 5. Decision Support & Planning → Multi-Step Analysis

**The Problem:** Complex decisions require analyzing multiple options, considering trade-offs, planning multi-step processes.

**The Solution:** Structured reasoning loops, option comparison frameworks, step-by-step plan generation with validation.

**Real-world scenario:**
```python
question = "Should we expand to the EU market this quarter?"

# Decision process:
# 1. Break down into sub-questions:
#    - Do we have budget? → Query finance API
#    - What's the competitive landscape? → Research
#    - Regulatory requirements? → Legal docs search
#    - Team capacity? → HR system query
# 2. Analyze each dimension
# 3. Score options (Go/Wait/No)
# 4. Identify blockers and dependencies
# 5. Recommend: "Wait until Q2 because..."
# 6. Provide reasoning chain for audit trail
```

**When it breaks:** Incomplete information, conflicting criteria with no clear priority, time-sensitive decisions with stale data.

---

## Quick Start (< 5 Minutes)

### Prerequisites
- Python 3.10+
- Anthropic API key (or OpenAI)
- 5 minutes

### Installation

```bash
# Clone the repository
git clone https://github.com/michalvalco/agentic_ai_development.git
cd agentic_ai_development

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment
cp .env.example .env
# Edit .env and add your API keys
```

### First Example: Prompt Routing

```python
from src.prompt_routing import Router, IntentClassifier

# Initialize
classifier = IntentClassifier(model="claude-sonnet-4-20250514")
router = Router(classifier)

# Route a user query
result = router.route("What were sales last quarter?")

print(f"Intent: {result.intent}")           # "database_query"
print(f"Confidence: {result.confidence}")   # 0.94
print(f"Handler: {result.handler}")         # "SQLQueryHandler"
```

**Expected output:**
```
Intent: database_query
Confidence: 0.94
Handler: SQLQueryHandler
Reasoning: Query asks for historical quantitative data, suggesting database lookup
```

**Next steps:** Check `examples/` for complete implementations of each capability.

---

## Documentation Map

### For Understanding
- **[PROJECT_INSTRUCTIONS_UPDATED.md](PROJECT_INSTRUCTIONS_UPDATED.md)** - Project philosophy, principles, working approach
- **[ARCHITECTURE.md](docs/ARCHITECTURE.md)** - System design, technology choices, data flow (884 lines)
- **[ROADMAP.md](ROADMAP.md)** - Development phases, milestones, current status

### For Implementation
- **[Personal Knowledge Base (PKB)](docs/PKB/)** - 12 comprehensive guides (12,000+ lines total):
  - `anthropic_tool_use.md` - Anthropic's tool use patterns (1,000 lines)
  - `langchain_agents.md` - LangChain agent architectures (800 lines)
  - `react_pattern.md` - ReAct pattern implementation (700 lines)
  - `pydantic_validation.md` - Structured outputs with Pydantic (900 lines)
  - `openai_function_calling.md` - OpenAI function calling (850 lines)
  - `rag_and_embeddings.md` - RAG architecture & vector stores (1,100 lines)
  - `langgraph_workflows.md` - LangGraph state machines (1,200 lines)
  - `agent_testing_evaluation_observability.md` - Testing & monitoring (1,400 lines)
  - ...and 4 more comprehensive guides

Each PKB document follows a rigorous format:
- Key Concepts (with examples)
- Implementation Patterns (production-ready)
- Common Pitfalls (and how to avoid them)
- Integration Points (across all 5 capabilities)
- Actionable Takeaways
- Implementation Checklist
- Testing Strategy

### For Contributing
- **[CONTRIBUTING.md](CONTRIBUTING.md)** - How to contribute (coming soon)
- **[CLAUDE_CODE_SPRINT_STRATEGY.md](CLAUDE_CODE_SPRINT_STRATEGY.md)** - 11-day implementation sprint plan

---

## Technology Stack

### Core
- **Python 3.10+** (match statements, modern type hints)
- **Pydantic 2.x** (structured outputs, validation)
- **LangChain** (agent patterns, tool abstractions)
- **LangGraph** (state machines, complex workflows)

### LLM Providers
- **Anthropic Claude** (primary: Sonnet 4)
- **OpenAI GPT** (alternative: GPT-4)

### Vector & Data
- **ChromaDB** (development)
- **Pinecone** (production option)
- **PostgreSQL** (query writing examples)
- **Redis** (caching, rate limiting)

### Observability
- **LangSmith** (LangChain tracing)
- **Phoenix** (Arize AI - cost tracking)
- **Python logging** (structured logs)

### Testing
- **pytest** (unit + integration)
- **pytest-asyncio** (async testing)
- **pytest-mock** (LLM mocking)

---

## Project Philosophy

### 1. Pragmatic Realism Over Hype

We document failure modes as prominently as success patterns. If a capability breaks under certain conditions, we tell you exactly when and why.

### 2. Production-Ready From Day One

Every implementation includes:
- Type hints and docstrings
- Error handling for common failures
- Cost tracking for LLM calls
- Unit and integration tests
- Real-world examples

### 3. Composition Over Complexity

Five capabilities that work independently but compose naturally. Start with one, add more as needed.

### 4. Context Propagation

Every layer passes context forward—user intent, cost tracking, error history—so downstream components have what they need.

### 5. Fail Explicitly, Recover Gracefully

No silent failures. Every error gets logged, surfaced, and handled. Recovery strategies are explicit.

---

## Development Status

**Current Phase:** Implementation (Days 2-11 of 11-day sprint)

| Capability | Status | Test Coverage | Docs |
|------------|--------|---------------|------|
| Prompt Routing | 🟢 Complete | 85% | ✅ |
| Query Writing | 🟡 In Progress | 60% | ✅ |
| Data Processing | 🟡 In Progress | 70% | ✅ |
| Tool Orchestration | 🔴 Not Started | - | ✅ |
| Decision Support | 🔴 Not Started | - | ✅ |

**PKB Documentation:** ✅ 12/12 complete (~12,000 lines)  
**Architecture:** ✅ Complete (884 lines)  
**Examples:** 🟡 3/10 complete  
**Integration Tests:** 🟡 Basic coverage  

---

## Real-World Use Cases

### Customer Support Automation
**Stack:** Routing + Query Writing + Tool Orchestration
- Route support requests by urgency/category
- Query historical customer data
- Create tickets, notify teams, log interactions

### Business Intelligence Assistant
**Stack:** Query Writing + Data Processing + Decision Support
- Generate SQL from natural language
- Transform results into visualizations
- Analyze trends, recommend actions

### Document Analysis Pipeline
**Stack:** Data Processing + Tool Orchestration + Decision Support
- Extract structured data from documents
- Chain OCR → parsing → validation → enrichment
- Route to appropriate downstream systems

### Multi-Source Research Agent
**Stack:** Routing + Tool Orchestration + Decision Support
- Route queries to internal docs vs external search
- Orchestrate multiple API calls (web, databases, vector stores)
- Synthesize findings, cite sources, recommend next steps

---

## Cost Expectations

Based on testing with Claude Sonnet 4 and GPT-4:

| Capability | Avg Cost/Request | Notes |
|------------|-----------------|-------|
| Prompt Routing | $0.002 - $0.005 | Single classification call |
| Query Writing | $0.008 - $0.015 | Schema retrieval + generation + validation |
| Data Processing | $0.005 - $0.012 | Depends on enrichment complexity |
| Tool Orchestration | $0.015 - $0.040 | Multiple tool calls + retry logic |
| Decision Support | $0.025 - $0.080 | Multi-step reasoning chains |

**Cost optimization strategies:**
- Cache classification results for similar prompts
- Use smaller models (Haiku, GPT-3.5) for simple tasks
- Implement request deduplication
- Set token limits per capability

All costs tracked automatically. See [observability guide](docs/PKB/agent_testing_evaluation_observability.md).

---

## Contributing

We welcome contributions that maintain the project's pragmatic, production-focused philosophy.

**Good contributions:**
- Implementations that work in production
- Documentation of edge cases and failure modes
- Tests that cover real-world scenarios
- Examples with complete context

**Less valuable:**
- Theoretical improvements without implementation
- Features without error handling
- Code without tests
- Examples that only work in demos

**How to contribute:**
1. Read [PROJECT_INSTRUCTIONS_UPDATED.md](PROJECT_INSTRUCTIONS_UPDATED.md)
2. Review relevant PKB documents
3. Open an issue describing your contribution
4. Submit PR with tests and documentation

---

## License

MIT License - See [LICENSE](LICENSE) for details.

---

## Acknowledgments

This project synthesizes patterns from:
- [Anthropic's tool use documentation](https://docs.anthropic.com/en/docs/build-with-claude/tool-use)
- [LangChain's agent frameworks](https://python.langchain.com/docs/modules/agents/)
- [LangGraph's workflow patterns](https://langchain-ai.github.io/langgraph/)
- [OpenAI's function calling guides](https://platform.openai.com/docs/guides/function-calling)
- [LlamaIndex's query engines](https://docs.llamaindex.ai/en/stable/module_guides/deploying/query_engine/)

Built with the understanding that:
- Agentic AI is a set of patterns, not a single technology
- Production systems require explicit error handling
- Documentation should prevent failures, not just explain successes
- Real value comes from composition, not individual capabilities

---

## Questions?

- **Implementation questions:** Check the [PKB documentation](docs/PKB/)
- **Architecture questions:** See [ARCHITECTURE.md](docs/ARCHITECTURE.md)
- **Issues or bugs:** [Open an issue](https://github.com/michalvalco/agentic_ai_development/issues)
- **Project philosophy:** Read [PROJECT_INSTRUCTIONS_UPDATED.md](PROJECT_INSTRUCTIONS_UPDATED.md)

---

**Status:** Active development | Phase 2 complete, Phase 3 in progress  
**Last updated:** 2025-11-07  
**Maintainer:** [@michalvalco](https://github.com/michalvalco)
