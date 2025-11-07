# Project Instructions: Agentic AI Development

## Project Identity

**Repository Name:** `agentic_ai_development`  
**Purpose:** Build practical, production-ready implementations of five core agentic AI capabilities  
**Philosophy:** Pragmatic realism over hype. Working code over theoretical frameworks. Documentation of failure modes alongside success patterns.

---

## Core Objectives

We are building five distinct but interconnected agentic capabilities:

1. **Prompt Routing** → Dynamic intent detection and routing logic
2. **Query Writing** → Self-constructing database/API queries
3. **Data Processing** → Transform raw inputs into usable outputs
4. **Tool Orchestration** → Chaining APIs with fallback handling
5. **Decision Support** → Multi-step planning and prioritization

Each capability must be:
- Independently testable
- Clearly documented (including when it breaks)
- Implementable in real projects
- Backed by examples from actual use cases

---

## Working Principles

### 1. **Documentation is Constitutive, Not Decorative**

Every pattern, every function, every architectural decision gets documented *as we build*, not after. 

The PKB (Personal Knowledge Base) serves as our conceptual foundation—we summarize external documentation into markdown files that become our reference material. This isn't busywork. It's how we internalize patterns and avoid cargo-cult implementations.

**✅ PKB STATUS: COMPLETE (12/12 documents)**

### 2. **Code Standards**

- **Python 3.10+** (for match statements, improved type hints)
- **Type hints everywhere** (Pydantic models for complex structures)
- **Docstrings follow Google style** (readable, practical)
- **Tests are non-negotiable** (pytest, with both unit and integration tests)
- **Error handling is explicit** (no silent failures, clear error messages)

### 3. **File Organization**

```
agentic_ai_development/
├── .github/
│   └── workflows/               # CI/CD pipelines
├── docs/
│   ├── PKB/                     # ✅ Personal Knowledge Base (12 docs complete)
│   │   ├── anthropic_tool_use.md
│   │   ├── langchain_agents.md
│   │   ├── react_pattern.md
│   │   ├── langchain_tools.md
│   │   ├── pydantic_validation.md
│   │   ├── openai_function_calling.md
│   │   ├── anthropic_prompt_engineering.md
│   │   ├── openai_prompt_engineering.md
│   │   ├── rag_and_embeddings.md
│   │   ├── llamaindex_query_engines.md
│   │   ├── langgraph_workflows.md
│   │   └── agent_testing_evaluation_observability.md
│   ├── ARCHITECTURE.md          # System architecture design
│   ├── PATTERNS.md              # Common patterns discovered
│   └── API.md                   # API documentation
├── src/
│   ├── __init__.py
│   ├── prompt_routing/
│   │   ├── __init__.py
│   │   ├── classifier.py
│   │   ├── router.py
│   │   └── handlers/
│   ├── query_writing/
│   │   ├── __init__.py
│   │   ├── sql_generator.py
│   │   ├── api_query_builder.py
│   │   └── schema_manager.py
│   ├── data_processing/
│   │   ├── __init__.py
│   │   ├── transformers.py
│   │   ├── validators.py
│   │   └── pipelines.py
│   ├── tool_orchestration/
│   │   ├── __init__.py
│   │   ├── orchestrator.py
│   │   ├── tool_registry.py
│   │   └── fallback_handler.py
│   ├── decision_support/
│   │   ├── __init__.py
│   │   ├── analyzer.py
│   │   ├── recommender.py
│   │   └── explainer.py
│   ├── common/
│   │   ├── __init__.py
│   │   ├── models.py            # Pydantic models
│   │   ├── exceptions.py
│   │   ├── config.py
│   │   └── utils.py
│   └── integrations/            # Observability, monitoring
│       ├── langsmith.py
│       ├── phoenix.py
│       └── cost_tracker.py
├── tests/
│   ├── unit/                    # Fast, mocked tests
│   ├── integration/             # Real LLM calls
│   ├── fixtures/                # Test data
│   └── conftest.py              # Pytest configuration
├── examples/                     # Working demonstrations
│   ├── basic_routing.py
│   ├── sql_query_generation.py
│   ├── data_pipeline.py
│   ├── tool_chain.py
│   └── decision_workflow.py
├── scripts/                      # Utility scripts
│   ├── setup_env.py
│   ├── run_evals.py
│   └── cost_report.py
├── .env.example
├── .gitignore
├── requirements.txt
├── requirements-dev.txt         # Development dependencies
├── pytest.ini
├── README.md
└── PROJECT_INSTRUCTIONS.md      # This file
```

### 4. **Naming Conventions**

- **Modules:** lowercase_with_underscores
- **Classes:** PascalCase
- **Functions:** lowercase_with_underscores
- **Constants:** UPPERCASE_WITH_UNDERSCORES
- **Private:** _leading_underscore
- **PKB files:** `service_topic.md` (e.g., `anthropic_tool_use.md`)

---

## Development Workflow

### ✅ Phase 1: Knowledge Foundation (COMPLETE)
1. ✅ Identified 12 key documentation sources
2. ✅ Summarized each into PKB markdown files
3. ✅ Extracted patterns, best practices, failure modes
4. ✅ Documented connections between different sources

**Deliverables:**
- 12 comprehensive PKB documents (700-1000 lines each)
- Clear understanding of all five capabilities
- Foundation for implementation decisions

---

### 🔄 Phase 2: Architecture Design (IN PROGRESS)

**Timeline:** 2-3 days

**Goals:**
1. Design overall system architecture
2. Define interfaces between components
3. Establish data flow patterns
4. Document design decisions and trade-offs
5. Create ARCHITECTURE.md

**Key Decisions to Document:**
- Which LLM providers to support (Anthropic, OpenAI, or both?)
- Vector database choice (ChromaDB for dev, Pinecone for prod?)
- State management strategy (LangGraph checkpoints?)
- Error handling patterns
- Testing strategy
- Observability approach (LangSmith, Phoenix, both?)

**Deliverables:**
- `docs/ARCHITECTURE.md` - System design document
- Interface definitions (Pydantic models)
- Sequence diagrams for key workflows
- Technology stack decisions documented

---

### 🚀 Phase 3: Implementation (RAPID EXECUTION - Claude Code)

**Timeline:** 7-8 days (using $1000 Claude Code credit)**

**Strategy:** Build all five capabilities in parallel using Claude Code's autonomous coding environment.

**Week 1 Sprint:**

**Day 1-2: Foundation & Routing**
- Set up project structure
- Implement common utilities (config, exceptions, models)
- Build Prompt Routing capability (classifier + router + handlers)
- Unit tests for routing logic
- Integration tests with real LLMs (budget: $20)

**Day 3-4: Query Writing & Data Processing**
- SQL/API query generation
- Schema management
- Data transformation pipelines
- Validation logic
- Tests for both capabilities (budget: $30)

**Day 5-6: Tool Orchestration & Decision Support**
- Tool registry and orchestrator
- Fallback handling
- Decision analyzer and recommender
- Integration tests (budget: $40)

**Day 7-8: Integration, Examples & Testing**
- Connect all capabilities
- Build working examples (5 complete demos)
- Comprehensive test suite
- Documentation finalization
- Performance testing (budget: $50)

**Total Budget:** ~$140 in LLM costs, rest for compute/iterations

---

### Phase 4: Production Readiness (POST-CREDIT)

**Timeline:** Ongoing

1. Performance optimization
2. Production deployment patterns
3. Real-world validation
4. Community feedback incorporation
5. Blog posts / documentation site

---

## Claude Code Strategy ($1000 Credit, 11 Days)

### Why Claude Code?

**Advantages for This Project:**
- Autonomous coding with full context of PKB
- Can read all 12 PKB documents and apply patterns
- Faster iteration than manual coding
- Better at maintaining consistency across modules
- Natural for implementing patterns from documentation

**Credit Usage:**
- $1000 credit = ~6.7M output tokens at $0.15/1M
- Or ~33M input tokens at $0.03/1M
- Realistically: ~200-300 hours of coding assistance
- Perfect for rapid implementation phase

### Execution Plan

**Day 1: Setup & Architecture**
- Push completed PKB to GitHub
- Initialize project structure
- Create ARCHITECTURE.md with Claude Code
- Set up development environment
- Budget: $50

**Days 2-3: Core Implementation**
- Implement common utilities
- Build prompt routing (capability #1)
- Budget: $150

**Days 4-5: Data Capabilities**
- Query writing (capability #2)
- Data processing (capability #3)
- Budget: $150

**Days 6-7: Advanced Capabilities**
- Tool orchestration (capability #4)
- Decision support (capability #5)
- Budget: $150

**Days 8-9: Integration & Testing**
- Connect all capabilities
- Comprehensive test suite
- Working examples
- Budget: $200

**Days 10-11: Polish & Documentation**
- Fix bugs discovered in testing
- Complete README
- API documentation
- Performance optimization
- Budget: $200

**Buffer:** $100 for unexpected iterations/debugging

### Daily Workflow with Claude Code

**Morning (Planning):**
1. Review previous day's work
2. Define today's specific goals
3. Create Claude Code task with PKB references
4. Example: "Implement SQL query generator using patterns from `query_writing.md` and `pydantic_validation.md`"

**Afternoon (Execution):**
1. Let Claude Code implement
2. Review generated code
3. Request refinements
4. Run tests

**Evening (Validation):**
1. Integration testing
2. Cost tracking
3. Update progress notes
4. Plan next day

### Cost Control

**Monitor Daily:**
```python
# Track Claude Code usage
daily_budget = 1000 / 11  # ~$91/day
if today_cost > daily_budget:
    # Pause, review, adjust strategy
```

**Optimization:**
- Use Claude Code for implementation
- Use manual review for validation
- Cache repeated patterns
- Batch related features

---

## Quality Standards

### Before Committing Code:
- [ ] Type hints present and accurate
- [ ] Docstrings explain *why*, not just *what*
- [ ] Tests pass (and actually test something meaningful)
- [ ] Error handling covers realistic failure modes
- [ ] Example usage included
- [ ] PKB references documented (which patterns informed this code)

### Before Marking a Capability "Complete":
- [ ] Works in isolation
- [ ] Integrates with at least one other capability
- [ ] Performance tested with realistic data volumes
- [ ] Failure modes documented
- [ ] Real-world example provided
- [ ] README updated

---

## GitHub Workflow

### Initial Push
```bash
# Initialize repo
cd C:\Users\Michal Valco\Documents\agentic_ai_development
git init
git add .
git commit -m "Initial commit: Complete PKB (12/12 docs)"

# Connect to GitHub
git remote add origin https://github.com/yourusername/agentic_ai_development.git
git branch -M main
git push -u origin main
```

### Development Branches
- `main` - Stable, working code only
- `dev` - Active development
- `feature/routing` - Individual capabilities
- `feature/testing` - Test infrastructure
- `docs/architecture` - Documentation work

### Commit Messages
```
feat(routing): Implement intent classifier with Anthropic
test(routing): Add integration tests for classifier
docs(architecture): Define system interfaces
fix(query): Handle SQL injection in query builder
refactor(common): Extract shared utilities to common module
```

### CI/CD (GitHub Actions)
```yaml
# .github/workflows/test.yml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.10'
      - name: Install dependencies
        run: pip install -r requirements-dev.txt
      - name: Run unit tests
        run: pytest tests/unit -v
      # Integration tests run only on main branch
      - name: Run integration tests
        if: github.ref == 'refs/heads/main'
        run: pytest tests/integration -v -m llm_integration
        env:
          ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
```

---

## Communication with Claude Code

### Effective Task Descriptions

**❌ Bad:**
"Build the routing system"

**✅ Good:**
"Implement prompt routing capability using patterns from docs/PKB/anthropic_prompt_engineering.md and docs/PKB/langchain_agents.md. Create:
1. IntentClassifier class (uses Claude to classify user intent)
2. Router class (routes to appropriate handler based on intent)
3. Handler interface (defines contract for route handlers)
4. Tests (unit + integration)

Follow patterns in PKB for error handling and structured outputs."

### Providing Context

**Reference PKB Documents:**
- "Using the ReAct pattern from react_pattern.md..."
- "Following the tool orchestration approach in langchain_tools.md..."
- "Implement error handling as described in anthropic_tool_use.md..."

**Specify Quality Requirements:**
- "Include type hints for all parameters"
- "Add docstrings in Google style"
- "Create both unit tests (mocked) and integration tests (real LLM calls)"
- "Handle rate limits and API failures gracefully"

---

## Success Metrics

This project succeeds when:

**Technical:**
- ✅ All 5 capabilities implemented and tested
- ✅ >80% code coverage
- ✅ All integration tests passing
- ✅ Production-ready error handling
- ✅ Comprehensive documentation

**Practical:**
- ✅ Each capability can be dropped into a real project
- ✅ Examples demonstrate real-world use cases
- ✅ Failure modes are predictable and recoverable
- ✅ Cost tracking shows efficient LLM usage

**Strategic:**
- ✅ $1000 Claude Code credit fully utilized
- ✅ Repository teaches as much as it provides
- ✅ Clear path for future contributors
- ✅ Portfolio-worthy project demonstrating AI engineering skills

---

## Current Status

**Completed:**
- ✅ Phase 1: PKB Development (12/12 documents, ~12,000 lines)
- ✅ Project structure defined
- ✅ Development workflow established
- ✅ Quality standards documented

**In Progress:**
- 🔄 Phase 2: Architecture Design (ARCHITECTURE.md in progress)
- 🔄 GitHub repository initialization

**Next Steps:**
1. **Immediate:** Push PKB to GitHub
2. **Day 1:** Create ARCHITECTURE.md
3. **Day 2-11:** Rapid implementation with Claude Code
4. **Ongoing:** Production readiness and optimization

---

**Last Updated:** 2025-11-07  
**Status:** Phase 1 Complete ✅ | Phase 2 In Progress 🔄 | Phase 3 Starting (Claude Code Sprint)  
**Credit Deadline:** November 18, 2025 (11 days remaining)  
**Budget:** $1000 Claude Code credit allocated for rapid implementation