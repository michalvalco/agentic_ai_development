# System Architecture: Agentic AI Development

**Version:** 1.0
**Last Updated:** 2025-11-07
**Status:** Phase 2 - Architecture Definition

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [System Overview](#system-overview)
3. [Core Capabilities](#core-capabilities)
4. [Technology Stack](#technology-stack)
5. [Data Flow Architecture](#data-flow-architecture)
6. [API Contracts](#api-contracts)
7. [Error Handling Strategy](#error-handling-strategy)
8. [Observability & Monitoring](#observability--monitoring)
9. [Security Considerations](#security-considerations)
10. [Deployment Architecture](#deployment-architecture)
11. [Design Decisions & Trade-offs](#design-decisions--trade-offs)

---

## Executive Summary

This document defines the architecture for a production-ready agentic AI system implementing five core capabilities: **Prompt Routing**, **Query Writing**, **Data Processing**, **Tool Orchestration**, and **Decision Support**.

The system is designed to be:
- **Modular**: Each capability operates independently but composes seamlessly
- **Observable**: Full tracing, logging, and cost tracking at every layer
- **Fault-tolerant**: Graceful degradation with explicit error handling
- **Extensible**: Plugin architecture for tools, models, and handlers
- **Production-ready**: Type-safe, tested, documented, and deployable

**Key Design Principles:**
1. Composition over complexity
2. Fail explicitly, recover gracefully
3. Context propagation throughout workflows
4. Async-first for I/O-bound operations
5. Cost awareness at every LLM call

---

## System Overview

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Client Application                        │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Agentic AI System                            │
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │   Prompt     │  │    Query     │  │    Data      │         │
│  │   Routing    │→ │   Writing    │→ │  Processing  │         │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
│         │                  │                  │                 │
│         ▼                  ▼                  ▼                 │
│  ┌──────────────────────────────────────────────────┐         │
│  │          Tool Orchestration Layer                │         │
│  └──────────────────────────────────────────────────┘         │
│         │                                                       │
│         ▼                                                       │
│  ┌──────────────────────────────────────────────────┐         │
│  │          Decision Support System                  │         │
│  └──────────────────────────────────────────────────┘         │
│                                                                  │
│  ┌──────────────────────────────────────────────────┐         │
│  │  Common Layer: Config, Models, Utils, Exceptions │         │
│  └──────────────────────────────────────────────────┘         │
└─────────────────────────────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│  External Services: LLMs, Databases, APIs, Vector Stores        │
└─────────────────────────────────────────────────────────────────┘
```

### Component Responsibilities

| Component | Responsibility | Dependencies |
|-----------|---------------|--------------|
| **Prompt Routing** | Classify intent, route to appropriate handler | Common, LLM Provider |
| **Query Writing** | Generate SQL/API queries from natural language | Common, LLM Provider, Schema Registry |
| **Data Processing** | Transform, validate, enrich data | Common, Pydantic |
| **Tool Orchestration** | Execute tool chains with fallbacks | Common, Tool Registry, all capabilities |
| **Decision Support** | Multi-step analysis and recommendations | Common, LLM Provider, all capabilities |
| **Common** | Shared utilities, models, configuration | - |

---

## Core Capabilities

### 1. Prompt Routing

**Purpose:** Intelligently classify user queries and route to appropriate handlers.

**Components:**
- `IntentClassifier`: Uses LLM to determine user intent
- `Router`: Maps intents to registered handlers
- `Handler`: Abstract interface for route implementations

**Flow:**
```
User Query → IntentClassifier → Router → Handler → Response
```

**Key Patterns (from PKB):**
- Few-shot classification ([anthropic_prompt_engineering.md](../docs/PKB/anthropic_prompt_engineering.md))
- Confidence scoring with threshold-based routing
- Fallback to human escalation for ambiguous queries

**Implementation:**
```python
# src/prompt_routing/classifier.py
class IntentClassifier:
    async def classify(self, query: str) -> Intent:
        """Classify user intent with confidence score."""

# src/prompt_routing/router.py
class Router:
    def register_handler(self, intent: str, handler: Handler)
    async def route(self, query: str) -> Response:
        """Route query to appropriate handler."""
```

---

### 2. Query Writing

**Purpose:** Generate parameterized SQL/API queries from natural language.

**Components:**
- `SQLQueryGenerator`: SQL query generation with injection prevention
- `APIQueryBuilder`: REST query construction from OpenAPI specs
- `SchemaManager`: Load and provide schema context to LLMs

**Flow:**
```
Natural Language → Schema Context → LLM → Query → Validation → Execution
```

**Key Patterns (from PKB):**
- Structured output with Pydantic ([pydantic_validation.md](../docs/PKB/pydantic_validation.md))
- Schema-aware prompting ([anthropic_prompt_engineering.md](../docs/PKB/anthropic_prompt_engineering.md))
- Query validation before execution

**Implementation:**
```python
# src/query_writing/sql_generator.py
class SQLQueryGenerator:
    async def generate(
        self,
        natural_query: str,
        schema: Schema
    ) -> SQLQuery:
        """Generate parameterized SQL from natural language."""

# src/query_writing/api_query_builder.py
class APIQueryBuilder:
    async def build(
        self,
        intent: str,
        api_spec: OpenAPISpec
    ) -> APIQuery:
        """Construct REST query from API specification."""
```

---

### 3. Data Processing

**Purpose:** Clean, transform, validate, and enrich data for downstream use.

**Components:**
- `DataTransformer`: Chainable transformation operations
- `Validator`: Pydantic-based validation
- `Pipeline`: Orchestrate multi-step transformations

**Flow:**
```
Raw Data → Transform → Validate → Enrich → Clean Data
```

**Key Patterns (from PKB):**
- Pipeline pattern with composable transformers ([langchain_tools.md](../docs/PKB/langchain_tools.md))
- Pydantic validation for type safety ([pydantic_validation.md](../docs/PKB/pydantic_validation.md))
- Error recovery with partial results

**Implementation:**
```python
# src/data_processing/transformers.py
class DataTransformer:
    def clean(self, data: Any) -> Any
    def normalize(self, data: Any) -> Any
    def enrich(self, data: Any, source: str) -> Any

# src/data_processing/pipelines.py
class Pipeline:
    def add_step(self, transformer: Callable)
    async def execute(self, data: Any) -> Result[Data, Error]
```

---

### 4. Tool Orchestration

**Purpose:** Coordinate multiple tools/APIs with dependency management and fallbacks.

**Components:**
- `ToolRegistry`: Register and discover available tools
- `Orchestrator`: Execute tool chains (sequential/parallel)
- `FallbackHandler`: Retry logic and alternative tool routing

**Flow:**
```
Task → Plan → Select Tools → Execute (parallel/sequential) → Handle Failures → Result
```

**Key Patterns (from PKB):**
- ReAct pattern for reasoning + acting ([react_pattern.md](../docs/PKB/react_pattern.md))
- Tool abstraction ([anthropic_tool_use.md](../docs/PKB/anthropic_tool_use.md), [langchain_tools.md](../docs/PKB/langchain_tools.md))
- Circuit breaker pattern for unreliable services
- State management ([langgraph_workflows.md](../docs/PKB/langgraph_workflows.md))

**Implementation:**
```python
# src/tool_orchestration/orchestrator.py
class Orchestrator:
    async def execute_chain(
        self,
        tools: List[str],
        input: Any,
        mode: ExecutionMode = "sequential"
    ) -> Result[Any, Error]

# src/tool_orchestration/fallback_handler.py
class FallbackHandler:
    async def retry(self, tool: Tool, max_attempts: int = 3)
    async def alternative(self, primary: Tool, fallback: Tool)
```

---

### 5. Decision Support

**Purpose:** Analyze complex situations and provide ranked recommendations with reasoning.

**Components:**
- `DecisionAnalyzer`: Break down complex decisions
- `Recommender`: Score and rank options
- `Explainer`: Provide transparent reasoning

**Flow:**
```
Situation → Decompose → Analyze → Score Options → Rank → Explain → Recommendation
```

**Key Patterns (from PKB):**
- Chain-of-thought reasoning ([openai_prompt_engineering.md](../docs/PKB/openai_prompt_engineering.md))
- Multi-step workflows ([langgraph_workflows.md](../docs/PKB/langgraph_workflows.md))
- RAG for historical context ([rag_and_embeddings.md](../docs/PKB/rag_and_embeddings.md))

**Implementation:**
```python
# src/decision_support/analyzer.py
class DecisionAnalyzer:
    async def analyze(
        self,
        situation: str,
        criteria: List[str]
    ) -> Analysis

# src/decision_support/recommender.py
class Recommender:
    async def recommend(
        self,
        options: List[Option],
        analysis: Analysis
    ) -> RankedRecommendations
```

---

## Technology Stack

### Core Technologies

| Category | Technology | Rationale | Alternatives Considered |
|----------|-----------|-----------|------------------------|
| **Language** | Python 3.10+ | Modern type hints, match statements, async support | Python 3.11 (not widely deployed) |
| **LLM Providers** | Anthropic Claude (primary), OpenAI (secondary) | Claude excels at structured output, tool use | OpenAI for specific use cases |
| **Validation** | Pydantic v2 | Type safety, validation, serialization | Marshmallow (less type-safe) |
| **Async Framework** | asyncio + aiohttp | Built-in, mature, well-documented | httpx (considered) |
| **Testing** | pytest + pytest-asyncio | De facto standard, great plugin ecosystem | unittest (less ergonomic) |
| **Vector DB (dev)** | ChromaDB | Lightweight, embeddable, no setup | FAISS (no persistence layer) |
| **Vector DB (prod)** | Pinecone | Managed, scalable, reliable | Weaviate, Qdrant (more complex) |
| **Observability** | LangSmith (prod), Phoenix (dev) | LangSmith for prod tracing, Phoenix for local debugging | Weights & Biases (overkill) |

### Dependencies

**Core:**
```
anthropic>=0.18.0
openai>=1.0.0
pydantic>=2.0.0
langchain>=0.1.0
langgraph>=0.0.20
aiohttp>=3.9.0
```

**Data & Embeddings:**
```
chromadb>=0.4.0
sentence-transformers>=2.2.0
tiktoken>=0.5.0
```

**Observability:**
```
langsmith>=0.1.0
arize-phoenix>=2.0.0
structlog>=23.1.0
```

**Development:**
```
pytest>=7.4.0
pytest-asyncio>=0.21.0
pytest-cov>=4.1.0
mypy>=1.5.0
black>=23.7.0
ruff>=0.0.287
```

---

## Data Flow Architecture

### Context Propagation

Every operation carries a `Context` object for tracing, logging, and state management:

```python
@dataclass
class Context:
    request_id: str
    user_id: Optional[str]
    session_id: Optional[str]
    metadata: Dict[str, Any]
    cost_tracker: CostTracker
    trace_id: str
    parent_span_id: Optional[str]
```

### Typical Request Flow

```
1. Client Request
   ↓
2. Create Context (request_id, trace_id)
   ↓
3. Prompt Routing (classify intent, route)
   ↓
4. Query Writing (generate SQL/API query) ← uses context for logging
   ↓
5. Data Processing (transform, validate) ← uses context for error tracking
   ↓
6. Tool Orchestration (execute tools) ← uses context for cost tracking
   ↓
7. Decision Support (analyze, recommend) ← uses context for reasoning trace
   ↓
8. Response + Metadata (cost, latency, confidence)
```

### Data Models

**Base Result Type:**
```python
from typing import Generic, TypeVar, Union

T = TypeVar('T')
E = TypeVar('E')

@dataclass
class Success(Generic[T]):
    value: T
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Failure(Generic[E]):
    error: E
    context: str
    recoverable: bool = False

Result = Union[Success[T], Failure[E]]
```

**Intent Model:**
```python
from pydantic import BaseModel, Field

class Intent(BaseModel):
    name: str = Field(..., description="Intent category")
    confidence: float = Field(..., ge=0.0, le=1.0)
    parameters: Dict[str, Any] = Field(default_factory=dict)
    fallback_required: bool = False
```

**Query Model:**
```python
class SQLQuery(BaseModel):
    query: str = Field(..., description="Parameterized SQL query")
    parameters: Dict[str, Any] = Field(default_factory=dict)
    schema_used: str
    safety_validated: bool = True

class APIQuery(BaseModel):
    endpoint: str
    method: str = Field(default="GET")
    params: Dict[str, Any] = Field(default_factory=dict)
    headers: Dict[str, str] = Field(default_factory=dict)
```

---

## API Contracts

### Capability Interfaces

Each capability exposes a consistent async interface:

```python
from abc import ABC, abstractmethod
from typing import Generic, TypeVar

Input = TypeVar('Input')
Output = TypeVar('Output')

class Capability(ABC, Generic[Input, Output]):
    """Base interface for all capabilities."""

    @abstractmethod
    async def execute(
        self,
        input: Input,
        context: Context
    ) -> Result[Output, Error]:
        """Execute capability with context."""
        pass

    @abstractmethod
    def health_check(self) -> HealthStatus:
        """Check capability health."""
        pass
```

### Common Response Format

All capabilities return a standardized response:

```python
@dataclass
class CapabilityResponse(Generic[T]):
    result: Result[T, Error]
    metadata: ResponseMetadata

@dataclass
class ResponseMetadata:
    latency_ms: float
    cost_usd: float
    tokens_used: TokenCount
    model_used: str
    confidence: Optional[float] = None
    trace_id: str
    timestamp: datetime
```

---

## Error Handling Strategy

### Error Hierarchy

```python
class AgenticError(Exception):
    """Base exception for all agentic errors."""
    def __init__(self, message: str, recoverable: bool = False):
        self.message = message
        self.recoverable = recoverable
        super().__init__(message)

# Capability-specific errors
class RoutingError(AgenticError): pass
class QueryGenerationError(AgenticError): pass
class DataProcessingError(AgenticError): pass
class ToolExecutionError(AgenticError): pass
class DecisionAnalysisError(AgenticError): pass

# Infrastructure errors
class LLMProviderError(AgenticError): pass
class RateLimitError(LLMProviderError):
    def __init__(self):
        super().__init__("Rate limit exceeded", recoverable=True)
```

### Error Handling Patterns

**1. Retry with Exponential Backoff:**
```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(RateLimitError)
)
async def call_llm(prompt: str) -> str:
    # LLM call implementation
    pass
```

**2. Circuit Breaker:**
```python
class CircuitBreaker:
    """Prevent cascading failures for unreliable services."""
    def __init__(self, failure_threshold: int = 5, timeout: int = 60):
        self.failure_count = 0
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half-open
```

**3. Fallback Chains:**
```python
async def execute_with_fallback(
    primary: Callable,
    fallback: Callable,
    context: Context
) -> Result:
    try:
        return await primary(context)
    except Exception as e:
        logger.warning(f"Primary failed: {e}, trying fallback")
        return await fallback(context)
```

---

## Observability & Monitoring

### Logging Strategy

**Structured Logging with Structlog:**
```python
import structlog

logger = structlog.get_logger()

logger.info(
    "query_generated",
    request_id=context.request_id,
    query_type="sql",
    tokens_used=150,
    cost_usd=0.002,
    latency_ms=245
)
```

### Tracing

**LangSmith Integration (Production):**
```python
from langsmith import Client, traceable

client = Client()

@traceable(name="prompt_routing", run_type="chain")
async def route_prompt(query: str, context: Context):
    # Automatically traces to LangSmith
    pass
```

**Phoenix Integration (Development):**
```python
import phoenix as px

px.launch_app()
px.trace_llm_calls()
```

### Cost Tracking

```python
class CostTracker:
    def __init__(self):
        self.total_cost = 0.0
        self.calls: List[LLMCall] = []

    def track_call(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int
    ) -> float:
        cost = calculate_cost(model, input_tokens, output_tokens)
        self.total_cost += cost
        self.calls.append(LLMCall(model, input_tokens, output_tokens, cost))
        return cost
```

### Metrics

**Key Metrics to Track:**
- **Latency:** p50, p95, p99 per capability
- **Cost:** Per request, per capability, daily total
- **Success Rate:** % of successful vs. failed requests
- **Confidence:** Distribution of confidence scores
- **Token Usage:** Input/output tokens per model

---

## Security Considerations

### 1. Prompt Injection Prevention

**Mitigation:**
- Input sanitization before LLM calls
- System/user message separation ([anthropic_prompt_engineering.md](../docs/PKB/anthropic_prompt_engineering.md))
- Validation of LLM outputs before execution

```python
def sanitize_input(user_input: str) -> str:
    """Remove potential prompt injection patterns."""
    # Remove special tokens, excessive newlines, etc.
    return sanitized
```

### 2. SQL Injection Prevention

**Mitigation:**
- Always use parameterized queries
- Validate generated SQL against allowed patterns
- Whitelist table/column names

```python
class SQLQueryGenerator:
    def __init__(self, allowed_tables: Set[str]):
        self.allowed_tables = allowed_tables

    def validate_query(self, query: SQLQuery) -> bool:
        # Validate table names, no dynamic SQL, etc.
        pass
```

### 3. API Key Management

**Best Practices:**
- Store keys in environment variables (never in code)
- Use separate keys for dev/staging/prod
- Rotate keys regularly
- Log key usage for audit trails

```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    anthropic_api_key: str
    openai_api_key: str
    pinecone_api_key: str

    class Config:
        env_file = ".env"
        case_sensitive = False
```

### 4. Data Privacy

**Considerations:**
- PII detection and redaction in logs
- User data isolation in multi-tenant scenarios
- Compliance with GDPR, CCPA requirements

---

## Deployment Architecture

### Development Environment

```
Local Machine
├── Python 3.10+ venv
├── ChromaDB (embedded)
├── Phoenix (localhost:6006)
└── .env (API keys)
```

### Production Environment (Recommended)

```
┌─────────────────────────────────────────────────┐
│              Load Balancer (ALB)                 │
└───────────────────┬─────────────────────────────┘
                    │
    ┌───────────────┴───────────────┐
    ▼                               ▼
┌─────────┐                   ┌─────────┐
│ API     │                   │ API     │
│ Service │                   │ Service │
│ (ECS)   │                   │ (ECS)   │
└────┬────┘                   └────┬────┘
     │                             │
     └──────────┬──────────────────┘
                ▼
    ┌──────────────────────┐
    │   External Services   │
    ├──────────────────────┤
    │ - Anthropic API      │
    │ - OpenAI API         │
    │ - Pinecone (vectors) │
    │ - PostgreSQL (state) │
    │ - Redis (cache)      │
    │ - LangSmith (obs)    │
    └──────────────────────┘
```

### Deployment Options

**Option 1: Serverless (AWS Lambda)**
- **Pros:** Auto-scaling, pay-per-use, no server management
- **Cons:** Cold start latency, 15min timeout limit
- **Best For:** Bursty, unpredictable workloads

**Option 2: Containerized (ECS/Kubernetes)**
- **Pros:** Full control, no timeouts, persistent connections
- **Cons:** More operational complexity
- **Best For:** Steady workloads, long-running tasks

**Option 3: Hybrid**
- **Pros:** Use Lambda for routing, ECS for heavy processing
- **Cons:** Increased architectural complexity
- **Best For:** Mixed workload patterns

---

## Design Decisions & Trade-offs

### Decision 1: Anthropic Claude as Primary LLM

**Rationale:**
- Superior tool use capabilities ([anthropic_tool_use.md](../docs/PKB/anthropic_tool_use.md))
- Better at following complex instructions
- Excellent structured output support
- Extended context window (200K tokens)

**Trade-off:**
- Single vendor dependency
- Higher cost than some alternatives

**Mitigation:**
- OpenAI as fallback provider
- Abstract LLM interface for easy swapping

---

### Decision 2: Pydantic v2 for Validation

**Rationale:**
- Type-safe data models
- Automatic validation
- Excellent LLM integration ([pydantic_validation.md](../docs/PKB/pydantic_validation.md))
- Serialization/deserialization built-in

**Trade-off:**
- Learning curve for complex validators
- Runtime overhead (minimal)

**Mitigation:**
- Comprehensive type hints reduce errors
- Performance is acceptable for our use case

---

### Decision 3: Async-First Architecture

**Rationale:**
- Most operations are I/O-bound (LLM calls, API requests)
- Significant performance improvement for concurrent requests
- Enables efficient tool orchestration

**Trade-off:**
- More complex code (async/await everywhere)
- Harder to debug

**Mitigation:**
- Clear async conventions
- Thorough testing
- Structured logging for debugging

---

### Decision 4: ChromaDB (Dev) / Pinecone (Prod)

**Rationale:**
- ChromaDB: Zero setup for development
- Pinecone: Managed, scalable for production

**Trade-off:**
- Different APIs between dev/prod

**Mitigation:**
- Abstract vector store interface
- Integration tests against both

---

### Decision 5: LangSmith for Production Observability

**Rationale:**
- Purpose-built for LLM applications
- Automatic tracing for LangChain/LangGraph
- Excellent debugging UI

**Trade-off:**
- Cost at scale
- Vendor lock-in

**Mitigation:**
- Abstract tracing interface
- Cost monitoring and alerts

---

## Future Enhancements

### Phase 3 (Post-Sprint)

1. **Caching Layer:** Redis for frequently-used prompts/responses
2. **Batch Processing:** Queue system for async, non-urgent requests
3. **Multi-tenancy:** Isolated workspaces per user/organization
4. **Custom Models:** Fine-tuned models for domain-specific tasks
5. **GraphQL API:** Flexible querying for frontend clients

### Phase 4 (Production Optimization)

1. **Auto-scaling:** Dynamic capacity based on load
2. **Geographic Distribution:** Edge deployments for low latency
3. **A/B Testing:** Compare prompts, models, strategies
4. **Cost Optimization:** Smaller models for simple tasks
5. **Self-healing:** Automatic recovery from common failures

---

## References

All architectural decisions are informed by patterns documented in the PKB:

- [anthropic_prompt_engineering.md](../docs/PKB/anthropic_prompt_engineering.md) - Prompting strategies
- [anthropic_tool_use.md](../docs/PKB/anthropic_tool_use.md) - Tool calling patterns
- [langchain_agents.md](../docs/PKB/langchain_agents.md) - Agent architectures
- [langgraph_workflows.md](../docs/PKB/langgraph_workflows.md) - State management
- [pydantic_validation.md](../docs/PKB/pydantic_validation.md) - Type safety
- [react_pattern.md](../docs/PKB/react_pattern.md) - Reasoning + acting
- [rag_and_embeddings.md](../docs/PKB/rag_and_embeddings.md) - Vector search
- [agent_testing_evaluation_observability.md](../docs/PKB/agent_testing_evaluation_observability.md) - Testing strategies

---

## Changelog

| Date | Version | Changes |
|------|---------|---------|
| 2025-11-07 | 1.0 | Initial architecture definition |

---

**Questions or feedback?** Open an issue on GitHub or refer to [PROJECT_INSTRUCTIONS_UPDATED.md](../PROJECT_INSTRUCTIONS_UPDATED.md)
