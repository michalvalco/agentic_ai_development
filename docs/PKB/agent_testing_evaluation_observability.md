# Agent Testing, Evaluation & Observability

**Sources:**
- LangSmith Documentation: https://docs.smith.langchain.com/
- LangSmith Evaluation Concepts: https://docs.langchain.com/langsmith/evaluation-concepts
- Phoenix (Arize AI) Documentation: https://docs.arize.com/phoenix
- LangChain Testing Guide: https://python.langchain.com/docs/guides/productionization/evaluation/
- OpenInference Specification: https://github.com/Arize-ai/openinference

**Date Accessed:** 2025-11-07

**Relevance:** This is where theory meets production reality. You can build the most sophisticated agent systemâ€”perfect routing, flawless query writing, elegant tool orchestrationâ€”and it will *still* fail in production if you can't test it, evaluate its quality, observe its behavior, or debug when things go wrong. Testing and observability aren't afterthoughts. They're the difference between a demo that impresses and a system that ships.

---

**Document Statistics:**
- Lines: ~990
- Code Examples: 15 comprehensive patterns
- Common Pitfalls: 10 detailed scenarios
- Integration Points: All 5 capabilities covered
- Our Takeaways: 15 actionable insights
- Implementation Checklist: 6-phase rollout
- Tool Comparisons: LangSmith vs Phoenix, testing approaches

---

## Key Concepts

### The Testing Challenge

**The Problem:** Agents are fundamentally non-deterministic. Same input, different outputs. How do you test that?

Traditional software testing assumes: *f(x) = y*. For any input x, you get output y. Every time.

LLM-based agents break this assumption: *f(x) Ã¢â€°Ë† y*. For input x, you get *approximately* y. Sometimes. If the moon is right.

**This changes everything about testing.**

You can't write assertions like `assert output == "expected_response"`. The LLM might:
- Paraphrase the same idea differently
- Include extra context
- Reorder information
- Use synonyms
- Hallucinate details
- Miss key points

**The Solution:** Test outcomes, not exact outputs. Evaluate quality, not equality.

### The Three Pillars

**1. Testing (Development Phase)**
- Unit tests for individual components
- Integration tests for workflows
- Mocking strategies to avoid LLM costs
- Regression tests to catch prompt changes

**2. Evaluation (Pre-Production)**
- Datasets with ground truth
- Automated evaluation metrics
- LLM-as-judge patterns
- Human review workflows
- A/B testing different versions

**3. Observability (Production)**
- Tracing every step
- Real-time monitoring
- Cost tracking per operation
- Alerting on failures
- Performance profiling

**Critical Understanding:** These aren't sequential phases. You need all three, all the time. Test during development. Evaluate before deploying. Observe in production. And feed production observations back into tests and evaluations.

---

## Implementation Patterns

### Pattern 1: Unit Testing Agent Components

Test individual nodes in isolation *without* calling LLMs.

```python
import pytest
from typing import Dict, Any

# Your agent node
def process_user_query(state: Dict[str, Any]) -> Dict[str, Any]:
    """Extract intent from user query."""
    query = state["query"]
    
    # Business logic (testable)
    if "price" in query.lower() or "cost" in query.lower():
        intent = "pricing"
    elif "how to" in query.lower() or "tutorial" in query.lower():
        intent = "documentation"
    else:
        intent = "general"
    
    return {"intent": intent, "processed": True}

# Unit test
def test_process_user_query_pricing():
    """Test intent detection for pricing queries."""
    state = {"query": "What's the price of your API?"}
    result = process_user_query(state)
    
    assert result["intent"] == "pricing"
    assert result["processed"] is True

def test_process_user_query_documentation():
    """Test intent detection for documentation queries."""
    state = {"query": "How to authenticate with OAuth?"}
    result = process_user_query(state)
    
    assert result["intent"] == "documentation"
    assert result["processed"] is True

def test_process_user_query_general():
    """Test fallback to general intent."""
    state = {"query": "Tell me about your company"}
    result = process_user_query(state)
    
    assert result["intent"] == "general"
    assert result["processed"] is True
```

**Key Insight:** Separate business logic from LLM calls. Test the logic, mock the LLM.

### Pattern 2: Integration Testing with Mocked LLMs

Test workflows end-to-end without real API calls.

```python
import pytest
from unittest.mock import Mock, patch
from langgraph.graph import StateGraph
from typing_extensions import TypedDict

class State(TypedDict):
    query: str
    context: str
    response: str

def retrieve_context(state: State) -> State:
    """Mock-friendly retrieval node."""
    # In production: calls vector database
    # In tests: mocked
    return {"context": "mock_context"}

def generate_response(state: State) -> State:
    """Mock-friendly generation node."""
    # In production: calls LLM
    # In tests: mocked
    return {"response": f"Generated from: {state['context']}"}

@pytest.fixture
def agent_graph():
    """Create test agent graph."""
    graph = StateGraph(State)
    graph.add_node("retrieve", retrieve_context)
    graph.add_node("generate", generate_response)
    graph.add_edge("retrieve", "generate")
    graph.set_entry_point("retrieve")
    graph.set_finish_point("generate")
    return graph.compile()

def test_agent_workflow_integration(agent_graph, monkeypatch):
    """Test full workflow with mocked components."""
    
    # Mock the retrieval
    def mock_retrieve(state):
        return {"context": "API authentication uses OAuth 2.0"}
    
    # Mock the generation
    def mock_generate(state):
        context = state.get("context", "")
        return {"response": f"Based on: {context}"}
    
    # Patch the nodes
    monkeypatch.setattr("retrieve_context", mock_retrieve)
    monkeypatch.setattr("generate_response", mock_generate)
    
    # Run the workflow
    result = agent_graph.invoke({
        "query": "How do I authenticate?",
        "context": "",
        "response": ""
    })
    
    # Verify the pipeline executed correctly
    assert "OAuth 2.0" in result["context"]
    assert "Based on:" in result["response"]
    assert result["response"] != ""
```

**Why This Works:** You're testing the *orchestration* (does retrieve â†’ generate work?) without testing LLM quality (which requires real evals).

### Pattern 3: Testing with Real LLM Calls (Budget-Conscious)

When you need to test actual LLM behavior, do it strategically.

```python
import pytest
import os
from anthropic import Anthropic

# Mark tests that cost money
pytestmark = pytest.mark.llm_integration

@pytest.fixture(scope="session")
def anthropic_client():
    """Shared client for all LLM tests."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        pytest.skip("ANTHROPIC_API_KEY not set")
    return Anthropic(api_key=api_key)

@pytest.mark.llm_integration
def test_intent_classification_real(anthropic_client):
    """Test intent classification with real LLM calls.
    
    Note: This test costs ~$0.001 per run.
    Run sparingly: pytest -m llm_integration
    """
    
    test_cases = [
        ("What's the pricing?", "pricing"),
        ("How do I get started?", "onboarding"),
        ("My API key isn't working", "support"),
    ]
    
    for query, expected_intent in test_cases:
        message = anthropic_client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=50,
            messages=[{
                "role": "user",
                "content": f"Classify this query into one of: pricing, onboarding, support.\n\nQuery: {query}\n\nIntent:"
            }]
        )
        
        response = message.content[0].text.strip().lower()
        
        # Flexible assertion (LLM might say "pricing query" not just "pricing")
        assert expected_intent in response, \
            f"Expected '{expected_intent}' in response, got: {response}"

# Run LLM tests only when explicitly requested
# pytest                        # Skips LLM tests
# pytest -m llm_integration     # Runs LLM tests
```

**Cost Management Strategy:**
```python
# conftest.py
def pytest_configure(config):
    """Configure test markers."""
    config.addinivalue_line(
        "markers", 
        "llm_integration: tests that call real LLMs (expensive, run sparingly)"
    )

# pytest.ini
[pytest]
markers =
    llm_integration: marks tests as integration (expensive)
    
# Run in CI only on main branch
# GitHub Actions:
# - name: Run LLM integration tests
#   if: github.ref == 'refs/heads/main'
#   run: pytest -m llm_integration
```

### Pattern 4: LangSmith Evaluation Setup

Evaluate agent quality systematically.

```python
import os
from langsmith import Client
from langsmith.evaluation import evaluate, LangChainStringEvaluator

# Initialize LangSmith
os.environ["LANGSMITH_API_KEY"] = "your-api-key"
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "agent-evaluations"

client = Client()

# Step 1: Create evaluation dataset
dataset_name = "routing-accuracy-v1"

# Create dataset if it doesn't exist
try:
    dataset = client.read_dataset(dataset_name=dataset_name)
except:
    dataset = client.create_dataset(dataset_name=dataset_name)
    
    # Add examples with ground truth
    examples = [
        {
            "inputs": {"query": "What's the API pricing?"},
            "outputs": {"expected_route": "pricing", "expected_confidence": "high"}
        },
        {
            "inputs": {"query": "How do I set up webhooks?"},
            "outputs": {"expected_route": "documentation", "expected_confidence": "high"}
        },
        {
            "inputs": {"query": "Getting 401 errors"},
            "outputs": {"expected_route": "support", "expected_confidence": "high"}
        },
        {
            "inputs": {"query": "What's your company mission?"},
            "outputs": {"expected_route": "general", "expected_confidence": "medium"}
        },
    ]
    
    client.create_examples(
        dataset_id=dataset.id,
        inputs=[ex["inputs"] for ex in examples],
        outputs=[ex["outputs"] for ex in examples]
    )

# Step 2: Define the system to evaluate
from anthropic import Anthropic

anthropic_client = Anthropic()

def route_query(inputs: dict) -> dict:
    """The routing system we're evaluating."""
    query = inputs["query"]
    
    message = anthropic_client.messages.create(
        model="claude-sonnet-4-5-20250929",
        max_tokens=100,
        messages=[{
            "role": "user",
            "content": f"""Classify this query into one route: pricing, documentation, support, or general.
            
Query: {query}

Respond with just the route name."""
        }]
    )
    
    route = message.content[0].text.strip().lower()
    
    return {
        "route": route,
        "model": "claude-sonnet-4-5"
    }

# Step 3: Define custom evaluator
def route_correctness(run, example):
    """Check if route matches expected route."""
    predicted_route = run.outputs.get("route", "").lower()
    expected_route = example.outputs.get("expected_route", "").lower()
    
    score = 1 if predicted_route == expected_route else 0
    
    return {
        "key": "route_accuracy",
        "score": score,
    }

# Step 4: Run evaluation
results = evaluate(
    route_query,
    data=dataset_name,
    evaluators=[route_correctness],
    experiment_prefix="routing-baseline",
    max_concurrency=2  # Control cost
)

print(f"Accuracy: {results['results'][0]['route_accuracy']:.1%}")
```

**Key Insight:** Build evaluation datasets incrementally. Start with 10-20 examples. Add edge cases as you discover them in production.

### Pattern 5: LLM-as-Judge Evaluation

Use an LLM to evaluate another LLM's outputs.

```python
from langsmith.evaluation import LangChainStringEvaluator
from langchain_anthropic import ChatAnthropic

# Initialize evaluator LLM (use cheaper model)
eval_llm = ChatAnthropic(model="claude-haiku-20250305")

# Built-in evaluator: correctness
correctness_evaluator = LangChainStringEvaluator(
    "cot_qa",  # Chain-of-thought QA evaluation
    config={"llm": eval_llm}
)

# Evaluate with reference answers
results = evaluate(
    your_rag_system,
    data="qa-dataset",
    evaluators=[correctness_evaluator],
    experiment_prefix="rag-quality-check"
)

# Custom LLM-as-judge evaluator
def custom_llm_judge(run, example):
    """Custom evaluation criteria."""
    
    predicted_answer = run.outputs.get("answer", "")
    reference_answer = example.outputs.get("reference_answer", "")
    query = example.inputs.get("query", "")
    
    judge_prompt = f"""Evaluate if the predicted answer correctly addresses the query.

Query: {query}

Reference Answer: {reference_answer}

Predicted Answer: {predicted_answer}

Criteria:
- Factual correctness (does it match reference?)
- Completeness (are key points covered?)
- Conciseness (no unnecessary fluff?)

Score 0-10 and explain why.

Format:
Score: [0-10]
Reasoning: [explanation]
"""
    
    message = eval_llm.invoke(judge_prompt)
    response = message.content
    
    # Parse score
    try:
        score_line = [line for line in response.split('\n') if 'Score:' in line][0]
        score = int(score_line.split(':')[1].strip()) / 10  # Normalize to 0-1
    except:
        score = 0
    
    return {
        "key": "llm_judge_quality",
        "score": score,
        "comment": response
    }
```

**When to Use LLM-as-Judge:**
- Evaluating open-ended generation (summaries, explanations)
- Checking semantic equivalence (not exact match)
- Assessing style/tone compliance
- Detecting hallucinations or factual errors

**When NOT to Use:**
- Evaluating structured outputs (use code-based assertions)
- Cost-sensitive scenarios (LLM judges are expensive)
- When ground truth is available (use exact match)

### Pattern 6: Phoenix Observability Setup

Instrument your agent for development-time tracing.

```python
import os
import phoenix as px
from phoenix.otel import register
from openinference.instrumentation.langchain import LangChainInstrumentor

# Option 1: Local Phoenix (development)
# Start Phoenix server: python -m phoenix.server.main serve
session = px.launch_app()
print(f"Phoenix UI: {session.url}")

# Option 2: Phoenix Cloud (team collaboration)
os.environ["PHOENIX_CLIENT_HEADERS"] = f"api_key={os.getenv('PHOENIX_API_KEY')}"
os.environ["PHOENIX_COLLECTOR_ENDPOINT"] = "https://app.phoenix.arize.com"

# Register tracer
tracer_provider = register(
    project_name="agent-development",
    endpoint="http://localhost:6006/v1/traces"  # Or Phoenix Cloud endpoint
)

# Instrument LangChain (auto-traces all LangChain operations)
LangChainInstrumentor().instrument(tracer_provider=tracer_provider)

# Now run your agent - everything is automatically traced
from langchain_anthropic import ChatAnthropic
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.tools import tool

@tool
def search_docs(query: str) -> str:
    """Search documentation."""
    # Simulated search
    return f"Documentation for: {query}"

llm = ChatAnthropic(model="claude-sonnet-4-5-20250929")

agent = create_tool_calling_agent(
    llm=llm,
    tools=[search_docs],
    prompt="You are a helpful assistant."
)

executor = AgentExecutor(agent=agent, tools=[search_docs])

# This call is automatically traced in Phoenix
result = executor.invoke({"input": "How do I authenticate?"})

# View traces at http://localhost:6006
```

**What You Get:**
- Every LLM call (input, output, latency, tokens)
- Tool invocations (which tools, when, results)
- Chain structure visualization
- Error stack traces
- Token usage per operation

### Pattern 7: Production Monitoring with LangSmith

Monitor live traffic in production.

```python
import os
from langsmith import Client
from langsmith.run_helpers import traceable

# Production configuration
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "production-monitoring"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"

client = Client()

# Decorate production functions
@traceable(
    run_type="chain",
    name="customer_support_agent",
    tags=["production", "support"],
    metadata={"version": "1.2.0"}
)
def handle_support_query(user_id: str, query: str) -> dict:
    """Production support agent with monitoring."""
    
    # All operations inside are traced
    intent = classify_intent(query)
    context = retrieve_context(intent, query)
    response = generate_response(context, query)
    
    return {
        "response": response,
        "intent": intent,
        "user_id": user_id
    }

@traceable(name="intent_classification", run_type="llm")
def classify_intent(query: str) -> str:
    """Classify user intent."""
    # LLM call here - automatically traced
    pass

@traceable(name="context_retrieval", run_type="retriever")
def retrieve_context(intent: str, query: str) -> str:
    """Retrieve relevant context."""
    # RAG retrieval here - automatically traced
    pass

@traceable(name="response_generation", run_type="llm")
def generate_response(context: str, query: str) -> str:
    """Generate final response."""
    # LLM call here - automatically traced
    pass

# Production usage
result = handle_support_query(
    user_id="user_123",
    query="My API key isn't working"
)

# View in LangSmith dashboard:
# - Request traces
# - Latency by operation
# - Token usage
# - Error rates
# - Cost per request
```

**Production Monitoring Checklist:**
- âœ… All user-facing operations traced
- âœ… User IDs logged (for debugging specific issues)
- âœ… Version tags (to compare deployments)
- âœ… Error handling with trace context
- âœ… Cost tracking per user/operation

### Pattern 8: Cost Tracking Implementation

Track costs at every level.

```python
from typing import Dict, Any
from dataclasses import dataclass
from datetime import datetime
import json

@dataclass
class CostMetrics:
    """Track costs for an operation."""
    operation: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    estimated_cost_usd: float
    timestamp: datetime
    
    def to_dict(self) -> dict:
        return {
            "operation": self.operation,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "estimated_cost_usd": self.estimated_cost_usd,
            "timestamp": self.timestamp.isoformat()
        }

class CostTracker:
    """Track and aggregate costs across operations."""
    
    # Pricing (as of Nov 2025 - update regularly!)
    PRICING = {
        "claude-sonnet-4-5-20250929": {
            "input": 0.003 / 1000,   # $3 per 1M tokens
            "output": 0.015 / 1000,  # $15 per 1M tokens
        },
        "claude-haiku-20250305": {
            "input": 0.00025 / 1000,  # $0.25 per 1M tokens
            "output": 0.00125 / 1000, # $1.25 per 1M tokens
        }
    }
    
    def __init__(self):
        self.metrics: list[CostMetrics] = []
    
    def track(self, operation: str, model: str, 
              input_tokens: int, output_tokens: int) -> CostMetrics:
        """Record cost for an operation."""
        
        pricing = self.PRICING.get(model, {"input": 0, "output": 0})
        
        cost = (
            input_tokens * pricing["input"] +
            output_tokens * pricing["output"]
        )
        
        metric = CostMetrics(
            operation=operation,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            estimated_cost_usd=cost,
            timestamp=datetime.now()
        )
        
        self.metrics.append(metric)
        return metric
    
    def get_total_cost(self) -> float:
        """Get total cost across all operations."""
        return sum(m.estimated_cost_usd for m in self.metrics)
    
    def get_cost_by_operation(self) -> Dict[str, float]:
        """Break down cost by operation type."""
        costs = {}
        for m in self.metrics:
            costs[m.operation] = costs.get(m.operation, 0) + m.estimated_cost_usd
        return costs
    
    def export_metrics(self, filepath: str):
        """Export metrics for analysis."""
        with open(filepath, 'w') as f:
            json.dump([m.to_dict() for m in self.metrics], f, indent=2)

# Usage in production
tracker = CostTracker()

def monitored_llm_call(prompt: str, model: str) -> str:
    """LLM call with cost tracking."""
    
    # Make the actual LLM call
    message = anthropic_client.messages.create(
        model=model,
        max_tokens=1000,
        messages=[{"role": "user", "content": prompt}]
    )
    
    # Track costs
    usage = message.usage
    metric = tracker.track(
        operation="user_query_processing",
        model=model,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens
    )
    
    # Log if expensive
    if metric.estimated_cost_usd > 0.10:  # $0.10 threshold
        print(f"âš ï¸ Expensive call: ${metric.estimated_cost_usd:.4f}")
    
    return message.content[0].text

# At end of session/day
print(f"Total cost today: ${tracker.get_total_cost():.2f}")
print("Cost by operation:")
for op, cost in tracker.get_cost_by_operation().items():
    print(f"  {op}: ${cost:.4f}")

tracker.export_metrics("costs_2025-11-07.json")
```

### Pattern 9: Testing Stateful Workflows with Checkpoints

Test agents that maintain state across interactions.

```python
import pytest
from langgraph.graph import StateGraph
from langgraph.checkpoint.memory import MemorySaver
from typing_extensions import TypedDict

class ConversationState(TypedDict):
    messages: list[dict]
    user_context: dict
    conversation_history: list[str]

def test_stateful_conversation_flow():
    """Test multi-turn conversation with state persistence."""
    
    # Create graph with checkpointing
    checkpointer = MemorySaver()
    
    def process_message(state: ConversationState) -> ConversationState:
        """Process a single message."""
        # Simulated processing
        last_message = state["messages"][-1]["content"]
        response = f"Processed: {last_message}"
        
        history = state.get("conversation_history", [])
        history.append(response)
        
        return {"conversation_history": history}
    
    graph = StateGraph(ConversationState)
    graph.add_node("process", process_message)
    graph.set_entry_point("process")
    graph.set_finish_point("process")
    
    app = graph.compile(checkpointer=checkpointer)
    
    # Thread ID for conversation continuity
    config = {"configurable": {"thread_id": "test-conversation-1"}}
    
    # Turn 1
    result1 = app.invoke(
        {
            "messages": [{"role": "user", "content": "Hello"}],
            "conversation_history": []
        },
        config=config
    )
    assert len(result1["conversation_history"]) == 1
    
    # Turn 2 (should remember turn 1)
    result2 = app.invoke(
        {
            "messages": [{"role": "user", "content": "How are you?"}],
        },
        config=config
    )
    assert len(result2["conversation_history"]) == 2
    assert "Hello" in str(result2["conversation_history"])
    
    # Turn 3 (should remember turns 1-2)
    result3 = app.invoke(
        {
            "messages": [{"role": "user", "content": "Goodbye"}],
        },
        config=config
    )
    assert len(result3["conversation_history"]) == 3
    
    # Verify state is actually persisting
    assert result3["conversation_history"][0] == "Processed: Hello"
    assert result3["conversation_history"][1] == "Processed: How are you?"
    assert result3["conversation_history"][2] == "Processed: Goodbye"

def test_checkpoint_recovery():
    """Test that we can recover from a checkpoint."""
    
    checkpointer = MemorySaver()
    # ... graph setup ...
    app = graph.compile(checkpointer=checkpointer)
    
    config = {"configurable": {"thread_id": "recovery-test"}}
    
    # Create some state
    app.invoke({"messages": [{"content": "Step 1"}]}, config=config)
    app.invoke({"messages": [{"content": "Step 2"}]}, config=config)
    
    # Get all checkpoints for this thread
    checkpoints = list(checkpointer.list(config))
    assert len(checkpoints) >= 2
    
    # Can recover state at any checkpoint
    for checkpoint in checkpoints:
        state = checkpointer.get(checkpoint.config)
        assert state is not None
```

### Pattern 10: Testing Human-in-the-Loop Patterns

Test interrupts and human feedback loops.

```python
import pytest
from langgraph.graph import StateGraph
from langgraph.checkpoint.memory import MemorySaver
from typing_extensions import TypedDict

class ApprovalState(TypedDict):
    action: str
    requires_approval: bool
    approved: bool
    result: str

def test_human_approval_workflow():
    """Test workflow that requires human approval."""
    
    def check_if_approval_needed(state: ApprovalState) -> ApprovalState:
        """Determine if action requires approval."""
        risky_keywords = ["delete", "transfer", "refund"]
        action = state["action"].lower()
        
        requires_approval = any(kw in action for kw in risky_keywords)
        
        return {"requires_approval": requires_approval}
    
    def execute_action(state: ApprovalState) -> ApprovalState:
        """Execute the action (only if approved or no approval needed)."""
        if state.get("requires_approval") and not state.get("approved"):
            return {"result": "ERROR: Awaiting approval"}
        
        return {"result": f"Executed: {state['action']}"}
    
    # Create graph with interrupt before sensitive actions
    graph = StateGraph(ApprovalState)
    graph.add_node("check_approval", check_if_approval_needed)
    graph.add_node("execute", execute_action)
    
    graph.add_edge("check_approval", "execute")
    graph.set_entry_point("check_approval")
    graph.set_finish_point("execute")
    
    checkpointer = MemorySaver()
    app = graph.compile(
        checkpointer=checkpointer,
        interrupt_before=["execute"]  # Pause before execution
    )
    
    config = {"configurable": {"thread_id": "approval-test"}}
    
    # Test 1: Safe action (no approval needed)
    result = app.invoke(
        {"action": "Get user profile", "approved": False},
        config=config
    )
    assert "Executed" in result["result"]
    
    # Test 2: Risky action (requires approval)
    config2 = {"configurable": {"thread_id": "approval-test-2"}}
    
    # First invocation - interrupted
    result = app.invoke(
        {"action": "Delete user account", "approved": False},
        config=config2
    )
    # Should be interrupted, not yet executed
    assert result.get("requires_approval") == True
    
    # Human approves (simulate by updating state)
    result = app.invoke(
        {"approved": True},  # Human approval
        config=config2
    )
    assert "Executed" in result["result"]
    assert "Delete user account" in result["result"]

def test_human_feedback_incorporation():
    """Test incorporating human feedback into agent behavior."""
    
    def generate_draft(state: dict) -> dict:
        return {"draft": "Initial draft content"}
    
    def incorporate_feedback(state: dict) -> dict:
        feedback = state.get("human_feedback", "")
        draft = state["draft"]
        
        if feedback:
            revised = f"{draft} [Revised based on: {feedback}]"
            return {"draft": revised, "revision_count": state.get("revision_count", 0) + 1}
        
        return {}
    
    graph = StateGraph(dict)
    graph.add_node("generate", generate_draft)
    graph.add_node("revise", incorporate_feedback)
    
    def should_continue(state: dict) -> str:
        if state.get("human_feedback"):
            return "revise"
        return "END"
    
    graph.add_conditional_edges("generate", should_continue, {
        "revise": "revise",
        "END": "END"
    })
    graph.add_edge("revise", "END")
    graph.set_entry_point("generate")
    
    app = graph.compile()
    
    # No feedback - ends immediately
    result = app.invoke({})
    assert result["draft"] == "Initial draft content"
    assert result.get("revision_count") is None
    
    # With feedback - incorporates it
    result = app.invoke({"human_feedback": "Make it more concise"})
    assert "Revised based on" in result["draft"]
    assert result["revision_count"] == 1
```

### Pattern 11: Regression Testing for Prompts

Catch when prompt changes break existing behavior.

```python
import pytest
from anthropic import Anthropic
import json
from pathlib import Path

class PromptRegressionTest:
    """Test that prompt changes don't break known-good outputs."""
    
    def __init__(self, test_cases_file: str = "prompt_regression_tests.json"):
        self.test_cases_file = Path(test_cases_file)
        self.client = Anthropic()
    
    def save_baseline(self, prompt_version: str, test_cases: list[dict]):
        """Save baseline outputs for regression testing."""
        baselines = []
        
        for case in test_cases:
            message = self.client.messages.create(
                model="claude-sonnet-4-5-20250929",
                max_tokens=500,
                messages=[{"role": "user", "content": case["input"]}]
            )
            
            baselines.append({
                "input": case["input"],
                "expected_output": message.content[0].text,
                "tags": case.get("tags", [])
            })
        
        data = {
            "prompt_version": prompt_version,
            "baselines": baselines
        }
        
        self.test_cases_file.write_text(json.dumps(data, indent=2))
        print(f"Saved {len(baselines)} baseline test cases")
    
    def run_regression_tests(self) -> dict:
        """Run regression tests against saved baselines."""
        if not self.test_cases_file.exists():
            raise FileNotFoundError("No baseline test cases found")
        
        data = json.loads(self.test_cases_file.read_text())
        baselines = data["baselines"]
        
        results = {
            "passed": 0,
            "failed": 0,
            "failures": []
        }
        
        for case in baselines:
            message = self.client.messages.create(
                model="claude-sonnet-4-5-20250929",
                max_tokens=500,
                messages=[{"role": "user", "content": case["input"]}]
            )
            
            actual_output = message.content[0].text
            expected_output = case["expected_output"]
            
            # Semantic similarity check (not exact match)
            is_similar = self._check_semantic_similarity(actual_output, expected_output)
            
            if is_similar:
                results["passed"] += 1
            else:
                results["failed"] += 1
                results["failures"].append({
                    "input": case["input"],
                    "expected": expected_output,
                    "actual": actual_output,
                    "tags": case["tags"]
                })
        
        return results
    
    def _check_semantic_similarity(self, text1: str, text2: str) -> bool:
        """Use LLM to check if outputs are semantically equivalent."""
        
        judge_prompt = f"""Are these two responses semantically equivalent?

Response 1:
{text1}

Response 2:
{text2}

Answer: yes or no"""

        message = self.client.messages.create(
            model="claude-haiku-20250305",  # Cheaper model for judging
            max_tokens=10,
            messages=[{"role": "user", "content": judge_prompt}]
        )
        
        answer = message.content[0].text.lower().strip()
        return "yes" in answer

# Usage
def test_prompt_regression():
    """Integration test that runs prompt regression suite."""
    tester = PromptRegressionTest()
    
    results = tester.run_regression_tests()
    
    # Assert no regressions
    assert results["failed"] == 0, \
        f"Regression test failures:\n" + \
        "\n".join([f"- {f['input']}: {f['tags']}" for f in results["failures"]])
    
    print(f"âœ… All {results['passed']} regression tests passed")

# Create baselines (run once when prompt is working well)
if __name__ == "__main__":
    tester = PromptRegressionTest()
    
    test_cases = [
        {"input": "What's your API pricing?", "tags": ["pricing", "critical"]},
        {"input": "How do I authenticate?", "tags": ["docs", "critical"]},
        {"input": "I'm getting a 500 error", "tags": ["support", "critical"]},
    ]
    
    tester.save_baseline("v1.0", test_cases)
```

### Pattern 12: A/B Testing Different Agent Versions

Compare agent versions systematically.

```python
from langsmith import Client
from langsmith.evaluation import evaluate
from typing import Callable, Dict, Any

client = Client()

def ab_test_agents(
    agent_a: Callable,
    agent_b: Callable,
    dataset_name: str,
    agent_a_name: str = "Agent A",
    agent_b_name: str = "Agent B"
) -> Dict[str, Any]:
    """A/B test two agent versions on same dataset."""
    
    # Run agent A
    results_a = evaluate(
        agent_a,
        data=dataset_name,
        experiment_prefix=f"ab-test-{agent_a_name}",
        max_concurrency=2
    )
    
    # Run agent B
    results_b = evaluate(
        agent_b,
        data=dataset_name,
        experiment_prefix=f"ab-test-{agent_b_name}",
        max_concurrency=2
    )
    
    # Compare metrics
    comparison = {
        "agent_a": {
            "name": agent_a_name,
            "metrics": results_a["results"][0] if results_a["results"] else {}
        },
        "agent_b": {
            "name": agent_b_name,
            "metrics": results_b["results"][0] if results_b["results"] else {}
        },
        "winner": None,
        "improvement": None
    }
    
    # Determine winner (example: by accuracy)
    if comparison["agent_a"]["metrics"] and comparison["agent_b"]["metrics"]:
        score_a = comparison["agent_a"]["metrics"].get("accuracy", 0)
        score_b = comparison["agent_b"]["metrics"].get("accuracy", 0)
        
        if score_b > score_a:
            comparison["winner"] = agent_b_name
            comparison["improvement"] = ((score_b - score_a) / score_a * 100) if score_a > 0 else float('inf')
        elif score_a > score_b:
            comparison["winner"] = agent_a_name
            comparison["improvement"] = ((score_a - score_b) / score_b * 100) if score_b > 0 else float('inf')
        else:
            comparison["winner"] = "Tie"
            comparison["improvement"] = 0
    
    return comparison

# Example: Test two routing strategies
def routing_agent_v1(inputs: dict) -> dict:
    """Baseline routing agent."""
    # Implementation
    pass

def routing_agent_v2(inputs: dict) -> dict:
    """Improved routing agent with better prompts."""
    # Implementation
    pass

results = ab_test_agents(
    routing_agent_v1,
    routing_agent_v2,
    dataset_name="routing-test-set",
    agent_a_name="baseline-v1",
    agent_b_name="improved-v2"
)

print(f"Winner: {results['winner']}")
print(f"Improvement: {results['improvement']:.1f}%")
```

### Pattern 13: Production Alerting Setup

Get notified when things go wrong.

```python
from typing import Dict, Any, Optional
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass, field

@dataclass
class AlertConfig:
    """Configuration for alerting thresholds."""
    error_rate_threshold: float = 0.05  # 5% error rate
    latency_p95_threshold: float = 5.0  # 5 seconds
    cost_per_hour_threshold: float = 10.0  # $10/hour
    tokens_per_request_threshold: int = 10000  # 10k tokens
    alert_cooldown_minutes: int = 15  # Don't spam alerts

@dataclass
class MetricWindow:
    """Sliding window for metrics."""
    values: list = field(default_factory=list)
    window_size: int = 100
    
    def add(self, value: Any):
        self.values.append(value)
        if len(self.values) > self.window_size:
            self.values.pop(0)
    
    def error_rate(self) -> float:
        if not self.values:
            return 0.0
        errors = sum(1 for v in self.values if v.get("error"))
        return errors / len(self.values)
    
    def p95_latency(self) -> float:
        if not self.values:
            return 0.0
        latencies = sorted([v.get("latency", 0) for v in self.values])
        idx = int(len(latencies) * 0.95)
        return latencies[idx] if idx < len(latencies) else latencies[-1]

class ProductionMonitor:
    """Monitor production metrics and alert on issues."""
    
    def __init__(self, config: AlertConfig = None):
        self.config = config or AlertConfig()
        self.metrics = MetricWindow()
        self.last_alert_time: Dict[str, datetime] = {}
        self.logger = logging.getLogger(__name__)
    
    def record_request(
        self,
        success: bool,
        latency: float,
        tokens: int,
        cost: float,
        operation: str,
        error: Optional[str] = None
    ):
        """Record a request and check thresholds."""
        
        metric = {
            "timestamp": datetime.now(),
            "success": success,
            "error": error,
            "latency": latency,
            "tokens": tokens,
            "cost": cost,
            "operation": operation
        }
        
        self.metrics.add(metric)
        
        # Check thresholds
        self._check_error_rate()
        self._check_latency()
        self._check_tokens(tokens)
        
        # Log the request
        if error:
            self.logger.error(f"Request failed: {operation} - {error}")
        else:
            self.logger.info(f"Request succeeded: {operation} ({latency:.2f}s)")
    
    def _check_error_rate(self):
        """Alert if error rate exceeds threshold."""
        error_rate = self.metrics.error_rate()
        
        if error_rate > self.config.error_rate_threshold:
            if self._should_alert("error_rate"):
                self._send_alert(
                    severity="HIGH",
                    message=f"Error rate {error_rate:.1%} exceeds threshold {self.config.error_rate_threshold:.1%}",
                    metric="error_rate",
                    value=error_rate
                )
    
    def _check_latency(self):
        """Alert if p95 latency exceeds threshold."""
        p95 = self.metrics.p95_latency()
        
        if p95 > self.config.latency_p95_threshold:
            if self._should_alert("latency"):
                self._send_alert(
                    severity="MEDIUM",
                    message=f"P95 latency {p95:.2f}s exceeds threshold {self.config.latency_p95_threshold:.2f}s",
                    metric="latency_p95",
                    value=p95
                )
    
    def _check_tokens(self, tokens: int):
        """Alert if single request uses excessive tokens."""
        if tokens > self.config.tokens_per_request_threshold:
            if self._should_alert("high_tokens"):
                self._send_alert(
                    severity="LOW",
                    message=f"Request used {tokens} tokens (threshold: {self.config.tokens_per_request_threshold})",
                    metric="token_usage",
                    value=tokens
                )
    
    def _should_alert(self, alert_type: str) -> bool:
        """Check if we should send alert (respects cooldown)."""
        last_alert = self.last_alert_time.get(alert_type)
        
        if not last_alert:
            return True
        
        cooldown = timedelta(minutes=self.config.alert_cooldown_minutes)
        return datetime.now() - last_alert > cooldown
    
    def _send_alert(self, severity: str, message: str, metric: str, value: Any):
        """Send alert (implement your alerting logic here)."""
        
        # Update last alert time
        self.last_alert_time[metric] = datetime.now()
        
        # Log alert
        self.logger.warning(f"ALERT [{severity}]: {message}")
        
        # TODO: Integrate with your alerting system
        # - Send to Slack
        # - Send to PagerDuty
        # - Send email
        # - Create incident ticket
        
        alert_data = {
            "severity": severity,
            "message": message,
            "metric": metric,
            "value": value,
            "timestamp": datetime.now().isoformat()
        }
        
        # Example: Slack webhook
        # requests.post(SLACK_WEBHOOK_URL, json={"text": message})
        
        print(f"ðŸš¨ ALERT: {alert_data}")

# Production usage
monitor = ProductionMonitor()

def production_agent_call(query: str) -> dict:
    """Production agent with monitoring."""
    start_time = datetime.now()
    
    try:
        # Make agent call
        response = your_agent.invoke(query)
        
        # Calculate metrics
        latency = (datetime.now() - start_time).total_seconds()
        tokens = response.get("total_tokens", 0)
        cost = tokens * 0.000003  # Example pricing
        
        # Record success
        monitor.record_request(
            success=True,
            latency=latency,
            tokens=tokens,
            cost=cost,
            operation="agent_query"
        )
        
        return response
        
    except Exception as e:
        # Record failure
        latency = (datetime.now() - start_time).total_seconds()
        
        monitor.record_request(
            success=False,
            latency=latency,
            tokens=0,
            cost=0,
            operation="agent_query",
            error=str(e)
        )
        
        raise
```

### Pattern 14: Evaluation Dataset Curation from Production

Turn production failures into test cases.

```python
from langsmith import Client
from datetime import datetime, timedelta
from typing import List, Dict

client = Client()

def curate_dataset_from_production(
    project_name: str,
    dataset_name: str,
    filters: Dict = None,
    limit: int = 100
) -> str:
    """Extract production traces and convert to evaluation dataset."""
    
    # Define filters for interesting traces
    default_filters = {
        "error": True,  # Failed requests
        # OR:
        # "feedback_score": {"lt": 0.5},  # Low user ratings
        # OR:
        # "latency": {"gt": 5.0},  # Slow requests
    }
    
    filters = filters or default_filters
    
    # Get traces from production
    runs = client.list_runs(
        project_name=project_name,
        filter=filters,
        limit=limit
    )
    
    # Create dataset
    try:
        dataset = client.read_dataset(dataset_name=dataset_name)
    except:
        dataset = client.create_dataset(dataset_name=dataset_name)
    
    examples = []
    for run in runs:
        # Extract input/output
        inputs = run.inputs
        outputs = run.outputs
        
        # Add metadata about why this was selected
        metadata = {
            "source": "production",
            "run_id": str(run.id),
            "timestamp": run.start_time.isoformat() if run.start_time else None,
            "error": run.error,
            "latency": run.latency,
        }
        
        examples.append({
            "inputs": inputs,
            "outputs": outputs,
            "metadata": metadata
        })
    
    # Upload to dataset
    client.create_examples(
        dataset_id=dataset.id,
        inputs=[ex["inputs"] for ex in examples],
        outputs=[ex["outputs"] for ex in examples],
        metadata=[ex["metadata"] for ex in examples]
    )
    
    print(f"Added {len(examples)} examples to dataset '{dataset_name}'")
    return dataset_name

def curate_edge_cases(
    project_name: str,
    dataset_name: str = "edge-cases-v1"
) -> str:
    """Curate edge cases from production for regression testing."""
    
    # Look for traces with specific characteristics
    interesting_patterns = [
        {"latency": {"gt": 10.0}},  # Very slow
        {"total_tokens": {"gt": 50000}},  # Token-heavy
        {"error": True},  # Errors
        {"feedback_score": {"lt": 0.3}},  # Very low ratings
    ]
    
    all_examples = []
    
    for pattern in interesting_patterns:
        runs = client.list_runs(
            project_name=project_name,
            filter=pattern,
            limit=25
        )
        all_examples.extend(runs)
    
    # Create dataset from edge cases
    dataset_name = curate_dataset_from_production(
        project_name=project_name,
        dataset_name=dataset_name,
        limit=len(all_examples)
    )
    
    return dataset_name

# Usage
# After a week in production
edge_case_dataset = curate_edge_cases(
    project_name="production-agent",
    dataset_name="production-edge-cases-week1"
)

# Run regression tests on these edge cases before next deployment
results = evaluate(
    your_improved_agent,
    data=edge_case_dataset,
    evaluators=[your_evaluators],
    experiment_prefix="pre-deployment-check"
)
```

### Pattern 15: Load Testing Agent Systems

Test performance under load.

```python
import asyncio
import time
from typing import List, Dict, Any
from dataclasses import dataclass
import statistics

@dataclass
class LoadTestResult:
    """Results from load testing."""
    total_requests: int
    successful_requests: int
    failed_requests: int
    avg_latency: float
    p50_latency: float
    p95_latency: float
    p99_latency: float
    requests_per_second: float
    errors: List[str]

async def load_test_agent(
    agent_fn: callable,
    test_queries: List[str],
    concurrent_users: int = 10,
    duration_seconds: int = 60
) -> LoadTestResult:
    """Load test an agent with concurrent requests."""
    
    start_time = time.time()
    end_time = start_time + duration_seconds
    
    results = {
        "latencies": [],
        "errors": [],
        "successful": 0,
        "failed": 0
    }
    
    async def make_request(query: str):
        """Make a single request and record metrics."""
        request_start = time.time()
        
        try:
            # Call agent
            response = await asyncio.to_thread(agent_fn, query)
            
            latency = time.time() - request_start
            results["latencies"].append(latency)
            results["successful"] += 1
            
        except Exception as e:
            latency = time.time() - request_start
            results["latencies"].append(latency)
            results["failed"] += 1
            results["errors"].append(str(e))
    
    async def user_simulation():
        """Simulate a single user making requests."""
        query_idx = 0
        
        while time.time() < end_time:
            query = test_queries[query_idx % len(test_queries)]
            await make_request(query)
            query_idx += 1
            
            # Small delay between requests (simulate thinking time)
            await asyncio.sleep(0.1)
    
    # Run concurrent users
    tasks = [user_simulation() for _ in range(concurrent_users)]
    await asyncio.gather(*tasks)
    
    # Calculate statistics
    total_time = time.time() - start_time
    total_requests = results["successful"] + results["failed"]
    
    latencies_sorted = sorted(results["latencies"])
    
    return LoadTestResult(
        total_requests=total_requests,
        successful_requests=results["successful"],
        failed_requests=results["failed"],
        avg_latency=statistics.mean(latencies_sorted) if latencies_sorted else 0,
        p50_latency=latencies_sorted[len(latencies_sorted)//2] if latencies_sorted else 0,
        p95_latency=latencies_sorted[int(len(latencies_sorted)*0.95)] if latencies_sorted else 0,
        p99_latency=latencies_sorted[int(len(latencies_sorted)*0.99)] if latencies_sorted else 0,
        requests_per_second=total_requests / total_time if total_time > 0 else 0,
        errors=results["errors"][:10]  # First 10 errors
    )

# Run load test
async def run_load_test():
    test_queries = [
        "What's the API pricing?",
        "How do I authenticate?",
        "Show me code examples",
        "I'm getting errors",
    ]
    
    result = await load_test_agent(
        agent_fn=your_agent_function,
        test_queries=test_queries,
        concurrent_users=10,
        duration_seconds=60
    )
    
    print(f"""
Load Test Results:
==================
Total Requests: {result.total_requests}
Successful: {result.successful_requests}
Failed: {result.failed_requests}
Success Rate: {result.successful_requests/result.total_requests*100:.1f}%

Latency:
  Average: {result.avg_latency:.2f}s
  P50: {result.p50_latency:.2f}s
  P95: {result.p95_latency:.2f}s
  P99: {result.p99_latency:.2f}s

Throughput: {result.requests_per_second:.1f} req/s

Errors ({len(result.errors)} shown):
""")
    for error in result.errors:
        print(f"  - {error}")

# Run it
if __name__ == "__main__":
    asyncio.run(run_load_test())
```

---

## Common Pitfalls

### 1. Testing Without LLM Calls (Pure Mocking)

**Problem:** Your tests pass, but the agent fails in production with real LLMs.

```python
# Ã¢Å’ BAD - Everything mocked, never touches real LLM
def test_agent_all_mocked():
    with mock.patch('llm.call') as mock_llm:
        mock_llm.return_value = "Perfect response"
        result = agent.run("query")
        assert result == "Perfect response"
```

**Reality Check:** The mock always returns perfect responses. Real LLMs don't.

**Solution:** Mix of testing strategies.

```python
# Ã¢Å“â€¦ GOOD - Test pyramid
# 1. Unit tests (90%): Mock everything, test logic
def test_routing_logic_unit():
    assert classify_intent("pricing") == "sales"

# 2. Integration tests (9%): Real LLMs, small test set
@pytest.mark.llm_integration
def test_routing_integration():
    result = agent.route("What's the price?")
    assert result["route"] in ["sales", "pricing"]

# 3. E2E tests (1%): Full system, critical paths only
@pytest.mark.e2e
def test_critical_user_flow():
    # Full agent with real LLM + real vector DB + real tools
    pass
```

**Testing Budget:**
- Unit tests: Fast, free, run on every commit
- Integration tests: Slower, ~$0.01 per run, run on PR
- E2E tests: Slowest, ~$0.50 per run, run before deploy

### 2. No Regression Testing for Prompts

**Problem:** You improve a prompt for one case, break three others.

**Why This Happens:** Prompt changes have non-obvious side effects. LLMs are sensitive to phrasing.

**Solution:** Maintain regression test suite.

```python
# Create baseline when prompt works well
test_cases = [
    ("What's the weather?", "external_search"),
    ("Company vacation policy?", "internal_docs"),
    ("Hello", "direct_response"),
]

# Before deploying prompt changes, run regression
for query, expected_route in test_cases:
    actual_route = agent.route(query)
    assert actual_route == expected_route, \
        f"Regression! {query} â†’ {actual_route} (expected {expected_route})"
```

**Automate:** Run regression suite in CI/CD pipeline.

### 3. Ignoring Stochastic Nature in Tests

**Problem:** Test fails randomly because LLM gave different (but valid) answer.

```python
# Ã¢Å’ BAD - Exact match on LLM output
def test_summarization():
    summary = agent.summarize("Long document...")
    assert summary == "This document discusses X, Y, and Z."
    # Fails because LLM said "The document covers X, Y, and Z." instead
```

**Solution:** Test properties, not exact outputs.

```python
# Ã¢Å“â€¦ GOOD - Test properties of output
def test_summarization_properties():
    summary = agent.summarize(long_document)
    
    # Test length constraint
    assert len(summary.split()) < 100, "Summary too long"
    
    # Test content inclusion
    assert "key concept 1" in summary.lower()
    assert "key concept 2" in summary.lower()
    
    # Test no hallucination (nothing not in source)
    # (Requires LLM-as-judge or manual verification)
```

### 4. Not Testing Error Paths

**Problem:** Agent works perfectly in happy path, crashes spectacularly on edge cases.

```python
# Ã¢Å’ BAD - Only test success cases
def test_query_agent():
    response = agent.query("Valid query")
    assert response is not None

# Ã¢Å“â€¦ GOOD - Test error handling
def test_query_agent_comprehensive():
    # Success case
    assert agent.query("Valid query") is not None
    
    # Empty input
    result = agent.query("")
    assert result["error"] == "Empty query"
    
    # Malformed input
    result = agent.query(None)
    assert result["error"] == "Invalid input type"
    
    # Tool failure
    with mock.patch('tool.execute', side_effect=Exception("API down")):
        result = agent.query("Query that needs tool")
        assert "error" in result
        assert "API down" in result["error"]
    
    # Rate limit
    with mock.patch('llm.call', side_effect=RateLimitError()):
        result = agent.query("Any query")
        assert result["error"] == "Rate limited, retry later"
```

**Test Matrix:**
- âœ… Happy path
- âœ… Empty input
- âœ… Malformed input
- âœ… Tool failures
- âœ… LLM failures (rate limits, timeouts)
- âœ… Invalid states
- âœ… Concurrent access

### 5. Evaluating Only on Clean, Perfect Data

**Problem:** Test dataset has perfect formatting, zero typos, clear intent. Production data is a mess.

```python
# Ã¢Å’ BAD - Pristine test set
test_set = [
    "What is the current price of your API service?",
    "How do I authenticate using OAuth 2.0?",
    "Please provide documentation for webhook integration."
]

# Ã¢Å“â€¦ GOOD - Realistic test set
test_set = [
    "wat the price??",  # Typos, bad grammar
    "auth???",  # Minimal query
    "i cant get webhooks working its broken!!!",  # Emotional, vague
    "ä½ ä»¬çš„ API ä»·æ ¼æ˜¯å¤šå°‘?",  # Non-English
    "How much $ for api",  # Mixed symbols
    "",  # Empty
    "a" * 10000,  # Absurdly long
]
```

**Curate from Production:** Export actual user queries as test cases.

### 6. No Cost Tracking in Tests

**Problem:** Test suite costs $50 every run. Nobody runs tests.

**Solution:** Track and budget test costs.

```python
# Track LLM costs in tests
class CostAwareTest:
    total_cost = 0.0
    cost_budget = 1.0  # $1 budget for test suite
    
    @classmethod
    def track_cost(cls, input_tokens: int, output_tokens: int):
        cost = (input_tokens * 0.000003) + (output_tokens * 0.000015)
        cls.total_cost += cost
        
        if cls.total_cost > cls.cost_budget:
            pytest.fail(f"Test suite exceeded cost budget: ${cls.total_cost:.2f}")
    
    @classmethod
    def teardown_class(cls):
        print(f"Total test cost: ${cls.total_cost:.4f}")

# Use in tests
def test_expensive_operation():
    response = llm.invoke(prompt)
    CostAwareTest.track_cost(
        response.usage.input_tokens,
        response.usage.output_tokens
    )
```

**Strategies:**
- Use cheaper models for tests (Haiku instead of Sonnet)
- Cache LLM responses (mock after first real call)
- Limit test dataset size
- Run expensive tests only on main branch

### 7. Not Separating Development and Production Tracing

**Problem:** Development traces pollute production dashboard. Can't find real issues.

**Solution:** Use separate projects/environments.

```python
# Ã¢Å’ BAD - Same project for everything
os.environ["LANGCHAIN_PROJECT"] = "my-agent"  # Dev, test, prod all mixed

# Ã¢Å“â€¦ GOOD - Environment-specific projects
import os

env = os.getenv("ENVIRONMENT", "development")

os.environ["LANGCHAIN_PROJECT"] = f"my-agent-{env}"
# â†’ my-agent-development
# â†’ my-agent-staging  
# â†’ my-agent-production
```

**Benefits:**
- Clear separation
- Different retention policies
- Different alerting thresholds
- Easier debugging

### 8. Metrics Without Context

**Problem:** Dashboard shows "95% success rate" but doesn't show *what* succeeded.

```python
# Ã¢Å’ BAD - Generic metrics
metrics = {"success_rate": 0.95}

# Ã¢Å“â€¦ GOOD - Contextual metrics
metrics = {
    "success_rate_by_intent": {
        "pricing": 0.98,
        "support": 0.87,  # â† Lower! Investigate
        "docs": 0.99
    },
    "success_rate_by_user_type": {
        "enterprise": 0.99,
        "free_tier": 0.92  # â† Lower! Maybe rate limiting?
    },
    "latency_by_operation": {
        "routing": 0.3,
        "retrieval": 2.1,  # â† Slow! Optimize
        "generation": 1.5
    }
}
```

**Break Down Metrics By:**
- Intent/category
- User segment
- Operation type
- Time of day
- Model version
- Geography

### 9. Alerting on Everything (Alert Fatigue)

**Problem:** 50 alerts per day. Team ignores all of them. Critical issue gets missed.

```python
# Ã¢Å’ BAD - Alert on every error
if error:
    send_alert("ERROR OCCURRED!")  # Called 1000x/day

# Ã¢Å“â€¦ GOOD - Alert on patterns, not individual errors
class SmartAlerting:
    def __init__(self):
        self.error_count = 0
        self.window_start = time.time()
    
    def check_error_rate(self, error_occurred: bool):
        # Track errors in 5-minute window
        if time.time() - self.window_start > 300:  # 5 minutes
            error_rate = self.error_count / 300
            
            if error_rate > 0.05:  # 5% error rate
                send_alert(f"High error rate: {error_rate:.1%} in last 5min")
            
            # Reset window
            self.error_count = 0
            self.window_start = time.time()
        
        if error_occurred:
            self.error_count += 1
```

**Alert Only On:**
- Sustained high error rates (not individual errors)
- Latency degradation (not single slow requests)
- Cost spikes (not normal variance)
- Critical path failures (not edge case bugs)

### 10. No Load Testing Before Production

**Problem:** Agent works great with 1 user. Falls over with 10.

**Why:**
- LLM API rate limits
- Vector database connection pools
- Memory leaks
- Concurrency bugs

**Solution:** Load test before deploying.

```python
# Simulate 100 concurrent users
async def load_test():
    tasks = [make_request() for _ in range(100)]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    errors = [r for r in results if isinstance(r, Exception)]
    
    if errors:
        print(f"Failed under load: {len(errors)}/100 requests failed")
        # Don't deploy!
```

**Test:**
- Concurrent users
- Sustained load (not just burst)
- Rate limit handling
- Queue backpressure
- Graceful degradation

---

## Integration Points

### Connection to Our Five Capabilities

Testing and observability apply differently to each capability:

**1. Prompt Routing**

**Testing:**
- Unit test: Does classifier logic work? (without LLM)
- Integration test: Does LLM correctly classify test queries?
- Regression test: Do prompt changes break known-good classifications?

**Evaluation:**
- Accuracy: % of correct routes
- Confidence calibration: High confidence = correct route?
- Edge case handling: Ambiguous queries

**Observability:**
- Route distribution (are some routes never used?)
- Misclassification patterns
- Latency by route

```python
# Testing prompt routing
def test_routing_accuracy():
    dataset = [
        ("pricing question", "sales"),
        ("technical question", "docs"),
    ]
    
    correct = 0
    for query, expected in dataset:
        actual = router.classify(query)
        if actual == expected:
            correct += 1
    
    accuracy = correct / len(dataset)
    assert accuracy > 0.90, f"Routing accuracy {accuracy:.1%} below 90%"
```

**2. Query Writing**

**Testing:**
- Unit test: Does query builder generate valid SQL syntax?
- Integration test: Do generated queries return correct results?
- Safety test: Does it reject SQL injection attempts?

**Evaluation:**
- Query correctness: Does it return expected rows?
- Query efficiency: Is it using indexes, avoiding full scans?
- Error rate: % of malformed queries

**Observability:**
- Query execution time
- Database errors
- Cost per query (for cloud databases)

```python
# Testing query writing
def test_query_generation():
    # Generate query
    query = query_writer.build_query(
        table="users",
        filters={"status": "active"},
        limit=10
    )
    
    # Validate syntax
    assert is_valid_sql(query)
    
    # Test against real database (integration)
    results = db.execute(query)
    assert len(results) <= 10
    assert all(r["status"] == "active" for r in results)
```

**3. Data Processing**

**Testing:**
- Unit test: Do transformations preserve data integrity?
- Property test: Does output always match expected schema?
- Round-trip test: Transform â†’ inverse transform â†’ original?

**Evaluation:**
- Data quality: % of valid outputs
- Schema compliance: Does output match Pydantic models?
- Performance: Throughput (records/sec)

**Observability:**
- Processing errors by step
- Data quality metrics
- Bottleneck identification

```python
# Testing data processing
def test_data_transformation():
    input_data = {"raw_value": "100.50"}
    
    # Transform
    output = processor.clean_and_normalize(input_data)
    
    # Validate schema
    assert isinstance(output["value"], float)
    assert output["value"] == 100.5
    
    # Test error handling
    bad_input = {"raw_value": "invalid"}
    result = processor.clean_and_normalize(bad_input)
    assert result["error"] is not None
```

**4. Tool Orchestration**

**Testing:**
- Unit test: Does orchestrator call tools in correct order?
- Integration test: Do tool chains work end-to-end?
- Failure test: Does it handle tool failures gracefully?

**Evaluation:**
- Tool selection accuracy: Right tool for the job?
- Orchestration efficiency: Minimum necessary tools?
- Recovery rate: % of failures with successful retry

**Observability:**
- Tool usage frequency
- Tool failure rates by type
- Tool latency distribution
- Cost per tool call

```python
# Testing tool orchestration
def test_tool_chain():
    # Mock tools
    with mock.patch('tool_a.execute') as mock_a, \
         mock.patch('tool_b.execute') as mock_b:
        
        mock_a.return_value = "result_a"
        mock_b.return_value = "final_result"
        
        # Run orchestration
        result = orchestrator.run(["tool_a", "tool_b"])
        
        # Verify order
        assert mock_a.called
        assert mock_b.called
        assert mock_a.call_count == 1
        
        # Verify result passed between tools
        assert mock_b.call_args[0][0] == "result_a"
```

**5. Decision Support**

**Testing:**
- Unit test: Does decision logic handle all inputs?
- Simulation test: Do decisions lead to good outcomes?
- Bias test: Are decisions fair across groups?

**Evaluation:**
- Decision quality: % of good recommendations (requires human judgment)
- Explanation quality: Are rationales clear?
- Consistency: Same inputs â†’ same decisions?

**Observability:**
- Decision distribution
- Human override rate (how often do humans disagree?)
- Outcome tracking (were decisions correct in hindsight?)

```python
# Testing decision support
def test_decision_recommendation():
    options = [
        {"name": "Option A", "cost": 100, "risk": "low"},
        {"name": "Option B", "cost": 50, "risk": "high"},
    ]
    
    criteria = {"budget": 75, "risk_tolerance": "low"}
    
    recommendation = decision_support.recommend(options, criteria)
    
    # Should recommend Option B (within budget)
    # But flag high risk as concern
    assert recommendation["choice"] == "Option B"
    assert "high risk" in recommendation["warnings"]
```

---

## Our Takeaways

### For agentic_ai_development

**1. Testing Is Not Optional. It's How You Build Confidence.**

You can't ship agents without testing. Period. The non-deterministic nature of LLMs makes testing *harder*, not less important. Your testing strategy is your safety net.

**Start simple:**
- Day 1: Unit tests for business logic
- Week 1: Integration tests with real LLMs (small test set)
- Month 1: Evaluation datasets with regression testing
- Production: Observability and monitoring

**2. The Test Pyramid for Agents**

Traditional test pyramid still applies, adjusted for LLM costs:

```
        /\
       /E2E\        1% of tests, full system, run before deploy
      /------\
     /  INT  \      9% of tests, real LLMs, run on PR
    /----------\
   /    UNIT    \   90% of tests, mocked, run on every commit
  /--------------\
```

**Don't invert this.** Running E2E tests on every file save will bankrupt you and slow development to a crawl.

**3. LangSmith vs Phoenix: Different Use Cases**

**LangSmith:**
- Best for: Production monitoring, evaluation, team collaboration
- Strengths: Datasets, experiments, human feedback, production scale
- Weaknesses: Paid service, requires cloud connectivity

**Phoenix:**
- Best for: Development debugging, local iteration, experimentation
- Strengths: Open source, runs locally, detailed span analysis
- Weaknesses: Not built for production scale

**Our Strategy:** Use both.
- Phoenix during development (detailed debugging, free, local)
- LangSmith in staging/production (monitoring, evaluation, team visibility)

**4. Evaluation Datasets Are Living Documents**

Your evaluation dataset is never "done." It evolves:

**Week 1:** 10 examples (happy path)
**Month 1:** 50 examples (edge cases discovered)
**Month 3:** 200 examples (production failures converted to tests)
**Month 6:** 500 examples (comprehensive coverage)

**Process:**
1. Start with obvious test cases
2. Add edge cases as you discover them
3. Convert production failures to test cases
4. Curate regularly (remove duplicates, outdated examples)

**5. LLM-as-Judge Is Powerful But Has Limits**

**Use LLM-as-judge for:**
- Semantic similarity (not exact match)
- Open-ended generation quality
- Style/tone compliance
- Hallucination detection

**Don't use LLM-as-judge for:**
- Structured outputs (use assertions)
- Binary facts (use exact match)
- When you have ground truth (use it!)
- Cost-sensitive scenarios (LLM judges are expensive)

**The Judge Can Be Wrong:** Always spot-check LLM judge decisions. Consider human review for high-stakes evaluations.

**6. Cost Tracking Is Non-Negotiable**

If you're not tracking costs, you'll get surprised. Track at every level:

- **Per operation:** What does routing cost vs. generation?
- **Per user:** Power users vs. casual users
- **Per feature:** Which capability is most expensive?
- **Over time:** Trending up or down?

**Set budgets and alerts:**
```python
if cost_per_hour > 10.0:
    alert("Cost spike detected!")
```

**7. Observability Begins Day One, Not When You Have Problems**

Don't wait for production issues to add observability. Instrument from the start:

- **Development:** Phoenix for debugging
- **Staging:** LangSmith for evaluation
- **Production:** LangSmith for monitoring + alerting

**The Earlier, The Better:** Traces from development help you understand normal behavior, making production anomalies easier to spot.

**8. Regression Testing Protects Against Prompt Changes**

Prompt engineering is iterative. You'll change prompts. Each change risks breaking something that worked.

**Protection:**
```python
# Before deploying new prompt
run_regression_tests(new_prompt, baseline_examples)

# If accuracy drops >5%, investigate
if new_accuracy < baseline_accuracy - 0.05:
    print("REGRESSION DETECTED")
    # Don't deploy
```

**9. Error Rates Are More Important Than Individual Errors**

A single error isn't a crisis. A sustained high error rate is.

**Don't alert on:**
- Single failed request
- One slow query
- Individual user complaint

**Do alert on:**
- Error rate >5% for 5 minutes
- P95 latency >2x baseline for 10 minutes
- Cost spike >3x normal

**Pattern Detection > Individual Events**

**10. Testing Human-in-the-Loop Is Tricky**

You can't fully automate testing of human approval flows. But you can test the mechanism:

- Does system correctly pause at approval points?
- Does it resume correctly after approval?
- Does it handle rejection gracefully?
- Are approval contexts clear?

**Test the plumbing, not the human decision.**

**11. Load Testing Reveals Different Bugs Than Functional Testing**

Your agent might work perfectly with 1 user and fail catastrophically with 10.

**What breaks under load:**
- Rate limits
- Connection pools
- Memory leaks
- Race conditions
- Queue backlogs

**Test at 2x expected production load** to have safety margin.

**12. Production Feedback Loop Is Essential**

```
Production â†’ Curate edge cases â†’ Add to test dataset â†’ Evaluate â†’ Deploy â†’ Production
```

This loop is how you improve systematically:

1. **Observe** production traces
2. **Identify** failures and edge cases
3. **Add** to evaluation dataset
4. **Fix** issues
5. **Verify** improvements
6. **Deploy** with confidence

**Break this loop, and you're flying blind.**

**13. Metrics Without Actionability Are Vanity Metrics**

Don't just collect metrics. Use them to make decisions:

**Vanity Metric:** "We have 10,000 traces!"
**Actionable Metric:** "Routing accuracy dropped 5% after prompt change v1.3"

**Each metric should answer:**
- Is this good or bad?
- What's the threshold for concern?
- What action do we take if threshold is breached?

**14. Testing Is Where Theory Meets Production Reality**

All the beautiful architecture, elegant prompts, and sophisticated orchestration mean nothing if:
- Tests don't exist
- Evaluations aren't run
- Observability isn't instrumented
- Production isn't monitored

**Testing isn't overhead. Testing is how you build systems that actually work.**

**15. Invest in Testing Infrastructure Early**

The cost of good testing infrastructure:
- Week 1: $0 (basic unit tests)
- Month 1: ~$100/month (LangSmith, evaluation datasets)
- Month 6: ~$500/month (comprehensive monitoring)

The cost of production failures:
- Lost customers: $$$$$
- Emergency debugging: Engineers working weekends
- Reputation damage: Priceless

**The ROI is obvious. Invest early.**

---

## Implementation Checklist

### Phase 1: Development Testing (Week 1)

**Setup:**
- [ ] pytest installed and configured
- [ ] Test directory structure created
- [ ] CI/CD integration (run tests on PR)

**Basic Tests:**
- [ ] Unit tests for business logic (no LLM calls)
- [ ] Mock-based integration tests
- [ ] Error handling tests

**Coverage Goal:** >80% code coverage for non-LLM code

### Phase 2: LLM Integration Testing (Week 2-3)

**Setup:**
- [ ] Separate test markers (`@pytest.mark.llm_integration`)
- [ ] Cost tracking in tests
- [ ] Test dataset curated (10-20 examples)

**Integration Tests:**
- [ ] Real LLM calls on critical paths
- [ ] Tool integration tests
- [ ] End-to-end workflow tests

**Budget:** <$1 per test run

### Phase 3: Evaluation Framework (Month 1)

**LangSmith Setup:**
- [ ] Account created and API key configured
- [ ] Evaluation datasets created (50+ examples)
- [ ] Custom evaluators defined
- [ ] Baseline experiments run

**Evaluation Types:**
- [ ] Correctness evaluation (LLM-as-judge or ground truth)
- [ ] Performance evaluation (latency, tokens)
- [ ] Cost evaluation

**Cadence:** Run evals weekly, before any deployment

### Phase 4: Development Observability (Month 1-2)

**Phoenix Setup:**
- [ ] Phoenix installed locally
- [ ] Auto-instrumentation configured
- [ ] Team trained on using Phoenix UI

**Instrumentation:**
- [ ] All LLM calls traced
- [ ] Tool calls traced
- [ ] Custom spans for business logic

**Usage:** Daily during active development

### Phase 5: Production Monitoring (Before First Deploy)

**LangSmith Production:**
- [ ] Separate production project
- [ ] All production operations traced
- [ ] User IDs and metadata logged
- [ ] Version tags on traces

**Monitoring:**
- [ ] Cost tracking per operation
- [ ] Latency monitoring
- [ ] Error rate dashboards
- [ ] Alerting configured

**Alerting Thresholds:**
- [ ] Error rate >5% for 5 minutes
- [ ] P95 latency >2x baseline
- [ ] Hourly cost >$X threshold

### Phase 6: Continuous Improvement (Ongoing)

**Feedback Loop:**
- [ ] Weekly review of production traces
- [ ] Edge case curation from production
- [ ] Regression test suite updated monthly
- [ ] A/B tests for major changes

**Optimization:**
- [ ] Performance profiling quarterly
- [ ] Cost optimization quarterly
- [ ] Evaluation dataset refresh quarterly

---

## Testing Strategy

### How to Test Your Testing Infrastructure

Meta-testing: Verify your observability and evaluation systems actually work.

**Test 1: Can you detect a regression?**
```python
def test_regression_detection():
    """Verify evaluation catches when quality drops."""
    
    # Intentionally broken agent
    def broken_agent(inputs):
        return {"answer": "I don't know"}
    
    # Run evaluation
    results = evaluate(broken_agent, data="test-dataset")
    
    # Should show poor performance
    assert results["accuracy"] < 0.3, \
        "Evaluation didn't detect broken agent!"
```

**Test 2: Are traces being recorded?**
```python
def test_tracing_works():
    """Verify LangSmith is capturing traces."""
    
    # Make a traced call
    result = traced_function(test_input)
    
    # Check trace was recorded
    time.sleep(2)  # Allow async trace to upload
    runs = client.list_runs(project_name="test", limit=1)
    assert len(list(runs)) > 0, "No traces recorded!"
```

**Test 3: Do alerts fire?**
```python
def test_alerting():
    """Verify alerting system works."""
    
    # Simulate high error rate
    for _ in range(100):
        monitor.record_request(success=False, ...)
    
    # Check alert was sent
    assert len(alert_system.get_recent_alerts()) > 0, \
        "No alert sent for high error rate!"
```

---

## Tool Comparison

### LangSmith vs Phoenix: When to Use Each

**LangSmith**

**Strengths:**
- Built for production scale
- Excellent dataset management
- Human feedback workflows
- Team collaboration features
- Experiment comparison UI
- Online and offline evaluation

**Weaknesses:**
- Paid service ($0 for limited use, paid beyond)
- Requires cloud connectivity
- Less detailed span analysis than Phoenix

**Best For:**
- Production monitoring
- Team-based evaluation
- Long-term experiment tracking
- When you need human review workflows

**When to Choose LangSmith:**
- You're deploying to production
- Multiple team members need access
- You need to track experiments over time
- Budget allows for paid observability

---

**Phoenix**

**Strengths:**
- Completely open source
- Runs locally (no data leaves your machine)
- Extremely detailed span analysis
- Great for debugging specific issues
- Fast iteration during development

**Weaknesses:**
- Not built for production scale
- Limited team collaboration features
- No built-in evaluation framework (just tracing)
- Local-only by default

**Best For:**
- Development and debugging
- Local experimentation
- Deep dive into specific traces
- When you need complete data privacy

**When to Choose Phoenix:**
- Early development phase
- Need offline/local-first tooling
- Want detailed debugging capabilities
- Free/open-source is required

---

**Our Recommendation: Use Both**

```python
# Development: Phoenix
if environment == "development":
    import phoenix as px
    px.launch_app()
    # Instrument with Phoenix

# Staging/Production: LangSmith  
if environment in ["staging", "production"]:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = f"myapp-{environment}"
    # Instrument with LangSmith
```

**Why Both:**
- Phoenix for deep debugging during development (free, local)
- LangSmith for evaluation and production monitoring (scalable, collaborative)

**You're Not Locked In:** Both use OpenTelemetry, so switching is straightforward.

---

### Testing Approaches: When to Use Each

**Unit Tests**
- **When:** Testing pure logic without LLMs
- **Cost:** Free
- **Speed:** Fast (~seconds)
- **Run:** Every commit
- **Coverage:** 90% of test suite

**Integration Tests (Mocked)**
- **When:** Testing orchestration without LLM costs
- **Cost:** Free
- **Speed:** Fast (~seconds)
- **Run:** Every commit
- **Coverage:** Supplementary to unit tests

**Integration Tests (Real LLMs)**
- **When:** Testing actual LLM behavior
- **Cost:** ~$0.01 per test
- **Speed:** Moderate (~10-30s)
- **Run:** On pull requests
- **Coverage:** 9% of test suite, critical paths

**Evaluation (Datasets)**
- **When:** Systematic quality measurement
- **Cost:** ~$0.10 per run (depends on dataset size)
- **Speed:** Slow (~minutes)
- **Run:** Weekly, before deploys
- **Coverage:** Representative sample of real use cases

**Load Tests**
- **When:** Testing performance under concurrent load
- **Cost:** ~$1-5 per run
- **Speed:** Very slow (~minutes to hours)
- **Run:** Before major releases
- **Coverage:** Production-like scenarios

**E2E Tests**
- **When:** Full system verification
- **Cost:** ~$0.50 per test
- **Speed:** Very slow (~minutes)
- **Run:** Before production deploys only
- **Coverage:** 1% of test suite, critical user flows

---

## Summary

Testing, evaluation, and observability are the bridge between demo and production. Your agent system is only as good as your ability to verify it works, measure its quality, debug when it fails, and monitor its behavior in production.

**The Core Principles:**

1. **Test outcomes, not exact outputs** - LLMs are non-deterministic
2. **Use the right tool for each phase** - Phoenix for dev, LangSmith for production
3. **Build evaluation datasets incrementally** - Start small, grow with production learnings
4. **Track costs everywhere** - Visibility prevents surprises
5. **Automate regression testing** - Protect against prompt changes
6. **Monitor patterns, not individual events** - Error rates > individual errors
7. **Close the feedback loop** - Production insights â†’ test cases â†’ improvements

**The Testing Pyramid:**
- 90% unit tests (fast, free, mocked)
- 9% integration tests (real LLMs, critical paths)
- 1% E2E tests (full system, before deploy)

**The Observability Stack:**
- Development: Phoenix (local debugging)
- Evaluation: LangSmith (datasets, experiments)
- Production: LangSmith (monitoring, alerting)

**The Feedback Loop:**
```
Develop â†’ Test â†’ Evaluate â†’ Deploy â†’ Observe â†’ Curate edge cases â†’ Improve â†’ Repeat
```

Break this loop, and quality degrades. Maintain this loop, and quality improves systematically.

**This is where most agent projects fail:** Not in architecture, not in prompt engineering, not in tool selectionâ€”but in the unglamorous work of testing, evaluation, and monitoring.

**Your advantage:** You're building this infrastructure from day one. Your agents will work in production because you've tested them. You'll know when they fail because you're observing them. You'll improve them systematically because you're measuring them.

**That's the difference between a demo and a product.**