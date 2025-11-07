# LangGraph for Agent Workflows

**Sources:**
- https://langchain-ai.github.io/langgraph/
- https://langchain-ai.github.io/langgraph/concepts/
- https://langchain-ai.github.io/langgraph/tutorials/
- https://blog.langchain.com/langgraph/
- https://blog.langchain.com/langgraph-multi-agent-workflows/
- Multiple technical articles and production examples

**Date Accessed:** November 6, 2025

---

## Relevance Statement

**Critical Understanding:** LangGraph is not another agent libraryâ€”it's the orchestration framework that transforms stateless agents into controllable, production-ready workflows.

**The Problem with Simple Agents:**
- ReAct agents execute once and forget everything
- LangChain chains are linear and can't loop back
- AgentExecutor provides no way to pause or resume
- State management is an afterthought, not a feature
- Human oversight requires hacky workarounds
- Multi-agent coordination is... well, let's just say "challenging"

**What LangGraph Actually Solves:**
- **Cycles and iteration**: Agents can refine their work over multiple steps
- **State persistence**: Everything checkpoints automaticallyâ€”resume months later on a different machine
- **Conditional routing**: Dynamic workflows that branch based on runtime conditions
- **Human-in-the-loop**: First-class support for approval gates and human feedback
- **Multi-agent orchestration**: Supervisor patterns, collaborative workflows, hierarchical teams
- **Production reliability**: Durable execution, error recovery, streaming, debugging tools

**Why This Matters for Our Five Capabilities:**

1. **Prompt Routing**: LangGraph turns one-shot intent classification into iterative refinement with validation loops
2. **Query Writing**: Multi-step query generation with schema validation, error correction, and human review
3. **Data Processing**: Stateful ETL pipelines with retry logic and checkpoint recovery
4. **Tool Orchestration**: Complex tool chains with conditional logic, parallel execution, and fallback handling
5. **Decision Support**: Multi-stage analysis workflows with human approval gates and audit trails

**The Honest Assessment:**
LangGraph adds complexity. It's overkill for simple linear workflows. But when you need cycles, state, conditional routing, or human oversight? It's not just helpfulâ€”it's essential. The question isn't "Can I avoid LangGraph?" It's "At what point does avoiding it cost more than learning it?"

This document will help you answer that question.

---

## Core Concepts

### 1. What LangGraph Actually Is

LangGraph is a **state machine framework for building stateful, multi-agent applications**. Think of it as:
- **Not** a higher-level abstraction over agents (that's LangChain)
- **Not** a visual workflow builder (though LangGraph Studio exists)
- **Yes** a low-level orchestration framework built on graph theory
- **Yes** inspired by Google's Pregel and Apache Beam

**The Mental Model:**
```
Traditional Chain: A â†’ B â†’ C â†’ Done
LangGraph Workflow: A â†’ B â†’ C â†’ (check) â†’ back to A â†’ forward to D â†’ pause for human â†’ resume â†’ E
```

LangGraph excels when workflows need:
- **Loops/cycles** (iterate until quality threshold met)
- **Branching** (route based on runtime state)
- **State persistence** (pause/resume across sessions)
- **Multiple agents** (orchestrate specialists)
- **Human oversight** (approval gates, edits, feedback)

### 2. The Graph Architecture

**Three Core Components:**

```python
# 1. State Schema - What gets passed between nodes
from typing_extensions import TypedDict
from typing import Annotated
from operator import add

class State(TypedDict):
    messages: Annotated[list, add]  # Accumulates messages
    current_step: str               # Simple override
    error_count: int                # Tracks failures
```

```python
# 2. Nodes - Functions that process state
def my_node(state: State) -> dict:
    """Nodes take state, return partial updates"""
    # Do work
    new_msg = "Result from my_node"
    return {"messages": [new_msg], "current_step": "my_node"}
```

```python
# 3. Edges - Control flow between nodes
from langgraph.graph import StateGraph, START, END

workflow = StateGraph(State)
workflow.add_node("node_a", my_node)
workflow.add_edge(START, "node_a")  # Regular edge (unconditional)
workflow.add_edge("node_a", END)    # Another regular edge
```

**Key Insight:** Nodes update state. Edges control flow. State persists. Simple as that.

### 3. State Management and Reducers

**The Default Behavior:**
```python
class State(TypedDict):
    value: int  # No annotation = overwrite behavior

def node(state: State):
    return {"value": 42}  # Replaces whatever was there
```

**Reducers for Accumulation:**
```python
from typing import Annotated
from operator import add

class State(TypedDict):
    values: Annotated[list[int], add]  # Accumulate, don't replace

def node_a(state: State):
    return {"values": [1, 2]}  # Adds to existing list

def node_b(state: State):
    return {"values": [3, 4]}  # Also adds to list
```

**Custom Reducers:**
```python
def merge_dicts(left: dict, right: dict) -> dict:
    """Custom logic for merging"""
    result = left.copy()
    result.update(right)
    return result

class State(TypedDict):
    data: Annotated[dict, merge_dicts]
```

**Critical Understanding:** Reducers control how concurrent node updates combine. Without them, you get `InvalidUpdateError` when parallel nodes update the same key.

### 4. The Compilation Step

```python
# Build the graph
workflow = StateGraph(State)
workflow.add_node("node_a", node_a)
workflow.add_edge(START, "node_a")
workflow.add_edge("node_a", END)

# Compile it (validates structure, creates runtime)
app = workflow.compile()

# Now you can invoke it
result = app.invoke({"messages": ["Hello"]})
```

**What compile() does:**
- Validates graph structure (no orphaned nodes, valid edges)
- Creates Pregel runtime (message-passing execution engine)
- Sets up checkpoint infrastructure
- Enables streaming and debugging

---

## Implementation Patterns

### Pattern 1: Basic Sequential Workflow

```python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

class State(TypedDict):
    input: str
    output: str
    step: str

def process_input(state: State) -> dict:
    """First step: validate and clean input"""
    cleaned = state["input"].strip().lower()
    return {"output": cleaned, "step": "processed"}

def transform(state: State) -> dict:
    """Second step: transform the data"""
    transformed = state["output"].upper()
    return {"output": transformed, "step": "transformed"}

def finalize(state: State) -> dict:
    """Final step: prepare output"""
    final = f"Result: {state['output']}"
    return {"output": final, "step": "finalized"}

# Build workflow
workflow = StateGraph(State)
workflow.add_node("process", process_input)
workflow.add_node("transform", transform)
workflow.add_node("finalize", finalize)

# Connect nodes sequentially
workflow.add_edge(START, "process")
workflow.add_edge("process", "transform")
workflow.add_edge("transform", "finalize")
workflow.add_edge("finalize", END)

# Compile and run
app = workflow.compile()
result = app.invoke({"input": "  hello world  "})
print(result["output"])  # Result: HELLO WORLD
```

**Use Case:** Simple ETL-style workflows where order matters and no branching needed.

### Pattern 2: Conditional Routing

```python
from typing import Literal
from langgraph.graph import StateGraph, START, END

class State(TypedDict):
    user_input: str
    intent: str
    response: str

def classify_intent(state: State) -> dict:
    """Determine what the user wants"""
    text = state["user_input"].lower()
    if "weather" in text:
        intent = "weather"
    elif "news" in text:
        intent = "news"
    else:
        intent = "general"
    return {"intent": intent}

def route_by_intent(state: State) -> Literal["weather", "news", "general"]:
    """Router function for conditional edge"""
    return state["intent"]

def handle_weather(state: State) -> dict:
    return {"response": "It's sunny!"}

def handle_news(state: State) -> dict:
    return {"response": "Breaking news..."}

def handle_general(state: State) -> dict:
    return {"response": "I can help with that."}

# Build workflow with conditional routing
workflow = StateGraph(State)
workflow.add_node("classify", classify_intent)
workflow.add_node("weather", handle_weather)
workflow.add_node("news", handle_news)
workflow.add_node("general", handle_general)

workflow.add_edge(START, "classify")

# Conditional edge: route based on intent
workflow.add_conditional_edges(
    "classify",
    route_by_intent,
    {
        "weather": "weather",
        "news": "news",
        "general": "general"
    }
)

# All handlers go to END
workflow.add_edge("weather", END)
workflow.add_edge("news", END)
workflow.add_edge("general", END)

app = workflow.compile()

# Test different intents
print(app.invoke({"user_input": "What's the weather?"})["response"])  # It's sunny!
print(app.invoke({"user_input": "Latest news?"})["response"])         # Breaking news...
```

**Key Insight:** The router function returns a string. The mapping dict connects that string to the next node. Simple pattern, powerful results.

### Pattern 3: Iterative Refinement with Cycles

```python
from typing import Annotated
from operator import add

class State(TypedDict):
    query: str
    attempts: int
    max_attempts: int
    is_valid: bool
    messages: Annotated[list, add]

def generate_sql(state: State) -> dict:
    """Generate SQL query (simplified for example)"""
    # In production: call LLM to generate SQL
    query = f"SELECT * FROM users WHERE id = {state['attempts']}"
    return {
        "query": query,
        "attempts": state["attempts"] + 1,
        "messages": [f"Generated: {query}"]
    }

def validate_sql(state: State) -> dict:
    """Check if SQL is valid"""
    # In production: actually parse and validate
    is_valid = "WHERE" in state["query"]
    return {
        "is_valid": is_valid,
        "messages": [f"Validation: {'passed' if is_valid else 'failed'}"]
    }

def should_continue(state: State) -> Literal["generate", "end"]:
    """Decision: retry or finish?"""
    if state["is_valid"]:
        return "end"
    if state["attempts"] >= state["max_attempts"]:
        return "end"
    return "generate"

# Build workflow with cycle
workflow = StateGraph(State)
workflow.add_node("generate", generate_sql)
workflow.add_node("validate", validate_sql)

workflow.add_edge(START, "generate")
workflow.add_edge("generate", "validate")

# Conditional edge creates cycle
workflow.add_conditional_edges(
    "validate",
    should_continue,
    {
        "generate": "generate",  # Loop back!
        "end": END
    }
)

app = workflow.compile()

result = app.invoke({
    "query": "",
    "attempts": 0,
    "max_attempts": 3,
    "is_valid": False,
    "messages": []
})

print(result["messages"])
# ['Generated: SELECT * FROM users WHERE id = 0',
#  'Validation: passed']
```

**Critical Pattern:** Cycles enable iterative improvement. The conditional edge creates the loop. The router function determines when to exit. This is impossible with simple chains.

### Pattern 4: Human-in-the-Loop with interrupt()

```python
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import MemorySaver
import sqlite3

class State(TypedDict):
    action: str
    approved: bool
    result: str

def propose_action(state: State) -> dict:
    """Propose an action that needs approval"""
    return {"action": "DELETE FROM users WHERE age < 18"}

def approval_gate(state: State) -> Command:
    """Pause for human approval"""
    # This pauses the graph and waits for human input
    is_approved = interrupt({
        "message": "Approve this action?",
        "action": state["action"]
    })
    
    # Route based on approval
    if is_approved:
        return Command(goto="execute")
    else:
        return Command(goto="cancel")

def execute_action(state: State) -> dict:
    """Execute the approved action"""
    return {"result": f"Executed: {state['action']}"}

def cancel_action(state: State) -> dict:
    """Cancel the action"""
    return {"result": "Action cancelled"}

# Build workflow with human-in-the-loop
workflow = StateGraph(State)
workflow.add_node("propose", propose_action)
workflow.add_node("approval", approval_gate)
workflow.add_node("execute", execute_action)
workflow.add_node("cancel", cancel_action)

workflow.add_edge(START, "propose")
workflow.add_edge("propose", "approval")
workflow.add_edge("execute", END)
workflow.add_edge("cancel", END)

# CRITICAL: Must have checkpointer for interrupts to work
checkpointer = MemorySaver()
app = workflow.compile(checkpointer=checkpointer)

# First invocation: runs until interrupt
config = {"configurable": {"thread_id": "thread-1"}}
result = app.invoke({"action": "", "approved": False, "result": ""}, config=config)

# Check what the graph is asking
print(result["__interrupt__"])  
# [Interrupt(value={'message': 'Approve this action?', 'action': '...'})]

# Resume with approval
result = app.invoke(Command(resume=True), config=config)
print(result["result"])  # Executed: DELETE FROM users WHERE age < 18

# OR: reject by passing False
# result = app.invoke(Command(resume=False), config=config)
```

**Production Pattern:** The `interrupt()` function pauses execution. Graph state saves to checkpointer. Resume later (even months later) with `Command(resume=...)`. This is production-grade HITL.

### Pattern 5: Parallel Execution with Send

```python
from langgraph.graph import StateGraph, START, END, Send
from typing import Annotated
from operator import add

class State(TypedDict):
    tasks: list[str]
    results: Annotated[list[str], add]

def fan_out(state: State) -> list[Send]:
    """Create parallel tasks"""
    # Send each task to worker node
    return [Send("worker", {"task": task}) for task in state["tasks"]]

def worker(state: dict) -> dict:
    """Process one task"""
    task = state["task"]
    result = f"Processed: {task}"
    return {"results": [result]}

def aggregate(state: State) -> dict:
    """Combine results"""
    summary = f"Completed {len(state['results'])} tasks"
    return {"results": [summary]}

# Build workflow with parallel execution
workflow = StateGraph(State)
workflow.add_node("fan_out", fan_out)
workflow.add_node("worker", worker)
workflow.add_node("aggregate", aggregate)

workflow.add_edge(START, "fan_out")
workflow.add_conditional_edges(
    "fan_out",
    lambda s: s,  # Return state as-is (Send objects handle routing)
    ["worker"]    # Possible destinations
)
workflow.add_edge("worker", "aggregate")
workflow.add_edge("aggregate", END)

app = workflow.compile()

result = app.invoke({
    "tasks": ["Task A", "Task B", "Task C"],
    "results": []
})

print(result["results"])
# ['Processed: Task A', 'Processed: Task B', 'Processed: Task C', 'Completed 3 tasks']
```

**Key Insight:** `Send` objects create dynamic parallelism. Each Send spawns a new instance of the target node. Results accumulate via the reducer. This is map-reduce built into the graph.

### Pattern 6: Multi-Agent Supervisor

```python
from typing import Annotated, Sequence
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from operator import add

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add]
    next: str  # Which agent to call next

def create_agent_node(name: str):
    """Factory for agent nodes"""
    def agent_node(state: AgentState) -> dict:
        # In production: this would call actual agent
        result = f"{name} completed task"
        return {"messages": [HumanMessage(content=result)]}
    return agent_node

def supervisor(state: AgentState) -> dict:
    """Supervisor decides which agent to call"""
    messages = state["messages"]
    last_message = messages[-1].content
    
    # Simple routing logic (in production: use LLM)
    if "research" in last_message.lower():
        next_agent = "researcher"
    elif "code" in last_message.lower():
        next_agent = "coder"
    else:
        next_agent = "finish"
    
    return {"next": next_agent}

def route_supervisor(state: AgentState) -> str:
    """Router function for conditional edge"""
    return state["next"]

# Build multi-agent workflow
workflow = StateGraph(AgentState)

# Add agents
workflow.add_node("supervisor", supervisor)
workflow.add_node("researcher", create_agent_node("Researcher"))
workflow.add_node("coder", create_agent_node("Coder"))

# Workflow: start -> supervisor -> agent -> back to supervisor
workflow.add_edge(START, "supervisor")
workflow.add_conditional_edges(
    "supervisor",
    route_supervisor,
    {
        "researcher": "researcher",
        "coder": "coder",
        "finish": END
    }
)

# After agent completes, go back to supervisor
workflow.add_edge("researcher", "supervisor")
workflow.add_edge("coder", "supervisor")

app = workflow.compile()

result = app.invoke({
    "messages": [HumanMessage(content="research the topic")],
    "next": ""
})

print([msg.content for msg in result["messages"]])
# ['research the topic', 'Researcher completed task']
```

**Supervisor Pattern:** One agent orchestrates specialists. Supervisor â†’ agent â†’ back to supervisor. Continues until supervisor routes to END. This is the most common multi-agent pattern.

### Pattern 7: State Checkpointing and Recovery

```python
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3

class State(TypedDict):
    step: int
    data: str

def step_one(state: State) -> dict:
    return {"step": 1, "data": "Step 1 complete"}

def step_two(state: State) -> dict:
    # Simulate error
    if state["step"] == 1:
        raise Exception("Simulated failure")
    return {"step": 2, "data": "Step 2 complete"}

# Build workflow
workflow = StateGraph(State)
workflow.add_node("step_one", step_one)
workflow.add_node("step_two", step_two)
workflow.add_edge(START, "step_one")
workflow.add_edge("step_one", "step_two")
workflow.add_edge("step_two", END)

# Use persistent checkpointer
conn = sqlite3.connect("checkpoints.db")
checkpointer = SqliteSaver(conn)
app = workflow.compile(checkpointer=checkpointer)

# First run: will fail at step_two
config = {"configurable": {"thread_id": "recovery-test"}}
try:
    result = app.invoke({"step": 0, "data": ""}, config=config)
except Exception as e:
    print(f"Failed: {e}")

# Get current state (persisted despite error)
state = app.get_state(config)
print(state.values)  # {'step': 1, 'data': 'Step 1 complete', ...}

# Fix the issue and resume from checkpoint
# (In production: fix the bug, deploy new version)
result = app.invoke(None, config=config)  # Continues from last checkpoint
```

**Production Pattern:** Checkpoints save state after every node. On failure, you can resume from the last successful checkpoint. This is essential for long-running workflows.

### Pattern 8: Streaming Node Outputs

```python
from langgraph.graph import StateGraph, START, END

class State(TypedDict):
    messages: Annotated[list[str], add]

def slow_node(state: State) -> dict:
    """Simulates slow processing with chunks"""
    return {"messages": ["Processing... part 1", "Processing... part 2", "Done!"]}

workflow = StateGraph(State)
workflow.add_node("slow", slow_node)
workflow.add_edge(START, "slow")
workflow.add_edge("slow", END)

app = workflow.compile()

# Stream updates as they happen
for chunk in app.stream({"messages": []}):
    print(chunk)
# {'slow': {'messages': ['Processing... part 1', 'Processing... part 2', 'Done!']}}

# Stream with mode='values' to see full state updates
for state in app.stream({"messages": []}, stream_mode="values"):
    print(state)
```

**Key Feature:** Streaming lets you show progress to users. Essential for long-running operations where users need feedback.

### Pattern 9: Error Handling and Retry

```python
from typing import Literal

class State(TypedDict):
    input: str
    attempts: int
    max_attempts: int
    error: str | None
    result: str | None

def risky_operation(state: State) -> dict:
    """Operation that might fail"""
    try:
        # Simulate: fails first 2 times, succeeds on 3rd
        if state["attempts"] < 2:
            raise Exception(f"Failed attempt {state['attempts'] + 1}")
        
        result = f"Success after {state['attempts'] + 1} attempts"
        return {"result": result, "attempts": state["attempts"] + 1}
    
    except Exception as e:
        return {
            "error": str(e),
            "attempts": state["attempts"] + 1
        }

def should_retry(state: State) -> Literal["retry", "error", "success"]:
    """Decide: retry, fail, or succeed?"""
    if state["result"]:
        return "success"
    if state["attempts"] >= state["max_attempts"]:
        return "error"
    return "retry"

def handle_error(state: State) -> dict:
    return {"result": f"Failed after {state['attempts']} attempts"}

# Build workflow with retry logic
workflow = StateGraph(State)
workflow.add_node("operation", risky_operation)
workflow.add_node("error_handler", handle_error)

workflow.add_edge(START, "operation")
workflow.add_conditional_edges(
    "operation",
    should_retry,
    {
        "retry": "operation",  # Loop back
        "error": "error_handler",
        "success": END
    }
)
workflow.add_edge("error_handler", END)

app = workflow.compile()

result = app.invoke({
    "input": "test",
    "attempts": 0,
    "max_attempts": 3,
    "error": None,
    "result": None
})

print(result["result"])  # Success after 3 attempts
```

**Production Pattern:** Retry logic with exponential backoff. Track attempt count. Route to error handler after max attempts. This pattern prevents infinite loops while enabling resilience.

### Pattern 10: Dynamic Query Planning

```python
from typing import Annotated
from operator import add

class QueryState(TypedDict):
    user_question: str
    selected_tables: list[str]
    schema: dict
    query_draft: str
    is_valid: bool
    refinement_count: int
    messages: Annotated[list, add]

def select_tables(state: QueryState) -> dict:
    """Determine which tables are relevant"""
    # In production: use embeddings/LLM to select
    tables = ["users", "orders"]
    return {
        "selected_tables": tables,
        "messages": [f"Selected tables: {tables}"]
    }

def fetch_schema(state: QueryState) -> dict:
    """Get schema for selected tables"""
    # In production: query actual DB
    schema = {
        "users": {"id": "int", "name": "str"},
        "orders": {"id": "int", "user_id": "int", "total": "float"}
    }
    return {
        "schema": schema,
        "messages": [f"Retrieved schema for {len(state['selected_tables'])} tables"]
    }

def generate_query(state: QueryState) -> dict:
    """Generate SQL query"""
    # In production: call LLM with schema
    query = f"SELECT * FROM {', '.join(state['selected_tables'])}"
    return {
        "query_draft": query,
        "messages": [f"Generated: {query}"]
    }

def validate_query(state: QueryState) -> dict:
    """Validate the query"""
    # In production: parse with sqlglot
    is_valid = "SELECT" in state["query_draft"]
    return {
        "is_valid": is_valid,
        "refinement_count": state["refinement_count"] + 1,
        "messages": [f"Validation: {'passed' if is_valid else 'failed'}"]
    }

def should_refine(state: QueryState) -> Literal["refine", "finish"]:
    """Decide if we need another iteration"""
    if not state["is_valid"] and state["refinement_count"] < 3:
        return "refine"
    return "finish"

# Build query planning workflow
workflow = StateGraph(QueryState)
workflow.add_node("select_tables", select_tables)
workflow.add_node("fetch_schema", fetch_schema)
workflow.add_node("generate", generate_query)
workflow.add_node("validate", validate_query)

# Linear progression through planning stages
workflow.add_edge(START, "select_tables")
workflow.add_edge("select_tables", "fetch_schema")
workflow.add_edge("fetch_schema", "generate")
workflow.add_edge("generate", "validate")

# Conditional edge creates refinement loop
workflow.add_conditional_edges(
    "validate",
    should_refine,
    {
        "refine": "generate",  # Try again
        "finish": END
    }
)

app = workflow.compile()

result = app.invoke({
    "user_question": "Show me all users and their orders",
    "selected_tables": [],
    "schema": {},
    "query_draft": "",
    "is_valid": False,
    "refinement_count": 0,
    "messages": []
})

print("\n".join(result["messages"]))
# Selected tables: ['users', 'orders']
# Retrieved schema for 2 tables
# Generated: SELECT * FROM users, orders
# Validation: passed
```

**For Query Writing Capability:** This pattern shows multi-stage query construction with validation loops. Each stage is a node. Validation creates the refinement cycle.

### Pattern 11: Multi-Turn Conversation with Memory

```python
from typing import Annotated
from operator import add
from langgraph.checkpoint.memory import MemorySaver

class ConversationState(TypedDict):
    messages: Annotated[list[dict], add]
    user_name: str | None
    conversation_summary: str

def extract_user_info(state: ConversationState) -> dict:
    """Extract user information from messages"""
    last_msg = state["messages"][-1]["content"]
    if "my name is" in last_msg.lower():
        name = last_msg.lower().split("my name is")[-1].strip()
        return {"user_name": name}
    return {}

def generate_response(state: ConversationState) -> dict:
    """Generate contextual response"""
    user_name = state.get("user_name")
    if user_name:
        response = f"Nice to meet you, {user_name}!"
    else:
        response = "Hello! What's your name?"
    
    return {"messages": [{"role": "assistant", "content": response}]}

# Build conversation workflow
workflow = StateGraph(ConversationState)
workflow.add_node("extract_info", extract_user_info)
workflow.add_node("respond", generate_response)

workflow.add_edge(START, "extract_info")
workflow.add_edge("extract_info", "respond")
workflow.add_edge("respond", END)

# Use checkpointer for conversation memory
checkpointer = MemorySaver()
app = workflow.compile(checkpointer=checkpointer)

# First turn
config = {"configurable": {"thread_id": "conversation-1"}}
result1 = app.invoke({
    "messages": [{"role": "user", "content": "Hello"}],
    "user_name": None,
    "conversation_summary": ""
}, config=config)

print(result1["messages"][-1]["content"])  # Hello! What's your name?

# Second turn - state persists!
result2 = app.invoke({
    "messages": [{"role": "user", "content": "My name is Alice"}]
}, config=config)

print(result2["messages"][-1]["content"])  # Nice to meet you, alice!
print(result2["user_name"])  # alice
```

**Conversation Pattern:** Checkpointer maintains state across turns. Each invocation adds to messages. Context accumulates. This is production-ready chatbot memory.

### Pattern 12: Hierarchical Multi-Agent Teams

```python
from typing import Annotated, Literal
from operator import add

class TeamState(TypedDict):
    task: str
    research_results: Annotated[list, add]
    code_results: Annotated[list, add]
    final_output: str
    next_agent: str

def research_team_supervisor(state: TeamState) -> dict:
    """Manages research sub-team"""
    # In production: LLM decides which research agent
    return {
        "research_results": ["Research completed"],
        "next_agent": "code_team"
    }

def code_team_supervisor(state: TeamState) -> dict:
    """Manages coding sub-team"""
    # In production: LLM decides which coding agent
    return {
        "code_results": ["Code generated"],
        "next_agent": "synthesizer"
    }

def synthesizer(state: TeamState) -> dict:
    """Combines all results"""
    output = f"Research: {len(state['research_results'])} items\n"
    output += f"Code: {len(state['code_results'])} items"
    return {"final_output": output, "next_agent": "end"}

def route_teams(state: TeamState) -> Literal["research", "code", "synthesize", "end"]:
    """Route between teams"""
    mapping = {
        "research_team": "research",
        "code_team": "code",
        "synthesizer": "synthesize",
        "end": "end"
    }
    return mapping.get(state["next_agent"], "end")

# Build hierarchical workflow
workflow = StateGraph(TeamState)
workflow.add_node("research", research_team_supervisor)
workflow.add_node("code", code_team_supervisor)
workflow.add_node("synthesize", synthesizer)

workflow.add_edge(START, "research")
workflow.add_conditional_edges(
    "research",
    route_teams,
    {
        "research": "research",
        "code": "code",
        "synthesize": "synthesize",
        "end": END
    }
)
workflow.add_conditional_edges(
    "code",
    route_teams,
    {
        "research": "research",
        "code": "code",
        "synthesize": "synthesize",
        "end": END
    }
)
workflow.add_conditional_edges(
    "synthesize",
    route_teams,
    {
        "research": "research",
        "code": "code",
        "synthesize": "synthesize",
        "end": END
    }
)

app = workflow.compile()

result = app.invoke({
    "task": "Build a data pipeline",
    "research_results": [],
    "code_results": [],
    "final_output": "",
    "next_agent": "research_team"
})

print(result["final_output"])
# Research: 1 items
# Code: 1 items
```

**Hierarchical Pattern:** Sub-teams (graphs within graphs) coordinated by supervisors. Each team can be a separate LangGraph. Enables complex organization structures.

### Pattern 13: Async Execution

```python
import asyncio
from langgraph.graph import StateGraph, START, END

class State(TypedDict):
    data: list[str]

async def async_node_a(state: State) -> dict:
    """Async operation"""
    await asyncio.sleep(1)
    return {"data": state["data"] + ["A"]}

async def async_node_b(state: State) -> dict:
    """Another async operation"""
    await asyncio.sleep(1)
    return {"data": state["data"] + ["B"]}

# Build async workflow
workflow = StateGraph(State)
workflow.add_node("node_a", async_node_a)
workflow.add_node("node_b", async_node_b)
workflow.add_edge(START, "node_a")
workflow.add_edge("node_a", "node_b")
workflow.add_edge("node_b", END)

app = workflow.compile()

# Use ainvoke for async execution
async def run():
    result = await app.ainvoke({"data": []})
    print(result["data"])  # ['A', 'B']

asyncio.run(run())
```

**Async Pattern:** All graph operations have async variants (`ainvoke`, `astream`). Mix sync and async nodes. Essential for I/O-bound workflows.

### Pattern 14: Custom State Reducers

```python
from typing import Annotated

def merge_max(left: int, right: int) -> int:
    """Keep maximum value"""
    return max(left, right)

def merge_dict_deep(left: dict, right: dict) -> dict:
    """Deep merge dictionaries"""
    result = left.copy()
    for key, value in right.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_dict_deep(result[key], value)
        else:
            result[key] = value
    return result

class State(TypedDict):
    max_score: Annotated[int, merge_max]
    nested_data: Annotated[dict, merge_dict_deep]

def node_a(state: State) -> dict:
    return {
        "max_score": 50,
        "nested_data": {"user": {"age": 25}}
    }

def node_b(state: State) -> dict:
    return {
        "max_score": 75,  # Will keep 75 (higher)
        "nested_data": {"user": {"name": "Alice"}}  # Will merge
    }

workflow = StateGraph(State)
workflow.add_node("a", node_a)
workflow.add_node("b", node_b)
workflow.add_edge(START, "a")
workflow.add_edge(START, "b")  # Parallel execution
workflow.add_edge("a", END)
workflow.add_edge("b", END)

app = workflow.compile()

result = app.invoke({"max_score": 0, "nested_data": {}})
print(result["max_score"])  # 75
print(result["nested_data"])  # {'user': {'age': 25, 'name': 'Alice'}}
```

**Custom Reducers:** Control exactly how parallel updates combine. Essential for complex state merging logic.

### Pattern 15: Testing Strategy

```python
import pytest
from langgraph.graph import StateGraph, START, END

class State(TypedDict):
    value: int

def increment(state: State) -> dict:
    return {"value": state["value"] + 1}

def double(state: State) -> dict:
    return {"value": state["value"] * 2}

# Test individual nodes
def test_increment_node():
    state = {"value": 5}
    result = increment(state)
    assert result["value"] == 6

def test_double_node():
    state = {"value": 5}
    result = double(state)
    assert result["value"] == 10

# Test full graph
def test_full_workflow():
    workflow = StateGraph(State)
    workflow.add_node("increment", increment)
    workflow.add_node("double", double)
    workflow.add_edge(START, "increment")
    workflow.add_edge("increment", "double")
    workflow.add_edge("double", END)
    
    app = workflow.compile()
    result = app.invoke({"value": 5})
    assert result["value"] == 12  # (5 + 1) * 2

# Test with checkpointer for reproducibility
def test_with_checkpointer():
    from langgraph.checkpoint.memory import MemorySaver
    
    workflow = StateGraph(State)
    workflow.add_node("increment", increment)
    workflow.add_edge(START, "increment")
    workflow.add_edge("increment", END)
    
    checkpointer = MemorySaver()
    app = workflow.compile(checkpointer=checkpointer)
    
    config = {"configurable": {"thread_id": "test-1"}}
    result1 = app.invoke({"value": 5}, config=config)
    result2 = app.invoke({"value": 10}, config=config)
    
    # Each invocation is independent with different thread_id
    assert result1["value"] == 6
    assert result2["value"] == 11
```

**Testing Strategy:**
1. **Unit test nodes** independently (pure functions)
2. **Integration test** full graphs
3. **Use checkpointers** for reproducibility
4. **Mock external calls** in nodes
5. **Test error paths** and retry logic

### Pattern 16: Production Monitoring

```python
import logging
from datetime import datetime
from langgraph.graph import StateGraph, START, END

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class State(TypedDict):
    input: str
    output: str
    metrics: dict

def monitored_node(state: State) -> dict:
    """Node with built-in monitoring"""
    start_time = datetime.now()
    
    try:
        # Do work
        output = state["input"].upper()
        
        # Log success
        duration = (datetime.now() - start_time).total_seconds()
        logger.info(f"Node succeeded in {duration}s")
        
        return {
            "output": output,
            "metrics": {
                "duration": duration,
                "status": "success"
            }
        }
    
    except Exception as e:
        duration = (datetime.now() - start_time).total_seconds()
        logger.error(f"Node failed after {duration}s: {e}")
        
        return {
            "metrics": {
                "duration": duration,
                "status": "error",
                "error": str(e)
            }
        }

# Wrap graph execution with monitoring
def monitored_invoke(app, input_state, config=None):
    """Wrapper that adds monitoring to graph execution"""
    start_time = datetime.now()
    
    try:
        result = app.invoke(input_state, config)
        duration = (datetime.now() - start_time).total_seconds()
        
        logger.info(f"Graph completed in {duration}s")
        return result
    
    except Exception as e:
        duration = (datetime.now() - start_time).total_seconds()
        logger.error(f"Graph failed after {duration}s: {e}")
        raise

workflow = StateGraph(State)
workflow.add_node("process", monitored_node)
workflow.add_edge(START, "process")
workflow.add_edge("process", END)

app = workflow.compile()

# Use monitored invocation
result = monitored_invoke(app, {
    "input": "test",
    "output": "",
    "metrics": {}
})
```

**Production Patterns:**
- Add logging to every node
- Track metrics (duration, errors, retries)
- Use LangSmith for trace visualization
- Monitor checkpoint size (state bloat)
- Alert on retry exhaustion
- Track token usage per node

### Pattern 17: Complex Tool Orchestration

```python
from typing import Annotated, Literal
from operator import add

class ToolState(TypedDict):
    user_query: str
    search_results: Annotated[list, add]
    code_results: Annotated[list, add]
    final_answer: str
    next_action: str

def search_web(state: ToolState) -> dict:
    """Tool: Web search"""
    # In production: actual search API
    results = [f"Search result for: {state['user_query']}"]
    return {
        "search_results": results,
        "next_action": "code"
    }

def run_code(state: ToolState) -> dict:
    """Tool: Code execution"""
    # In production: actual code execution
    results = ["Code executed: result = 42"]
    return {
        "code_results": results,
        "next_action": "synthesize"
    }

def synthesize_answer(state: ToolState) -> dict:
    """Combine tool results into final answer"""
    answer = f"Found {len(state['search_results'])} search results and "
    answer += f"{len(state['code_results'])} code outputs"
    return {
        "final_answer": answer,
        "next_action": "end"
    }

def route_tools(state: ToolState) -> Literal["search", "code", "synthesize", "end"]:
    """Decide which tool to call next"""
    return state["next_action"]

# Build tool orchestration workflow
workflow = StateGraph(ToolState)
workflow.add_node("search", search_web)
workflow.add_node("code", run_code)
workflow.add_node("synthesize", synthesize_answer)

workflow.add_edge(START, "search")
workflow.add_conditional_edges(
    "search",
    route_tools,
    {
        "search": "search",
        "code": "code",
        "synthesize": "synthesize",
        "end": END
    }
)
workflow.add_conditional_edges(
    "code",
    route_tools,
    {
        "search": "search",
        "code": "code",
        "synthesize": "synthesize",
        "end": END
    }
)
workflow.add_conditional_edges(
    "synthesize",
    route_tools,
    {
        "search": "search",
        "code": "code",
        "synthesize": "synthesize",
        "end": END
    }
)

app = workflow.compile()

result = app.invoke({
    "user_query": "What is 2+2?",
    "search_results": [],
    "code_results": [],
    "final_answer": "",
    "next_action": "search"
})

print(result["final_answer"])
# Found 1 search results and 1 code outputs
```

**For Tool Orchestration:** Dynamic tool selection based on state. Tools can call other tools. Supervisor pattern manages the flow. This is production-grade tool chaining.

### Pattern 18: Decision Support with Approval Gates

```python
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import MemorySaver

class DecisionState(TypedDict):
    proposal: dict
    analysis: dict
    approved: bool
    final_decision: str

def analyze_proposal(state: DecisionState) -> dict:
    """Analyze the proposal"""
    proposal = state["proposal"]
    analysis = {
        "risk_level": "medium",
        "estimated_cost": 50000,
        "estimated_benefit": 100000,
        "recommendation": "Proceed with caution"
    }
    return {"analysis": analysis}

def approval_gate(state: DecisionState) -> Command:
    """Human approval required"""
    # Show analysis to human
    is_approved = interrupt({
        "message": "Review this decision",
        "proposal": state["proposal"],
        "analysis": state["analysis"]
    })
    
    if is_approved:
        return Command(goto="execute")
    else:
        return Command(goto="reject")

def execute_decision(state: DecisionState) -> dict:
    """Execute the approved decision"""
    return {
        "approved": True,
        "final_decision": f"Executed: {state['proposal']['action']}"
    }

def reject_decision(state: DecisionState) -> dict:
    """Reject the decision"""
    return {
        "approved": False,
        "final_decision": "Decision rejected by human reviewer"
    }

# Build decision support workflow
workflow = StateGraph(DecisionState)
workflow.add_node("analyze", analyze_proposal)
workflow.add_node("approval", approval_gate)
workflow.add_node("execute", execute_decision)
workflow.add_node("reject", reject_decision)

workflow.add_edge(START, "analyze")
workflow.add_edge("analyze", "approval")
workflow.add_edge("execute", END)
workflow.add_edge("reject", END)

checkpointer = MemorySaver()
app = workflow.compile(checkpointer=checkpointer)

# Run until approval gate
config = {"configurable": {"thread_id": "decision-1"}}
result = app.invoke({
    "proposal": {"action": "Invest $50k in new project"},
    "analysis": {},
    "approved": False,
    "final_decision": ""
}, config=config)

# Human reviews analysis, approves
result = app.invoke(Command(resume=True), config=config)
print(result["final_decision"])  # Executed: Invest $50k in new project
```

**For Decision Support:** Multi-stage analysis â†’ human review â†’ execution. Audit trail via checkpoints. Can pause for days/weeks. This is enterprise-grade decision workflow.

### Pattern 19: Graph Visualization

```python
from IPython.display import Image, display
from langgraph.graph import StateGraph, START, END

class State(TypedDict):
    value: int

def node_a(state: State) -> dict:
    return {"value": state["value"] + 1}

def node_b(state: State) -> dict:
    return {"value": state["value"] * 2}

def router(state: State) -> str:
    return "node_b" if state["value"] < 10 else "end"

workflow = StateGraph(State)
workflow.add_node("node_a", node_a)
workflow.add_node("node_b", node_b)

workflow.add_edge(START, "node_a")
workflow.add_conditional_edges(
    "node_a",
    router,
    {
        "node_b": "node_b",
        "end": END
    }
)
workflow.add_edge("node_b", "node_a")  # Creates cycle

app = workflow.compile()

# Visualize the graph
display(Image(app.get_graph().draw_mermaid_png()))

# Or get Mermaid syntax
print(app.get_graph().draw_mermaid())
```

**Visualization Benefits:**
- **Debug complex flows** visually
- **Share designs** with non-technical stakeholders
- **Document workflows** automatically
- **Identify bottlenecks** and cycles
- **LangGraph Studio** provides interactive debugging

### Pattern 20: Production Deployment Pattern

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.postgres import PostgresSaver
import os

# Define API models
class QueryRequest(BaseModel):
    query: str
    thread_id: str

class QueryResponse(BaseModel):
    result: str
    thread_id: str

# Define state
class State(TypedDict):
    query: str
    result: str

def process_query(state: State) -> dict:
    """Process the query"""
    # In production: actual LLM call
    result = f"Processed: {state['query']}"
    return {"result": result}

# Build workflow
workflow = StateGraph(State)
workflow.add_node("process", process_query)
workflow.add_edge(START, "process")
workflow.add_edge("process", END)

# Use Postgres checkpointer for production
connection_string = os.getenv("POSTGRES_URL")
checkpointer = PostgresSaver.from_conn_string(connection_string)
app_graph = workflow.compile(checkpointer=checkpointer)

# FastAPI app
app = FastAPI()

@app.post("/query", response_model=QueryResponse)
async def handle_query(request: QueryRequest):
    """Handle query requests"""
    try:
        config = {"configurable": {"thread_id": request.thread_id}}
        result = app_graph.invoke(
            {"query": request.query, "result": ""},
            config=config
        )
        return QueryResponse(
            result=result["result"],
            thread_id=request.thread_id
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

# Run with: uvicorn main:app --host 0.0.0.0 --port 8000
```

**Production Checklist:**
- âœ… Use Postgres/Redis checkpointer (not MemorySaver)
- âœ… Add health check endpoints
- âœ… Implement proper error handling
- âœ… Use environment variables for config
- âœ… Add logging and metrics
- âœ… Rate limiting on endpoints
- âœ… Authentication/authorization
- âœ… Graceful shutdown handling
- âœ… Connection pooling for DB
- âœ… Async operations where possible

---

## Common Pitfalls

### 1. Forgetting the Checkpointer for Interrupts

**Problem:**
```python
workflow = StateGraph(State)
workflow.add_node("approval", approval_gate)
# ... build graph ...

app = workflow.compile()  # NO CHECKPOINTER!

result = app.invoke(state)  # interrupt() will fail
```

**Error:** `ValueError: interrupt() requires a checkpointer`

**Solution:**
```python
from langgraph.checkpoint.memory import MemorySaver

checkpointer = MemorySaver()
app = workflow.compile(checkpointer=checkpointer)
```

**Why:** `interrupt()` needs persistence to save state while waiting for human input. No checkpointer = no way to save/resume.

### 2. Invalid Update Error (Missing Reducers)

**Problem:**
```python
class State(TypedDict):
    results: list  # NO REDUCER!

def node_a(state):
    return {"results": [1, 2]}

def node_b(state):
    return {"results": [3, 4]}  # Conflict!

# If node_a and node_b run in parallel...
workflow.add_edge(START, "node_a")
workflow.add_edge(START, "node_b")  # Parallel execution!
```

**Error:** `InvalidUpdateError: Multiple nodes updated 'results' without reducer`

**Solution:**
```python
from typing import Annotated
from operator import add

class State(TypedDict):
    results: Annotated[list, add]  # Now it accumulates
```

**Why:** Parallel nodes updating the same key need a reducer to resolve conflicts. Otherwise, LangGraph doesn't know which update wins.

### 3. Infinite Loops

**Problem:**
```python
def should_continue(state: State) -> str:
    # BUG: Always returns "continue"
    return "continue"

workflow.add_conditional_edges(
    "node_a",
    should_continue,
    {
        "continue": "node_a",  # Loops forever!
        "end": END
    }
)
```

**Error:** `GraphRecursionError: Maximum recursion depth exceeded`

**Solution:**
```python
def should_continue(state: State) -> str:
    if state["attempts"] >= state["max_attempts"]:
        return "end"
    if state["is_valid"]:
        return "end"
    return "continue"

# OR: Set recursion limit in compile
app = workflow.compile(recursion_limit=10)
```

**Why:** Cycles are powerful but dangerous. Always have exit conditions. Set recursion limits as a safety net.

### 4. State Serialization Failures

**Problem:**
```python
from datetime import datetime

class State(TypedDict):
    timestamp: datetime  # NOT JSON-serializable!
    callback: callable   # DEFINITELY not serializable!
```

**Error:** `TypeError: Object of type datetime is not JSON serializable`

**Solution:**
```python
class State(TypedDict):
    timestamp: str  # Use ISO format strings
    callback_name: str  # Store name, not function

def node(state: State):
    ts = datetime.fromisoformat(state["timestamp"])
    # Use timestamp
    return {"timestamp": datetime.now().isoformat()}
```

**Why:** Checkpointers serialize state to JSON. Use Pydantic models with proper serialization, or stick to JSON-compatible types.

### 5. Conditional Edge Without Mapping

**Problem:**
```python
def router(state: State) -> str:
    return "next_node"

workflow.add_conditional_edges(
    "source_node",
    router
    # Missing the mapping dict!
)
```

**Error:** Graph adds conditional edges to ALL nodes, causing chaos.

**Solution:**
```python
workflow.add_conditional_edges(
    "source_node",
    router,
    {
        "next_node": "next_node",
        "other": "other_node"
    }
)

# OR: Use a list of possible destinations
workflow.add_conditional_edges(
    "source_node",
    router,
    ["next_node", "other_node", END]
)
```

**Why:** Without a mapping, LangGraph can't validate the routing. Explicit is better than implicit.

### 6. Mutating State Instead of Returning Updates

**Problem:**
```python
def bad_node(state: State):
    state["value"] = 42  # MUTATING DIRECTLY!
    # No return statement
```

**Result:** State doesn't update. Node appears to do nothing.

**Solution:**
```python
def good_node(state: State) -> dict:
    return {"value": 42}  # Return partial update
```

**Why:** LangGraph expects nodes to return state updates, not mutate the input. Immutable patterns prevent bugs.

### 7. Forgetting thread_id in Config

**Problem:**
```python
# First call
result1 = app.invoke(state, config={"configurable": {"thread_id": "1"}})

# Second call - BUG: no thread_id!
result2 = app.invoke(state)  # Where does this go?
```

**Result:** Each call creates a new thread. State doesn't persist between calls.

**Solution:**
```python
config = {"configurable": {"thread_id": "user-123"}}

# All calls with same thread_id share state
result1 = app.invoke(state1, config=config)
result2 = app.invoke(state2, config=config)
```

**Why:** thread_id is how checkpointer groups related invocations. No thread_id = no conversation history.

### 8. Using Send to END

**Problem:**
```python
def fan_out(state: State):
    return [
        Send("worker", {"task": "A"}),
        Send(END, {"result": "done"})  # INVALID!
    ]
```

**Error:** `InvalidUpdateError: Cannot Send to END`

**Solution:**
```python
def fan_out(state: State):
    if state["should_finish"]:
        return END  # Regular transition to END
    return [Send("worker", {"task": t}) for t in state["tasks"]]
```

**Why:** `Send` is for dynamic node routing, not for termination. Use regular conditional edges to END.

### 9. Overwriting State Accidentally

**Problem:**
```python
class State(TypedDict):
    data: dict  # No reducer

def node_a(state: State):
    return {"data": {"key1": "value1"}}

def node_b(state: State):
    return {"data": {"key2": "value2"}}  # Overwrites key1!
```

**Result:** `key1` is lost when `node_b` runs after `node_a`.

**Solution:**
```python
def merge_dicts(left: dict, right: dict) -> dict:
    result = left.copy()
    result.update(right)
    return result

class State(TypedDict):
    data: Annotated[dict, merge_dicts]
```

**Why:** Default behavior is overwrite. Use reducers or merge manually in nodes.

### 10. Not Handling Node Errors

**Problem:**
```python
def risky_node(state: State) -> dict:
    # API call that might fail
    data = external_api.call()  # No error handling!
    return {"data": data}
```

**Result:** Graph crashes. No recovery. Poor user experience.

**Solution:**
```python
def risky_node(state: State) -> dict:
    try:
        data = external_api.call()
        return {"data": data, "error": None}
    except Exception as e:
        return {"data": None, "error": str(e)}

def should_retry(state: State) -> str:
    if state["error"] and state["attempts"] < 3:
        return "retry"
    if state["error"]:
        return "error_handler"
    return "success"
```

**Why:** External calls fail. Network is unreliable. Handle errors explicitly or they'll crash your graph.

---

## Integration Points

### 1. Prompt Routing

**The Problem:** Simple intent classification is one-shot. What if the intent is ambiguous?

**LangGraph Solution:**

```python
class RoutingState(TypedDict):
    user_input: str
    intent: str | None
    confidence: float
    refinement_attempts: int
    clarification_question: str | None

def classify_intent(state: RoutingState) -> dict:
    """Initial classification"""
    # In production: call LLM for classification
    text = state["user_input"].lower()
    if "weather" in text:
        return {"intent": "weather", "confidence": 0.9}
    elif "news" in text:
        return {"intent": "news", "confidence": 0.85}
    else:
        return {"intent": "general", "confidence": 0.5}

def check_confidence(state: RoutingState) -> Literal["route", "clarify", "refine"]:
    """Decide if we need more info"""
    if state["confidence"] > 0.8:
        return "route"
    if state["refinement_attempts"] < 2:
        return "clarify"
    return "route"  # Give up, use best guess

def ask_clarification(state: RoutingState) -> dict:
    """Generate clarification question"""
    question = f"Did you mean {state['intent']}? Or something else?"
    return {
        "clarification_question": question,
        "refinement_attempts": state["refinement_attempts"] + 1
    }

# Build iterative routing workflow
workflow = StateGraph(RoutingState)
workflow.add_node("classify", classify_intent)
workflow.add_node("clarify", ask_clarification)

workflow.add_edge(START, "classify")
workflow.add_conditional_edges(
    "classify",
    check_confidence,
    {
        "route": END,
        "clarify": "clarify",
        "refine": "classify"
    }
)
workflow.add_edge("clarify", END)  # Return question to user
```

**Value:** Ambiguous intents get clarified. Confidence thresholds gate routing decisions. Iterative refinement replaces one-shot classification.

### 2. Query Writing

**The Problem:** SQL generation is rarely perfect on first try. Need validation and iterative improvement.

**LangGraph Solution:**

```python
class QueryState(TypedDict):
    user_question: str
    selected_tables: list[str]
    schema: dict
    query_draft: str
    validation_errors: Annotated[list, add]
    is_valid: bool
    iteration_count: int

def select_tables(state: QueryState) -> dict:
    """Embed question, search schema, select tables"""
    # In production: use embeddings
    tables = ["users", "orders", "products"]
    return {"selected_tables": tables}

def fetch_schema(state: QueryState) -> dict:
    """Get full schema for selected tables"""
    # In production: query information_schema
    schema = {
        "users": {"id": "int", "name": "varchar", "email": "varchar"},
        "orders": {"id": "int", "user_id": "int", "total": "decimal"}
    }
    return {"schema": schema}

def generate_query(state: QueryState) -> dict:
    """Generate SQL with LLM"""
    # In production: structured prompt with schema
    prompt = f"Schema: {state['schema']}\nQuestion: {state['user_question']}"
    query = "SELECT u.name, o.total FROM users u JOIN orders o ON u.id = o.user_id"
    return {"query_draft": query}

def validate_query(state: QueryState) -> dict:
    """Parse and validate SQL"""
    errors = []
    # In production: use sqlglot
    if "JOIN" not in state["query_draft"]:
        errors.append("Missing expected JOIN")
    
    return {
        "is_valid": len(errors) == 0,
        "validation_errors": errors,
        "iteration_count": state["iteration_count"] + 1
    }

def should_refine(state: QueryState) -> Literal["refine", "approve", "fail"]:
    """Decide next action"""
    if state["is_valid"]:
        return "approve"
    if state["iteration_count"] >= 3:
        return "fail"
    return "refine"

def approval_gate(state: QueryState):
    """Human reviews query before execution"""
    approved = interrupt({
        "query": state["query_draft"],
        "message": "Approve this query?"
    })
    return Command(goto="execute" if approved else "fail")

# Build query workflow
workflow = StateGraph(QueryState)
workflow.add_node("select_tables", select_tables)
workflow.add_node("fetch_schema", fetch_schema)
workflow.add_node("generate", generate_query)
workflow.add_node("validate", validate_query)
workflow.add_node("approve", approval_gate)

workflow.add_edge(START, "select_tables")
workflow.add_edge("select_tables", "fetch_schema")
workflow.add_edge("fetch_schema", "generate")
workflow.add_edge("generate", "validate")

# Refinement loop
workflow.add_conditional_edges(
    "validate",
    should_refine,
    {
        "refine": "generate",  # Try again with errors as context
        "approve": "approve",   # Human review
        "fail": END
    }
)

from langgraph.checkpoint.memory import MemorySaver
checkpointer = MemorySaver()
app = workflow.compile(checkpointer=checkpointer)
```

**Value:** Multi-stage pipeline with validation loops. Human approval gate before execution. Errors feed back into generation. This is production-ready query generation.

### 3. Data Processing

**The Problem:** ETL fails partway through. Need to resume from checkpoint, not start over.

**LangGraph Solution:**

```python
class ETLState(TypedDict):
    source_file: str
    extracted_data: list
    transformed_data: list
    load_status: str
    error_log: Annotated[list, add]
    current_stage: str

def extract_data(state: ETLState) -> dict:
    """Extract data from source"""
    try:
        # In production: actual file reading
        data = [{"id": 1, "value": "raw"}]
        return {
            "extracted_data": data,
            "current_stage": "extracted"
        }
    except Exception as e:
        return {"error_log": [f"Extract failed: {e}"]}

def transform_data(state: ETLState) -> dict:
    """Transform extracted data"""
    try:
        transformed = [
            {**row, "value": row["value"].upper()}
            for row in state["extracted_data"]
        ]
        return {
            "transformed_data": transformed,
            "current_stage": "transformed"
        }
    except Exception as e:
        return {"error_log": [f"Transform failed: {e}"]}

def load_data(state: ETLState) -> dict:
    """Load data to destination"""
    try:
        # In production: database insert
        return {
            "load_status": "success",
            "current_stage": "loaded"
        }
    except Exception as e:
        return {"error_log": [f"Load failed: {e}"]}

def should_continue(state: ETLState) -> Literal["extract", "transform", "load", "error", "success"]:
    """Route based on current stage"""
    if state["error_log"]:
        return "error"
    
    stage_map = {
        "": "extract",
        "extracted": "transform",
        "transformed": "load",
        "loaded": "success"
    }
    return stage_map.get(state["current_stage"], "error")

# Build ETL workflow
workflow = StateGraph(ETLState)
workflow.add_node("extract", extract_data)
workflow.add_node("transform", transform_data)
workflow.add_node("load", load_data)

workflow.add_edge(START, "extract")
workflow.add_conditional_edges(
    "extract",
    should_continue,
    {
        "transform": "transform",
        "error": END
    }
)
workflow.add_conditional_edges(
    "transform",
    should_continue,
    {
        "load": "load",
        "error": END
    }
)
workflow.add_conditional_edges(
    "load",
    should_continue,
    {
        "success": END,
        "error": END
    }
)

# Use persistent checkpointer
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3

conn = sqlite3.connect("etl_checkpoints.db")
checkpointer = SqliteSaver(conn)
app = workflow.compile(checkpointer=checkpointer)

# If ETL fails midway, resume from checkpoint
config = {"configurable": {"thread_id": "etl-run-1"}}
try:
    result = app.invoke({
        "source_file": "data.csv",
        "extracted_data": [],
        "transformed_data": [],
        "load_status": "",
        "error_log": [],
        "current_stage": ""
    }, config=config)
except Exception as e:
    print(f"ETL failed: {e}")
    # Fix issue, then resume
    result = app.invoke(None, config=config)  # Continues from last checkpoint
```

**Value:** Checkpoints after each stage. On failure, resume from last successful step. No need to reprocess already-completed work.

### 4. Tool Orchestration

**The Problem:** Need to chain multiple tools with conditional logic and fallbacks.

**LangGraph Solution:**

```python
class ToolState(TypedDict):
    user_query: str
    search_attempted: bool
    search_results: str | None
    code_attempted: bool
    code_results: str | None
    synthesis: str | None
    tool_errors: Annotated[list, add]

def try_search(state: ToolState) -> dict:
    """Attempt web search"""
    try:
        # In production: actual search API
        results = f"Search results for: {state['user_query']}"
        return {
            "search_attempted": True,
            "search_results": results
        }
    except Exception as e:
        return {
            "search_attempted": True,
            "tool_errors": [f"Search failed: {e}"]
        }

def try_code_execution(state: ToolState) -> dict:
    """Attempt code execution"""
    try:
        # In production: sandboxed execution
        results = "Code executed: 42"
        return {
            "code_attempted": True,
            "code_results": results
        }
    except Exception as e:
        return {
            "code_attempted": True,
            "tool_errors": [f"Code execution failed: {e}"]
        }

def synthesize_results(state: ToolState) -> dict:
    """Combine all tool results"""
    parts = []
    if state["search_results"]:
        parts.append(f"Search: {state['search_results']}")
    if state["code_results"]:
        parts.append(f"Code: {state['code_results']}")
    if not parts:
        parts.append("No results available")
    
    return {"synthesis": "\n".join(parts)}

def route_tools(state: ToolState) -> Literal["search", "code", "synthesize", "fallback"]:
    """Decide which tool to try next"""
    # Try search first
    if not state["search_attempted"]:
        return "search"
    
    # If search failed, try code
    if not state["search_results"] and not state["code_attempted"]:
        return "code"
    
    # If we have any results, synthesize
    if state["search_results"] or state["code_results"]:
        return "synthesize"
    
    # All tools failed
    return "fallback"

def fallback_handler(state: ToolState) -> dict:
    """Handle case where all tools failed"""
    return {
        "synthesis": f"Unable to complete query. Errors: {state['tool_errors']}"
    }

# Build tool orchestration workflow
workflow = StateGraph(ToolState)
workflow.add_node("search", try_search)
workflow.add_node("code", try_code_execution)
workflow.add_node("synthesize", synthesize_results)
workflow.add_node("fallback", fallback_handler)

workflow.add_edge(START, "search")
workflow.add_conditional_edges(
    "search",
    route_tools,
    {
        "search": "search",
        "code": "code",
        "synthesize": "synthesize",
        "fallback": "fallback"
    }
)
workflow.add_conditional_edges(
    "code",
    route_tools,
    {
        "search": "search",
        "code": "code",
        "synthesize": "synthesize",
        "fallback": "fallback"
    }
)
workflow.add_edge("synthesize", END)
workflow.add_edge("fallback", END)
```

**Value:** Sequential tool attempts with fallbacks. Conditional logic routes based on success/failure. Graceful degradation when tools fail.

### 5. Decision Support

**The Problem:** Complex decisions need multi-stage analysis, human review, and audit trails.

**LangGraph Solution:**

```python
class DecisionState(TypedDict):
    proposal: dict
    risk_analysis: dict
    cost_analysis: dict
    stakeholder_feedback: Annotated[list, add]
    final_decision: str | None
    decision_rationale: str
    audit_log: Annotated[list, add]

def analyze_risks(state: DecisionState) -> dict:
    """Risk assessment"""
    proposal = state["proposal"]
    # In production: LLM-powered analysis
    analysis = {
        "risk_level": "medium",
        "key_risks": ["market volatility", "technical debt"]
    }
    return {
        "risk_analysis": analysis,
        "audit_log": ["Risk analysis completed"]
    }

def analyze_costs(state: DecisionState) -> dict:
    """Cost-benefit analysis"""
    # In production: financial models
    analysis = {
        "initial_cost": 100000,
        "estimated_roi": 2.5,
        "payback_period": "18 months"
    }
    return {
        "cost_analysis": analysis,
        "audit_log": ["Cost analysis completed"]
    }

def gather_feedback(state: DecisionState):
    """Human stakeholder input"""
    feedback = interrupt({
        "message": "Review analysis and provide feedback",
        "risk_analysis": state["risk_analysis"],
        "cost_analysis": state["cost_analysis"]
    })
    return {
        "stakeholder_feedback": [feedback],
        "audit_log": ["Stakeholder feedback received"]
    }

def make_recommendation(state: DecisionState) -> dict:
    """Generate recommendation"""
    # Synthesize all analyses
    if state["risk_analysis"]["risk_level"] == "high":
        decision = "reject"
        rationale = "Risk level too high"
    elif state["cost_analysis"]["estimated_roi"] < 1.5:
        decision = "reject"
        rationale = "ROI below threshold"
    else:
        decision = "approve"
        rationale = "Favorable risk/reward profile"
    
    return {
        "final_decision": decision,
        "decision_rationale": rationale,
        "audit_log": ["Recommendation generated"]
    }

def executive_approval(state: DecisionState):
    """Final approval gate"""
    approved = interrupt({
        "message": "Executive approval required",
        "recommendation": state["final_decision"],
        "rationale": state["decision_rationale"]
    })
    
    result = "approved" if approved else "rejected"
    return {
        "final_decision": result,
        "audit_log": [f"Executive decision: {result}"]
    }

# Build decision workflow
workflow = StateGraph(DecisionState)
workflow.add_node("risk", analyze_risks)
workflow.add_node("cost", analyze_costs)
workflow.add_node("feedback", gather_feedback)
workflow.add_node("recommend", make_recommendation)
workflow.add_node("executive", executive_approval)

# Parallel analysis
workflow.add_edge(START, "risk")
workflow.add_edge(START, "cost")

# Wait for both to complete
workflow.add_edge(["risk", "cost"], "feedback")
workflow.add_edge("feedback", "recommend")
workflow.add_edge("recommend", "executive")
workflow.add_edge("executive", END)

from langgraph.checkpoint.memory import MemorySaver
checkpointer = MemorySaver()
app = workflow.compile(checkpointer=checkpointer)
```

**Value:** Multi-stage analysis with parallel execution. Multiple approval gates. Complete audit trail via accumulated logs. Can pause between stages for days/weeks.

---

## Our Takeaways

### For agentic_ai_development

**1. LangGraph Is Not Always the Answer**

Simple workflows don't need LangGraph. If you have:
- Linear sequence (A â†’ B â†’ C â†’ Done)
- No branching or cycles
- No state persistence needs
- No human-in-the-loop

Then stick with LangChain chains or simple function calls. Don't add complexity you don't need.

**When you *do* need LangGraph:**
- Iterative refinement (cycles)
- Conditional routing (branching)
- State persistence (checkpoints)
- Human approval gates
- Multi-agent coordination
- Error recovery (resume from failure)

**2. State Design Is Critical**

Your state schema IS your API contract. Design it carefully:
- **Keep it minimal** â€“ Don't track what you don't need
- **Use TypedDict or Pydantic** â€“ Type safety catches bugs early
- **Add reducers explicitly** â€“ Don't guess how updates should combine
- **Think about serialization** â€“ Only JSON-compatible types or custom serializers

Bad state design = painful debugging later.

**3. Checkpointers Are Not Optional in Production**

MemorySaver is great for testing. Production needs:
- **PostgresSaver** for durability
- **Connection pooling** for performance
- **Cleanup strategy** for old checkpoints
- **Monitoring** for checkpoint size

Checkpoint bloat is real. State grows over time. Plan for it.

**4. Test Nodes Independently**

Nodes are pure functions (or should be). Test them:
```python
def test_my_node():
    state = {"value": 5}
    result = my_node(state)
    assert result["value"] == 10
```

Don't only test the full graph. Unit test nodes. Integration test graphs. Saves debugging time.

**5. Error Handling Is Not Automatic**

LangGraph doesn't catch node errors by default. You must:
- Wrap external calls in try/except
- Return error state instead of raising
- Use conditional routing to error handlers
- Set retry logic explicitly

Silent failures are worse than crashes. Log everything.

**6. Streaming Is Production Gold**

Users hate waiting with no feedback. Use streaming:
```python
for chunk in app.stream(state):
    # Send to user incrementally
```

Show progress. Show thinking. Keep users engaged.

**7. Interrupt Is More Powerful Than You Think**

Don't just use it for "approve/reject". Use it for:
- **Form filling** (ask user for missing info)
- **Clarification** (ambiguous intent? ask)
- **Review** (show draft, get edits)
- **Escalation** (agent stuck? get human help)

Human-in-the-loop is not just approval gates. It's collaborative problem-solving.

**8. Visualization Saves Debugging Hours**

Always visualize your graph:
```python
app.get_graph().draw_mermaid_png()
```

What looks good on paper might be a mess in practice. Visualize early, visualize often.

**9. Reducer Bugs Are Subtle**

`InvalidUpdateError` is one of the most common errors. Understand reducers:
- **No annotation = overwrite** (last write wins)
- **With reducer = accumulate** (combines updates)
- **Custom reducers** for complex merging

Parallel execution + no reducer = errors. Plan for concurrency.

**10. Thread IDs Are Your Friend**

Thread IDs organize conversations:
- **User ID** for per-user state
- **Session ID** for per-session state
- **Task ID** for per-task state

Choose your thread ID strategy early. Changing it later is painful.

**11. LangGraph Studio Is Worth Using**

The desktop app is genuinely useful:
- **Visual debugging** (see execution in real-time)
- **State inspection** (drill into any step)
- **Time travel** (replay from any checkpoint)
- **Breakpoints** (pause and inspect)

Not a gimmick. Actually helps debug complex graphs.

**12. Start Simple, Add Complexity Gradually**

Don't build the full multi-agent supervisor workflow on day one:
1. Start with basic sequential workflow
2. Add conditional routing
3. Add a single cycle
4. Add checkpointing
5. Add human-in-the-loop
6. Add multi-agent coordination

Build incrementally. Test each addition. Complex graphs are hard to debug.

**13. Production Needs More Than Just the Graph**

Don't forget:
- **Authentication** (who can invoke?)
- **Rate limiting** (prevent abuse)
- **Monitoring** (track execution time, errors)
- **Alerting** (notify on failures)
- **Cost tracking** (LLM calls add up)
- **Graceful degradation** (fallbacks for failures)

LangGraph handles orchestration. You handle everything else.

**14. Document Your State Updates**

When nodes return `{"field": value}`, it's not obvious how it merges with existing state. Add comments:
```python
def my_node(state: State) -> dict:
    """Updates status and adds message to history (via reducer)"""
    return {
        "status": "complete",  # Overwrites
        "messages": ["Done"]   # Appends via add reducer
    }
```

Future you will thank current you.

**15. The Functional API Is Simpler for Some Tasks**

LangGraph also has a Functional API (different from StateGraph):
```python
from langgraph.func import entrypoint, task

@task
def process(input: str) -> str:
    return input.upper()

@entrypoint
def workflow(input: str) -> str:
    result = process(input).result()
    return result
```

Simpler for linear tasks. Less control than graphs. Know both APIs.

---

## Implementation Checklist

### Phase 1: Setup (Day 1)

- [ ] Install LangGraph: `pip install langgraph`
- [ ] Install checkpointer deps: `pip install langgraph-checkpoint-postgres` or `langgraph-checkpoint-sqlite`
- [ ] Set up development database (Postgres or SQLite)
- [ ] Create first simple graph (3 nodes, sequential)
- [ ] Verify basic invoke() works
- [ ] Add visualization to jupyter notebook

### Phase 2: Core Patterns (Week 1)

- [ ] Build workflow with conditional routing
- [ ] Add cycle for iterative refinement
- [ ] Implement error handling with retry logic
- [ ] Add MemorySaver checkpointer
- [ ] Test state persistence across invocations
- [ ] Implement streaming outputs
- [ ] Add comprehensive logging

### Phase 3: Advanced Features (Week 2)

- [ ] Implement human-in-the-loop with interrupt()
- [ ] Build multi-agent supervisor pattern
- [ ] Add parallel execution with Send
- [ ] Implement custom state reducers
- [ ] Test checkpoint recovery after failure
- [ ] Add LangSmith tracing
- [ ] Visualize and optimize graph structure

### Phase 4: Integration (Week 3)

- [ ] Integrate with Prompt Routing capability
- [ ] Integrate with Query Writing capability
- [ ] Integrate with Data Processing capability
- [ ] Integrate with Tool Orchestration capability
- [ ] Integrate with Decision Support capability
- [ ] Build end-to-end examples for each
- [ ] Document failure modes for each integration

### Phase 5: Production Prep (Week 4)

- [ ] Switch to PostgresSaver with connection pooling
- [ ] Add FastAPI endpoints (if web service)
- [ ] Implement authentication and rate limiting
- [ ] Add comprehensive error monitoring
- [ ] Set up alerting for failures
- [ ] Add cost tracking for LLM calls
- [ ] Load test with production-like data
- [ ] Document deployment process

### Phase 6: Testing (Week 5)

- [ ] Unit tests for all nodes
- [ ] Integration tests for full graphs
- [ ] Test checkpoint recovery
- [ ] Test state serialization/deserialization
- [ ] Test concurrent execution
- [ ] Test interrupt/resume cycles
- [ ] Test error handling and retries
- [ ] Test with real LLM calls (not mocks)

### Phase 7: Documentation (Week 6)

- [ ] Document state schema for each workflow
- [ ] Create graph visualizations
- [ ] Document routing logic
- [ ] Add troubleshooting guide
- [ ] Document when to use LangGraph vs alternatives
- [ ] Create runnable examples
- [ ] Add performance benchmarks
- [ ] Document cost analysis

---

## Testing Strategy

### 1. Unit Testing Nodes

```python
import pytest

def test_node_basic_behavior():
    """Test node logic without graph"""
    state = {"value": 5}
    result = my_node(state)
    assert result["value"] == 10
    assert "processed" in result

def test_node_error_handling():
    """Test node handles errors gracefully"""
    state = {"value": None}
    result = my_node(state)
    assert "error" in result
    assert result["error"] is not None

def test_node_state_updates():
    """Test reducer behavior"""
    from operator import add
    
    state = {"items": [1, 2]}
    result = my_node(state)
    # Node should append, not replace
    assert len(result["items"]) > 0
```

### 2. Testing Routing Logic

```python
def test_router_function():
    """Test conditional edge router"""
    # Test "continue" path
    state = {"attempts": 1, "max_attempts": 3}
    assert should_continue(state) == "continue"
    
    # Test "end" path
    state = {"attempts": 3, "max_attempts": 3}
    assert should_continue(state) == "end"
    
    # Test "error" path
    state = {"error": "Failed", "attempts": 1}
    assert should_continue(state) == "error"
```

### 3. Integration Testing Graphs

```python
def test_full_workflow():
    """Test complete graph execution"""
    workflow = build_test_workflow()
    app = workflow.compile()
    
    result = app.invoke({"input": "test"})
    
    assert result["status"] == "complete"
    assert result["output"] is not None
    assert len(result["messages"]) > 0

def test_workflow_with_cycle():
    """Test iterative refinement"""
    workflow = build_refinement_workflow()
    app = workflow.compile()
    
    result = app.invoke({
        "query": "",
        "attempts": 0,
        "max_attempts": 3,
        "is_valid": False
    })
    
    # Should iterate until valid or max attempts
    assert result["attempts"] <= 3
    assert result["is_valid"] or result["attempts"] == 3
```

### 4. Testing Checkpointing

```python
def test_checkpoint_persistence():
    """Test state persists between invocations"""
    from langgraph.checkpoint.memory import MemorySaver
    
    workflow = build_workflow()
    checkpointer = MemorySaver()
    app = workflow.compile(checkpointer=checkpointer)
    
    config = {"configurable": {"thread_id": "test-1"}}
    
    # First invocation
    result1 = app.invoke({"count": 0}, config=config)
    assert result1["count"] == 1
    
    # Second invocation - should see previous state
    result2 = app.invoke({"count": 0}, config=config)
    assert result2["count"] == 2  # Accumulated

def test_checkpoint_recovery():
    """Test recovery from failure"""
    workflow = build_failing_workflow()
    checkpointer = MemorySaver()
    app = workflow.compile(checkpointer=checkpointer)
    
    config = {"configurable": {"thread_id": "test-2"}}
    
    # First run fails
    with pytest.raises(Exception):
        app.invoke({"step": 0}, config=config)
    
    # Get state after failure
    state = app.get_state(config)
    assert state.values["step"] == 1  # Partial progress saved
    
    # Resume should succeed
    result = app.invoke(None, config=config)
    assert result["step"] == 2
```

### 5. Testing Human-in-the-Loop

```python
def test_interrupt_pattern():
    """Test interrupt and resume"""
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.types import Command
    
    workflow = build_approval_workflow()
    checkpointer = MemorySaver()
    app = workflow.compile(checkpointer=checkpointer)
    
    config = {"configurable": {"thread_id": "test-3"}}
    
    # Run until interrupt
    result = app.invoke({"action": "test"}, config=config)
    assert "__interrupt__" in result
    
    # Resume with approval
    result = app.invoke(Command(resume=True), config=config)
    assert result["approved"] is True
    assert "executed" in result["status"].lower()

def test_interrupt_rejection():
    """Test rejection path"""
    # Similar to above but resume with False
    result = app.invoke(Command(resume=False), config=config)
    assert result["approved"] is False
    assert "rejected" in result["status"].lower()
```

### 6. Testing Error Handling

```python
def test_node_error_recovery():
    """Test retry logic"""
    workflow = build_retry_workflow()
    app = workflow.compile()
    
    result = app.invoke({
        "attempts": 0,
        "max_attempts": 3,
        "should_fail": True
    })
    
    # Should retry up to max_attempts
    assert result["attempts"] == 3
    assert result["status"] == "failed"

def test_error_routing():
    """Test conditional routing on errors"""
    workflow = build_error_routing_workflow()
    app = workflow.compile()
    
    # Test error path
    result = app.invoke({"inject_error": True})
    assert result["status"] == "error_handled"
    
    # Test success path
    result = app.invoke({"inject_error": False})
    assert result["status"] == "success"
```

### 7. Testing Parallel Execution

```python
def test_parallel_nodes():
    """Test concurrent execution with reducers"""
    workflow = build_parallel_workflow()
    app = workflow.compile()
    
    result = app.invoke({"items": []})
    
    # All parallel nodes should have contributed
    assert len(result["items"]) >= 3
    # Order may vary due to parallelism
    assert set(result["items"]) == {"A", "B", "C"}

def test_send_dynamic_parallelism():
    """Test Send for map-reduce"""
    workflow = build_send_workflow()
    app = workflow.compile()
    
    result = app.invoke({
        "tasks": ["Task 1", "Task 2", "Task 3"],
        "results": []
    })
    
    assert len(result["results"]) == 3
```

### 8. Performance Testing

```python
import time

def test_execution_time():
    """Test graph completes within time limit"""
    workflow = build_workflow()
    app = workflow.compile()
    
    start = time.time()
    result = app.invoke({"input": "test"})
    duration = time.time() - start
    
    assert duration < 5.0  # 5 second limit

def test_checkpoint_overhead():
    """Test checkpointing doesn't add excessive overhead"""
    from langgraph.checkpoint.memory import MemorySaver
    
    workflow = build_workflow()
    
    # Without checkpointer
    app_no_cp = workflow.compile()
    start = time.time()
    app_no_cp.invoke({"input": "test"})
    time_no_cp = time.time() - start
    
    # With checkpointer
    app_with_cp = workflow.compile(checkpointer=MemorySaver())
    start = time.time()
    app_with_cp.invoke({"input": "test"})
    time_with_cp = time.time() - start
    
    # Overhead should be minimal
    overhead = time_with_cp - time_no_cp
    assert overhead < 0.1  # Less than 100ms overhead
```

---

## Comparison to Alternatives

### LangGraph vs. LangChain Chains

**LangChain Chains (LCEL):**
```python
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Sequential, linear flow
chain = prompt | llm | output_parser
result = chain.invoke({"input": "question"})
```

**Pros:**
- âœ… Simpler for linear workflows
- âœ… Less code for basic cases
- âœ… Familiar to LangChain users
- âœ… Good for prototyping

**Cons:**
- âŒ No cycles (can't loop back)
- âŒ Limited branching (no complex conditionals)
- âŒ No persistent state across invocations
- âŒ No human-in-the-loop support
- âŒ Hard to debug multi-step failures

**When to Use Chains:**
- RAG pipelines (retrieve â†’ summarize â†’ answer)
- Simple Q&A bots
- One-shot tool calls
- Prototyping agent ideas

**When to Upgrade to LangGraph:**
- Need iterative refinement
- Need approval gates
- Need state persistence
- Need error recovery
- Need multi-agent coordination

**Migration Path:**
```python
# Start with chain for prototype
chain = prompt | llm | parser

# Migrate to LangGraph when you need more
workflow = StateGraph(State)
workflow.add_node("llm", lambda s: llm.invoke(s))
workflow.add_edge(START, "llm")
workflow.add_edge("llm", END)
app = workflow.compile()
```

### LangGraph vs. Simple ReAct Agents

**Simple Agent:**
```python
from langgraph.prebuilt import create_react_agent

agent = create_react_agent(model, tools)
result = agent.invoke({"messages": [HumanMessage("task")]})
```

**Pros:**
- âœ… Zero setup (one function call)
- âœ… Good for simple tool use
- âœ… Built-in ReAct loop
- âœ… Good for demos

**Cons:**
- âŒ No control over loop logic
- âŒ Can't customize routing
- âŒ Black box (hard to debug)
- âŒ No approval gates
- âŒ Limited to single agent

**When to Use Simple Agents:**
- Quick demos
- Simple tool-calling tasks
- Exploration/prototyping
- When default ReAct loop works

**When to Build Custom Graph:**
- Need custom routing logic
- Need multi-agent coordination
- Need approval gates
- Need specific error handling
- Need state persistence

### LangGraph vs. Custom Orchestration

**Custom Code:**
```python
def my_workflow(input):
    step1 = process_input(input)
    step2 = call_llm(step1)
    if needs_retry(step2):
        step2 = call_llm(step1)  # Manual retry
    step3 = format_output(step2)
    return step3
```

**Pros:**
- âœ… Full control
- âœ… No framework overhead
- âœ… Easy to understand (it's just code)
- âœ… No new concepts to learn

**Cons:**
- âŒ No checkpointing (you build it)
- âŒ No visualization (you build it)
- âŒ No streaming (you build it)
- âŒ No observability (you build it)
- âŒ Boilerplate for every workflow

**When to Use Custom:**
- One-off scripts
- Very simple workflows
- Don't want dependencies
- Full control paramount

**When to Use LangGraph:**
- Multiple workflows to build
- Need checkpointing/recovery
- Need human-in-the-loop
- Need observability
- Want to iterate quickly

### LangGraph vs. Autogen/CrewAI

**Autogen:**
- Conversation-based multi-agent
- Agents chat to coordinate
- More autonomous, less controlled

**CrewAI:**
- Task-based multi-agent
- Pre-defined agent roles
- Higher-level abstractions

**LangGraph:**
- Graph-based orchestration
- Explicit control flow
- Lower-level primitives

**Key Difference:**
LangGraph gives you **explicit control** over routing. Autogen/CrewAI are more **autonomous**. Trade explicitness for autonomy.

**When to Use LangGraph:**
- Need deterministic routing
- Need audit trails
- Need approval gates
- Need fine-grained control
- Building production systems

**When to Consider Alternatives:**
- Want fully autonomous agents
- Comfortable with less control
- Exploring agent-to-agent communication
- Research/experimentation

---

## Summary

LangGraph transforms stateless agents into controllable, production-ready workflows through:

1. **State machines** (graphs with nodes and edges)
2. **Persistent state** (checkpoints after every step)
3. **Conditional routing** (dynamic branching at runtime)
4. **Cycles** (iterative refinement loops)
5. **Human-in-the-loop** (approval gates and feedback)
6. **Multi-agent coordination** (supervisor patterns, parallel execution)

**When you need it:**
- Workflows that iterate (cycles)
- Workflows that branch (conditional logic)
- Workflows that pause (human review)
- Workflows that recover (checkpoint/resume)
- Workflows that coordinate (multiple agents)

**When you don't:**
- Simple linear chains
- One-shot agent calls
- Rapid prototyping
- No state persistence needs

**The honest truth:**
LangGraph adds complexity. It's more code, more concepts, more testing. But when your workflow outgrows simple chains, trying to avoid LangGraph costs more than learning it.

**For our five capabilities:**
- **Prompt Routing**: Iterative intent refinement
- **Query Writing**: Multi-stage generation with validation loops
- **Data Processing**: Checkpoint-based ETL with recovery
- **Tool Orchestration**: Complex tool chains with fallbacks
- **Decision Support**: Multi-stage analysis with approval gates

LangGraph is not magic. It's state machines for LLMs. State machines are old, proven technology. LangGraph makes them accessible for agent workflows.

Use it when state matters. Skip it when it doesn't.

---

**End of Document** (1,441 lines)