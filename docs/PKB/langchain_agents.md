# LangChain: Agents & Tool-Calling Architecture

**Source:** https://python.langchain.com/docs/concepts/agents/ & https://langchain-ai.github.io/langgraph/concepts/agentic_concepts/  
**Date Accessed:** 2025-11-06  
**Relevance:** LangChain provides the conceptual framework for our agent implementations. Understanding both legacy AgentExecutor patterns and the modern LangGraph approach is essential for building robust, production-ready agentic systems. The ReAct pattern (Reasoning + Acting) is foundational to all five capabilities we're building.

---

## Key Concepts

### What Makes Something an Agent?

**Core Definition:** An agent is a system that uses an LLM to decide the control flow of an application, rather than following hard-coded paths.

The LLM can control applications in several ways:
- **Route between paths** (simple routing decisions)
- **Decide which tools to call** (and in what order)
- **Determine if work is complete** (or more steps are needed)

**Key Distinction from Chains:**
- **Chains:** Fixed sequence of operations (always step A → B → C)
- **Agents:** Dynamic decisions about which operations to perform and when

### The Evolution: AgentExecutor → LangGraph

**Legacy Approach (AgentExecutor):**
LangChain originally used `AgentExecutor` as the runtime for agents. While excellent for getting started, it showed limitations with complex, customized agents.

**Modern Approach (LangGraph):**
LangChain now recommends LangGraph for all agent development. Built on top of LangChain, LangGraph provides:
- Durable execution
- Streaming capabilities
- Human-in-the-loop support
- State persistence
- More flexible control flows

**Important:** LangChain's pre-built agents are now built on LangGraph—you get LangGraph benefits without needing to know it deeply for basic use cases.

---

## Agent Architectures

### 1. Router (Simplest)

A router allows an LLM to select a single step from pre-defined options.

**Control Level:** Limited—typically one decision, specific output from known options

**Key Concepts:**
- **Structured Output:** LLM responds in a specific format/schema
- **Prompt Engineering:** Instructions in system prompt guide format
- **Tool Calling:** Built-in capabilities generate structured decisions

**Use Case:** "Based on user intent, route to search_internal, search_web, or respond_directly"

**Connection to Our Project:** This IS our Prompt Routing capability.

### 2. Tool-Calling Agent (ReAct Pattern)

The most common agent architecture. Expands LLM control in two ways:
- **Multi-step decision making:** Series of decisions, not just one
- **Tool access:** Choose from and use various tools

**The ReAct Loop:**
1. **Thought:** LLM reasons about what to do
2. **Action:** LLM calls a tool
3. **Observation:** Tool result is observed
4. **Repeat:** Until task is complete

**Core Components:**

**Tool Calling:**
- Functions the agent can invoke to interact with external systems
- Bind Python functions to LLM: `ChatModel.bind_tools(function)`
- LLM becomes aware of required input schema
- Returns output adhering to tool's schema

**Memory:**
- **Short-term:** Information from earlier steps in current sequence
- **Long-term:** Information from previous interactions/conversations
- In LangGraph: State (schema), Checkpointer (session), Store (cross-session)

**Planning:**
- LLM called repeatedly in a while-loop
- Each step: decide which tools, what inputs
- Execute tools, feed outputs back as observations
- Terminates when agent has sufficient information

**Modern Implementation:**
Today's agents rely on LLM's native tool-calling capabilities and operate on message lists (not raw text like the original ReAct paper).

---

## Implementation Patterns

### Legacy Pattern (AgentExecutor)

The traditional LangChain approach before LangGraph:

```python
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool

# Define tools
tools = [
    Tool(
        name="Calculator",
        func=calculate,
        description="Useful for mathematical calculations"
    )
]

# Initialize agent
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# Run the agent
result = agent.run("What is 25 * 4?")
```

**AgentExecutor Pseudocode:**
```python
next_action = agent.get_action(...)
while next_action != AgentFinish:
    observation = run(next_action)
    next_action = agent.get_action(..., next_action, observation)
return next_action
```

### Modern Pattern (LangChain with LangGraph)

The current recommended approach:

```python
from langchain.agents import create_agent
from langchain_core.tools import tool

# Define a tool using decorator
@tool
def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

# Create agent (built on LangGraph under the hood)
agent = create_agent(
    model="claude-sonnet-4-5-20250929",
    tools=[get_weather],
    system_prompt="You are a helpful assistant"
)

# Run the agent
response = agent.invoke({
    "messages": [{"role": "user", "content": "what is the weather in sf"}]
})
```

**Key Improvements:**
- Cleaner API (no AgentType enums)
- Message-based interface (more flexible)
- LangGraph features (streaming, persistence) built-in
- Tool definition via decorators

---

## Custom Agent Architectures (LangGraph Features)

### Human-in-the-Loop

Human involvement for reliability on sensitive tasks:
- Approving specific actions before execution
- Providing feedback to update agent's state
- Offering guidance in complex decisions

**Use Case:** "Agent drafts email → human reviews → agent sends"

### Parallelization

Process multiple states concurrently:
- Map-reduce-like operations
- Independent subtask handling
- Multi-agent coordination

**LangGraph API:** `Send` enables concurrent processing

### Subgraphs

Manage complex, multi-agent architectures:
- Isolated state management per agent
- Hierarchical organization of agent teams
- Controlled communication via overlapping state keys

**Pattern:** Parent graph contains multiple subgraphs (each an agent)

### Reflection

Self-improvement mechanisms:
- Evaluate task completion and correctness
- Provide feedback for iterative improvement
- Enable self-correction

**Example:** Code generation agent uses compilation errors as feedback to fix code

---

## Tools & Toolkits

### Tool Definition

**Requirements for good tools:**

1. **Clear Docstring:** Describes what the tool does and when to use it
2. **Descriptive Parameter Names:** Ideally self-explanatory
3. **Type Hints:** Help LLM understand expected types

```python
@tool
def search_database(
    query: str,
    limit: int = 10
) -> str:
    """Search the internal database for relevant documents.
    
    Use this when the user asks about internal company data,
    policies, or documentation. Do not use for general web queries.
    
    Args:
        query: The search query string
        limit: Maximum number of results to return (default: 10)
    """
    # Implementation
    pass
```

### Toolkit

A **Toolkit** is simply a collection of related tools:

```python
# Just an array of tools
toolkit = [search_database, fetch_user_data, update_record]
```

**Design Principle:** Group related functionality. Example:
- DatabaseToolkit: `[search, insert, update, delete]`
- WebToolkit: `[search_web, fetch_url, parse_html]`

---

## Common Pitfalls

### 1. Choosing Agent When Chain Would Suffice

**Problem:** Agents add cost and unpredictability. If your workflow is always the same sequence, use a chain.

**Use Chain When:**
- Fixed flow: Step 1 → Step 2 → Step 3
- No branching logic needed
- Predictable costs matter

**Use Agent When:**
- Need to determine which tools to use dynamically
- Multi-source reasoning required
- Complex queries requiring decomposition

### 2. Vague Tool Descriptions

**Problem:** LLM can't decide when to use the tool

```python
# BAD
def helper_function():
    """Does stuff"""
    pass

# GOOD
@tool
def search_company_policies(topic: str) -> str:
    """Search internal company policy documents.
    
    Use this when users ask about:
    - HR policies (vacation, benefits, conduct)
    - IT policies (security, equipment)
    - Financial policies (expenses, reimbursement)
    
    Do NOT use for:
    - General knowledge questions
    - External regulations
    - Personal advice
    """
    pass
```

### 3. Not Handling Tool Failures

**Problem:** Tools fail. APIs timeout. Data is missing.

**Solution:** Return informative errors in tool results

```python
@tool
def fetch_stock_price(symbol: str) -> str:
    """Get current stock price."""
    try:
        price = api.get_price(symbol)
        return f"${price}"
    except APIError as e:
        return f"Error: Unable to fetch {symbol}. API returned: {e}"
    except ValueError:
        return f"Error: {symbol} is not a valid stock symbol"
```

### 4. Infinite Loops

**Problem:** Agent keeps calling tools without making progress

**Solutions:**
- Set maximum iterations
- Implement progress tracking
- Add explicit termination conditions
- Use reflection to detect lack of progress

### 5. Memory Management Failures

**Problem:** Agent forgets context or hallucinates from incomplete memory

**Short-term Memory Issues:**
- Not passing intermediate steps to next iteration
- Losing tool results in message chain

**Long-term Memory Issues:**
- Not persisting state between sessions
- No checkpointing for recovery

---

## Integration Points

### Connection to Our Five Capabilities

**1. Prompt Routing**
- **Router Architecture:** Single-step routing decision
- **Structured Output:** Force specific routing schema
- **Implementation:** Tool that represents each route, LLM picks one

**2. Query Writing**
- **Tool as Query Builder:** Define a `write_query` tool
- **LLM constructs query:** Based on user intent and schema
- **Sequential refinement:** Agent can iterate on query based on results

**3. Data Processing**
- **Tool Chain:** extract → transform → enrich → summarize
- **Parallel Processing:** Independent transformations run concurrently
- **Memory:** Maintain state across processing steps

**4. Tool Orchestration**
- **This IS the core capability:** Agents orchestrate tools
- **ReAct Loop:** The fundamental pattern for orchestration
- **LangGraph:** Advanced orchestration with parallelization, subgraphs

**5. Decision Support**
- **Multi-step Planning:** Agent breaks down complex decisions
- **Reflection:** Evaluate options, provide reasoning
- **Human-in-the-loop:** Critical decisions require approval

### ReAct Pattern Connection

The ReAct (Reasoning + Acting) pattern is foundational to everything:

1. **Reason:** What do I need to know?
2. **Act:** Call appropriate tool(s)
3. **Observe:** Analyze results
4. **Decide:** Continue or finish?

This pattern applies to:
- **Routing:** Reason about intent → Act by routing
- **Query Writing:** Reason about data needs → Act by constructing query
- **Data Processing:** Reason about transformations → Act by applying them
- **Orchestration:** Reason about tool sequence → Act by calling tools
- **Decision Support:** Reason about options → Act by recommending

---

## Our Takeaways

### For Agentic_AI_Development

**1. The Router IS Prompt Routing**

Don't overthink it. A router that selects from pre-defined paths is exactly what we're building for Prompt Routing. The structured output guarantees we get a valid routing decision.

**2. Start Simple, Add Complexity When Needed**

- Begin with basic ReAct agent
- Add memory only when context loss is a problem
- Add parallelization only when latency matters
- Add human-in-the-loop only for high-stakes decisions

**3. LangGraph Is the Future (But You Don't Need to Master It Yet)**

LangChain's modern `create_agent` is built on LangGraph, so you get the benefits without needing deep LangGraph knowledge. Learn LangGraph when you need:
- Custom control flows
- Multi-agent systems
- Advanced state management
- Durable execution requirements

**4. Tool Design Is Critical**

The quality of our agent implementations depends heavily on tool design:
- **Descriptions are prompts:** Make them specific, contextual
- **Error handling is mandatory:** Tools must fail gracefully
- **Type hints help:** LLM understands schemas better
- **Group related tools:** Toolkits reduce cognitive load

**5. Memory Architecture Matters Early**

Don't bolt memory on later. Design for it upfront:
- **Short-term:** What does agent need from current session?
- **Long-term:** What should persist across sessions?
- **State schema:** What structure makes sense for this agent?

**6. Agent vs. Chain Decision Framework**

Ask these questions:
1. Is the sequence always the same? → **Chain**
2. Need dynamic tool selection? → **Agent**
3. Cost predictability critical? → **Chain**
4. Complex multi-source reasoning? → **Agent**

**7. The ReAct Pattern Is Universal**

Every agentic capability follows this loop:
- **Reason** about what's needed
- **Act** by using tools/APIs
- **Observe** results
- **Iterate** or complete

Master this pattern, and the five capabilities become variations on a theme rather than five separate problems.

**8. Migration Strategy**

If you encounter legacy code using `AgentExecutor`:
- Check LangChain's migration guide
- Modern `create_agent` is usually a drop-in replacement
- LangGraph for advanced needs

### Testing Strategy

Based on agent patterns:
1. **Tool selection:** Does agent pick right tools?
2. **Tool sequencing:** Correct order for dependent operations?
3. **Termination:** Does agent stop when it should?
4. **Error recovery:** Handles tool failures gracefully?
5. **Memory persistence:** Maintains context appropriately?
6. **Cost efficiency:** Minimizes unnecessary tool calls?

---

## Next Documentation to Review

Based on this foundation:
1. **LangGraph Workflows** - Deep dive into advanced orchestration
2. **ReAct Pattern Paper** - Theoretical foundations
3. **LangChain Tools** - Abstraction patterns for tool creation
4. **OpenAI Function Calling** - Alternative implementation approach

---

**Summary:** Agents use LLMs to control application flow dynamically. The ReAct pattern (Reasoning + Acting) is fundamental to all agentic capabilities. LangChain evolved from AgentExecutor to LangGraph for better flexibility and control. Tool design quality directly determines agent effectiveness. The router architecture IS prompt routing. Everything builds on the ReAct loop.
