# LangChain: Tools & Tool Abstraction Layer

**Source:** https://python.langchain.com/docs/concepts/tools/  
**Date Accessed:** 2025-11-06  
**Relevance:** LangChain Tools provide the abstraction layer that makes tool creation practical and maintainable. After understanding the theory (ReAct), the implementation (Anthropic), and the agent framework (LangChain Agents), this is how we actually build tools in production. This abstraction is what we'll use for all five capabilities.

---

## The Core Abstraction

### What Is a LangChain Tool?

A **Tool** is a Python function wrapped with a schema that defines:
- **Name:** What to call it
- **Description:** When and how to use it (this is the prompt for the LLM)
- **Arguments:** JSON schema specifying expected inputs

**Key Insight:** Tools bridge the gap between natural language (LLM) and structured code (Python functions).

### The Tool Interface (BaseTool)

All tools in LangChain inherit from `BaseTool`, which implements the Runnable Interface.

**Key Attributes (Schema):**
```python
tool.name        # String identifier
tool.description # Natural language description (the LLM reads this)
tool.args        # JSON schema for arguments
```

**Key Methods (Execution):**
```python
tool.invoke(inputs)   # Synchronous execution
tool.ainvoke(inputs)  # Asynchronous execution
```

---

## Creating Tools: The @tool Decorator

The **recommended** way to create tools. Simple, powerful, handles most use cases.

### Basic Example

```python
from langchain_core.tools import tool

@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b

# Use it directly
result = multiply.invoke({"a": 2, "b": 3})  # Returns 6

# Inspect its schema
print(multiply.name)         # "multiply"
print(multiply.description)  # "Multiply two numbers."
print(multiply.args)         # JSON schema with a, b as integers
```

**What Just Happened:**
- `@tool` decorator converted function to Tool object
- Name inferred from function name
- Description extracted from docstring
- Arguments inferred from type hints
- JSON schema automatically generated

### Customizing the Schema

You can override defaults:

```python
@tool("multiply_numbers")
def multiply(a: int, b: int) -> int:
    """Multiply two numbers together.
    
    Use this when the user asks for multiplication or product calculations.
    Do NOT use for addition or other operations.
    """
    return a * b
```

**Best Practice:** Make descriptions specific, contextual, and include:
- What the tool does
- When to use it
- When NOT to use it
- Examples of appropriate queries

### Type Hints = Better Tools

Type hints do two things:
1. **Schema generation:** LLM knows what types to provide
2. **Runtime validation:** Pydantic validates inputs

```python
from typing import Optional

@tool
def search_database(
    query: str,
    limit: int = 10,
    include_metadata: bool = False
) -> str:
    """Search the internal database for documents.
    
    Args:
        query: Search terms to look for
        limit: Maximum results to return (default: 10)
        include_metadata: Whether to include document metadata
    """
    # Implementation
    pass
```

The LLM now knows:
- `query` is required (no default)
- `limit` is optional with default 10
- `include_metadata` is optional boolean

---

## Special Type Annotations

LangChain provides annotations that modify runtime behavior without changing the schema.

### InjectedToolArg (Hidden from LLM)

Some arguments should NOT be controlled by the LLM:

```python
from langchain_core.tools import tool, InjectedToolArg

@tool
def user_specific_action(
    action: str,
    user_id: InjectedToolArg  # NOT in schema
) -> str:
    """Perform action for a specific user."""
    return f"User {user_id} performed {action}"

# At runtime, you inject the value:
result = user_specific_action.invoke(
    {"action": "logout"},
    user_id="user123"  # Injected, not from LLM
)
```

**Use Cases:**
- User authentication tokens
- Session IDs
- Internal system parameters
- API keys
- Database connections

**Key Point:** LLM cannot see or set these—they're injected at runtime by your code.

### RunnableConfig (Access Configuration)


Access runtime configuration:

```python
from langchain_core.runnables import RunnableConfig

@tool
async def fetch_data(
    query: str,
    config: RunnableConfig  # NOT in schema
) -> str:
    """Fetch data from external API."""
    # Access config values
    api_key = config.get("configurable", {}).get("api_key")
    # Use it...
    pass

# Invoke with config
await fetch_data.ainvoke(
    {"query": "test"},
    config={"configurable": {"api_key": "secret"}}
)
```

**Use Case:** Passing runtime values that aren't tool arguments (API keys, rate limits, etc.)

### Annotated (Add Descriptions)

Enhance argument descriptions in the schema:

```python
from typing import Annotated

@tool
def analyze_sentiment(
    text: Annotated[str, "The text to analyze for emotional tone"],
    language: Annotated[str, "Two-letter language code (e.g., 'en', 'es')"] = "en"
) -> str:
    """Analyze the sentiment of text."""
    pass
```

The LLM gets detailed descriptions for each argument, improving parameter selection.

---

## Tool Artifacts (Dual Output)

Sometimes tools produce:
1. **Content for LLM** (summary, status, message)
2. **Artifact for downstream use** (full object, dataframe, image)

```python
from typing import Tuple, Any

@tool(response_format="content_and_artifact")
def fetch_report(report_id: str) -> Tuple[str, Any]:
    """Fetch a detailed report.
    
    Returns:
        Tuple of (summary for LLM, full report object)
    """
    full_report = database.get_report(report_id)
    summary = f"Retrieved report {report_id} with {len(full_report.data)} rows"
    
    return summary, full_report  # (content, artifact)

# Later in your workflow
result = fetch_report.invoke({"report_id": "Q3-2024"})
print(result.content)   # "Retrieved report Q3-2024 with 1000 rows"
print(result.artifact)  # Full report object
```

**Use Case:**
- Image generation (return description + image)
- Database queries (return summary + full results)
- File operations (return status + file handle)

**Why This Matters:**
- LLM doesn't need raw data (saves tokens, improves reasoning)
- Downstream tools/code can access full artifacts
- Clean separation of concerns

---

## Toolkits (Grouped Tools)

A **Toolkit** is a collection of related tools designed to work together.

### Interface

All toolkits expose `get_tools()`:

```python
from langchain_community.agent_toolkits import SQLDatabaseToolkit

# Initialize toolkit with dependencies
toolkit = SQLDatabaseToolkit(db=database, llm=llm)

# Get all tools
tools = toolkit.get_tools()  # Returns list of Tool objects

# Pass to agent
agent = create_agent(model=llm, tools=tools)
```

### Common Toolkits

**Built-in Examples:**
- **SQLDatabaseToolkit:** Query, describe, check SQL databases
- **FileManagementToolkit:** Read, write, list, delete files
- **PythonREPLToolkit:** Execute Python code
- **APIToolkit:** Interact with REST APIs

**Pattern:** Toolkit groups tools that share resources (database connection, API client, file system access).

---

## Best Practices

### 1. Tool Naming

**Good Names:**
- Descriptive: `search_customer_database`
- Action-oriented: `send_email`, `calculate_tax`
- Specific: `fetch_weather_forecast` not `get_data`

**Bad Names:**
- Generic: `tool1`, `helper`, `utility`
- Vague: `process`, `handle`, `do_thing`

### 2. Tool Descriptions

**Structure:**
1. **What it does** (one sentence)
2. **When to use it** (specific scenarios)
3. **When NOT to use it** (prevent misuse)
4. **Examples** (if complex)

```python
@tool
def query_sales_database(
    start_date: str,
    end_date: str,
    region: Optional[str] = None
) -> str:
    """Query sales data from the internal database.
    
    Use this tool when:
    - User asks about sales figures, revenue, or transactions
    - Questions involve time periods (e.g., 'Q3 sales', 'this month')
    - Regional breakdowns are requested
    
    Do NOT use this tool for:
    - Product inventory (use inventory_database instead)
    - Customer information (use customer_database instead)
    - External market data (use market_data_api instead)
    
    Args:
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        region: Optional region filter (e.g., 'west', 'east', 'midwest')
    """
    pass
```

### 3. Narrow Scope

**Principle:** One tool = one clear purpose

**Bad (Too Broad):**
```python
@tool
def database_operations(operation: str, table: str, data: dict) -> str:
    """Perform any database operation."""
    # Too flexible, hard for LLM to use correctly
```

**Good (Narrow & Specific):**
```python
@tool
def insert_customer(name: str, email: str) -> str:
    """Insert a new customer record."""
    pass

@tool
def update_customer_email(customer_id: int, new_email: str) -> str:
    """Update existing customer's email."""
    pass
```

**Why:** Narrow tools are easier for LLMs to understand and use correctly.

### 4. Error Handling

Tools should return informative errors:

```python
@tool
def fetch_user_data(user_id: int) -> str:
    """Fetch user data from database."""
    try:
        user = database.get_user(user_id)
        return f"User: {user.name}, Email: {user.email}"
    except UserNotFoundError:
        return f"Error: User {user_id} not found in database"
    except DatabaseConnectionError as e:
        return f"Error: Database unavailable - {str(e)}"
    except Exception as e:
        return f"Error: Unexpected failure - {str(e)}"
```

**Critical:** Return error messages as strings, not exceptions. The LLM needs to read and reason about errors.

### 5. Return String or Structured Data

**Preferred:** Return strings (LLMs process text)

```python
@tool
def get_weather(city: str) -> str:
    """Get weather for a city."""
    return "Temperature: 72°F, Conditions: Sunny"
```

**Alternative:** Use tool artifacts for structured data

```python
@tool(response_format="content_and_artifact")
def get_weather_detailed(city: str) -> Tuple[str, dict]:
    """Get detailed weather data."""
    weather_data = api.fetch_weather(city)
    summary = f"{city}: {weather_data['temp']}°F, {weather_data['conditions']}"
    return summary, weather_data  # String for LLM, dict for downstream
```

---

## Integration Points

### Connection to Our Five Capabilities

**1. Prompt Routing**
```python
@tool
def route_to_internal() -> str:
    """Route query to internal knowledge base."""
    return "Routing to internal search"

@tool
def route_to_external() -> str:
    """Route query to web search."""
    return "Routing to external search"

# Agent picks which route based on query
```

**2. Query Writing**
```python
@tool
def construct_sql_query(
    table: str,
    filters: dict,
    order_by: Optional[str] = None
) -> str:
    """Construct and execute SQL query dynamically."""
    query = build_query(table, filters, order_by)
    results = execute(query)
    return format_results(results)
```

**3. Data Processing**
```python
@tool
def clean_data(raw_data: str) -> str:
    """Remove nulls and standardize format."""
    pass

@tool
def enrich_data(cleaned_data: str, api_source: str) -> str:
    """Enrich data with external API."""
    pass

@tool
def summarize_data(enriched_data: str) -> str:
    """Generate summary statistics."""
    pass

# Agent chains these tools sequentially
```

**4. Tool Orchestration**
```python
# Tools ARE the orchestration
# The agent (using ReAct) decides which tools to call and in what order

@tool
def step_one() -> str:
    """First operation."""
    pass

@tool
def step_two(result_from_one: str) -> str:
    """Second operation, uses result from first."""
    pass

# Agent: Thought → Action(step_one) → Observation → Thought → Action(step_two)
```

**5. Decision Support**
```python
@tool
def analyze_option(option_name: str) -> str:
    """Analyze a specific option with pros/cons."""
    pass

@tool
def compare_options(option_a: str, option_b: str) -> str:
    """Compare two options directly."""
    pass

@tool
def recommend_best(criteria: dict) -> str:
    """Recommend best option based on criteria."""
    pass
```

---

## Our Takeaways

### For Agentic_AI_Development

**1. The @tool Decorator Is Your Primary Interface**

Don't manually implement BaseTool unless you have very specific needs. The decorator:
- Infers schemas automatically
- Handles validation
- Integrates seamlessly with agents
- Is the pattern LangChain recommends

**2. Descriptions Are Mini-Prompts**

Tool descriptions are how you communicate with the LLM about when and how to use the tool. Invest time here:
- Be explicit about use cases
- Include negative examples (when NOT to use)
- Provide context about what the tool returns
- Mention any prerequisites or dependencies

**3. InjectedToolArg Solves the Security Problem**

Never let the LLM control:
- Authentication tokens
- User IDs
- API keys
- Database connections
- System-level parameters

Use `InjectedToolArg` to hide these from the schema and inject them at runtime.

**4. Tool Artifacts = Clean Architecture**

When a tool produces both a summary and detailed data:
- Content: What the LLM needs to reason
- Artifact: What downstream code needs to process

This separation keeps token usage down and maintains clean interfaces.

**5. One Tool = One Clear Action**

Avoid "Swiss Army Knife" tools that do many things. Better to have:
- 5 specific tools that do one thing each
- Than 1 general tool that requires complex parameters

**Reasoning:** LLMs are better at selecting the right tool than configuring a complex one.

**6. Error Messages Are Part of the Interface**

Don't raise exceptions. Return error strings that:
- Explain what went wrong
- Suggest what to try next
- Help the LLM recover or reformulate

Remember: The LLM is reading these errors and using them to decide next steps (ReAct observation).

**7. Type Hints = Better Agent Performance**

Type hints do double duty:
- Generate accurate JSON schemas
- Enable runtime validation

Always use them. The small effort pays dividends in agent reliability.

**8. Toolkits Reduce Cognitive Load**

When tools share dependencies (database, API client, file system):
- Group them into toolkits
- Initialize shared resources once
- Pass as a unit to agents

This is better than individual tool management.

**9. The Abstraction Enables Portability**

LangChain's tool abstraction works across:
- Multiple LLM providers (Anthropic, OpenAI, Google)
- Different agent architectures (ReAct, Plan-and-Execute)
- Various frameworks (LangChain, LangGraph)

Write tools once, use everywhere.

**10. Testing Tools Is Trivial**

Tools are just Python functions:

```python
def test_multiply():
    result = multiply.invoke({"a": 3, "b": 4})
    assert result == 12
    
def test_multiply_schema():
    assert "a" in multiply.args["properties"]
    assert "b" in multiply.args["properties"]
    assert multiply.args["properties"]["a"]["type"] == "integer"
```

Test both function logic AND schema correctness.

---

## Implementation Checklist

When building tools for our five capabilities:

### Tool Design
- [ ] Use @tool decorator (not BaseTool)
- [ ] Include comprehensive docstring
- [ ] Add type hints for all parameters
- [ ] Use Annotated for detailed arg descriptions
- [ ] Mark injected args with InjectedToolArg

### Schema Quality
- [ ] Tool name is action-oriented and specific
- [ ] Description includes "when to use" and "when NOT to use"
- [ ] Arguments have clear descriptions
- [ ] Required vs. optional parameters are correct
- [ ] Default values are sensible

### Implementation
- [ ] Error handling returns string messages (not exceptions)
- [ ] Returns string or uses tool artifacts appropriately
- [ ] Tool scope is narrow (one clear purpose)
- [ ] Shared resources grouped into toolkits when applicable

### Security
- [ ] Authentication tokens use InjectedToolArg
- [ ] User IDs use InjectedToolArg
- [ ] API keys use InjectedToolArg or RunnableConfig
- [ ] No sensitive data in schemas

### Testing
- [ ] Unit tests for function logic
- [ ] Tests verify schema structure
- [ ] Tests cover error cases
- [ ] Integration tests with actual agent

---

## Comparison to Anthropic's Native Approach

**Anthropic Tool Use (Raw):**
```python
tools = [{
    "name": "get_weather",
    "description": "Get weather...",
    "input_schema": {
        "type": "object",
        "properties": {...},
        "required": [...]
    }
}]

# Manual schema definition
# No type safety
# No validation
```

**LangChain Tools (Abstraction):**
```python
@tool
def get_weather(city: str) -> str:
    """Get weather..."""
    return weather_data

# Schema inferred automatically
# Type safety via hints
# Pydantic validation
# Portable across providers
```

**Takeaway:** LangChain's abstraction eliminates boilerplate while adding safety and portability.

---

## Next Documentation to Review

Based on this foundation:
1. **Pydantic Validation** - Type safety deep dive
2. **LangGraph Workflows** - Advanced orchestration
3. **Anthropic/OpenAI Prompt Engineering** - Making tools work better

---

## The Universal Tool Pattern

Across all implementations (Anthropic, OpenAI, LangChain), tools follow the same fundamental structure:

```
Tool = {
    name: string,
    description: string (prompt for LLM),
    schema: JSON (parameter validation),
    function: executable code
}
```

**LangChain's Contribution:** A Python-native abstraction that:
- Infers schemas from code
- Adds type safety
- Enables portability
- Reduces boilerplate
- Maintains full control

---

## Real-World Pattern: The Tool Development Cycle

1. **Start Simple:** Basic @tool decorator, minimal description
2. **Test with Agent:** See how LLM uses it
3. **Refine Description:** Add "when to use" and "when NOT to use"
4. **Add Type Hints:** Improve schema precision
5. **Handle Errors:** Return informative error strings
6. **Optimize:** Consider artifacts, injection, narrow scope

**Key Insight:** Tools evolve through use. Don't over-engineer upfront—let agent interaction guide refinement.

---

**Summary:** LangChain Tools provide a Python-native abstraction layer over raw tool schemas. The @tool decorator infers schemas from type hints and docstrings, eliminating boilerplate. InjectedToolArg solves security concerns. Tool artifacts separate LLM-facing content from downstream data. Toolkits group related tools. The abstraction is portable across LLM providers and agent architectures. This is the practical interface for building all five of our capabilities.
