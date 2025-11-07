# Anthropic: Tool Use

**Source:** https://docs.claude.com/en/docs/agents-and-tools/tool-use/overview  
**Date Accessed:** 2025-11-06  
**Relevance:** Core capability for all five agentic AI patterns. Tool use is the mechanism that enables prompt routing, query writing, tool orchestration, and decision support. Understanding Anthropic's implementation is essential since Claude is our primary development platform.

---

## Key Concepts

### Two Tool Types

**Client Tools** (Execute on your system):
- **User-defined tools:** Custom functions you create and implement
- **Anthropic-defined tools:** Like `computer_use` and `text_editor` that require your implementation
- You define the tool schema, Claude decides when to use it, you execute and return results

**Server Tools** (Execute on Anthropic's servers):
- **Pre-built tools:** `web_search_20250305`, `web_fetch_20250305`
- Claude both decides to use them AND executes them automatically
- No client-side execution required—just specify them in your request

### The Tool Use Loop (Client Tools)

This is the fundamental pattern:

1. **You provide:** Tools (name, description, schema) + user prompt
2. **Claude decides:** Whether any tool can help, constructs tool_use request
3. **You execute:** Extract tool name/input, run actual function, return results
4. **Claude synthesizes:** Uses tool results to craft final response

Key insight: There's a **conversation round-trip** here. Unlike server tools, client tools require you to handle the execution and return results in a new message.

### Stop Reasons

When Claude wants to use a tool, the API response includes:
- `stop_reason: "tool_use"` (for client tools)
- `content` array with `tool_use` blocks containing tool name and input

This is your signal to execute the tool and continue the conversation.

---

## Implementation Patterns

### Basic Tool Definition

Tools are defined with three components:

```python
tool = {
    "name": "get_weather",  # Clear, descriptive function name
    "description": "Get current weather in a given location",  # What it does, when to use it
    "input_schema": {  # JSON Schema for validation
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "City and state, e.g. San Francisco, CA"
            },
            "unit": {
                "type": "string",
                "enum": ["celsius", "fahrenheit"],
                "description": "Temperature unit"
            }
        },
        "required": ["location"]  # Only location is required, unit is optional
    }
}
```

**Critical:** Descriptions matter. Claude uses them to decide when to invoke the tool. Make them specific and context-rich.

### Parallel Tool Use

Claude can request multiple tools in a single response when operations are independent:

```python
# Claude's response might include multiple tool_use blocks:
{
    "content": [
        {"type": "tool_use", "id": "toolu_01A", "name": "get_weather", "input": {...}},
        {"type": "tool_use", "id": "toolu_01B", "name": "get_time", "input": {...}}
    ],
    "stop_reason": "tool_use"
}

# You must return all results in a single user message:
{
    "role": "user",
    "content": [
        {"type": "tool_result", "tool_use_id": "toolu_01A", "content": "72°F, sunny"},
        {"type": "tool_result", "tool_use_id": "toolu_01B", "content": "2:30 PM PST"}
    ]
}
```

**Pattern insight:** Parallel execution is for *independent* operations. If tool B depends on tool A's output, Claude will call them sequentially across multiple messages.

### Sequential Tool Chaining

When one tool's output feeds into another:

```python
# Turn 1: Claude requests location
{"type": "tool_use", "name": "get_location", "input": {}}

# Turn 2: You return location
{"type": "tool_result", "content": "San Francisco, CA"}

# Turn 3: Claude requests weather with that location
{"type": "tool_use", "name": "get_weather", "input": {"location": "San Francisco, CA"}}

# Turn 4: You return weather
{"type": "tool_result", "content": "59°F, mostly cloudy"}

# Turn 5: Claude synthesizes final answer
"Based on your location in San Francisco, CA, it's 59°F and mostly cloudy..."
```

**Pattern insight:** This is the ReAct loop in practice—reasoning about what's needed, acting to get it, then using results to inform next actions.

### Chain of Thought for Tool Use

Force Claude to think before using tools:

```python
system_prompt = """
Answer using relevant tools if available. Before calling a tool:
1. Think about which tool is relevant
2. Check if user provided all required parameters
3. If missing required params, ask user (don't guess with fillers)
4. If all params present or inferable, proceed with tool call

DO NOT ask for optional parameters if not provided.
"""
```

**Why this matters:** Prevents Claude from making bad guesses or calling tools with incomplete information. Forces explicit reasoning about tool necessity and parameter completeness.

### Forcing Tool Use (tool_choice parameter)

Control when Claude uses tools:

```python
# Auto (default): Claude decides
tool_choice = {"type": "auto"}

# Any: Must use at least one tool
tool_choice = {"type": "any"}

# Specific tool: Must use this exact tool
tool_choice = {"type": "tool", "name": "get_weather"}

# None: Prevent all tool use
tool_choice = {"type": "none"}
```

**Use case for "any":** JSON mode. Define a single `record_summary` tool and force its use to guarantee structured output.

---

## Code Examples

### Minimal Working Example

```python
import anthropic

client = anthropic.Anthropic(api_key="your-key")

tools = [{
    "name": "get_weather",
    "description": "Get current weather in a given location",
    "input_schema": {
        "type": "object",
        "properties": {
            "location": {"type": "string", "description": "City and state"}
        },
        "required": ["location"]
    }
}]

# Initial request
response = client.messages.create(
    model="claude-sonnet-4-5",
    max_tokens=1024,
    tools=tools,
    messages=[{"role": "user", "content": "What's the weather in SF?"}]
)

# Check if Claude wants to use a tool
if response.stop_reason == "tool_use":
    tool_use = next(block for block in response.content if block.type == "tool_use")
    
    # Execute your actual function
    weather_data = get_weather_from_api(tool_use.input["location"])
    
    # Return results to Claude
    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=1024,
        tools=tools,
        messages=[
            {"role": "user", "content": "What's the weather in SF?"},
            {"role": "assistant", "content": response.content},
            {
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": tool_use.id,
                    "content": weather_data
                }]
            }
        ]
    )
    
    print(response.content[0].text)
```

---

## Common Pitfalls

### 1. Forgetting to Include Prior Messages

**Problem:** Each API call is stateless. If you don't include the full conversation history, Claude loses context.

```python
# WRONG - loses context
response2 = client.messages.create(
    messages=[{"role": "user", "content": [tool_result]}]  # Missing previous turns!
)

# RIGHT - maintains context
response2 = client.messages.create(
    messages=[
        {"role": "user", "content": "What's the weather in SF?"},
        {"role": "assistant", "content": response1.content},
        {"role": "user", "content": [tool_result]}
    ]
)
```

### 2. Not Handling Missing Parameters Gracefully

**Problem:** Claude (especially Sonnet) sometimes guesses required parameters instead of asking.

**Solution:** Use chain-of-thought prompting (see above) to force explicit parameter checking.

### 3. Tool Descriptions Are Too Vague

**Problem:** Generic descriptions lead to incorrect tool selection.

```python
# BAD
"description": "Gets data"

# GOOD
"description": "Retrieves real-time stock price and volume for a given ticker symbol from NYSE/NASDAQ exchanges. Returns current price, 24h change, and volume. Use when user asks about stock prices or market data."
```

**Rule:** Descriptions should answer: What does this do? When should it be used? What does it return?

### 4. Ignoring stop_reason

**Problem:** Assuming every response contains text. Some responses are ONLY tool use requests.

```python
# WRONG - crashes when response is only tool_use
text = response.content[0].text

# RIGHT - check stop_reason first
if response.stop_reason == "tool_use":
    handle_tool_use(response)
elif response.stop_reason == "end_turn":
    text = response.content[0].text
```

### 5. Not Matching tool_result IDs

**Problem:** Each tool_use has a unique ID. Your tool_result MUST reference the correct ID.

```python
# Extract ID from tool use
tool_use_id = tool_use.id

# Return result with matching ID
{"type": "tool_result", "tool_use_id": tool_use_id, "content": result}
```

### 6. Parallel vs Sequential Confusion

**Problem:** Expecting Claude to always call dependent tools in sequence. Sometimes it tries parallel execution when it shouldn't.

**Solution:** Design tools to be as independent as possible. If dependencies exist, make them explicit in tool descriptions.

---

## Integration Points

### Connection to Our Five Capabilities

**1. Prompt Routing**
- Tools can represent different routing destinations (search_internal, search_web, respond_directly)
- Tool descriptions become routing logic: "Use search_internal when user asks about company data..."
- tool_choice can force specific routing paths

**2. Query Writing**
- Define a `write_query` tool that accepts query parameters
- Claude constructs the query based on user intent
- Tool description guides Claude on query syntax and capabilities

**3. Data Processing**
- Tools like `transform_data`, `enrich_data`, `summarize_data`
- Input schemas enforce data structure requirements
- Sequential chaining: extract → transform → enrich → summarize

**4. Tool Orchestration**
- THIS IS THE FOUNDATION: Tool use IS tool orchestration
- Parallel execution for independent operations
- Sequential chaining for dependent workflows
- Error handling via tool_result content

**5. Decision Support**
- Tools like `compare_options`, `analyze_tradeoffs`, `recommend_next_action`
- Chain-of-thought prompting forces explicit reasoning before tool use
- Sequential tool use mirrors multi-step decision processes

### Versioning Pattern

Note the `_20250305` suffix on server tools like `web_search_20250305`. This is a versioning strategy that:
- Ensures backward compatibility
- Allows API changes without breaking existing code
- Signals to users which tool version they're using

**Our implementation:** Consider adopting similar versioning for custom tools, especially for complex query writers or data processors that might evolve.

---

## Our Takeaways

### For agentic_ai_development

**1. Tool Use Is Our Core Primitive**

Everything we're building—routing, query writing, data processing, orchestration, decision support—is implemented through tool use. Master this, and we master agentic AI.

**2. Descriptions Are Prompts**

Tool descriptions aren't just documentation. They're instructions to Claude about *when* and *how* to use tools. Treat them like mini-prompts:
- Be specific about use cases
- Include examples in descriptions
- Specify expected input/output formats
- Clarify edge cases

**3. The Conversation Loop Is Sacred**

Never break the message chain. Always include full history. This isn't just good practice—it's essential for context maintenance.

**4. Build for Parallel When Possible**

Independent operations should be parallelizable. Design tools with minimal dependencies. This improves latency and user experience dramatically.

**5. Explicit > Implicit**

Don't rely on Claude's ability to infer parameters or decide tool usage. Use:
- Chain-of-thought prompting for critical decisions
- `tool_choice` to force specific behaviors
- Detailed schemas with examples
- System prompts that set clear expectations

**6. Error Handling Must Be Built-In**

Tool execution fails. APIs timeout. Data is malformed. Our tool implementations need:
- Try-catch blocks with descriptive errors
- Fallback strategies
- Clear error messages returned in tool_result
- Logging for debugging

**7. Token Economics Matter**

Tool use adds token overhead:
- 346 tokens for system prompt (auto/none mode)
- 313 tokens (any/tool mode)
- Plus tool schemas, tool_use blocks, tool_result blocks

For high-frequency operations, this adds up. Design tools to minimize round-trips when possible.

### Pricing Reality Check

From the docs:
- Claude Sonnet 4.5: 346 tokens overhead (auto/none), 313 tokens (any/tool)
- Every tool definition adds tokens (name + description + schema)
- Every tool_use and tool_result block adds tokens

**Implication for our project:** 
- Keep tool descriptions concise but precise
- Minimize the number of tools provided per request when possible
- Consider tool_choice="none" for simple queries that don't need tools
- Group related functionality into single tools rather than many micro-tools

### Testing Strategy

Based on patterns observed:
1. **Test tool selection:** Does Claude choose the right tool?
2. **Test parameter extraction:** Does Claude correctly parse user intent into tool inputs?
3. **Test parallel execution:** Do independent tools execute in parallel?
4. **Test sequential chaining:** Do dependent tools chain correctly?
5. **Test error recovery:** How does Claude handle tool failures?
6. **Test missing parameters:** Does Claude ask or guess?

These tests should be part of our quality standards for each capability.

---

## Next Documentation to Review

Based on this foundation:
1. **LangChain Tools** - Abstraction patterns over tool use
2. **Pydantic Validation** - Type-safe tool input/output handling
3. **LangGraph Workflows** - State machines for complex tool orchestration
4. **ReAct Pattern** - The theoretical foundation of reasoning + acting

---

**Summary:** Tool use is the bridge between Claude's reasoning and real-world actions. Everything in agentic_ai_development builds on this foundation. Master the conversation loop, design great tool descriptions, handle errors explicitly, and respect token economics.
