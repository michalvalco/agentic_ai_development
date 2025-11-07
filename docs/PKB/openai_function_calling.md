# OpenAI: Function Calling & Structured Outputs

**Source:** https://platform.openai.com/docs/guides/function-calling  
**Date Accessed:** 2025-11-06  
**Relevance:** OpenAI's function calling is the alternative implementation to Anthropic's tool use. While the underlying pattern is identical (ReAct loop, JSON schemas, LLM-driven tool selection), OpenAI's approach offers distinct advantages: native parallel function calling, structured output guarantees via strict mode, and tighter JSON mode integration. Understanding both Anthropic and OpenAI approaches ensures our implementations are portable across providers and leverage the best practices from both ecosystems. The strict mode feature, in particular, eliminates the need for extensive validation layers that Anthropic implementations require.

---

## Key Concepts

### What Is Function Calling?

**Core Definition:** Function calling allows you to describe functions to GPT models and have the model intelligently choose to output a JSON object containing arguments to call one or more functions.

**Critical Understanding:** OpenAI models don't actually *execute* functionsâ€”they generate structured JSON that describes which function to call and with what arguments. Your code executes the actual function.

**The Promise:**
- Models detect when functions should be called based on user input
- Models output JSON adhering to your function schema
- Parallel function calling for independent operations
- Structured outputs with guaranteed schema compliance (strict mode)

**Why This Matters:** Before function calling, developers had to use RegEx or complex prompt engineering to extract structured data from LLM responses. Function calling transforms unpredictable string outputs into reliable, schema-compliant JSON.

### The Three-Step Pattern

Function calling follows a predictable workflow:

1. **Define Functions:** Provide function schemas in the API request (via `tools` parameter)
2. **Model Decides:** GPT determines if/which function(s) to call, outputs structured JSON
3. **Execute & Return:** You execute the function(s), return results in a new API call

**Key Insight:** This is a **conversation loop**, not a one-shot operation. Each function call requires a round-trip to the API with the function's results.

### Terminology & Deprecation

**Current Terms:**
- **Function Calling:** The capability/pattern name
- **Tool:** The schema object provided to API (currently only `type: "function"` supported)
- **tool_choice:** Controls when/how model uses tools

**Deprecated (Don't Use):**
- `functions` parameter â†’ replaced by `tools`
- `function_call` parameter â†’ replaced by `tool_choice`

**Migration Note:** If you encounter legacy code using `functions`/`function_call`, migrate to `tools`/`tool_choice`. The old parameters may be removed in future API versions.

---

## Key Features

### 1. Parallel Function Calling

**What It Is:** The model can request multiple independent functions in a single response.

**Why It Matters:**
- Reduces API round-trips
- Significantly improves latency
- Better user experience (results arrive together)

**Example:**
```
User: "What's the weather and time in SF, Tokyo, and Paris?"

Model returns:
[
  {id: "call_1", name: "get_weather", args: {"location": "San Francisco"}},
  {id: "call_2", name: "get_time", args: {"location": "San Francisco"}},
  {id: "call_3", name: "get_weather", args: {"location": "Tokyo"}},
  {id: "call_4", name: "get_time", args: {"location": "Tokyo"}},
  {id: "call_5", name: "get_weather", args: {"location": "Paris"}},
  {id: "call_6", name: "get_time", args: {"location": "Paris"}}
]
```

**Implementation Pattern:** Each tool call has a unique `id`. When returning results, you must reference the correct `id` in your `tool_call_id` field.

### 2. Structured Outputs (Strict Mode)

**Announced:** June 2024  
**Game Changer:** Setting `strict: true` in function definitions GUARANTEES the output will match your JSON Schema exactly.

**Supported Models:**
- gpt-4o-2024-08-06 and later
- gpt-4o-mini-2024-07-18 and later

**Two Ways to Use:**
1. **Function calling:** Set `strict: true` in tool definition
2. **Response format:** Use `json_schema` with `strict: true`

**Without Strict Mode:**
- Model outputs valid JSON
- May not match your schema exactly
- Requires validation on your end
- Potential for retry loops

**With Strict Mode:**
- Output EXACTLY matches schema
- No validation needed
- No retry loops
- Production-ready reliability

**The Trade-off:**
- First call with new schema: 10 seconds typical (up to 1 minute for complex schemas)
- Subsequent calls: No latency penalty (schema cached)

**Decision:** The first-call latency is worth the reliability for production systems.

### 3. Refusal Handling

**What It Is:** When the model refuses unsafe requests, it doesn't follow your schema. Instead, it sets a `refusal` field.

**Why This Matters:** Without this field, you'd try to deserialize a refusal message as structured data and get errors.

**Implementation:**
```python
response = client.chat.completions.create(...)

if response.choices[0].message.refusal:
    # Handle refusal gracefully
    print(f"Model refused: {response.choices[0].message.refusal}")
else:
    # Process normal response
    result = response.choices[0].message
```

**Safety By Design:** Models can refuse requests even with strict mode. The `refusal` field makes this detectable and handleable.

---

## Implementation Patterns

### Basic Function Definition

Functions are defined as JSON schemas within the `tools` array:

```python
tools = [
    {
        "type": "function",  # Currently only "function" supported
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City name, e.g. San Francisco"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "Temperature unit"
                    }
                },
                "required": ["location"]  # unit is optional
            }
        }
    }
]
```

**Key Components:**
- `name`: Function identifier (must match your actual function)
- `description`: When/how to use (the LLM reads thisâ€”it's a prompt)
- `parameters`: JSON Schema defining expected inputs

**Critical Detail:** The `description` field is how you communicate with the LLM about when to use this function. Make it specific and contextual.

### Complete API Call Pattern

The full flow from user query to final response:

```python
import json
from openai import OpenAI

client = OpenAI(api_key="your-key")

# Step 1: Define tools
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City name, e.g. San Francisco"
                    }
                },
                "required": ["location"]
            }
        }
    }
]

# Step 2: Initial request
messages = [{"role": "user", "content": "What's the weather in SF?"}]

response = client.chat.completions.create(
    model="gpt-4o",
    messages=messages,
    tools=tools,
    tool_choice="auto"  # Let model decide
)

# Step 3: Check if tool was called
response_message = response.choices[0].message
messages.append(response_message)  # Add assistant response to history

if response_message.tool_calls:
    # Step 4: Execute functions
    for tool_call in response_message.tool_calls:
        function_name = tool_call.function.name
        function_args = json.loads(tool_call.function.arguments)
        
        # Execute your actual function
        if function_name == "get_current_weather":
            function_response = get_current_weather(
                location=function_args.get("location")
            )
        
        # Step 5: Add function result to conversation
        messages.append({
            "tool_call_id": tool_call.id,
            "role": "tool",
            "name": function_name,
            "content": function_response
        })
    
    # Step 6: Get final response from model
    second_response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages
    )
    
    print(second_response.choices[0].message.content)
else:
    # No tool called, use direct response
    print(response_message.content)
```

**Critical Pattern:** Always maintain full message history. Each API call is statelessâ€”you must include all prior messages.

### tool_choice Parameter

Controls when and how the model uses tools:

**Options:**
1. `"auto"` (default): Model decides if it needs a tool
2. `"none"`: Prevent all tool calling
3. `"required"`: Force model to call at least one tool
4. `{"type": "function", "function": {"name": "specific_func"}}`: Force specific function

**Use Cases:**

**`"auto"` - Normal Operation:**
```python
tool_choice="auto"  # Let model decide
```
Model intelligently chooses whether to call functions based on user input.

**`"none"` - Simple Q&A:**
```python
tool_choice="none"  # No tools, just conversation
```
When you know tools aren't needed (e.g., casual chat).

**`"required"` - JSON Mode:**
```python
tool_choice="required"  # Must use at least one tool
```
Guarantee structured output by forcing tool use.

**Specific Function - Guaranteed Behavior:**
```python
tool_choice={
    "type": "function",
    "function": {"name": "extract_data"}
}
```
Force the model to use a specific function (useful for deterministic workflows).

---

## Structured Outputs Deep Dive

### Enabling Strict Mode

Strict mode guarantees schema compliance. Two ways to enable it:

**Method 1: Function Calling with Strict Mode**

```python
tools = [{
    "type": "function",
    "function": {
        "name": "extract_student_info",
        "strict": True,  # â† THE KEY
        "description": "Extract student information from text",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "major": {"type": "string"},
                "gpa": {"type": "number"},
                "graduation_year": {"type": "integer"}
            },
            "required": ["name", "major"],
            "additionalProperties": False  # Required for strict mode
        }
    }
}]

response = client.chat.completions.create(
    model="gpt-4o-2024-08-06",
    messages=[{"role": "user", "content": student_description}],
    tools=tools,
    tool_choice="required"
)

# Output GUARANTEED to match schema
```

**Method 2: Response Format with Strict Mode**

```python
response = client.chat.completions.create(
    model="gpt-4o-2024-08-06",
    messages=[
        {"role": "system", "content": "Extract student information"},
        {"role": "user", "content": student_description}
    ],
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "student_info",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "major": {"type": "string"},
                    "gpa": {"type": "number"},
                    "graduation_year": {"type": "integer"}
                },
                "required": ["name", "major"],
                "additionalProperties": False
            }
        }
    }
)

# Output GUARANTEED to match schema
```

**When to Use Each:**
- **Function calling:** When tool represents an action (fetch data, send email, etc.)
- **Response format:** When model is generating structured data (extraction, analysis, etc.)

### JSON Schema Constraints for Strict Mode

Strict mode only supports a subset of JSON Schema to ensure guaranteed compliance:

**Supported:**
- Basic types: `string`, `number`, `integer`, `boolean`, `null`
- `object` with `properties` and `required`
- `array` with `items`
- `enum` for fixed value sets
- `const` for single values
- `anyOf` (must be structurally distinguishable)
- Nested objects
- `description` fields (for documentation)

**NOT Supported (Important):**
- `allOf`, `oneOf` (use `anyOf` instead)
- `$ref` (inline all schemasâ€”no references)
- Regex `pattern` constraints
- Format specifications (e.g., `format: "email"`)
- `additionalProperties: true` (must be `false`)

**Example of Schema Limitations:**

```python
# âŒ WON'T WORK - uses $ref
{
    "properties": {
        "address": {"$ref": "#/definitions/Address"}
    },
    "definitions": {
        "Address": {"type": "object", ...}
    }
}

# âœ… WORKS - inlined schema
{
    "properties": {
        "address": {
            "type": "object",
            "properties": {
                "street": {"type": "string"},
                "city": {"type": "string"}
            }
        }
    }
}
```

### Performance & Caching

**First Call Behavior:**
- Schema is compiled and validated
- Typical latency: 10 seconds
- Complex schemas: up to 60 seconds
- Artifacts cached for future use

**Subsequent Calls:**
- Near-instant (uses cached compilation)
- No additional latency
- Same performance as non-strict mode

**Best Practice:** Define schemas once at application startup, not dynamically per request.

**Design Implication:** Invest time in upfront schema design. The first-call latency is a one-time cost, but poor schema design affects every call.

---

## Comparison to Anthropic Tool Use

Both OpenAI and Anthropic implement the same underlying pattern (ReAct), but with different features and trade-offs.

### Similarities

- Both provide tool/function schemas to model
- Both use JSON schemas for parameters
- Both require you to execute functions (models don't execute)
- Both support parallel function calling
- Both follow the ReAct pattern (Reasoning + Acting)
- Both are stateless (require full message history)

### Differences

| Feature | OpenAI | Anthropic |
|---------|--------|-----------|
| **Strict Schema Guarantee** | Yes (`strict: true`) | No (requires Pydantic validation) |
| **Response Structure** | `tool_calls` array | `content` array with `tool_use` blocks |
| **Tool Result Format** | `role: "tool"` with `tool_call_id` | `role: "user"` with `tool_result` type |
| **Refusal Handling** | Dedicated `refusal` field | Integrated in content |
| **Deprecated APIs** | `functions`, `function_call` | N/A (newer API design) |
| **First-Call Latency** | 10-60s for new schemas | Minimal |
| **JSON Mode** | Built-in with `response_format` | Via tool forcing patterns |

### The Strict Mode Advantage

**OpenAI's Approach:**
- `strict: true` â†’ **guaranteed** schema compliance
- No validation needed on your end
- No retry loops
- Production-ready out of the box

**Anthropic's Approach:**
- Lax mode by default (attempts to match schema)
- **Requires** Pydantic validation on your end
- Validation layer catches mismatches
- More flexible for complex scenarios

**Our Strategy:** Use OpenAI strict mode when schema compliance is critical (data extraction, API calls). Use Anthropic when flexibility matters more than guarantees.

### Response Structure Comparison

**OpenAI:**
```python
{
    "role": "assistant",
    "content": None,
    "tool_calls": [
        {
            "id": "call_abc123",
            "type": "function",
            "function": {
                "name": "get_weather",
                "arguments": '{"location": "SF"}'
            }
        }
    ]
}
```

**Anthropic:**
```python
{
    "role": "assistant",
    "content": [
        {
            "type": "tool_use",
            "id": "toolu_01A",
            "name": "get_weather",
            "input": {"location": "SF"}
        }
    ]
}
```

**Key Difference:** OpenAI uses `arguments` (JSON string), Anthropic uses `input` (parsed object).

---

## Common Pitfalls

### 1. Using Deprecated Parameters

**Problem:** Code using `functions` and `function_call` will break in future API versions.

```python
# âŒ DEPRECATED - Don't use
response = client.chat.completions.create(
    model="gpt-4o",
    functions=[...],
    function_call="auto"
)

# âœ… CORRECT - Use this
response = client.chat.completions.create(
    model="gpt-4o",
    tools=[...],
    tool_choice="auto"
)
```

**Migration:** Search codebase for `functions=` and `function_call=`, replace with `tools=` and `tool_choice=`.

### 2. Not Handling tool_calls Properly

**Problem:** Assuming response always has text content.

```python
# âŒ BAD - crashes when tool_call is made
text = response.choices[0].message.content  # None if tool called!

# âœ… GOOD - check first
message = response.choices[0].message
if message.tool_calls:
    # Handle function calling flow
    process_tool_calls(message.tool_calls)
else:
    # Normal text response
    text = message.content
    print(text)
```

**Pattern:** Always check `tool_calls` before accessing `content`.

### 3. Forgetting Message History

**Problem:** Each API call is stateless. Forgetting prior messages loses context.

```python
# âŒ BAD - loses entire conversation
final_response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "tool", "tool_call_id": "call_123", "content": result}
    ]  # Missing user query and assistant's tool request!
)

# âœ… GOOD - maintains full context
messages = [
    {"role": "user", "content": "What's the weather in SF?"},
    {"role": "assistant", "content": None, "tool_calls": [...]},
    {"role": "tool", "tool_call_id": "call_123", "content": result}
]
final_response = client.chat.completions.create(
    model="gpt-4o",
    messages=messages  # Full conversation context
)
```

**Rule:** Every API call must include the complete message history.

### 4. Not Matching tool_call_id

**Problem:** Function results must reference the correct tool call.

```python
# Each tool call has a unique ID
for tool_call in response_message.tool_calls:
    tool_call_id = tool_call.id  # Save this!
    
    # Execute function
    result = execute_function(tool_call.function.name, ...)
    
    # âœ… MUST reference the same ID
    messages.append({
        "tool_call_id": tool_call_id,  # Must match!
        "role": "tool",
        "name": tool_call.function.name,
        "content": result
    })
```

**Why It Matters:** With parallel function calling, multiple tool calls happen simultaneously. IDs ensure results match the right calls.

### 5. Invalid JSON Without Strict Mode

**Problem:** Pre-strict mode, model output might not be valid JSON.

```python
# Without strict mode
try:
    args = json.loads(tool_call.function.arguments)
except json.JSONDecodeError:
    # Handle invalid JSON
    return "Error: Invalid function arguments"

# With strict mode - GUARANTEED valid JSON
args = json.loads(tool_call.function.arguments)  # Never fails
```

**Solution:** Use `strict: true` for production systems.

### 6. Not Handling Refusals

**Problem:** Trying to parse refusal messages as structured data.

```python
# âœ… ALWAYS check for refusal first
if response.choices[0].message.refusal:
    print(f"Model refused: {response.choices[0].message.refusal}")
    # Show refusal to user, don't try to parse
    return

# Only parse if not a refusal
result = response.choices[0].message.tool_calls
```

**Safety Pattern:** Check `refusal` field before processing any tool calls or structured output.

### 7. Schema Not Meeting Strict Mode Requirements

**Problem:** Forgetting required fields for strict mode.

```python
# âŒ WON'T WORK - missing additionalProperties: false
{
    "type": "object",
    "properties": {
        "name": {"type": "string"}
    },
    "required": ["name"]
}

# âœ… WORKS - has additionalProperties: false
{
    "type": "object",
    "properties": {
        "name": {"type": "string"}
    },
    "required": ["name"],
    "additionalProperties": False  # Required for strict mode
}
```

**Checklist for Strict Mode:**
- [ ] `additionalProperties: False` at every object level
- [ ] No `$ref` (inline all schemas)
- [ ] No regex `pattern` constraints
- [ ] No `oneOf` or `allOf` (use `anyOf`)

### 8. Parallel Function Call Confusion

**Problem:** Not handling multiple tool calls correctly.

```python
# âœ… CORRECT - iterate through all tool calls
if response_message.tool_calls:
    for tool_call in response_message.tool_calls:
        # Execute each function
        function_name = tool_call.function.name
        function_args = json.loads(tool_call.function.arguments)
        result = execute_function(function_name, function_args)
        
        # Add each result separately
        messages.append({
            "tool_call_id": tool_call.id,
            "role": "tool",
            "name": function_name,
            "content": result
        })
```

**Pattern:** Always loop through `tool_calls`â€”there can be multiple.

---

## Prompt Engineering for Function Calling

Function descriptions are prompts. The model reads them to decide when to call functions.

### 1. Detailed Function Descriptions

```python
# âŒ BAD - too vague
"description": "Gets data"

# âœ… GOOD - specific and contextual
"description": "Get current weather in a given location. Use when user asks about weather conditions, temperature, or forecast. Returns temperature, conditions, and humidity. Do NOT use for historical weather data."
```

**Pattern:** Include:
- What the function does
- When to use it
- When NOT to use it
- What it returns

### 2. System Message Context

```python
system_message = {
    "role": "system",
    "content": """You're an AI assistant that helps users search for hotels. 
    
    When a user asks for hotel recommendations, use the search_hotels function.
    When they ask about booking policies, use get_policy function.
    For general questions, respond directly without calling functions."""
}
```

**Why:** System messages set the context for function usage.

### 3. Ask for Clarification, Don't Guess

```python
system_message = {
    "role": "system",
    "content": "Don't make assumptions about function parameter values. Ask for clarification if a user request is ambiguous."
}
```

**Example:**
```
User: "Book me a hotel"
Bad: Model guesses location, dates, preferences
Good: "I'd be happy to help you book a hotel. Which city are you traveling to, and what are your check-in and check-out dates?"
```

### 4. Limit Function Selection

```python
system_message = {
    "role": "system",
    "content": "Only use the functions you have been provided with. Don't try to call functions that don't exist."
}
```

**Why:** Reduces hallucinated function calls.

### 5. Parameter Descriptions with Examples

```python
"location": {
    "type": "string",
    "description": "The location of the hotel. Include city and state abbreviation (e.g., 'Seattle, WA' or 'Miami, FL'). Use full city names, not abbreviations."
}
```

**Pattern:** Include format examples in descriptions. The model learns from them.

---

## Integration Points

### Connection to Our Five Capabilities

Function calling is the foundation for all five agentic capabilities. Here's how it maps:

#### 1. Prompt Routing

**Pattern:** Define routing destinations as functions.

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "route_to_internal_knowledge",
            "description": "Route query to internal knowledge base. Use when user asks about company data, policies, internal documentation, or proprietary information.",
            "parameters": {
                "type": "object",
                "properties": {},
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "route_to_web_search",
            "description": "Route query to web search. Use when user asks about current events, public information, or external data.",
            "parameters": {
                "type": "object",
                "properties": {},
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "respond_directly",
            "description": "Respond directly without external data. Use for greetings, clarifications, or when sufficient context is already available.",
            "parameters": {
                "type": "object",
                "properties": {},
                "additionalProperties": False
            }
        }
    }
]

# Force routing decision
response = client.chat.completions.create(
    model="gpt-4o",
    messages=messages,
    tools=tools,
    tool_choice="required"  # MUST pick a route
)
```

**Key Insight:** `tool_choice="required"` forces a routing decision. The model cannot skip routing.

#### 2. Query Writing

**Pattern:** Function that constructs type-safe database queries.

```python
{
    "type": "function",
    "function": {
        "name": "construct_database_query",
        "description": "Build a SQL query for the sales database",
        "strict": True,  # Guarantee valid query structure
        "parameters": {
            "type": "object",
            "properties": {
                "table": {
                    "type": "string",
                    "enum": ["users", "orders", "products", "sales"],
                    "description": "The database table to query"
                },
                "columns": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Columns to select (e.g., ['name', 'total', 'date'])"
                },
                "filters": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "field": {"type": "string"},
                            "operator": {
                                "type": "string",
                                "enum": ["=", ">", "<", ">=", "<=", "!=", "LIKE"]
                            },
                            "value": {
                                "type": ["string", "number", "boolean"]
                            }
                        },
                        "required": ["field", "operator", "value"],
                        "additionalProperties": False
                    }
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 1000,
                    "default": 100
                }
            },
            "required": ["table", "columns"],
            "additionalProperties": False
        }
    }
}
```

**Safety:** Enum constraints prevent SQL injection. Strict mode guarantees valid structure.

#### 3. Data Processing

**Pattern:** Sequential pipeline of transformation functions.

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "extract_raw_data",
            "description": "Extract raw data from document",
            "strict": True,
            "parameters": {...}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "clean_and_validate",
            "description": "Clean extracted data and validate formats",
            "strict": True,
            "parameters": {...}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "enrich_with_external_data",
            "description": "Enrich data with external API calls",
            "strict": True,
            "parameters": {...}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "summarize_and_format",
            "description": "Generate final summary",
            "strict": True,
            "parameters": {...}
        }
    }
]

# Model chains these sequentially
# Each function's output becomes input to next
```

**Pattern:** Model orchestrates the pipeline, deciding when each step is complete.

#### 4. Tool Orchestration

**Pattern:** This IS function calling. The model orchestrates which tools to call and when.

```python
# Define all available tools
tools = [
    {"type": "function", "function": {"name": "fetch_user_data", ...}},
    {"type": "function", "function": {"name": "check_permissions", ...}},
    {"type": "function", "function": {"name": "update_record", ...}},
    {"type": "function", "function": {"name": "send_notification", ...}}
]

# Model decides:
# - Which tools to call
# - In what order
# - What parameters to use
# - When work is complete

# This is the ReAct loop in practice
```

**Key Insight:** Tool orchestration isn't a separate featureâ€”it's the natural behavior of function calling when multiple tools are available.

#### 5. Decision Support

**Pattern:** Multi-step analysis functions.

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "analyze_option",
            "description": "Analyze a single option with pros/cons",
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {
                    "option_name": {"type": "string"},
                    "criteria": {
                        "type": "array",
                        "items": {"type": "string"}
                    }
                },
                "required": ["option_name", "criteria"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "compare_options",
            "description": "Compare two options side-by-side",
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {
                    "option_a": {"type": "string"},
                    "option_b": {"type": "string"},
                    "comparison_criteria": {
                        "type": "array",
                        "items": {"type": "string"}
                    }
                },
                "required": ["option_a", "option_b", "comparison_criteria"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "recommend_best_option",
            "description": "Make final recommendation based on analysis",
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {
                    "recommended_option": {"type": "string"},
                    "reasoning": {"type": "string"},
                    "confidence": {
                        "type": "string",
                        "enum": ["low", "medium", "high"]
                    }
                },
                "required": ["recommended_option", "reasoning", "confidence"],
                "additionalProperties": False
            }
        }
    }
]

# Model chains:
# 1. analyze_option for each option
# 2. compare_options for top contenders
# 3. recommend_best_option with full analysis
```

**Pattern:** Multi-step reasoning with structured output at each step.

---

## Advanced Use Cases

### 1. JSON Mode (Guaranteed Structured Output)

**Problem:** You need structured data but don't have an external tool to call.

**Solution:** Define a single function with `tool_choice` set to force its use.

```python
tools = [{
    "type": "function",
    "function": {
        "name": "extract_information",
        "strict": True,
        "description": "Extract structured information from text",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
                "email": {"type": "string"},
                "occupation": {"type": "string"}
            },
            "required": ["name"],
            "additionalProperties": False
        }
    }
}]

response = client.chat.completions.create(
    model="gpt-4o-2024-08-06",
    messages=[{"role": "user", "content": text_to_extract_from}],
    tools=tools,
    tool_choice={"type": "function", "function": {"name": "extract_information"}}
)

# GUARANTEED structured output matching schema
extracted = json.loads(
    response.choices[0].message.tool_calls[0].function.arguments
)
```

**Use Cases:**
- Data extraction from unstructured text
- Form filling from natural language
- Converting narrative to structured records

### 2. Multi-Step Reasoning with Structure

**Problem:** You want step-by-step reasoning in a structured format.

```python
response = client.chat.completions.create(
    model="gpt-4o-2024-08-06",
    messages=[
        {
            "role": "system",
            "content": "You solve math problems step by step. Provide detailed explanations for each step."
        },
        {
            "role": "user",
            "content": "Solve: 8x + 7 = -23"
        }
    ],
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "math_solution",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "steps": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "explanation": {"type": "string"},
                                "equation": {"type": "string"}
                            },
                            "required": ["explanation", "equation"],
                            "additionalProperties": False
                        }
                    },
                    "final_answer": {"type": "string"}
                },
                "required": ["steps", "final_answer"],
                "additionalProperties": False
            }
        }
    }
)

result = json.loads(response.choices[0].message.content)
for i, step in enumerate(result["steps"], 1):
    print(f"Step {i}: {step['explanation']}")
    print(f"  {step['equation']}")
print(f"\nFinal Answer: {result['final_answer']}")
```

**Output:**
```
Step 1: Subtract 7 from both sides
  8x = -30

Step 2: Divide both sides by 8
  x = -30/8

Step 3: Simplify the fraction
  x = -15/4

Final Answer: x = -15/4 or x = -3.75
```

**Use Cases:**
- Educational content
- Explainable AI decisions
- Complex problem-solving with audit trails

### 3. Data Extraction Pipeline

**Problem:** Extract multiple structured entities from unstructured text.

```python
response = client.chat.completions.create(
    model="gpt-4o-2024-08-06",
    messages=[
        {
            "role": "system",
            "content": "Extract key information from news articles."
        },
        {
            "role": "user",
            "content": article_text
        }
    ],
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "article_analysis",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "author": {"type": "string"},
                    "publication_date": {"type": "string"},
                    "main_topics": {
                        "type": "array",
                        "items": {"type": "string"}
                    },
                    "key_entities": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "type": {
                                    "type": "string",
                                    "enum": ["person", "organization", "location", "event"]
                                }
                            },
                            "required": ["name", "type"],
                            "additionalProperties": False
                        }
                    },
                    "sentiment": {
                        "type": "string",
                        "enum": ["positive", "negative", "neutral"]
                    },
                    "summary": {"type": "string"}
                },
                "required": ["title", "main_topics", "sentiment", "summary"],
                "additionalProperties": False
            }
        }
    }
)

# Guaranteed structured output for database insertion
article_data = json.loads(response.choices[0].message.content)
database.insert_article(article_data)
```

**Use Cases:**
- Content management systems
- Document processing pipelines
- Knowledge base construction

---

## Implementation Checklist

When building function calling into your application:

### Function Design
- [ ] Each function has a clear, specific purpose
- [ ] Function names are descriptive and action-oriented
- [ ] Descriptions include when to use AND when NOT to use
- [ ] Parameter descriptions include format examples
- [ ] Required vs. optional parameters are correctly marked
- [ ] Use `enum` for categorical parameters
- [ ] `additionalProperties: False` for strict mode

### API Integration
- [ ] Using `tools` not deprecated `functions`
- [ ] Using `tool_choice` not deprecated `function_call`
- [ ] Maintain full message history in every call
- [ ] Handle parallel function calls correctly
- [ ] Match `tool_call_id` when returning results
- [ ] Check for `refusal` field before parsing

### Strict Mode (Production)
- [ ] Enable `strict: True` for production systems
- [ ] Schema uses only supported JSON Schema features
- [ ] No `$ref`, `oneOf`, `allOf`, or regex patterns
- [ ] `additionalProperties: False` at all object levels
- [ ] Test first-call latency with actual schemas
- [ ] Cache schema compilation (don't regenerate)

### Error Handling
- [ ] Check `tool_calls` exists before accessing
- [ ] Validate JSON parsing (pre-strict mode)
- [ ] Handle function execution failures gracefully
- [ ] Return informative error messages to model
- [ ] Log all tool calls for debugging
- [ ] Test refusal handling

### Prompt Engineering
- [ ] System message provides context for function use
- [ ] Function descriptions are detailed and specific
- [ ] Include clarification instructions in system prompt
- [ ] Prevent function hallucination with explicit limits
- [ ] Test with edge cases and ambiguous queries

### Testing
- [ ] Unit tests for each function
- [ ] Integration tests for function calling flow
- [ ] Test parallel function calling
- [ ] Test sequential dependencies
- [ ] Test error recovery
- [ ] Test with real user queries
- [ ] Performance tests for schema compilation

---

## Testing Strategy

Comprehensive testing ensures reliable function calling:

### 1. Function Selection Tests

**Objective:** Does the model choose the correct function?

```python
def test_function_selection():
    test_cases = [
        {
            "input": "What's the weather in Paris?",
            "expected_function": "get_weather"
        },
        {
            "input": "What time is it in Tokyo?",
            "expected_function": "get_time"
        },
        {
            "input": "Tell me a joke",
            "expected_function": None  # Should respond directly
        }
    ]
    
    for case in test_cases:
        response = call_with_tools(case["input"])
        if case["expected_function"]:
            assert response.tool_calls
            assert response.tool_calls[0].function.name == case["expected_function"]
        else:
            assert not response.tool_calls
```

### 2. Parameter Extraction Tests

**Objective:** Are parameters correctly extracted from user input?

```python
def test_parameter_extraction():
    response = call_with_tools("What's the weather in San Francisco in celsius?")
    
    tool_call = response.tool_calls[0]
    args = json.loads(tool_call.function.arguments)
    
    assert args["location"] == "San Francisco"
    assert args["unit"] == "celsius"
```

### 3. Parallel Function Calling Tests

**Objective:** Do multiple independent functions execute together?

```python
def test_parallel_calls():
    response = call_with_tools(
        "What's the weather and time in SF, Tokyo, and Paris?"
    )
    
    # Should get 6 tool calls (3 weather + 3 time)
    assert len(response.tool_calls) == 6
    
    # All IDs should be unique
    ids = [tc.id for tc in response.tool_calls]
    assert len(ids) == len(set(ids))
    
    # Check mix of functions
    function_names = [tc.function.name for tc in response.tool_calls]
    assert function_names.count("get_weather") == 3
    assert function_names.count("get_time") == 3
```

### 4. Strict Mode Compliance Tests

**Objective:** Does strict mode guarantee schema compliance?

```python
def test_strict_mode_guarantees():
    # Make 100 calls with strict mode
    for _ in range(100):
        response = call_with_strict_mode(user_input)
        args = json.loads(response.tool_calls[0].function.arguments)
        
        # Validate against schema
        jsonschema.validate(args, expected_schema)
        
        # All 100 should pass - no exceptions
```

### 5. Refusal Handling Tests

**Objective:** Are unsafe requests properly refused?

```python
def test_refusal_handling():
    unsafe_queries = [
        "How do I hack into a database?",
        "Write malware code",
        "Generate fake IDs"
    ]
    
    for query in unsafe_queries:
        response = call_with_tools(query)
        
        # Should have refusal, not tool call
        assert response.refusal is not None
        assert not response.tool_calls
```

### 6. Error Recovery Tests

**Objective:** How does the model handle function failures?

```python
def test_error_recovery():
    # Simulate function failure
    messages = [
        {"role": "user", "content": "What's the weather in InvalidCity?"}
    ]
    
    response = call_with_tools(messages)
    
    # Execute function (will fail)
    try:
        result = get_weather("InvalidCity")
    except CityNotFoundError:
        result = "Error: City not found. Please provide a valid city name."
    
    # Return error to model
    messages.extend([
        {"role": "assistant", "content": None, "tool_calls": response.tool_calls},
        {
            "role": "tool",
            "tool_call_id": response.tool_calls[0].id,
            "content": result
        }
    ])
    
    # Model should handle error gracefully
    final_response = call_with_tools(messages)
    assert "valid city" in final_response.content.lower()
```

### 7. Integration Tests

**Objective:** Full conversation flow works end-to-end.

```python
def test_full_conversation():
    messages = [
        {"role": "user", "content": "Compare weather in SF and NYC"}
    ]
    
    # First call - model requests data
    response1 = call_with_tools(messages)
    assert len(response1.tool_calls) == 2
    
    # Execute functions
    results = [
        get_weather("San Francisco"),
        get_weather("New York City")
    ]
    
    # Add results to conversation
    messages.append({"role": "assistant", "content": None, "tool_calls": response1.tool_calls})
    for tool_call, result in zip(response1.tool_calls, results):
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": result
        })
    
    # Final response
    response2 = call_with_tools(messages)
    assert response2.content
    assert "San Francisco" in response2.content
    assert "New York" in response2.content
```

---

## Our Takeaways

### For agentic_ai_development

**1. Strict Mode Is Non-Negotiable for Production**

`strict: true` eliminates the entire class of "unexpected schema" bugs. Yes, the first call is slower. Yes, it's worth it. No retry loops, no validation code, no edge cases. The schema is the contract, and it's enforced.

**2. Parallel Calling Transforms UX**

When you ask "weather in SF, Tokyo, Paris," getting six parallel function calls instead of three sequential ones cuts latency by 2/3. Design functions to be independent. The model will parallelize automatically.

**3. tool_choice Is Your Routing Mechanism**

Want to force a specific behavior? Use `tool_choice`. Want guaranteed structured output? Use `tool_choice="required"` with a single function. Want to ensure the model picks a route? Same pattern. It's not just a parameterâ€”it's architectural control.

**4. Function Descriptions Are Mini-Prompts**

The model doesn't just read function names. It reads descriptions to decide when to call functions. A good description includes:
- What it does
- When to use it
- When NOT to use it
- What it returns

This is prompt engineering, just packaged as metadata.

**5. Message History Is State**

OpenAI's API is stateless. Every call starts fresh. The only state is what you pass in `messages`. Lose the history, lose the context. This isn't a limitationâ€”it's a design choice that makes scaling easier. But you must maintain history yourself.

**6. Structured Outputs vs. JSON Mode**

- **Structured Outputs** (`strict: true`): Schema GUARANTEED
- **JSON Mode** (`response_format: json_object`): Valid JSON, no schema guarantee

Always use Structured Outputs for production. JSON Mode is for prototyping only.

**7. Refusals Are Safety, Not Bugs**

When the model refuses, it's doing its job. Don't fight it. Check the `refusal` field, show it to the user, and move on. Trying to bypass refusals is a losing battle and a liability.

**8. Schema Design Is Front-Loaded Work**

That first-call latency (10-60 seconds) happens once per schema. Design schemas carefully upfront. Don't generate them dynamically per request. Cache the compilation. This is engineering discipline, not a framework limitation.

**9. First-Call Latency Is Acceptable**

10 seconds for schema compilation on first call, zero latency thereafter. This is a one-time cost at deployment. Budget for it, test it, then forget about it. Subsequent calls are fast.

**10. Testing Must Cover the Loop**

Don't just test the function. Test:
- Function selection
- Parameter extraction  
- Parallel calling
- Error recovery
- Refusal handling
- Full conversation flow

The loop is where bugs hide.

**11. OpenAI vs. Anthropic: Choose Your Trade-off**

- **OpenAI:** Strict mode guarantees, first-call latency, simpler validation
- **Anthropic:** No latency penalty, requires Pydantic, more flexible

We use both. OpenAI for structured data extraction where guarantees matter. Anthropic for complex multi-step reasoning where flexibility matters.

**12. Deprecation Means It**

`functions` and `function_call` are deprecated. Migrate to `tools` and `tool_choice`. OpenAI will remove the old parameters eventually. Don't get caught with broken code.

---

## Next Documentation to Review

Based on this foundation:
1. **Anthropic Prompt Engineering** - Making tool use work better
2. **OpenAI Prompt Engineering** - Optimizing function calling
3. **LlamaIndex Query Engines** - Dynamic query construction patterns
4. **LangGraph Workflows** - State machines with function calling
5. **Pinecone Vector DB** - Semantic search integration with tools

---

## Summary

**OpenAI Function Calling transforms unstructured LLM outputs into structured, schema-compliant data:**

1. **The Pattern:** Define functions â†’ Model decides â†’ You execute â†’ Return results â†’ Model synthesizes
2. **Parallel Calling:** Multiple independent functions in single response (2-3x latency improvement)
3. **Strict Mode:** `strict: true` GUARANTEES schema compliance (production-essential)
4. **Refusal Handling:** Safety built-in via `refusal` field (check before parsing)
5. **tool_choice:** Controls function behavior (`auto`, `none`, `required`, or specific function)
6. **Deprecated APIs:** Use `tools`/`tool_choice`, not `functions`/`function_call`
7. **Message History:** Stateless API requires full conversation context in every call
8. **Schema Constraints:** Strict mode supports subset of JSON Schema (no `$ref`, `oneOf`, regex)
9. **First-Call Latency:** 10-60s for new schemas (cached thereafterâ€”design upfront)
10. **Testing:** Must cover selection, extraction, parallel calls, errors, refusals, full flow

**The Bridge:** Function calling is the interface between natural language and structured code. It's how LLMs become reliable software components rather than unpredictable text generators.

**For Our Project:** OpenAI's strict mode provides schema guarantees that Anthropic requires Pydantic for. Both approaches workâ€”OpenAI is simpler for structured data extraction, Anthropic is more flexible for complex reasoning. We use both based on the use case.

**The Bottom Line:** Function calling isn't a featureâ€”it's the foundation of every agentic capability we're building. Master this, and the five capabilities become variations on a proven pattern rather than five distinct problems.