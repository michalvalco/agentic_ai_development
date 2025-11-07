# Pydantic: Data Validation & Type Safety

**Source:** https://docs.pydantic.dev/latest/  
**Date Accessed:** 2025-11-06  
**Relevance:** Pydantic provides the type safety foundation for all agentic AI tool implementations. It ensures tool inputs/outputs conform to expected schemas, generates JSON schemas automatically from Python types, and bridges the gap between LLM-generated structured data and Python code execution. This is the validation layer that makes our five capabilities reliable and production-ready.

---

## Key Concepts

### What Is Pydantic?

**Core Definition:** Pydantic is a data validation library that uses Python type hints to define data schemas and automatically validates data at runtime.

**Key Insight:** It's primarily a *parsing* library, not just validation. Pydantic guarantees the **types and constraints of the output model**, not necessarily the input data. If input can be coerced to the correct type, Pydantic will do so.

**The Pydantic Promise:**
- Define schemas using pure Python 3.9+ syntax (type hints)
- Automatic validation on instantiation
- Guaranteed type correctness of output
- JSON schema generation
- Serialization/deserialization

### BaseModel: The Foundation

All Pydantic models inherit from `BaseModel`:

```python
from pydantic import BaseModel
from datetime import datetime

class User(BaseModel):
    id: int                      # Required field
    name: str = 'John Doe'       # Optional with default
    signup_ts: datetime | None   # Optional (can be None)
    
# Automatic validation on instantiation
user = User(id=123, signup_ts='2019-06-01 12:22')
```

**What Just Happened:**
- `id` must be provided and will be coerced to int if possible
- `name` uses default if not provided
- `signup_ts` accepts ISO 8601 string or Unix timestamp, converts to `datetime`
- If validation fails, raises `ValidationError` with detailed breakdown

### Validation vs. Parsing

**Parsing:** Converting input data to the correct type
- `'123'` → `123` (string to int)
- `'2019-06-01'` → `datetime(2019, 6, 1)` (string to datetime)
- `b'hello'` → `'hello'` (bytes to string)

**Validation:** Ensuring data meets constraints
- `age: PositiveInt` → must be > 0
- `email: EmailStr` → must be valid email format
- `name: str = Field(max_length=50)` → max 50 characters

**Pydantic does both** in a single instantiation step.



### Strict vs. Lax Mode

Pydantic can operate in two modes:

**Lax Mode (Default):**
- Attempts type coercion when safe
- `'123'` becomes `123` for an `int` field
- `'true'` becomes `True` for a `bool` field
- More forgiving, better for external data

**Strict Mode:**
- No coercion, types must match exactly
- `'123'` raises error for an `int` field (must be actual int)
- Useful for internal APIs where types are controlled

```python
from pydantic import BaseModel, Field

class StrictModel(BaseModel):
    strict_int: int = Field(strict=True)
    lax_int: int
    
# This works
model = StrictModel(strict_int=123, lax_int='456')

# This fails - strict_int doesn't accept string
# model = StrictModel(strict_int='123', lax_int='456')
```

**When to use strict mode:**
- Internal APIs where you control data types
- Performance-critical paths (no coercion overhead)
- When type mismatches indicate serious bugs

---

## Implementation Patterns

### Basic Model Definition

```python
from pydantic import BaseModel, PositiveInt, EmailStr
from datetime import datetime

class Employee(BaseModel):
    id: int
    name: str
    email: EmailStr                    # Validates email format
    age: PositiveInt                   # Must be > 0
    hire_date: datetime
    salary: float | None = None        # Optional
    
# Create instance
emp = Employee(
    id='123',                          # Coerced to int
    name='Jane Doe',
    email='jane@company.com',
    age=30,
    hire_date='2023-01-15'             # Coerced to datetime
)

print(emp.id)         # 123 (int, not string)
print(emp.hire_date)  # datetime.datetime(2023, 1, 15, 0, 0)
```

**Key Points:**
- Type hints define expected types
- Pydantic coerces when possible
- Special types (`EmailStr`, `PositiveInt`) add validation
- `| None` makes fields optional
- Default values make fields optional

### Field Constraints with Field()


The `Field()` function adds constraints and metadata:

```python
from pydantic import BaseModel, Field
from decimal import Decimal

class Product(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    price: Decimal = Field(gt=0, max_digits=10, decimal_places=2)
    quantity: int = Field(ge=0, le=1000)  # >= 0, <= 1000
    description: str = Field(default='', max_length=500)
    
# Field() parameters:
# - gt, ge, lt, le: Greater/less than (equal)
# - min_length, max_length: String/collection length
# - max_digits, decimal_places: Decimal precision
# - pattern: Regex pattern for strings
# - default: Default value
# - alias: Alternative field name for input
```

**Common Field Constraints:**

| Type | Constraints |
|------|-------------|
| `int`, `float` | `gt`, `ge`, `lt`, `le` (numerical bounds) |
| `str` | `min_length`, `max_length`, `pattern` |
| `Decimal` | `max_digits`, `decimal_places` |
| `list`, `dict` | `min_length`, `max_length` |

### Custom Validators


Pydantic provides multiple ways to add custom validation logic:

#### Field Validators (@field_validator)

```python
from pydantic import BaseModel, field_validator
from datetime import datetime

class Employee(BaseModel):
    name: str
    birth_date: datetime
    
    @field_validator('name')
    @classmethod
    def name_must_not_be_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError('Name cannot be empty')
        return v.strip()
    
    @field_validator('birth_date')
    @classmethod
    def must_be_adult(cls, v: datetime) -> datetime:
        age = (datetime.now() - v).days // 365
        if age < 18:
            raise ValueError('Employee must be at least 18 years old')
        return v
```

**Validator Types:**

1. **After Validators** (default): Run after Pydantic's validation
   - Type is already validated
   - Generally safer and easier to implement

2. **Before Validators** (`mode='before'`): Run before Pydantic's validation
   - Receive raw input (any type)
   - Can transform data before type validation


3. **Wrap Validators** (`mode='wrap'`): Wrap around Pydantic's validation
   - Full control over validation process
   - Can catch and handle ValidationErrors

```python
from typing import Any
from pydantic import BaseModel, field_validator, ValidationError
from pydantic.functional_validators import ValidatorFunctionWrapHandler

class Model(BaseModel):
    my_string: str = Field(max_length=5)
    
    @field_validator('my_string', mode='wrap')
    @classmethod
    def truncate_if_too_long(
        cls, 
        value: Any, 
        handler: ValidatorFunctionWrapHandler
    ) -> str:
        try:
            return handler(value)  # Try normal validation
        except ValidationError as e:
            if e.errors()[0]['type'] == 'string_too_long':
                return value[:5]   # Truncate instead of error
            raise
```

#### Annotated Validators

Cleaner syntax using `Annotated`:

```python
from typing import Annotated
from pydantic import AfterValidator, BaseModel

def is_even(value: int) -> int:
    if value % 2 == 1:
        raise ValueError(f'{value} is not even')
    return value

EvenNumber = Annotated[int, AfterValidator(is_even)]

class Model(BaseModel):
    my_number: EvenNumber
    list_of_evens: list[EvenNumber]  # Validates each list item
```


**Benefit:** Reusable across multiple models. Define once, use everywhere.

#### Model Validators (@model_validator)

Validate entire model, checking relationships between fields:

```python
from typing_extensions import Self
from pydantic import BaseModel, model_validator

class UserRegistration(BaseModel):
    password: str
    password_confirm: str
    email: str
    
    @model_validator(mode='after')
    def passwords_match(self) -> Self:
        if self.password != self.password_confirm:
            raise ValueError('Passwords do not match')
        return self  # MUST return self
```

**Use Cases for Model Validators:**
- Cross-field validation
- Computing derived fields
- Complex business logic involving multiple fields

### Validation with Context

Pass runtime context to validators:

```python
from pydantic import BaseModel, field_validator, FieldValidationInfo

class Document(BaseModel):
    text: str
    
    @field_validator('text')
    @classmethod
    def remove_stopwords(cls, v: str, info: FieldValidationInfo):
        if context := info.context:
            stopwords = context.get('stopwords', set())
            return ' '.join(w for w in v.split() if w not in stopwords)
        return v

# Use with context
doc = Document.model_validate(
    {'text': 'This is a test'},
    context={'stopwords': {'is', 'a'}}
)
print(doc.text)  # "This test"
```

**Use Case:** Dynamic validation rules based on runtime conditions.



---

## JSON Schema Generation

### Automatic Schema Generation

Pydantic models automatically generate JSON schemas:

```python
from pydantic import BaseModel, Field

class FooBar(BaseModel):
    count: int
    size: float | None = None

class MainModel(BaseModel):
    """This is the description of the main model"""
    foo_bar: FooBar
    snap: int = Field(
        default=42,
        title='The Snap',
        description='this is the value of snap',
        gt=30, 
        lt=50
    )

# Generate JSON schema
schema = MainModel.model_json_schema()
```

**Output:**
```json
{
  "title": "MainModel",
  "description": "This is the description of the main model",
  "type": "object",
  "properties": {
    "foo_bar": {"$ref": "#/$defs/FooBar"},
    "snap": {
      "type": "integer",
      "title": "The Snap",
      "description": "this is the value of snap",
      "exclusiveMinimum": 30,
      "exclusiveMaximum": 50,
      "default": 42
    }
  },
  "required": ["foo_bar"],
  "$defs": {
    "FooBar": {
      "type": "object",
      "properties": {
        "count": {"type": "integer"},
        "size": {"type": "number", "nullable": true}
      },
      "required": ["count"]
    }
  }
}
```

**Key Features:**
- Compliant with JSON Schema Draft 2020-12 and OpenAPI 3.1.0
- Nested models included in `$defs`
- Field descriptions from `Field()` or docstrings
- Enums properly represented
- Required fields automatically identified

### Schema Generation Modes

```python
from pydantic import TypeAdapter

# For validation (input schema)
validation_schema = TypeAdapter(MainModel).json_schema(mode='validation')

# For serialization (output schema)
serialization_schema = TypeAdapter(MainModel).json_schema(mode='serialization')
```

**Use Cases:**
- `mode='validation'`: Schema for tool inputs (what LLM provides)
- `mode='serialization'`: Schema for tool outputs (what tool returns)

### Customizing Schema Templates

For OpenAPI compliance, customize reference templates:

```python
adapter = TypeAdapter(MainModel)
schema = adapter.json_schema(
    ref_template='#/components/schemas/{model}'
)
```

**Integration Point for Tool Use:**
Pydantic models auto-generate the `input_schema` needed for Anthropic/OpenAI tool definitions. No manual JSON schema writing required—type hints become the single source of truth.

---

## Serialization

### Core Serialization Methods

Pydantic provides methods to convert models to dicts and JSON:

```python
from pydantic import BaseModel

class FooBarModel(BaseModel):
    banana: float
    foo: str
    bar: dict

m = FooBarModel(banana=3.14, foo='hello', bar={'whatever': (1, 2)})

# To dict with Python objects
print(m.model_dump())  
# {'banana': 3.14, 'foo': 'hello', 'bar': {'whatever': (1, 2)}}

# To JSON string
print(m.model_dump_json(indent=2))
```

### Serialization Parameters

Control what gets serialized:

```python
class User(BaseModel):
    name: str
    email: str
    password: str
    active: bool = True

user = User(name='John', email='john@example.com', password='secret')

# Exclude specific fields
user.model_dump(exclude={'password'})

# Only include specific fields
user.model_dump(include={'name', 'email'})

# Exclude unset fields
user.model_dump(exclude_unset=True)

# Exclude defaults
user.model_dump(exclude_defaults=True)

# Exclude None values
user.model_dump(exclude_none=True)

# Use field aliases
user.model_dump(by_alias=True)
```

**Key Parameters:**
- `by_alias=True` - Use field aliases instead of names
- `exclude_unset=True` - Omit fields not explicitly provided
- `exclude_defaults=True` - Omit fields with default values  
- `exclude_none=True` - Omit None fields
- `mode='json'` - Ensure JSON-compatible types

### Custom Serializers

#### Field-Level Serialization

```python
from pydantic import BaseModel, field_serializer

class StudentModel(BaseModel):
    courses: set[str]
    
    @field_serializer('courses', when_used='json')
    def serialize_courses_in_order(self, courses: set[str]):
        return sorted(courses)  # Sets → sorted lists in JSON

student = StudentModel(courses={'math', 'physics', 'chemistry'})
print(student.model_dump_json())  
# {"courses": ["chemistry", "math", "physics"]}
```

#### Model-Level Serialization

```python
from pydantic import BaseModel, model_serializer

class Employee(BaseModel):
    id: int
    name: str
    salary: float
    
    @model_serializer
    def serialize_model(self) -> dict:
        return {
            "id": self.id,
            "name": self.name
            # Salary deliberately excluded
        }
```

#### Functional Serialization (Annotated)

```python
from typing import Annotated
from pydantic import PlainSerializer

# Define a custom type with serialization logic
CustomStr = Annotated[
    list,
    PlainSerializer(lambda x: ' '.join(x), return_type=str)
]

class Model(BaseModel):
    words: CustomStr

m = Model(words=['hello', 'world'])
print(m.model_dump_json())  # {"words": "hello world"}
```

### Supported Serialization Types

Pydantic handles these types automatically:
- Standard JSON types (str, int, float, bool, None)
- `datetime`, `date`, `time` (ISO 8601 format)
- `UUID` (string representation)
- `bytes` (base64 encoding)
- `Decimal` (string representation)
- `Path` (string representation)
- Sets (converted to lists)
- Nested models (recursive serialization)

**Tool Use Connection:**
Tools return data to LLMs, which must be serializable. Use `model_dump()` for clean dicts or `model_dump_json()` for JSON strings. For artifacts pattern: return summary string for LLM + full object for downstream processing.

---

## Error Handling

### ValidationError Structure

When validation fails, Pydantic raises `ValidationError` with detailed information:

```python
from pydantic import BaseModel, ValidationError
from datetime import datetime

class User(BaseModel):
    id: int
    signup_ts: datetime
    tastes: dict[str, int]

try:
    User(id='not an int', tastes={})
except ValidationError as e:
    print(e.errors())
```

**Output:**
```python
[
    {
        'type': 'int_parsing',
        'loc': ('id',),
        'msg': 'Input should be a valid integer, unable to parse string as an integer',
        'input': 'not an int',
        'url': 'https://errors.pydantic.dev/2/v/int_parsing'
    },
    {
        'type': 'missing',
        'loc': ('signup_ts',),
        'msg': 'Field required',
        'input': {'id': 'not an int', 'tastes': {}},
        'url': 'https://errors.pydantic.dev/2/v/missing'
    }
]
```

**Error Dictionary Fields:**
- `type` - Error type identifier (e.g., 'int_parsing', 'missing')
- `loc` - Tuple showing field path (supports nested fields)
- `msg` - Human-readable error message  
- `input` - The invalid input value
- `url` - Link to error documentation
- `ctx` - Additional context (optional)

### Custom Error Messages

#### Method 1: Custom Error Types

```python
from pydantic import BaseModel, field_validator
from pydantic_core import PydanticCustomError

class Model(BaseModel):
    foo: str
    
    @field_validator('foo')
    @classmethod
    def value_must_equal_bar(cls, v: str) -> str:
        if v != 'bar':
            raise PydanticCustomError(
                'not_a_bar',
                'value is not "bar", got "{wrong_value}"',
                {'wrong_value': v}
            )
        return v
```

#### Method 2: Post-Processing Errors

```python
def customize_error_messages(e: ValidationError) -> list[dict]:
    """Add custom, user-friendly error messages."""
    CUSTOM_MESSAGES = {
        'int_parsing': 'This is not an integer! 🤦',
        'url_scheme': 'Hey, use the right URL scheme!',
        'missing': 'This field is required'
    }
    
    new_errors = []
    for error in e.errors():
        custom_message = CUSTOM_MESSAGES.get(error['type'])
        if custom_message:
            ctx = error.get('ctx', {})
            error['msg'] = (
                custom_message.format(**ctx) 
                if ctx else custom_message
            )
        new_errors.append(error)
    return new_errors
```

### Formatting Error Locations

Convert nested error locations to readable paths:

```python
def loc_to_dot_sep(loc: tuple) -> str:
    """Convert ('items', 1, 'key') to 'items[1].key'"""
    path = ''
    for i, x in enumerate(loc):
        if isinstance(x, str):
            if i > 0:
                path += '.'
            path += x
        elif isinstance(x, int):
            path += f'[{x}]'
    return path

# Example
loc = ('items', 1, 'name')
print(loc_to_dot_sep(loc))  # "items[1].name"
```

### Best Practices for Error Handling

**DO:**
- Raise `ValueError` or `AssertionError` in validators (Pydantic catches and wraps)
- Return informative error strings to LLMs
- Include suggestions for correction in error messages
- Use custom errors for business logic violations

**DON'T:**
- Raise `ValidationError` directly in validators
- Return raw exceptions to LLMs
- Use generic error messages like "invalid input"
- Ignore validation errors silently

### Tool Use Connection

For agent tool execution, wrap calls in try/except:

```python
from pydantic import ValidationError

def execute_tool(raw_input: dict) -> str:
    """Execute tool with validation."""
    try:
        validated_input = ToolInput(**raw_input)
        result = perform_tool_operation(validated_input)
        return result
    except ValidationError as e:
        # Return detailed error for LLM to understand and retry
        errors = e.errors()
        error_msg = errors[0]['msg']
        field = errors[0]['loc'][0]
        return f"Error in field '{field}': {error_msg}"
    except Exception as e:
        return f"Tool execution failed: {str(e)}"
```

The LLM reads the error message and can reformulate the tool call with corrected parameters.

---

## Common Pitfalls

### 1. Not Understanding Lax vs Strict Mode

**Pitfall:** Assuming no type coercion happens by default.

```python
# This works in lax mode (default)
class Model(BaseModel):
    number: int

m = Model(number='123')  # String coerced to int
print(m.number)  # 123 (int)
```

**Solution:** Use `Field(strict=True)` when exact types are required:
```python
class StrictModel(BaseModel):
    number: int = Field(strict=True)

# This now fails
# m = StrictModel(number='123')  # ValidationError!
```

**Context:** LLM outputs are often strings. Lax mode helps, but be aware of coercion behavior.

### 2. Forgetting to Validate Before Use

**Pitfall:** Passing raw dicts directly to business logic without validation.

```python
# BAD - no validation
def process_user(user_data: dict):
    # What if user_data is malformed?
    name = user_data['name']  # Could raise KeyError
    age = user_data['age']    # Could be wrong type
```

**Solution:** Always instantiate model first:
```python
# GOOD - validation first
def process_user(user_data: dict):
    validated_user = User(**user_data)  # Validates structure and types
    # Now guaranteed: validated_user.name exists and is correct type
    perform_operation(validated_user)
```

### 3. Overcomplicating Validators

**Pitfall:** Complex multi-step logic in a single validator makes debugging difficult.

```python
# BAD - too complex
@field_validator('data')
@classmethod
def transform_and_validate(cls, v):
    v = v.strip().lower()
    v = re.sub(r'\s+', ' ', v)
    if len(v) < 5:
        raise ValueError('too short')
    if not v.startswith('valid'):
        raise ValueError('wrong prefix')
    return v.upper()
```

**Solution:** Break into multiple field validators:
```python
# GOOD - single responsibility
@field_validator('data', mode='before')
@classmethod
def normalize_whitespace(cls, v):
    return v.strip().lower()

@field_validator('data')
@classmethod
def check_length(cls, v):
    if len(v) < 5:
        raise ValueError('too short')
    return v
```

### 4. Not Handling Optional Fields Correctly

**Pitfall:** Confusion between `None`, `Optional`, and defaults.

```python
# These are DIFFERENT
class Model1(BaseModel):
    field: str | None           # Field required, can be None
    
class Model2(BaseModel):
    field: str | None = None    # Field optional, defaults to None
    
class Model3(BaseModel):
    field: str = 'default'      # Field optional, defaults to 'default'

# Model1 requires explicit None
Model1(field=None)  # ✓ OK
# Model1()          # ✗ Error - field required

# Model2 and Model3 don't require field
Model2()  # ✓ OK - field is None
Model3()  # ✓ OK - field is 'default'
```

**For Tools:** Be explicit about required vs. optional parameters. The LLM needs to know what it must provide.

### 5. Poor Error Messages

**Pitfall:** Generic errors like "validation failed" don't help LLMs correct mistakes.

```python
# BAD
@field_validator('age')
@classmethod
def check_age(cls, v):
    if v < 18:
        raise ValueError('invalid')  # Unhelpful!
    return v
```

**Solution:** Provide specific, actionable guidance:
```python
# GOOD
@field_validator('age')
@classmethod
def check_age(cls, v):
    if v < 18:
        raise ValueError(
            'Age must be at least 18. '
            'Provided value was {v}. '
            'Please provide a valid adult age.'
        )
    return v
```

The LLM reads error messages and uses them to reformulate requests.

### 6. Serialization Surprises

**Pitfall:** Not checking what format data actually serializes to.

```python
from datetime import datetime

class Event(BaseModel):
    timestamp: datetime

event = Event(timestamp=datetime.now())
serialized = event.model_dump()
print(serialized['timestamp'])  # datetime object (not string!)

json_serialized = event.model_dump_json()
# NOW it's an ISO string: "2025-11-06T10:30:00"
```

**Solution:** Always test serialized output format. Use `model_dump_json()` when you need JSON strings.

### 7. Validation in Wrong Mode

**Pitfall:** Using `mode='before'` when `mode='after'` would work better.

```python
# BAD - before validator receives untyped data
@field_validator('count', mode='before')
@classmethod
def double_count(cls, v):
    return v * 2  # What if v is a string? Runtime error!
```

**Solution:** Use `mode='after'` by default for type-safe validation:
```python
# GOOD - after validator receives validated int
@field_validator('count')
@classmethod
def double_count(cls, v: int) -> int:
    return v * 2  # v is guaranteed to be int
```

**Rule:** Use 'after' by default (type-safe). Only use 'before' when you need to transform raw input before type validation.

---

## Integration Points

### Connection to Our Five Capabilities

#### 1. Prompt Routing

**Pydantic's Role:** Validates routing decisions, ensures type-safe routing logic.

```python
from pydantic import BaseModel, Field
from typing import Literal

class RouteDecision(BaseModel):
    """Decision about where to route user query."""
    destination: Literal['internal', 'external', 'direct']
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str = Field(min_length=10)

# LLM outputs this, Pydantic validates it
raw_llm_output = {
    'destination': 'internal',
    'confidence': 0.85,
    'reasoning': 'Query asks about company policy'
}

decision = RouteDecision(**raw_llm_output)
# Guaranteed: valid destination, confidence in range, reasoning provided
```

**Pattern:** Route selection becomes a typed, validated decision rather than string matching.

#### 2. Query Writing

**Pydantic's Role:** Validates query parameters, prevents SQL injection via type checking.

```python
from pydantic import BaseModel, Field

class QueryParams(BaseModel):
    """Parameters for constructing safe database query."""
    table: str = Field(pattern=r'^[a-zA-Z_][a-zA-Z0-9_]*$')
    filters: dict[str, str | int | float] = Field(default_factory=dict)
    limit: int = Field(ge=1, le=1000, default=100)
    order_by: str | None = Field(default=None, pattern=r'^[a-zA-Z_]+$')
    
    @field_validator('table')
    @classmethod
    def validate_table_exists(cls, v: str) -> str:
        ALLOWED_TABLES = {'users', 'products', 'orders'}
        if v not in ALLOWED_TABLES:
            raise ValueError(f'Table {v} not allowed. Use: {ALLOWED_TABLES}')
        return v

# LLM constructs query params
params = QueryParams(
    table='users',
    filters={'age': 25, 'city': 'NYC'},
    limit=50
)
# Safe to construct SQL from validated params
```

**Pattern:** Query construction with guaranteed safety and type correctness.

#### 3. Data Processing

**Pydantic's Role:** Enforces input/output contracts, guarantees data quality through pipeline.

```python
class RawData(BaseModel):
    """Input contract for data processing."""
    text: str = Field(min_length=1)
    metadata: dict[str, str] = Field(default_factory=dict)

class ProcessedData(BaseModel):
    """Output contract after processing."""
    cleaned_text: str
    entities: list[str]
    sentiment: float = Field(ge=-1.0, le=1.0)
    word_count: int = Field(gt=0)
    
def process_data(raw: RawData) -> ProcessedData:
    """Type-safe data processing pipeline."""
    # Input guaranteed valid by Pydantic
    cleaned = raw.text.strip().lower()
    entities = extract_entities(cleaned)
    sentiment = analyze_sentiment(cleaned)
    
    # Output validated before return
    return ProcessedData(
        cleaned_text=cleaned,
        entities=entities,
        sentiment=sentiment,
        word_count=len(cleaned.split())
    )
```

**Pattern:** Data transformations with guaranteed input/output contracts at each step.

#### 4. Tool Orchestration

**Pydantic's Role:** THIS IS THE FOUNDATION. Every tool has Pydantic input/output models.

```python
# Tool A
class ToolAInput(BaseModel):
    param1: str

class ToolAOutput(BaseModel):
    result: str
    metadata: dict

# Tool B (depends on Tool A)
class ToolBInput(BaseModel):
    input_from_a: str  # Must match ToolAOutput.result type
    additional_param: int = 10

# Type-safe tool chaining
def orchestrate_tools(initial_input: str) -> str:
    # Step 1
    tool_a_result = tool_a(ToolAInput(param1=initial_input))
    
    # Step 2 - type-safe input from A's output
    tool_b_result = tool_b(ToolBInput(
        input_from_a=tool_a_result.result,
        additional_param=20
    ))
    
    return tool_b_result
```

**Pattern:** Tool chaining with compile-time type safety and runtime validation.

#### 5. Decision Support

**Pydantic's Role:** Structures options, criteria, and recommendations with validation.

```python
from decimal import Decimal

class Option(BaseModel):
    """Represents a decision option."""
    name: str = Field(min_length=1, max_length=100)
    cost: Decimal = Field(gt=0, max_digits=10, decimal_places=2)
    features: list[str] = Field(min_length=1)
    rating: float = Field(ge=0.0, le=5.0)

class DecisionCriteria(BaseModel):
    """Criteria for evaluating options."""
    max_cost: Decimal = Field(gt=0)
    required_features: list[str]
    min_rating: float = Field(ge=0.0, le=5.0)

class Recommendation(BaseModel):
    """Final recommendation with reasoning."""
    chosen_option: Option
    reasoning: str = Field(min_length=50)
    alternatives: list[Option] = Field(max_length=3)
    confidence: float = Field(ge=0.0, le=1.0)
    
    @model_validator(mode='after')
    def chosen_must_meet_criteria(self) -> 'Recommendation':
        # Ensure recommendation is logically sound
        if self.confidence < 0.5:
            raise ValueError(
                'Cannot recommend option with confidence below 0.5'
            )
        return self

# LLM provides structured decision
recommendation = Recommendation(
    chosen_option=Option(
        name='Vendor B',
        cost=Decimal('99.99'),
        features=['feature1', 'feature2'],
        rating=4.5
    ),
    reasoning='Best balance of cost and features...',
    alternatives=[...],
    confidence=0.85
)
```

**Pattern:** Multi-step decision analysis with validated options and recommendations.

---

## Our Takeaways

### For agentic_ai_development

**1. Pydantic IS the Type Safety Layer**

Everything flows through Pydantic:
- LLM outputs → Pydantic models (validation)
- Tool inputs → Pydantic validation
- Tool outputs → Pydantic serialization
- Error messages → Pydantic ValidationError

Without Pydantic, there's no type safety. Without type safety, there's runtime chaos.

**2. JSON Schema Generation = Zero Manual Work**

```python
# This is ALL you write:
class ToolInput(BaseModel):
    query: str
    limit: int = 10

# This is auto-generated:
tool_schema = ToolInput.model_json_schema()
# Ready for Anthropic/OpenAI tool definition
```

**Implication:** Type hints become the single source of truth. Change types, schema updates automatically.

**3. Validation Happens at the Boundary**

```
LLM Output (unstructured, untrusted)
    ↓
[Pydantic Validation] ← THE BOUNDARY
    ↓
Python Code (type-safe, trusted)
```

**Design Principle:** Validate early at the boundary, trust the types thereafter. Don't re-validate inside business logic.

**4. Error Messages Are Part of the ReAct Loop**

```
LLM → Tool Call (invalid parameters)
    ↓
Pydantic → ValidationError (detailed, specific)
    ↓
LLM reads error → Reformulates call
    ↓
Tool Call (valid) → Success
```

**Key Insight:** Error messages guide LLM corrections. Make them specific, actionable, educational.

**5. Lax Mode for LLMs, Strict for Internal APIs**

- **LLM outputs:** Use lax mode (coercion helpful—`'123'` → `123`)
- **Internal APIs:** Use strict mode (exact types required)
- **Reason:** LLMs output strings primarily. Internal code controls types precisely.

**6. Serialization = Tool Return Values**

Tools must return clean, serializable data:
- Simple responses: `str`
- Structured responses: `model.model_dump()`
- JSON responses: `model.model_dump_json()`
- Never return raw Python objects to LLM

**7. Field Descriptions Are Mini-Prompts for LLMs**

```python
class SearchInput(BaseModel):
    query: str = Field(
        description=(
            "Search query using specific keywords. "
            "Example: 'sales data Q3 2024 filtered by region'"
        )
    )
    limit: int = Field(
        default=10,
        description="Maximum results to return (1-100)"
    )
```

**The LLM reads these descriptions** when deciding tool parameters. Treat them like prompts.

**8. Validators Can Guide LLM Behavior**

```python
@field_validator('date')
@classmethod
def date_must_be_future(cls, v: datetime) -> datetime:
    if v < datetime.now():
        raise ValueError(
            'Date must be in the future. '
            'Current date is {datetime.now().date()}. '
            'Please provide a date after today.'
        )
    return v
```

**Error message becomes instruction** for the LLM on next attempt.

**9. Model Validators for Business Logic**

Use `@model_validator` when validation involves multiple fields:
- Cross-field constraints
- Complex business rules
- Derived field computation
- Logical consistency checks

Don't try to cram this into field validators.

**10. Testing Is Trivial**

```python
def test_tool_input():
    # Valid case
    valid = ToolInput(query="test", limit=10)
    assert valid.limit == 10
    
    # Invalid case
    with pytest.raises(ValidationError):
        ToolInput(query="", limit=-1)
    
    # Schema test
    schema = ToolInput.model_json_schema()
    assert 'query' in schema['properties']
```

Test both function logic AND schema correctness. Pydantic makes both straightforward.

---

## Implementation Checklist

When building tools for our five capabilities:

### Model Design
- [ ] Use `BaseModel` for all tool inputs
- [ ] Use `BaseModel` for structured outputs
- [ ] Include `Field()` with descriptions (LLM guidance)
- [ ] Add constraints (`min_length`, `gt`, `pattern`, etc.)
- [ ] Use `Literal` for enum-like fields
- [ ] Mark optional fields explicitly (`| None` or `= default`)
- [ ] Add model docstrings for context

### Validation
- [ ] Use lax mode for LLM-facing inputs (default)
- [ ] Use strict mode for internal APIs when appropriate
- [ ] Add custom validators only when needed
- [ ] Keep validators simple and focused
- [ ] Use `@model_validator` for cross-field logic
- [ ] Test validation with invalid inputs
- [ ] Verify error messages are helpful

### Schema Generation
- [ ] Verify JSON schema output
- [ ] Check that descriptions are clear and actionable
- [ ] Ensure required fields marked correctly
- [ ] Test with actual LLM tool calls
- [ ] Use `mode='validation'` for input schemas
- [ ] Use `mode='serialization'` for output schemas

### Serialization
- [ ] Test serialized output format
- [ ] Use `model_dump()` for dicts
- [ ] Use `model_dump_json()` for JSON strings
- [ ] Handle `datetime`/`UUID`/`Decimal` correctly
- [ ] Exclude unset fields when appropriate (`exclude_unset=True`)
- [ ] Test JSON roundtrip (serialize → deserialize)

### Error Handling
- [ ] Wrap tool execution in `try/except`
- [ ] Catch `ValidationError` separately from other exceptions
- [ ] Return informative error strings to LLM
- [ ] Customize error messages when needed
- [ ] Log validation errors for debugging
- [ ] Test error recovery in agent loop

### Documentation
- [ ] Docstrings on all models
- [ ] Field descriptions for all parameters
- [ ] Examples in docstrings
- [ ] Type hints on all fields
- [ ] Document `model_config` if customized

---

## Testing Strategy

### Unit Tests for Models

```python
import pytest
from pydantic import ValidationError

def test_tool_input_validation():
    # Valid input
    valid = ToolInput(query="test", limit=10)
    assert valid.query == "test"
    assert valid.limit == 10
    
    # Invalid input - empty query
    with pytest.raises(ValidationError) as exc_info:
        ToolInput(query="", limit=10)
    assert 'query' in str(exc_info.value)
    
    # Invalid input - negative limit
    with pytest.raises(ValidationError):
        ToolInput(query="test", limit=-1)
    
    # Type coercion (lax mode)
    coerced = ToolInput(query="test", limit="10")
    assert isinstance(coerced.limit, int)
    assert coerced.limit == 10
```

### Schema Tests

```python
def test_json_schema_generation():
    schema = ToolInput.model_json_schema()
    
    # Check structure
    assert 'properties' in schema
    assert 'query' in schema['properties']
    assert 'limit' in schema['properties']
    
    # Check required fields
    assert 'required' in schema
    assert 'query' in schema['required']
    
    # Check defaults
    assert schema['properties']['limit']['default'] == 10
    
    # Check types
    assert schema['properties']['query']['type'] == 'string'
    assert schema['properties']['limit']['type'] == 'integer'
```

### Serialization Tests

```python
def test_serialization():
    model = ToolOutput(
        result="test",
        data=[1, 2, 3],
        metadata={'key': 'value'}
    )
    
    # Dict serialization
    d = model.model_dump()
    assert isinstance(d, dict)
    assert d['result'] == "test"
    
    # JSON serialization
    j = model.model_dump_json()
    assert isinstance(j, str)
    assert '"result": "test"' in j
    
    # Roundtrip test
    reloaded = ToolOutput.model_validate_json(j)
    assert reloaded == model
    assert reloaded.result == model.result
```

### Integration Tests with LLM

```python
def test_tool_with_llm():
    """Test that LLM-generated input validates correctly."""
    # Simulate LLM output
    llm_output = {
        'query': 'search term',
        'limit': '25'  # Note: string from LLM
    }
    
    # Validate with Pydantic
    try:
        validated = ToolInput(**llm_output)
        assert isinstance(validated.limit, int)  # Coerced
        
        # Execute tool
        result = execute_tool(validated)
        assert result is not None
    except ValidationError as e:
        pytest.fail(f"LLM generated invalid input: {e.errors()}")
```

### Validator Tests

```python
def test_custom_validators():
    """Test that custom validators work correctly."""
    
    # Should pass
    valid = DateModel(event_date=datetime(2026, 1, 1))
    assert valid.event_date.year == 2026
    
    # Should fail - date in past
    with pytest.raises(ValidationError) as exc_info:
        DateModel(event_date=datetime(2020, 1, 1))
    
    error = exc_info.value.errors()[0]
    assert 'future' in error['msg'].lower()
```

---

## Comparison to Alternatives

### Pydantic vs. Raw JSON Schema

**Raw JSON Schema (Manual):**
```python
tool_schema = {
    "name": "search",
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "limit": {"type": "integer", "default": 10}
        },
        "required": ["query"]
    }
}

# No validation, no type safety, manual updates
```

**Pydantic (Automatic):**
```python
class SearchInput(BaseModel):
    query: str
    limit: int = 10

# Schema auto-generated
schema = SearchInput.model_json_schema()

# Validation built-in
validated = SearchInput(**llm_output)
```

**Takeaway:** Pydantic eliminates boilerplate, adds type safety, and keeps schemas in sync with code.

### Pydantic vs. Dataclasses

**Dataclasses:**
```python
from dataclasses import dataclass

@dataclass
class User:
    name: str
    age: int

# No validation
user = User(name=123, age="not a number")  # Silently wrong!
```

**Pydantic:**
```python
class User(BaseModel):
    name: str
    age: int

# Validation automatic
# user = User(name=123, age="not a number")  # ValidationError!
```

**Takeaway:** Dataclasses provide structure but no validation. Pydantic provides both.

---

## Next Documentation to Review

Based on this foundation:
1. **OpenAI Function Calling** - Alternative implementation approach
2. **LlamaIndex Query Engines** - Dynamic query construction patterns
3. **LangGraph Workflows** - State machines with Pydantic state models
4. **Anthropic Prompt Engineering** - Writing prompts that work with Pydantic schemas

---

## Summary

**Pydantic provides the validation and type safety foundation for agentic AI systems:**

1. **Automatic JSON Schema Generation** from type hints eliminates manual schema writing
2. **Runtime Validation** catches LLM output errors before code execution
3. **Type Safety** guarantees correct types throughout tool pipelines
4. **Clear Error Messages** guide LLM to correct mistakes in the ReAct loop
5. **Serialization** handles complex Python types → JSON conversion cleanly
6. **Lax Mode** forgives LLM string outputs via intelligent type coercion
7. **Field Descriptions** act as prompts guiding LLM parameter selection
8. **Validators** enforce business rules and provide correction guidance

**The Bridge:**
Pydantic bridges the gap between unstructured LLM outputs and structured code execution. It's the boundary where untrusted external data becomes trusted, typed internal data.

**For Our Project:**
Pydantic is non-negotiable. Every tool input, every tool output, every routing decision, every query parameter—all flow through Pydantic models. It's the foundation that makes agentic AI reliable enough for production.

Master this bridge, and tools become trustworthy. Skip it, and you're building on quicksand.

---

**Final Insight:** Type safety isn't pedantry—it's the difference between a system that occasionally works and one that consistently works. Pydantic transforms "hope the LLM got it right" into "guarantee the types are correct." That's the difference between a demo and a product.
