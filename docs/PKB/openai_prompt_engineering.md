# OpenAI: Prompt Engineering Guide

**Source:** https://platform.openai.com/docs/guides/prompt-engineering  
**Date Accessed:** 2025-11-06  
**Relevance:** OpenAI's prompt engineering guide represents the complementary perspective to Anthropic's approachâ€”where Anthropic emphasizes structure (XML tags) and chains, OpenAI emphasizes clarity and systematic testing. Both approaches target the same goal: reliable, production-ready outputs from LLMs. Understanding OpenAI's six strategies alongside Anthropic's techniques creates a complete mental model for prompt engineering across providers. The differences matter: OpenAI's native JSON mode and temperature controls vs. Anthropic's prefilling and XML patterns. For our five capabilities, we need both playbooksâ€”OpenAI excels at structured extraction and grounding, Anthropic at complex reasoning chains. This isn't theoreticalâ€”these are battle-tested patterns from billions of production API calls.

---

## Key Concepts

### The Six-Strategy Framework

OpenAI organizes prompt engineering around six core strategies. Unlike Anthropic's hierarchical approach (basic â†’ advanced), OpenAI presents these as complementary techniques that can be combined.

**The Six Strategies:**
1. **Write Clear Instructions** â†’ Specificity eliminates ambiguity
2. **Provide Reference Text** â†’ Grounding prevents hallucination
3. **Split Complex Tasks** â†’ Decomposition improves accuracy
4. **Give Time to Think** â†’ Chain-of-thought unlocks reasoning
5. **Use External Tools** â†’ Compensate for model limitations
6. **Test Changes Systematically** â†’ Measure don't guess

**Key Insight:** These aren't stepsâ€”they're dimensions. A single prompt can (and often should) employ multiple strategies simultaneously.

### The Fundamental Model Behavior

**Critical Understanding:** GPT models are completion engines. They predict "What comes next?" based on the input. There's no special "Q&A mode" or "instruction mode"â€”the model simply generates the most likely continuation.

**This means:**
- Prompts aren't commands; they're context that shapes what's "likely"
- The model doesn't understand your intent; it pattern-matches to training data
- Ambiguous prompts â†’ ambiguous outputs (the model can't read your mind)
- Clear patterns in training data â†’ reliable completions

**Example:**
```
Input: "Four score and seven years ago our"
Output: "fathers brought forth on this continent, a new nation..."
# Model recognizes famous text and continues it
```

**Practical Implication:** Don't ask "Why isn't the model following my instructions?" Ask "What completion does my prompt make most likely given the training data?"

### System vs User vs Assistant Messages

The Chat Completions API structures conversations with three message types:

**System Message:**
- Sets behavior, personality, constraints
- Persists across conversation
- Think: "Rules of engagement"
- Example: "You are a helpful assistant that answers concisely"

**User Message:**
- The human's input
- Can contain instructions, questions, or content
- Think: "What the human wants"

**Assistant Message:**
- The model's response
- Used in conversation history for context
- Can be prefilled to guide format (less direct than Anthropic)
- Think: "What the AI has said"

**Key Difference from Anthropic:** OpenAI doesn't have native prefilling support like Anthropic. You can approximate it by including incomplete assistant messages, but it's not the primary pattern.

### Model Parameters: Temperature and Top_p

These control randomness/creativity in outputs:

**Temperature (0-2):**
- **Low (0-0.3):** Deterministic, focused, factual
  - Use for: Data extraction, code generation, factual Q&A
- **Medium (0.7-1.0):** Balanced creativity and coherence
  - Use for: General conversation, content writing
- **High (1.0-2.0):** Creative, diverse, unpredictable
  - Use for: Brainstorming, creative writing, varied responses

**Top_p (0-1):**
- Alternative to temperature (don't adjust both)
- **Low (0.1-0.5):** Narrow token selection
- **High (0.9-1.0):** Wide token selection

**Best Practice:** Start with temperature=0 for production tasks, increase only if outputs are too repetitive.

---

## Strategy 1: Write Clear Instructions

### The Core Principle

Specificity compounds. Every vague word multiplies possible interpretations. Models can't read your mindâ€”if you want specific behavior, specify it explicitly.

### Tactics

#### Tactic 1: Include Details

**Weak:**
```python
prompt = "Write a summary of this article"
```

**Strong:**
```python
prompt = """Summarize the following article in exactly 3 bullet points.
Each bullet point should:
- Be 1-2 sentences
- Focus on actionable insights
- Start with a verb (e.g., "Consider", "Implement", "Avoid")

Article:
{article_text}

Summary:
"""
```

**Why This Works:** The model knows the exact format, length, and style required. No guesswork.

#### Tactic 2: Ask the Model to Adopt a Persona

**Pattern:**
```python
system_message = """You are an experienced DevOps engineer with 10 years working with Kubernetes.
Your communication style is:
- Direct and technical
- Assumes audience knows basic concepts
- Prioritizes security and reliability over convenience
- Provides specific commands and configs, not theory"""

user_message = "How should I deploy a stateful application on K8s?"
```

**Why This Works:** Persona sets implicit context that would otherwise require explicit instructions for every response.

**Use Cases:**
- **Code reviews:** "You are a senior code reviewer focused on security vulnerabilities"
- **Customer support:** "You are an empathetic customer service agent who never makes promises about features"
- **Technical writing:** "You are a technical writer who explains complex concepts to non-technical audiences"

#### Tactic 3: Use Delimiters

**Pattern:**
```python
prompt = """Analyze the following customer feedback for sentiment and key issues.

###FEEDBACK###
{customer_message}
###END FEEDBACK###

###INSTRUCTIONS###
1. Rate sentiment: Positive, Neutral, or Negative
2. Extract top 3 issues mentioned
3. Suggest resolution priority (High, Medium, Low)
###END INSTRUCTIONS###

Output format:
Sentiment: ...
Issues: ...
Priority: ...
"""
```

**Supported Delimiters:**
- Triple quotes: `"""..."""`
- XML-style tags: `<feedback>...</feedback>`
- Markdown headers: `### Feedback ###`
- Triple hashes: `###...###`
- Triple backticks: ` ```...``` `

**Why This Works:** Clear boundaries prevent the model from confusing instructions with content. Especially critical when content contains text that could be misinterpreted as instructions.

#### Tactic 4: Specify Steps

**Pattern:**
```python
prompt = """Extract product recommendations from this review following these exact steps:

Step 1: Read the review and identify all product mentions
Step 2: For each product, determine if the reviewer recommends it (yes/no)
Step 3: Extract the specific reason for recommendation or warning
Step 4: Rate confidence in the recommendation (high/medium/low)
Step 5: Format output as JSON

Review:
{review_text}

Now execute each step:
"""
```

**Why This Works:** Sequential steps guide the model through a process, reducing the chance of skipping critical analysis.

#### Tactic 5: Provide Examples

**Few-Shot Pattern:**
```python
prompt = """Classify the urgency of customer support tickets.

Example 1:
Ticket: "My account is locked and I can't access my data"
Urgency: HIGH
Reasoning: Customer is blocked from using the product

Example 2:
Ticket: "The export button is slightly misaligned on mobile"
Urgency: LOW
Reasoning: Minor UI issue with available workarounds

Example 3:
Ticket: "Seeing intermittent timeouts when uploading files >100MB"
Urgency: MEDIUM
Reasoning: Impacts functionality but only for specific use case

Now classify this ticket:
Ticket: {new_ticket}
Urgency:
"""
```

**How Many Examples:**
- 0 (zero-shot): For simple, unambiguous tasks
- 1-2 (few-shot): For most classification/extraction tasks
- 3-5 (multi-shot): For nuanced tasks requiring pattern recognition

**Why This Works:** Examples are more effective than descriptions. "Show don't tell" principle.

#### Tactic 6: Specify Output Length

**Pattern:**
```python
# Vague
prompt = "Summarize this report concisely"

# Specific  
prompt = "Summarize this report in exactly 100 words"

# More specific
prompt = "Summarize this report in 2-3 paragraphs (approximately 200-250 words)"

# Most specific
prompt = """Summarize this report following this structure:
- Executive summary: 1 paragraph (50 words)
- Key findings: 3 bullet points (each 20-30 words)
- Recommendations: 2 bullet points (each 20-30 words)
Total: approximately 200 words"""
```

**Why This Works:** Word/sentence counts are concrete constraints the model can follow.

### Implementation Example: Complete Clear Instructions

```python
from openai import OpenAI

client = OpenAI(api_key="your-key")

system_message = """You are a technical documentation writer.
Your goal is to make complex concepts understandable to junior developers.

Guidelines:
- Use simple, concrete language
- Provide code examples for every concept
- Define jargon before using it
- Organize content with clear headings"""

user_message = """Explain how OAuth 2.0 authorization code flow works.

###REQUIREMENTS###
- Length: 300-400 words
- Include a sequence diagram using Mermaid syntax
- Provide a Python code example
- Explain security considerations
###END REQUIREMENTS###

###FORMAT###
## Overview
[explanation]

## Sequence Diagram
```mermaid
[diagram]
```

## Code Example
```python
[code]
```

## Security Considerations
[bullet points]
###END FORMAT###
"""

response = client.chat.completions.create(
    model="gpt-4",
    temperature=0.3,  # Low for factual content
    messages=[
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message}
    ]
)

output = response.choices[0].message.content
```

---

## Strategy 2: Provide Reference Text

### The Core Principle

Grounding prevents fabrication. When the model has authoritative source material, it can answer from facts rather than from patterns in training data (which might be outdated or incorrect).

**The Promise:** Fewer hallucinations, more trustworthy outputs.

### Tactics

#### Tactic 1: Instruct to Answer Using Reference

**Pattern:**
```python
reference_text = """
[DOCUMENT: Product Specifications v2.1]
Model XR-500 supports:
- WiFi 6E (802.11ax)
- Bluetooth 5.2
- USB-C charging (18W)
- Battery life: 12 hours typical use
Launch date: March 2024
Price: $399
"""

prompt = f"""Using ONLY the information in the document below, answer the customer question.
If the answer isn't in the document, respond with "This information isn't available in our documentation."

DOCUMENT:
{reference_text}

QUESTION: What's the battery life of Model XR-500?

ANSWER:"""

response = client.chat.completions.create(
    model="gpt-4",
    temperature=0,  # Zero for factual accuracy
    messages=[{"role": "user", "content": prompt}]
)
```

**Why This Works:** Explicit instruction to stay grounded + providing source material dramatically reduces hallucination.

#### Tactic 2: Instruct to Answer with Citations

**Pattern:**
```python
documents = """
[1] Study by Johnson et al (2023): "Regular exercise reduces cardiovascular disease risk by 30%"
[2] WHO Report (2024): "150 minutes of moderate exercise per week recommended for adults"  
[3] Harvard Medical (2023): "Strength training twice weekly reduces injury risk by 40%"
"""

prompt = f"""Answer the following question using the provided research documents.
You MUST cite your sources using [X] notation after each claim.

DOCUMENTS:
{documents}

QUESTION: What are the benefits of regular exercise?

REQUIREMENTS:
- Every factual claim must have a citation
- Use multiple sources when available
- If information conflicts, mention both views
- If asked information isn't covered, state that clearly

ANSWER:"""
```

**Expected Output:**
```
Regular exercise provides significant health benefits. It reduces cardiovascular 
disease risk by 30% [1] and injury risk by 40% when combined with strength 
training [3]. Health organizations recommend at least 150 minutes of moderate 
exercise weekly [2]. The evidence consistently shows exercise is crucial for 
maintaining health across multiple dimensions.
```

**Why This Works:** Citations create accountability. The model must:
1. Find relevant information in sources
2. Attribute claims to specific sources
3. Avoid making up "facts" (no source = no claim)

### Implementation Example: Building a Grounded Q&A System

```python
def grounded_qa(question: str, documents: list[str]) -> dict:
    """Answer questions with citations from provided documents."""
    
    # Number documents for citation
    numbered_docs = "\n\n".join([
        f"[{i+1}] {doc}" for i, doc in enumerate(documents)
    ])
    
    system_message = """You are a research assistant that answers questions strictly based on provided sources.

    Rules:
    - Every claim must be cited with [X] notation
    - If the answer isn't in the sources, say so explicitly
    - If sources conflict, present both views with citations
    - Don't add information from your training data"""
    
    user_message = f"""SOURCES:
{numbered_docs}

QUESTION: {question}

Provide an answer with inline citations [X] for every claim.
If the question cannot be answered from the sources, respond: "The provided sources don't contain information about this question."

ANSWER:"""
    
    response = client.chat.completions.create(
        model="gpt-4",
        temperature=0,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]
    )
    
    answer = response.choices[0].message.content
    
    # Extract cited sources
    import re
    citations = set(re.findall(r'\[(\d+)\]', answer))
    cited_docs = [documents[int(c)-1] for c in citations if int(c) <= len(documents)]
    
    return {
        "answer": answer,
        "cited_sources": cited_docs,
        "sources_used": len(citations)
    }

# Usage
result = grounded_qa(
    question="What's the recommended dosage?",
    documents=[
        "Adults: Take 1 tablet daily with food",
        "Children under 12: Consult a physician",
        "Do not exceed 2 tablets in 24 hours"
    ]
)
```

### Best Practices for Reference Text

**1. Position Matters:**
```python
# LESS EFFECTIVE - reference at end
prompt = f"Answer: {question}\n\nReference: {long_document}"

# MORE EFFECTIVE - reference before question
prompt = f"Reference: {long_document}\n\nAnswer: {question}"
```

**Why:** Models are trained to use earlier context more heavily. Put critical information first.

**2. Format for Scannability:**
```python
# LESS EFFECTIVE - wall of text
reference = "Product A costs $100 and Product B costs $200..."

# MORE EFFECTIVE - structured
reference = """
Product A:
- Price: $100
- Features: X, Y, Z
- Availability: In stock

Product B:
- Price: $200  
- Features: A, B, C
- Availability: Backordered
"""
```

**3. Length Management:**
- GPT-4: ~8K tokens for reference text (safe)
- GPT-4 Turbo: ~32K tokens for reference text
- For longer documents: Use semantic search to retrieve relevant chunks first

---

## Strategy 3: Split Complex Tasks

### The Core Principle

Models perform better on focused sub-tasks than monolithic complex tasks. Decomposition reduces error propagation and makes debugging easier.

**The Promise:** Higher accuracy through task simplification.

### Tactics

#### Tactic 1: Intent Classification for Routing

**Pattern:**
```python
# Step 1: Classify intent
classification_prompt = f"""Classify the customer query into exactly one category:
- technical_support: Issues with product functionality
- billing: Payment, refunds, subscriptions
- product_info: Questions about features or specs
- feedback: Compliments or complaints

Query: {user_query}

Category:"""

intent = client.chat.completions.create(
    model="gpt-4",
    temperature=0,
    messages=[{"role": "user", "content": classification_prompt}],
    max_tokens=20
).choices[0].message.content.strip()

# Step 2: Route to specialized handler
if intent == "technical_support":
    response = handle_technical(user_query)
elif intent == "billing":
    response = handle_billing(user_query)
# ... etc
```

**Why This Works:** Each handler can have specialized prompts, tools, and context. Better than one prompt trying to handle everything.

#### Tactic 2: Summarize Long Conversations

**Pattern:**
```python
def manage_conversation_context(messages: list, max_tokens: int = 2000):
    """Summarize old messages when conversation gets too long."""
    
    # Calculate current token count (rough estimate)
    current_tokens = sum(len(m["content"]) // 4 for m in messages)
    
    if current_tokens > max_tokens:
        # Summarize older messages
        old_messages = messages[:-5]  # Keep last 5 messages
        
        summary_prompt = f"""Summarize this conversation history concisely:
{json.dumps(old_messages, indent=2)}

Summary (200 words max):"""
        
        summary = client.chat.completions.create(
            model="gpt-4",
            temperature=0,
            messages=[{"role": "user", "content": summary_prompt}],
            max_tokens=400
        ).choices[0].message.content
        
        # Replace old messages with summary
        messages = [
            {"role": "system", "content": f"Previous conversation summary: {summary}"},
            *messages[-5:]  # Recent messages
        ]
    
    return messages
```

**Why This Works:** Keeps context within limits while preserving critical information.

#### Tactic 3: Piecewise Document Summarization

**Pattern:**
```python
def summarize_long_document(document: str, chunk_size: int = 4000) -> str:
    """Summarize document in chunks, then summarize summaries."""
    
    # Split into chunks
    chunks = [document[i:i+chunk_size] for i in range(0, len(document), chunk_size)]
    
    # Summarize each chunk
    chunk_summaries = []
    for i, chunk in enumerate(chunks):
        prompt = f"""Summarize this section concisely (100 words):

Section {i+1}/{len(chunks)}:
{chunk}

Summary:"""
        
        summary = client.chat.completions.create(
            model="gpt-4",
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200
        ).choices[0].message.content
        
        chunk_summaries.append(summary)
    
    # Combine summaries
    if len(chunk_summaries) > 1:
        combined_prompt = f"""Create a cohesive summary from these section summaries:

{' '.join([f"Section {i+1}: {s}" for i, s in enumerate(chunk_summaries)])}

Final summary (300 words):"""
        
        final_summary = client.chat.completions.create(
            model="gpt-4",
            temperature=0,
            messages=[{"role": "user", "content": combined_prompt}],
            max_tokens=500
        ).choices[0].message.content
        
        return final_summary
    
    return chunk_summaries[0]
```

**Why This Works:** Recursive summarization maintains detail while respecting token limits. Known as "MapReduce" pattern in LLM literature.

### Implementation Example: Multi-Step Data Extraction

```python
def extract_structured_data(text: str) -> dict:
    """Extract structured data using multi-step pipeline."""
    
    # Step 1: Extract entities
    extract_prompt = f"""Extract all entities from this text:
- Persons (names)
- Organizations (companies, institutions)
- Locations (cities, countries)
- Dates
- Monetary amounts

Text: {text}

Output as JSON:
```

{
  "persons": [...],
  "organizations": [...],
  "locations": [...],
  "dates": [...],
  "amounts": [...]
}
```"""
    
    entities = client.chat.completions.create(
        model="gpt-4",
        temperature=0,
        messages=[{"role": "user", "content": extract_prompt}]
    ).choices[0].message.content
    
    # Parse JSON
    import json
    entities_dict = json.loads(entities.strip("```json\n").strip("\n```"))
    
    # Step 2: Extract relationships
    relationship_prompt = f"""Given these entities:
{json.dumps(entities_dict, indent=2)}

And this text:
{text}

Extract relationships in format: (entity1, relationship_type, entity2)

Examples:
- (John Smith, works_for, Acme Corp)
- (Acme Corp, located_in, New York)

Relationships:"""
    
    relationships = client.chat.completions.create(
        model="gpt-4",
        temperature=0,
        messages=[{"role": "user", "content": relationship_prompt}]
    ).choices[0].message.content
    
    # Step 3: Validate and structure
    validation_prompt = f"""Review this extracted data for accuracy:

Entities: {json.dumps(entities_dict)}
Relationships: {relationships}

Original text: {text}

Corrections needed? If yes, provide corrected JSON. If no, respond "Valid".
"""
    
    validation = client.chat.completions.create(
        model="gpt-4",
        temperature=0,
        messages=[{"role": "user", "content": validation_prompt}]
    ).choices[0].message.content
    
    return {
        "entities": entities_dict,
        "relationships": relationships,
        "validation": validation
    }
```

**Why Multi-Step Wins:**
- Each step has a focused goal (extract vs. relate vs. validate)
- Errors are isolated to specific steps (easier debugging)
- Can optimize temperature per step (0 for extraction, 0.3 for relationships)
- Can parallelize independent steps

---

## Strategy 4: Give Time to Think

### The Core Principle

Models make more errors when forced to answer immediately. By explicitly requesting step-by-step reasoning, you unlock the model's ability to work through problems systematically.

**This is Chain-of-Thought (CoT) prompting.**

**The Promise:** Dramatically improved accuracy on reasoning tasks (math, logic, analysis).

### Tactics

#### Tactic 1: Instruct to Work Step-by-Step

**Pattern:**
```python
prompt = """Solve this math problem step-by-step:

Problem: A store sells apples for $1.20 each and oranges for $0.80 each.
If Sarah buys 7 apples and 12 oranges, and uses a 15% discount coupon,
how much does she pay?

Show your work step by step before giving the final answer.

Solution:"""

response = client.chat.completions.create(
    model="gpt-4",
    temperature=0,
    messages=[{"role": "user", "content": prompt}]
)
```

**Expected Output:**
```
Step 1: Calculate apple cost: 7 apples Ã— $1.20 = $8.40
Step 2: Calculate orange cost: 12 oranges Ã— $0.80 = $9.60  
Step 3: Calculate subtotal: $8.40 + $9.60 = $18.00
Step 4: Calculate discount: $18.00 Ã— 0.15 = $2.70
Step 5: Calculate final cost: $18.00 - $2.70 = $15.30

Final answer: $15.30
```

**Why This Works:** The model generates intermediate steps, which:
- Forces systematic thinking (not guessing)
- Makes errors detectable (you can see where logic breaks)
- Improves final answer accuracy by 20-30% on complex problems

#### Tactic 2: Use Inner Monologue (Hidden Reasoning)

**Pattern:**
```python
system_message = """When solving problems:
1. First, think through the solution in <thinking> tags
2. Then provide your final answer in <answer> tags

The user will only see the <answer> section."""

user_message = """A train travels 120 miles in 2 hours. At this rate, how long
will it take to travel 450 miles?"""

response = client.chat.completions.create(
    model="gpt-4",
    temperature=0,
    messages=[
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message}
    ]
)

# Extract just the answer
import re
full_response = response.choices[0].message.content
answer = re.search(r'<answer>(.*?)</answer>', full_response, re.DOTALL).group(1)
```

**Expected Output:**
```
<thinking>
Speed = 120 miles / 2 hours = 60 mph
Time needed = 450 miles / 60 mph = 7.5 hours
</thinking>

<answer>
It will take 7.5 hours (7 hours and 30 minutes) to travel 450 miles.
</answer>
```

**Why This Works:**
- Model still gets reasoning benefits of CoT
- User sees clean, concise answer
- Thinking section available for debugging

#### Tactic 3: Ask for Alternative Approaches

**Pattern:**
```python
prompt = """Solve this problem using at least 2 different methods:

Problem: Find the area of a triangle with sides 5, 12, and 13.

For each method:
1. Explain the approach
2. Show calculations
3. Verify the answer

Method 1:"""
```

**Why This Works:** Multiple approaches:
- Catch errors (methods should agree)
- Build confidence in answer
- Teach reasoning patterns

### Implementation Example: Reasoning-Based Code Review

```python
def code_review_with_reasoning(code: str, filename: str) -> dict:
    """Review code with explicit reasoning about issues."""
    
    system_message = """You are a senior code reviewer.
    When reviewing code:
    1. First analyze in <reasoning> tags (potential issues, best practices, security)
    2. Then provide verdict in <review> tags (approve/needs-changes/reject)"""
    
    user_message = f"""Review this code:

Filename: {filename}
```python
{code}
```

Provide:
<reasoning>
- Code quality analysis
- Potential bugs or issues
- Security considerations
- Performance concerns
</reasoning>

<review>
- Verdict: [approve/needs-changes/reject]
- Priority issues: [list]
- Recommendations: [list]
</review>"""
    
    response = client.chat.completions.create(
        model="gpt-4",
        temperature=0.3,  # Slight creativity for finding edge cases
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]
    )
    
    full_response = response.choices[0].message.content
    
    # Extract sections
    reasoning = re.search(r'<reasoning>(.*?)</reasoning>', full_response, re.DOTALL).group(1)
    review = re.search(r'<review>(.*?)</review>', full_response, re.DOTALL).group(1)
    
    return {
        "reasoning": reasoning.strip(),
        "review": review.strip(),
        "full_response": full_response
    }

# Usage
result = code_review_with_reasoning(
    code="""
def process_payment(amount, user_id):
    query = f"UPDATE accounts SET balance = balance - {amount} WHERE id = {user_id}"
    db.execute(query)
    return True
    """,
    filename="payment.py"
)
```

**Expected Reasoning:**
```
<reasoning>
Critical security issues:
1. SQL injection vulnerability: String formatting directly into query
2. No input validation on amount or user_id
3. No error handling for database failures
4. No transaction management (balance could go negative)
5. No logging of payment operations

Best practice violations:
- Hardcoded SQL instead of parameterized queries
- No type hints
- Missing docstring
</reasoning>

<review>
Verdict: REJECT
Priority issues:
1. SQL injection (CRITICAL) - Use parameterized queries
2. No validation (HIGH) - Validate amount > 0, user_id exists
3. No error handling (HIGH) - Wrap in try/except, use transactions

Recommendations:
- Rewrite using parameterized queries
- Add input validation
- Implement transaction management
- Add comprehensive error handling
- Include audit logging
</review>
```

---

## Strategy 5: Use External Tools

### The Core Principle

Models have inherent limitations: they can't browse the web, perform precise calculations, access real-time data, or execute code. External tools compensate for these weaknesses.

**The Promise:** Expand capabilities beyond LLM limitations.

### When to Use External Tools

**Model Limitations:**
1. **Stale knowledge:** Training data cutoff (no current events)
2. **Arithmetic errors:** Floating point math, complex calculations
3. **No real-time data:** Stock prices, weather, current dates
4. **Can't execute code:** Limited to describing code, not running it
5. **No private data access:** Your databases, internal systems

**Tool Categories:**
- **Search/Retrieval:** Web search, vector DB, internal wikis
- **Computation:** Code execution, calculator, symbolic math
- **APIs:** Weather, stock data, CRM systems
- **File Operations:** Read/write files, parse documents
- **Database:** SQL execution, data querying

### Tactics

#### Tactic 1: Embeddings for Knowledge Retrieval

**Pattern:**
```python
from openai import OpenAI
import numpy as np

client = OpenAI()

# Build knowledge base
documents = [
    "Product X supports WiFi 6 and Bluetooth 5.2",
    "Product X battery lasts 12 hours on typical use",
    "Product Y is waterproof up to 50 meters",
    # ... thousands more
]

# Create embeddings once
embeddings = []
for doc in documents:
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=doc
    )
    embeddings.append(response.data[0].embedding)

# Store embeddings (use vector DB in production)
knowledge_base = list(zip(documents, embeddings))

# Query time: Find relevant docs
def find_relevant_docs(query: str, top_k: int = 3) -> list[str]:
    # Embed query
    query_response = client.embeddings.create(
        model="text-embedding-3-small",
        input=query
    )
    query_embedding = query_response.data[0].embedding
    
    # Calculate similarity
    similarities = []
    for doc, emb in knowledge_base:
        similarity = np.dot(query_embedding, emb)  # Cosine similarity
        similarities.append((doc, similarity))
    
    # Return top K
    similarities.sort(key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in similarities[:top_k]]

# Use in prompt
query = "What's the battery life?"
relevant_docs = find_relevant_docs(query)

prompt = f"""Using only these documents, answer the question:

DOCUMENTS:
{chr(10).join(f"- {doc}" for doc in relevant_docs)}

QUESTION: {query}

ANSWER:"""
```

**Why This Works:** Semantic search finds relevant information efficiently. Scales to millions of documents.

#### Tactic 2: Code Execution for Calculations

**Pattern:**
```python
def calculate_with_code(problem: str) -> dict:
    """Let the model write code, then execute it for accurate results."""
    
    # Step 1: Generate code
    code_prompt = f"""Write Python code to solve this problem:

Problem: {problem}

Requirements:
- Use only standard library
- Print the final answer
- Include comments explaining logic

```python"""
    
    response = client.chat.completions.create(
        model="gpt-4",
        temperature=0,
        messages=[{"role": "user", "content": code_prompt}]
    )
    
    code = response.choices[0].message.content.strip("```python\n").strip("\n```")
    
    # Step 2: Execute code (SAFELY - sandbox this in production)
    import sys
    from io import StringIO
    
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    
    try:
        exec(code)
        output = sys.stdout.getvalue()
        error = None
    except Exception as e:
        output = None
        error = str(e)
    finally:
        sys.stdout = old_stdout
    
    return {
        "code": code,
        "output": output,
        "error": error
    }

# Usage
result = calculate_with_code(
    "What's the compound interest on $10,000 at 5% annually for 30 years?"
)
```

**Why This Works:** 
- Models are better at writing code than doing math
- Code execution is perfectly accurate (no floating point errors)
- Verifiable (you can inspect the code)

**CRITICAL:** Sandbox code execution in production (use Docker, restricted environment, or services like CodeSandbox API).

#### Tactic 3: Function Calling for Tool Integration

This is covered extensively in our function calling PKB doc, but here's the connection:

**Pattern:**
```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "search_database",
            "description": "Search internal customer database",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "limit": {"type": "integer", "default": 10}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"}
                },
                "required": ["location"]
            }
        }
    }
]

response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "What's the weather in the customer's city?"}],
    tools=tools
)

# Model decides which tools to call
# You execute them and return results
# See function_calling PKB for full implementation
```

**Why This Works:** Model becomes an orchestrator, deciding which external systems to query and in what order.

### Implementation Example: Multi-Tool Query System

```python
class ToolOrchestrator:
    """Orchestrate multiple external tools."""
    
    def __init__(self):
        self.tools = {
            "search_web": self.search_web,
            "query_database": self.query_database,
            "calculate": self.calculate,
            "get_current_time": self.get_current_time
        }
    
    def search_web(self, query: str) -> str:
        """Simulate web search."""
        # In production: call actual search API
        return f"Search results for: {query}"
    
    def query_database(self, sql: str) -> str:
        """Execute safe SQL query."""
        # In production: validate and execute against DB
        return "Query results: [...]"
    
    def calculate(self, expression: str) -> float:
        """Safely evaluate math expression."""
        # In production: use ast.literal_eval or sympy
        try:
            return eval(expression, {"__builtins__": {}})
        except:
            return "Error in calculation"
    
    def get_current_time(self) -> str:
        """Get current time."""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def execute_query(self, user_query: str) -> str:
        """Execute user query with tools."""
        
        # Let model plan tool usage
        planning_prompt = f"""Determine which tools to use for this query:

Available tools: {', '.join(self.tools.keys())}

Query: {user_query}

Respond with JSON:
{{
  "tools_needed": ["tool1", "tool2"],
  "reasoning": "why these tools"
}}"""
        
        plan_response = client.chat.completions.create(
            model="gpt-4",
            temperature=0,
            messages=[{"role": "user", "content": planning_prompt}]
        )
        
        # Parse plan
        import json
        plan = json.loads(plan_response.choices[0].message.content)
        
        # Execute tools
        tool_results = {}
        for tool_name in plan["tools_needed"]:
            if tool_name in self.tools:
                # In production: extract parameters from conversation
                result = self.tools[tool_name]("example_input")
                tool_results[tool_name] = result
        
        # Synthesize final answer
        synthesis_prompt = f"""Using these tool results, answer the user query:

Query: {user_query}

Tool Results:
{json.dumps(tool_results, indent=2)}

Answer:"""
        
        final_response = client.chat.completions.create(
            model="gpt-4",
            temperature=0.3,
            messages=[{"role": "user", "content": synthesis_prompt}]
        )
        
        return final_response.choices[0].message.content

# Usage
orchestrator = ToolOrchestrator()
answer = orchestrator.execute_query(
    "What's the current time and what's 15% of our Q3 revenue?"
)
```

---

## Strategy 6: Test Changes Systematically

### The Core Principle

Prompt engineering is empirical, not theoretical. What "should" work often doesn't. The only way to know if a change improves performance is to measure it.

**The Promise:** Confidence in prompt changes through data, not intuition.

### Tactics

#### Tactic 1: Evaluate Against Gold-Standard Answers

**Pattern:**
```python
def evaluate_prompt(prompt_template: str, test_cases: list[dict]) -> dict:
    """Measure prompt performance against known-good answers."""
    
    results = {
        "correct": 0,
        "incorrect": 0,
        "partial": 0,
        "failed_cases": []
    }
    
    for case in test_cases:
        # Generate response
        prompt = prompt_template.format(**case["input"])
        
        response = client.chat.completions.create(
            model="gpt-4",
            temperature=0,
            messages=[{"role": "user", "content": prompt}]
        )
        
        actual = response.choices[0].message.content.strip()
        expected = case["expected"]
        
        # Compare
        if actual == expected:
            results["correct"] += 1
        elif expected.lower() in actual.lower():
            results["partial"] += 1
        else:
            results["incorrect"] += 1
            results["failed_cases"].append({
                "input": case["input"],
                "expected": expected,
                "actual": actual
            })
    
    # Calculate metrics
    total = len(test_cases)
    results["accuracy"] = results["correct"] / total
    results["partial_accuracy"] = (results["correct"] + results["partial"]) / total
    
    return results

# Usage
test_cases = [
    {
        "input": {"text": "The product is amazing!"},
        "expected": "Positive"
    },
    {
        "input": {"text": "Terrible experience, would not recommend"},
        "expected": "Negative"
    },
    # ... 100 more test cases
]

prompt_v1 = "Classify sentiment: {text}\nSentiment:"
prompt_v2 = """Classify the sentiment of this review as Positive, Negative, or Neutral.

Review: {text}

Sentiment (one word):"""

results_v1 = evaluate_prompt(prompt_v1, test_cases)
results_v2 = evaluate_prompt(prompt_v2, test_cases)

print(f"v1 accuracy: {results_v1['accuracy']:.2%}")
print(f"v2 accuracy: {results_v2['accuracy']:.2%}")
```

**Why This Works:** 
- Objective measurement (not subjective judgment)
- Catches regressions (new prompt might be worse)
- Identifies failure patterns (specific cases that break)

#### Tactic 2: Use Evals Framework

OpenAI provides an evals framework for systematic testing:

```bash
# Install
pip install evals

# Create eval YAML
cat > my_eval.yaml << EOF
my_sentiment_eval:
  class: evals.elsuite.basic.match:Match
  args:
    samples_jsonl: sentiment_samples.jsonl
EOF

# Create samples
cat > sentiment_samples.jsonl << EOF
{"input": [{"role": "user", "content": "Classify sentiment: Amazing product!"}], "ideal": "Positive"}
{"input": [{"role": "user", "content": "Classify sentiment: Terrible quality"}], "ideal": "Negative"}
EOF

# Run eval
evals run my_sentiment_eval --model gpt-4
```

**Why This Works:** Standardized evaluation across prompt versions, models, and parameters.

#### Tactic 3: A/B Test in Production

**Pattern:**
```python
import random

class PromptABTest:
    """A/B test prompt variations in production."""
    
    def __init__(self, variants: dict):
        self.variants = variants  # {"control": prompt_a, "test": prompt_b}
        self.results = {key: {"calls": 0, "successes": 0} for key in variants}
    
    def get_prompt(self, user_id: str) -> tuple[str, str]:
        """Return prompt and variant name."""
        # Consistent assignment per user
        random.seed(user_id)
        variant = random.choice(list(self.variants.keys()))
        return self.variants[variant], variant
    
    def record_result(self, variant: str, success: bool):
        """Record whether prompt succeeded."""
        self.results[variant]["calls"] += 1
        if success:
            self.results[variant]["successes"] += 1
    
    def get_stats(self) -> dict:
        """Calculate success rates."""
        stats = {}
        for variant, data in self.results.items():
            if data["calls"] > 0:
                success_rate = data["successes"] / data["calls"]
                stats[variant] = {
                    **data,
                    "success_rate": success_rate
                }
        return stats

# Usage
ab_test = PromptABTest({
    "control": "Classify sentiment: {text}\nSentiment:",
    "test": "Classify the sentiment (Positive/Negative/Neutral): {text}\nAnswer:"
})

# In request handler
prompt, variant = ab_test.get_prompt(user_id)
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": prompt.format(text=user_text)}]
)

# Record success (however you define it)
user_satisfied = check_user_feedback()
ab_test.record_result(variant, user_satisfied)

# After 1000 requests
stats = ab_test.get_stats()
print(f"Control: {stats['control']['success_rate']:.2%}")
print(f"Test: {stats['test']['success_rate']:.2%}")
```

**Why This Works:** Real-world performance data. Users don't lie.

### Implementation Example: Complete Evaluation Pipeline

```python
class PromptEvaluator:
    """Comprehensive prompt evaluation system."""
    
    def __init__(self, model: str = "gpt-4"):
        self.model = model
        self.client = OpenAI()
    
    def evaluate_accuracy(self, 
                         prompt_template: str,
                         test_cases: list[dict],
                         ) -> dict:
        """Measure accuracy against gold-standard answers."""
        correct = 0
        total = len(test_cases)
        errors = []
        
        for case in test_cases:
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=0,
                messages=[{"role": "user", "content": prompt_template.format(**case["input"])}]
            )
            
            actual = response.choices[0].message.content.strip()
            expected = case["expected"]
            
            if self._match(actual, expected):
                correct += 1
            else:
                errors.append({
                    "input": case["input"],
                    "expected": expected,
                    "actual": actual
                })
        
        return {
            "accuracy": correct / total,
            "correct": correct,
            "total": total,
            "errors": errors
        }
    
    def evaluate_latency(self,
                        prompt_template: str,
                        sample_inputs: list[dict],
                        runs: int = 10) -> dict:
        """Measure average latency."""
        import time
        
        latencies = []
        for _ in range(runs):
            for input_data in sample_inputs:
                start = time.time()
                
                self.client.chat.completions.create(
                    model=self.model,
                    temperature=0,
                    messages=[{"role": "user", "content": prompt_template.format(**input_data)}]
                )
                
                latencies.append(time.time() - start)
        
        return {
            "mean_latency": sum(latencies) / len(latencies),
            "min_latency": min(latencies),
            "max_latency": max(latencies),
            "p50_latency": sorted(latencies)[len(latencies)//2],
            "p95_latency": sorted(latencies)[int(len(latencies)*0.95)]
        }
    
    def evaluate_cost(self,
                     prompt_template: str,
                     sample_inputs: list[dict]) -> dict:
        """Estimate cost per request."""
        # Rough token estimation
        avg_prompt_tokens = sum(
            len(prompt_template.format(**inp)) // 4 
            for inp in sample_inputs
        ) / len(sample_inputs)
        
        # Sample completion to estimate response tokens
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            messages=[{"role": "user", "content": prompt_template.format(**sample_inputs[0])}]
        )
        
        completion_tokens = response.usage.completion_tokens
        
        # GPT-4 pricing (as of 2024)
        cost_per_1k_input = 0.03
        cost_per_1k_output = 0.06
        
        cost_per_request = (
            (avg_prompt_tokens / 1000) * cost_per_1k_input +
            (completion_tokens / 1000) * cost_per_1k_output
        )
        
        return {
            "avg_prompt_tokens": avg_prompt_tokens,
            "avg_completion_tokens": completion_tokens,
            "cost_per_request": cost_per_request,
            "cost_per_1000_requests": cost_per_request * 1000
        }
    
    def compare_prompts(self,
                       prompts: dict[str, str],
                       test_cases: list[dict],
                       sample_inputs: list[dict]) -> dict:
        """Comprehensive comparison of prompt variants."""
        results = {}
        
        for name, prompt in prompts.items():
            print(f"Evaluating {name}...")
            
            accuracy = self.evaluate_accuracy(prompt, test_cases)
            latency = self.evaluate_latency(prompt, sample_inputs, runs=5)
            cost = self.evaluate_cost(prompt, sample_inputs)
            
            results[name] = {
                "accuracy": accuracy["accuracy"],
                "latency_p50": latency["p50_latency"],
                "cost_per_request": cost["cost_per_request"],
                "errors": len(accuracy["errors"]),
                "full_results": {
                    "accuracy": accuracy,
                    "latency": latency,
                    "cost": cost
                }
            }
        
        return results
    
    def _match(self, actual: str, expected: str) -> bool:
        """Flexible matching."""
        # Exact match
        if actual.strip() == expected.strip():
            return True
        # Case-insensitive
        if actual.strip().lower() == expected.strip().lower():
            return True
        # Contains
        if expected.lower() in actual.lower():
            return True
        return False

# Usage
evaluator = PromptEvaluator()

prompts = {
    "baseline": "Classify: {text}\nSentiment:",
    "detailed": """Analyze the sentiment of the following text.
    
Text: {text}

Classify as: Positive, Negative, or Neutral

Classification:""",
    "with_reasoning": """Classify the sentiment with reasoning:

Text: {text}

First, explain your reasoning:
<reasoning>
[Your analysis]
</reasoning>

Then classify:
<classification>
[Positive/Negative/Neutral]
</classification>"""
}

test_cases = load_test_cases("sentiment_test_set.json")
sample_inputs = [{"text": "Great product!"}, {"text": "Disappointed"}]

comparison = evaluator.compare_prompts(prompts, test_cases, sample_inputs)

# Print report
for name, metrics in comparison.items():
    print(f"\n{name}:")
    print(f"  Accuracy: {metrics['accuracy']:.2%}")
    print(f"  Latency (p50): {metrics['latency_p50']:.2f}s")
    print(f"  Cost: ${metrics['cost_per_request']:.4f}/request")
    print(f"  Errors: {metrics['errors']}")
```

---

## Common Pitfalls

### 1. Using Temperature > 0 for Factual Tasks

**Problem:** Temperature adds randomness. For tasks requiring consistency (classification, extraction), randomness causes variability.

```python
# âŒ BAD
response = client.chat.completions.create(
    model="gpt-4",
    temperature=0.7,  # Will give different answers each time!
    messages=[{"role": "user", "content": "Extract the invoice number from: INV-12345"}]
)

# âœ… GOOD
response = client.chat.completions.create(
    model="gpt-4",
    temperature=0,  # Deterministic
    messages=[{"role": "user", "content": "Extract the invoice number from: INV-12345"}]
)
```

**When to Use Temperature:**
- **0**: Classification, extraction, factual Q&A
- **0.3-0.5**: Balanced content generation
- **0.7-1.0**: Creative writing, brainstorming
- **1.0+**: Maximum diversity (rarely useful)

### 2. Not Using System Messages Effectively

**Problem:** Putting everything in user messages makes prompts harder to maintain and doesn't leverage the system message's persistent context.

```python
# âŒ BAD - instructions in every user message
messages = [
    {"role": "user", "content": "You are a code reviewer. Review this: def foo(): pass"},
    {"role": "user", "content": "You are a code reviewer. Review this: def bar(): return 1"}
]

# âœ… GOOD - instructions in system message
messages = [
    {"role": "system", "content": "You are a code reviewer focused on security and best practices"},
    {"role": "user", "content": "Review: def foo(): pass"},
    {"role": "user", "content": "Review: def bar(): return 1"}
]
```

**System Message Best Practices:**
- Define persona and expertise
- Set output format requirements
- Establish safety constraints
- Provide persistent context

### 3. Providing Too Many Examples

**Problem:** More examples â‰  better performance. After 3-5 examples, you hit diminishing returns and waste tokens.

```python
# âŒ BAD - 15 examples
prompt = """
Example 1: ...
Example 2: ...
... [13 more examples] ...
Example 15: ...

Now classify: ...
"""

# âœ… GOOD - 2-3 diverse examples
prompt = """
Example 1: "Great service" â†’ Positive
Example 2: "Slow and buggy" â†’ Negative
Example 3: "It works" â†’ Neutral

Now classify: ...
"""
```

**Optimal Example Count:**
- **0 (zero-shot):** Simple, unambiguous tasks
- **1-2 (few-shot):** Most classification tasks
- **3-5 (multi-shot):** Complex pattern recognition
- **5+:** Diminishing returns, wasted tokens

### 4. Not Handling Long Context Properly

**Problem:** Blindly stuffing all context into prompts leads to exceeding limits or poor performance.

```python
# âŒ BAD - entire 50-page document in prompt
prompt = f"Summarize: {entire_document}"  # Might exceed limits or be expensive

# âœ… GOOD - chunk and process
def process_long_document(document: str) -> str:
    chunks = split_into_chunks(document, chunk_size=4000)
    chunk_summaries = [summarize_chunk(c) for c in chunks]
    return summarize_summaries(chunk_summaries)
```

**Strategies for Long Context:**
1. **Chunk and summarize:** Process in pieces
2. **Semantic search:** Extract relevant sections only
3. **Recursive summarization:** Summarize summaries
4. **Compression:** Remove boilerplate, keep core content

### 5. Ignoring Cost Optimization

**Problem:** Not considering token usage leads to unnecessary expenses.

```python
# âŒ BAD - wasteful
system = """You are a helpful assistant who provides detailed, comprehensive 
answers with lots of examples and thorough explanations..."""  # 200 tokens

# âœ… GOOD - concise
system = "You are a helpful assistant who provides concise, accurate answers."  # 15 tokens
```

**Cost Optimization Tactics:**
- Use shorter system messages
- Set `max_tokens` appropriately (don't ask for 2000 tokens when 100 suffice)
- Cache reusable prompts
- Use gpt-3.5-turbo for simple tasks
- Batch requests when possible

### 6. Not Validating Structured Outputs

**Problem:** Assuming the model will always return valid JSON/format.

```python
# âŒ BAD - no validation
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Return JSON: {name, age}"}]
)
data = json.loads(response.choices[0].message.content)  # Might crash!

# âœ… GOOD - with validation
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Return JSON: {name, age}"}]
)

try:
    content = response.choices[0].message.content
    # Strip markdown code blocks if present
    if content.startswith("```"):
        content = content.split("```")[1]
        if content.startswith("json"):
            content = content[4:]
    data = json.loads(content.strip())
    # Validate schema
    assert "name" in data and "age" in data
except (json.JSONDecodeError, AssertionError) as e:
    print(f"Invalid response: {e}")
    # Retry or use fallback
```

**Better: Use Function Calling with Strict Mode** (OpenAI-specific)
```python
tools = [{
    "type": "function",
    "function": {
        "name": "extract_person",
        "strict": True,  # GUARANTEED schema compliance
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"}
            },
            "required": ["name", "age"],
            "additionalProperties": False
        }
    }
}]

response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Extract: John Doe, 30"}],
    tools=tools,
    tool_choice={"type": "function", "function": {"name": "extract_person"}}
)

# GUARANTEED valid JSON matching schema
data = json.loads(response.choices[0].message.tool_calls[0].function.arguments)
```

### 7. Not Testing with Edge Cases

**Problem:** Only testing happy paths. Production inputs are messy.

```python
# âŒ BAD - only test ideal inputs
test_cases = [
    {"input": "Great product!", "expected": "Positive"},
    {"input": "Terrible service", "expected": "Negative"}
]

# âœ… GOOD - test edge cases
test_cases = [
    # Happy paths
    {"input": "Great product!", "expected": "Positive"},
    {"input": "Terrible service", "expected": "Negative"},
    
    # Edge cases
    {"input": "", "expected": "Error: Empty input"},
    {"input": "It's okay I guess", "expected": "Neutral"},
    {"input": "Good but expensive", "expected": "Mixed"},  # Mixed sentiment
    {"input": "Great! Jk it's terrible", "expected": "Negative"},  # Sarcasm
    {"input": "Product is product", "expected": "Neutral"},  # Meaningless
    {"input": "a" * 10000, "expected": "Error: Input too long"}  # Length limit
]
```

**Edge Case Categories:**
- Empty/null inputs
- Very long inputs
- Mixed sentiment
- Sarcasm/irony
- Typos and misspellings
- Multiple languages
- Special characters
- Ambiguous cases

### 8. Using JSON Mode Incorrectly

**Problem:** Not understanding JSON mode limitations.

```python
# âŒ BAD - JSON mode doesn't guarantee schema
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Give me data"}],
    response_format={"type": "json_object"}
)
# Might return: {"random": "structure"} - valid JSON, wrong schema!

# âœ… GOOD - specify schema in prompt
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": """Return JSON in this exact format:
{
  "name": "string",
  "age": integer,
  "email": "string"
}

Data: John, 30, john@example.com
"""}],
    response_format={"type": "json_object"}
)

# âœ… BETTER - use function calling with strict mode
# (See pitfall #6 above)
```

### 9. Not Handling Rate Limits

**Problem:** No retry logic for rate limit errors.

```python
# âŒ BAD - crashes on rate limit
response = client.chat.completions.create(...)

# âœ… GOOD - exponential backoff
import time
from openai import RateLimitError

def call_with_retry(max_retries=3):
    for attempt in range(max_retries):
        try:
            return client.chat.completions.create(...)
        except RateLimitError:
            if attempt == max_retries - 1:
                raise
            wait = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
            print(f"Rate limited. Waiting {wait}s...")
            time.sleep(wait)
```

### 10. Mixing Concerns in Single Prompt

**Problem:** Trying to do extraction, validation, formatting, and analysis in one prompt.

```python
# âŒ BAD - too many responsibilities
prompt = """Extract all dates, validate they're in the past, format as ISO 8601,
calculate days since each date, and rank by importance.

Text: {text}"""

# âœ… GOOD - separate prompts for each concern
dates_raw = extract_dates(text)
dates_validated = validate_dates(dates_raw)
dates_formatted = format_dates(dates_validated)
dates_analyzed = calculate_and_rank(dates_formatted)
```

**Why Separation Wins:**
- Each step is testable independently
- Errors are isolated
- Can optimize temperature per step
- Easier debugging
- Can cache/reuse intermediate results

---

## Integration Points

### Connection to Our Five Capabilities

OpenAI's strategies aren't separate from our capabilitiesâ€”they're how we implement them. Here's the mapping:

#### 1. Prompt Routing

**OpenAI Strategies Applied:**

**Clear Instructions (Strategy 1):**
```python
system = """You are a query router. Classify each query into exactly one category:

INTERNAL: Company data, policies, internal documentation
EXTERNAL: Current events, web search, public information
DIRECT: Greetings, help requests, general knowledge

Rules:
- Choose ONE category
- Output only the category name
- No explanation needed"""

user_query = "What's our vacation policy?"

response = client.chat.completions.create(
    model="gpt-4",
    temperature=0,  # Deterministic routing
    messages=[
        {"role": "system", "content": system},
        {"role": "user", "content": f"Classify: {user_query}\nCategory:"}
    ],
    max_tokens=10
)

route = response.choices[0].message.content.strip()
```

**Provide Examples (Strategy 1, Tactic 5):**
```python
system = """Classify queries into routes:

Examples:
"What's our refund policy?" â†’ INTERNAL
"Who won the Super Bowl?" â†’ EXTERNAL
"Hello" â†’ DIRECT

Query: {query}
Route:"""
```

**Use External Tools (Strategy 5):**
```python
# After routing, use appropriate tool
if route == "INTERNAL":
    answer = search_internal_kb(query)
elif route == "EXTERNAL":
    answer = web_search(query)
else:
    answer = respond_directly(query)
```

#### 2. Query Writing

**OpenAI Strategies Applied:**

**Clear Instructions + Reference Text (Strategies 1 & 2):**
```python
database_schema = """
Table: users
- id (INTEGER, primary key)
- email (VARCHAR, unique)
- created_at (TIMESTAMP)

Table: orders
- id (INTEGER, primary key)
- user_id (INTEGER, foreign key -> users.id)
- amount (DECIMAL)
- status (ENUM: pending, completed, cancelled)
"""

system = """You generate safe, efficient SQL queries.

Rules:
- Use parameterized queries (? placeholders)
- Include LIMIT clauses
- Use table aliases in JOINs
- Never use SELECT *"""

user_request = "Find all users who placed orders over $100 last month"

prompt = f"""Database schema:
{database_schema}

Request: {user_request}

Generate SQL query:"""

response = client.chat.completions.create(
    model="gpt-4",
    temperature=0,
    messages=[
        {"role": "system", "content": system},
        {"role": "user", "content": prompt}
    ]
)

sql_query = response.choices[0].message.content
```

**Split Complex Tasks (Strategy 3):**
```python
# Step 1: Identify tables needed
tables_prompt = f"Which tables are needed for: {user_request}\nTables:"
tables = client.chat.completions.create(...).choices[0].message.content

# Step 2: Identify filters
filters_prompt = f"What filters are needed for: {user_request}\nFilters:"
filters = client.chat.completions.create(...).choices[0].message.content

# Step 3: Generate query
query_prompt = f"Generate SQL using tables={tables}, filters={filters}"
query = client.chat.completions.create(...).choices[0].message.content
```

**Give Time to Think (Strategy 4):**
```python
system = """Generate SQL queries step-by-step.

Process:
1. Identify tables needed
2. Determine JOIN conditions
3. Identify WHERE filters
4. Consider ORDER BY and LIMIT
5. Write final query

Show your work in <thinking> tags, then provide query in <sql> tags."""
```

#### 3. Data Processing

**OpenAI Strategies Applied:**

**Split Complex Tasks (Strategy 3):**
```python
def process_data_pipeline(raw_data: str) -> dict:
    """Multi-step data processing."""
    
    # Step 1: Extract structured data
    extract_prompt = f"Extract key-value pairs from: {raw_data}"
    extracted = client.chat.completions.create(
        model="gpt-4",
        temperature=0,
        messages=[{"role": "user", "content": extract_prompt}]
    ).choices[0].message.content
    
    # Step 2: Validate extracted data
    validate_prompt = f"Validate this data, flag issues: {extracted}"
    validated = client.chat.completions.create(
        model="gpt-4",
        temperature=0,
        messages=[{"role": "user", "content": validate_prompt}]
    ).choices[0].message.content
    
    # Step 3: Enrich with additional info
    enrich_prompt = f"Enrich this data with missing fields: {validated}"
    enriched = client.chat.completions.create(
        model="gpt-4",
        temperature=0,
        messages=[{"role": "user", "content": enrich_prompt}]
    ).choices[0].message.content
    
    # Step 4: Format for output
    format_prompt = f"Format as JSON: {enriched}"
    formatted = client.chat.completions.create(
        model="gpt-4",
        temperature=0,
        messages=[{"role": "user", "content": format_prompt}],
        response_format={"type": "json_object"}
    ).choices[0].message.content
    
    return json.loads(formatted)
```

**Use External Tools (Strategy 5):**
```python
# Use code execution for data transformations
def transform_with_pandas(data: str) -> str:
    """Let model write pandas code, execute it."""
    
    code_prompt = f"""Write Python pandas code to:
1. Load this CSV data
2. Clean null values
3. Calculate summary statistics
4. Return JSON

Data:
{data}

Code:"""
    
    code = client.chat.completions.create(
        model="gpt-4",
        temperature=0,
        messages=[{"role": "user", "content": code_prompt}]
    ).choices[0].message.content
    
    # Execute safely
    result = execute_code_safely(code)
    return result
```

#### 4. Tool Orchestration

**OpenAI Strategies Applied:**

**Use External Tools (Strategy 5):**
```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "search_database",
            "description": "Search internal database for customer info",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "send_email",
            "description": "Send email to customer",
            "parameters": {
                "type": "object",
                "properties": {
                    "to": {"type": "string"},
                    "subject": {"type": "string"},
                    "body": {"type": "string"}
                },
                "required": ["to", "subject", "body"]
            }
        }
    }
]

# Model decides which tools to call and when
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Find John's email and send him a welcome message"}],
    tools=tools
)

# Execute tool calls in order
# (See function calling PKB for full implementation)
```

**Give Time to Think (Strategy 4):**
```python
system = """You orchestrate tools to solve user requests.

Process:
1. Analyze the request in <planning> tags
2. Determine which tools are needed
3. Decide on tool call order
4. Execute tools sequentially
5. Synthesize results

Always plan before acting."""
```

#### 5. Decision Support

**OpenAI Strategies Applied:**

**Provide Reference Text + Give Time to Think (Strategies 2 & 4):**
```python
system = """You provide decision support through structured analysis.

Process:
<analysis>
1. List all options
2. Evaluate each against criteria
3. Identify trade-offs
4. Note risks
</analysis>

<recommendation>
- Best option: [X]
- Reasoning: [why]
- Confidence: [high/medium/low]
- Action items: [list]
</recommendation>"""

reference_data = """
Option A: Cost $100K, Time 6 months, Risk medium
Option B: Cost $50K, Time 3 months, Risk high
Option C: Cost $150K, Time 12 months, Risk low
"""

user_question = "Which option should we choose for our database migration?"

prompt = f"""Reference data:
{reference_data}

Decision criteria:
- Budget: $120K max
- Timeline: Complete within 9 months
- Risk tolerance: Low to medium

Question: {user_question}

Provide structured analysis and recommendation:"""

response = client.chat.completions.create(
    model="gpt-4",
    temperature=0.3,  # Slight creativity for edge case consideration
    messages=[
        {"role": "system", "content": system},
        {"role": "user", "content": prompt}
    ]
)
```

**Split Complex Tasks (Strategy 3):**
```python
# Step 1: Gather information about each option
option_analyses = []
for option in ["A", "B", "C"]:
    analysis = analyze_single_option(option, criteria)
    option_analyses.append(analysis)

# Step 2: Compare options pairwise
comparisons = []
for i in range(len(options)):
    for j in range(i+1, len(options)):
        comparison = compare_two_options(options[i], options[j])
        comparisons.append(comparison)

# Step 3: Synthesize final recommendation
recommendation = synthesize_recommendation(option_analyses, comparisons)
```

---

## Comparison to Anthropic Prompt Engineering

Both OpenAI and Anthropic provide prompt engineering guidance, but with different emphases and techniques.

### Universal Principles (Both Agree)

**Be Specific:**
- Both emphasize clear, detailed instructions
- Both recommend avoiding vague language
- Both advocate for concrete success criteria

**Use Examples:**
- Both strongly recommend few-shot prompting
- Both suggest 2-4 examples for most tasks
- Both note that example quality > quantity

**Chain of Thought:**
- Both support CoT prompting for complex reasoning
- Both show accuracy improvements on math/logic tasks
- Both recommend outputting reasoning before conclusions

**System Messages:**
- Both support system prompts to set behavior
- Both recommend using them for consistent personality
- Both allow system prompts to define safety constraints

### OpenAI-Specific Techniques

**JSON Mode:**
- **OpenAI:** Native `response_format={"type": "json_object"}` parameter
- **Anthropic:** No native JSON mode, use prefilling or tool forcing

```python
# OpenAI
response = client.chat.completions.create(
    model="gpt-4",
    messages=messages,
    response_format={"type": "json_object"}  # Native support
)

# Anthropic equivalent
messages = [
    {"role": "user", "content": "Return JSON"},
    {"role": "assistant", "content": "{"}  # Prefill to force JSON
]
```

**Strict Mode Function Calling:**
- **OpenAI:** `strict: true` GUARANTEES schema compliance
- **Anthropic:** No strict mode, requires Pydantic validation

```python
# OpenAI - guaranteed schema
tools = [{
    "type": "function",
    "function": {
        "name": "extract",
        "strict": True,  # Guaranteed compliance
        "parameters": {...}
    }
}]

# Anthropic - requires validation
tool_result = anthropic_response.tool_use
try:
    validated = ToolInput(**tool_result.input)  # Pydantic validates
except ValidationError:
    # Handle schema violation
```

**Temperature Defaults:**
- **OpenAI:** Default temperature = 1.0 (more creative)
- **Anthropic:** Default temperature = 1.0 (same)
- **Both:** Recommend temperature = 0 for factual tasks

### Anthropic-Specific Techniques

**XML Tags for Structure:**
- **Anthropic:** Explicitly recommends XML tags, trained specifically on them
- **OpenAI:** No special recommendation for XML vs. other delimiters

```python
# Anthropic style
prompt = """<context>{data}</context>
<task>Analyze the data above</task>
<output_format>
<analysis>...</analysis>
<conclusion>...</conclusion>
</output_format>"""

# OpenAI style (more flexible)
prompt = """[Context]
{data}

[Task]
Analyze the data above

[Output Format]
Analysis: ...
Conclusion: ...
"""
```

**Prefilling Assistant Responses:**
- **Anthropic:** Direct support via assistant message in Chat Completions
- **OpenAI:** Less direct, can approximate but not primary pattern

```python
# Anthropic - direct prefilling
messages = [
    {"role": "user", "content": "Write JSON"},
    {"role": "assistant", "content": "{"}  # Prefill starts response
]

# OpenAI - less direct
# Use system prompt to guide or function calling to force format
```

**Prompt Chaining Emphasis:**
- **Anthropic:** Heavily emphasizes chaining as core technique
- **OpenAI:** Mentions it in "split tasks" but doesn't prioritize as much

**Hierarchical Technique Ordering:**
- **Anthropic:** Explicit hierarchy (Basic â†’ Advanced)
- **OpenAI:** Six strategies as complementary techniques

### Style Differences

**Anthropic's Approach:**
- More prescriptive (do X for Y)
- Structured hierarchically
- XML-centric for complex prompts
- Emphasis on chains over monolithic prompts
- Thinking tags pattern (`<thinking>...</thinking>`)

**OpenAI's Approach:**
- More flexible (try X or Y)
- Six-strategy framework
- Format-agnostic
- Emphasis on systematic testing
- Direct focus on evaluation frameworks

### When to Use Which Approach

**Use OpenAI Techniques When:**
- Need guaranteed JSON output (strict mode)
- Want native JSON mode without prefilling
- Prefer parameter tuning (temperature, top_p) over prompt structure
- Building cross-model systems (OpenAI patterns more universal)
- Need comprehensive evaluation framework

**Use Anthropic Techniques When:**
- Building complex, multi-step workflows (chaining)
- Need highly structured prompts (XML tags)
- Want to extract specific sections programmatically
- Optimizing specifically for Claude
- Prefer thinking tags for debugging

**Best Practice for Our Project:**
Combine both:
- Use OpenAI's six strategies as conceptual framework
- Use Anthropic's XML structure for complex prompts
- Use OpenAI's strict mode when available
- Use Anthropic's prefilling when needed
- Test across both providers
- Abstract provider-specific features behind interfaces

---

## Our Takeaways

### For agentic_ai_development

**1. OpenAI's Six Strategies Are Orthogonal, Not Sequential**

Unlike Anthropic's hierarchy (basic â†’ advanced), OpenAI's strategies are dimensions you can combine:
- Strategy 1 (clear instructions) + Strategy 2 (reference text) + Strategy 4 (CoT) = powerful combination
- Don't think "which strategy?" Think "which combination?"
- Production prompts often use 4-5 strategies simultaneously

**2. Temperature=0 for Production, Period**

For factual tasks (classification, extraction, routing, query writing), use temperature=0:
- Eliminates randomness
- Guarantees consistency
- Reduces debugging difficulty
- Only increase temperature for creative tasks where variability is desired

**3. System Messages Are Underutilized**

Most developers put everything in user messages. System messages are gold:
- Persistent context across conversation
- Define behavioral contracts
- Specify output requirements
- Establish safety guardrails
- Don't waste themâ€”design thoughtfully

**4. Few-Shot > Zero-Shot, But 3-5 Examples Is the Sweet Spot**

- 0 examples: Only for trivial tasks
- 1-2 examples: Most classification/extraction
- 3-5 examples: Complex pattern recognition
- 5+ examples: Diminishing returns, wasted tokens

Counter-intuitive: Sometimes 2 examples outperform 10. Quality and diversity matter more than quantity.

**5. Reference Text Is the Anti-Hallucination Weapon**

Grounding prevents fabrication:
- Provide authoritative source material
- Instruct to cite sources
- Force model to justify claims with references
- "I don't know" becomes acceptable answer when info isn't in sources

Production systems handling facts MUST use reference text.

**6. Task Decomposition Is Not Optional**

Complex tasks need breaking down:
- Easier to debug (isolate failures)
- Easier to optimize (different temperature per step)
- Easier to test (validate each step independently)
- Higher accuracy (focused sub-tasks perform better)

One 500-token monolithic prompt < Five 100-token focused prompts.

**7. Chain-of-Thought Unlocks 20-30% Accuracy Gains on Complex Tasks**

For reasoning tasks (math, logic, analysis), CoT is mandatory:
- Force explicit step-by-step thinking
- Makes errors detectable (see where reasoning breaks)
- Reduces guessing (model works through problem)
- Cost: 2-3x tokens, Benefit: Dramatically better accuracy

Trade-off is worth it for high-stakes decisions.

**8. External Tools Aren't Optional Extrasâ€”They're Requirements**

Models have inherent limitations:
- Stale training data (use semantic search, RAG)
- Arithmetic errors (use code execution)
- No real-time data (use APIs)
- No private access (use database tools)

Production agentic systems MUST integrate external tools.

**9. You Cannot Trust Without Testing**

Prompt engineering is empirical:
- "It should work" â‰  "It works"
- Build eval sets with 50-100 test cases
- Measure accuracy, latency, cost
- A/B test variants in production
- Regression test on every prompt change

We need evaluation infrastructure from day one.

**10. JSON Mode â‰  Schema Guarantee (But Strict Mode Does)**

- `response_format={"type": "json_object"}` â†’ Valid JSON, NO schema guarantee
- `strict: true` in function calling â†’ GUARANTEED schema compliance
- For structured extraction, use function calling with strict mode, not JSON mode
- JSON mode is for when you want JSON format but schema doesn't matter

This distinction is critical.

**11. Delimiters Prevent Prompt Injection**

Clear boundaries matter:
```python
# Without delimiters - vulnerable
prompt = f"Summarize: {user_input}"
# If user_input = "Ignore previous instructions...", prompt breaks

# With delimiters - protected
prompt = f"Summarize the text in ###DELIMITERS###:\n###{user_input}###"
```

Use `###`, `<tags>`, or triple backticks to separate instructions from content.

**12. Cost Optimization Starts with Prompt Design**

Token efficiency matters:
- Shorter system messages
- Fewer examples (3-5, not 15)
- Appropriate `max_tokens` limits
- gpt-3.5-turbo for simple tasks
- Cache reusable prompts

At scale, prompt optimization saves thousands in API costs.

**13. The Six Strategies Map Perfectly to Our Five Capabilities**

- **Prompt Routing:** Strategies 1 (clear instructions) + 5 (external tools)
- **Query Writing:** Strategies 1 + 2 (reference text) + 3 (split tasks)
- **Data Processing:** Strategy 3 (split tasks) + 5 (external tools)
- **Tool Orchestration:** Strategy 5 (external tools) + 4 (time to think)
- **Decision Support:** Strategies 2 (reference text) + 4 (time to think)

These aren't separate domainsÃ¢â‚¬"they're applications of the same principles.

**14. OpenAI's Evaluation Focus Is Their Differentiator**

Where Anthropic emphasizes prompt structure (XML, chains), OpenAI emphasizes systematic testing:
- Evals framework for standardized testing
- Gold-standard answer comparison
- A/B testing guidance
- Regression testing methodology

Both matter, but testing infrastructure is undervalued in most projects.

**15. Prompts Are Codeâ€”Treat Them Like Code**

Prompts should be:
- Version controlled (git)
- Tested (pytest with eval suite)
- Deployed via CI/CD
- Monitored in production
- Documented (what it does, why it works)
- Rolled back when they break

If you're editing prompts in a web UI, you're doing it wrong.

---

## Implementation Checklist

When building prompts for our five capabilities:

### Basic Prompt Hygiene
- [ ] Task clearly defined in first sentence
- [ ] All required context included
- [ ] No typos or grammatical errors
- [ ] Output format explicitly specified
- [ ] Edge case handling described
- [ ] Using temperature=0 for factual tasks
- [ ] Using appropriate model (gpt-4 vs gpt-3.5-turbo)

### Strategy Application
- [ ] Clear instructions (Strategy 1) with specificity
- [ ] Reference text (Strategy 2) for grounding when applicable
- [ ] Complex tasks split (Strategy 3) into steps
- [ ] CoT (Strategy 4) for reasoning tasks
- [ ] External tools (Strategy 5) integrated
- [ ] Evaluation framework (Strategy 6) in place

### System Messages
- [ ] Persona and expertise defined
- [ ] Output format requirements specified
- [ ] Safety constraints established
- [ ] Persistent context provided
- [ ] Behavioral contracts clear

### Examples (When Needed)
- [ ] 2-5 diverse examples provided
- [ ] Examples show desired format
- [ ] Examples cover edge cases
- [ ] Examples are realistic, not toy cases

### Structured Outputs
- [ ] Using function calling with `strict: true` for schema compliance
- [ ] OR using JSON mode with explicit schema in prompt
- [ ] Validation logic in place for parsing
- [ ] Error handling for malformed outputs
- [ ] Fallback strategies defined

### Testing
- [ ] Eval set with 50+ test cases
- [ ] Happy path tests
- [ ] Edge case tests (empty input, very long input, etc.)
- [ ] Accuracy measured against gold-standard
- [ ] Latency measured
- [ ] Cost calculated
- [ ] Regression tests prevent breakage

### External Tools Integration
- [ ] Tool descriptions are clear and specific
- [ ] Tool parameters are well-defined
- [ ] Error handling for tool failures
- [ ] Retry logic for transient failures
- [ ] Logging for debugging

### Production Readiness
- [ ] Prompt stored in version control
- [ ] Evaluation suite exists
- [ ] Monitoring/logging implemented
- [ ] Rate limit handling (exponential backoff)
- [ ] Cost tracking per request
- [ ] A/B test framework (if needed)

---

## Testing Strategy

Comprehensive testing ensures reliable prompt performance:

### 1. Accuracy Testing

**Objective:** Measure correctness against known-good answers.

```python
def test_accuracy():
    test_cases = load_test_cases("eval_set.json")
    
    correct = 0
    total = len(test_cases)
    
    for case in test_cases:
        response = call_prompt(case["input"])
        if response == case["expected"]:
            correct += 1
    
    accuracy = correct / total
    assert accuracy >= 0.85, f"Accuracy too low: {accuracy:.2%}"
```

### 2. Consistency Testing

**Objective:** Ensure deterministic behavior with temperature=0.

```python
def test_consistency():
    input_text = "Classify: Great product!"
    
    results = set()
    for _ in range(10):
        response = call_prompt(input_text)
        results.add(response)
    
    # With temperature=0, should be identical
    assert len(results) == 1, f"Non-deterministic: {results}"
```

### 3. Edge Case Testing

**Objective:** Handle unusual inputs gracefully.

```python
def test_edge_cases():
    edge_cases = [
        "",  # Empty
        "a" * 10000,  # Very long
        "It's good but also bad",  # Mixed sentiment
        "Great! (sarcasm)",  # Sarcasm marker
        "ðŸŽ‰ Amazing ðŸŽ‰",  # Emojis
    ]
    
    for case in edge_cases:
        response = call_prompt(case)
        assert response is not None
        assert len(response) > 0
```

### 4. Format Compliance Testing

**Objective:** Verify structured output matches schema.

```python
def test_json_format():
    response = call_prompt_with_json_mode("Extract: John, 30")
    
    # Should be valid JSON
    data = json.loads(response)
    
    # Should have required fields
    assert "name" in data
    assert "age" in data
    
    # Should have correct types
    assert isinstance(data["name"], str)
    assert isinstance(data["age"], int)
```

### 5. Latency Testing

**Objective:** Measure response time.

```python
def test_latency():
    import time
    
    latencies = []
    for _ in range(20):
        start = time.time()
        call_prompt("Test input")
        latencies.append(time.time() - start)
    
    p50 = sorted(latencies)[len(latencies)//2]
    p95 = sorted(latencies)[int(len(latencies)*0.95)]
    
    assert p50 < 2.0, f"Median latency too high: {p50:.2f}s"
    assert p95 < 5.0, f"P95 latency too high: {p95:.2f}s"
```

### 6. Cost Testing

**Objective:** Track token usage and API costs.

```python
def test_cost():
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": "Test"}]
    )
    
    prompt_tokens = response.usage.prompt_tokens
    completion_tokens = response.usage.completion_tokens
    
    # GPT-4 pricing (as of 2024)
    cost = (
        (prompt_tokens / 1000) * 0.03 +
        (completion_tokens / 1000) * 0.06
    )
    
    assert cost < 0.10, f"Cost per request too high: ${cost:.4f}"
```

### 7. Regression Testing

**Objective:** Prevent prompt changes from breaking functionality.

```python
def test_regression():
    """Run after every prompt change."""
    baseline_accuracy = 0.85
    
    current_accuracy = evaluate_accuracy()
    
    assert current_accuracy >= baseline_accuracy, \
        f"Regression: {current_accuracy:.2%} < {baseline_accuracy:.2%}"
```

### 8. A/B Testing Framework

**Objective:** Compare prompt variants in production.

```python
class PromptABTest:
    def __init__(self, variants: dict):
        self.variants = variants
        self.stats = {v: {"calls": 0, "successes": 0} for v in variants}
    
    def get_variant(self, user_id: str) -> str:
        # Consistent per-user assignment
        return hash(user_id) % len(self.variants)
    
    def record_success(self, variant: str, success: bool):
        self.stats[variant]["calls"] += 1
        if success:
            self.stats[variant]["successes"] += 1
    
    def get_winner(self) -> str:
        success_rates = {
            v: s["successes"] / s["calls"] if s["calls"] > 0 else 0
            for v, s in self.stats.items()
        }
        return max(success_rates, key=success_rates.get)
```

---

## Summary

**OpenAI's prompt engineering guide provides six complementary strategies for reliable LLM outputs:**

1. **Write Clear Instructions** â†’ Specificity eliminates ambiguity
2. **Provide Reference Text** â†’ Grounding prevents hallucination  
3. **Split Complex Tasks** â†’ Decomposition improves accuracy
4. **Give Time to Think** â†’ Chain-of-thought unlocks reasoning
5. **Use External Tools** â†’ Compensate for model limitations
6. **Test Changes Systematically** â†’ Measure don't guess

**Key Differences from Anthropic:**
- OpenAI emphasizes testing and evaluation frameworks
- Native JSON mode (but not schema-guaranteed without strict mode)
- Strict mode function calling GUARANTEES schema compliance
- Format-agnostic (no XML preference)
- Six strategies as orthogonal dimensions, not hierarchy

**Universal Principles (Both Providers):**
- Be specific and detailed
- Use 2-4 examples for most tasks
- Chain-of-thought for reasoning tasks
- System messages set behavior
- External tools are mandatory for production

**For Our Five Capabilities:**
All five capabilities are applications of these six strategies:
- **Routing:** Clear instructions + Few-shot examples
- **Query Writing:** Reference text (schema) + Task splitting
- **Data Processing:** Task splitting + External tools
- **Tool Orchestration:** External tools + Chain-of-thought
- **Decision Support:** Reference text + Chain-of-thought

**The Bottom Line:**
OpenAI's approach is empirical and test-driven. Where Anthropic says "structure your prompts this way," OpenAI says "try these strategies and measure which combinations work." Both are right. Production systems need structure (Anthropic) AND measurement (OpenAI).

**Critical for Our Project:**
- Use OpenAI's six strategies as conceptual framework
- Apply Anthropic's structural patterns (XML) when building complex prompts
- Leverage OpenAI's strict mode for guaranteed schema compliance
- Build comprehensive evaluation infrastructure from day one
- Version control prompts like code
- Test systematically, deploy confidently

Master OpenAI's six strategies, combine with Anthropic's structural techniques, and you have a complete prompt engineering toolkit. Theory without testing is hope. Testing without theory is chaos. Together, they're production-ready agentic AI.