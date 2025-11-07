# LlamaIndex Query Engines

**Primary Sources:**
- LlamaIndex Query Engine Documentation: https://docs.llamaindex.ai/en/stable/module_guides/deploying/query_engine/
- Router Query Engine: https://docs.llamaindex.ai/en/stable/examples/query_engine/RouterQueryEngine/
- Sub Question Query Engine: https://docs.llamaindex.ai/en/stable/examples/query_engine/sub_question_query_engine/
- Response Synthesizer: https://docs.llamaindex.ai/en/stable/module_guides/querying/response_synthesizers/
- Query Transformations: https://docs.llamaindex.ai/en/stable/optimizing/advanced_retrieval/query_transformations/
- Routing: https://docs.llamaindex.ai/en/stable/module_guides/querying/router/

**Date Accessed:** November 6, 2025

---

## Relevance Statement

Query engines are LlamaIndex's abstraction layer over RAG. We've built RAG from scratch (document 9). Now we need to decide: when should we use LlamaIndex's abstractions vs. our custom implementation?

**Why LlamaIndex Query Engines Matter for agentic_ai_development:**

Query engines provide orchestration patterns we'd otherwise build ourselves:
- **Router engines** for multi-index querying (routing between different data sources)
- **Sub-question engines** for complex query decomposition
- **Response synthesis modes** for different quality/latency tradeoffs
- **Query transformations** (HyDE, rewriting, decomposition) for improved retrieval
- **Built-in streaming and async** for production UX

**The Hard Truth:** LlamaIndex query engines are NOT always the answer. For simple RAG (single index, straightforward queries), they're overkill. For complex multi-source queries or agentic workflows, they save weeks of development. The key is knowing when the abstraction helps vs. when it hurts.

**Critical Understanding:** Query engines sit between deterministic retrieval and agentic reasoning. They're more sophisticated than raw RAG but less flexible than full agents. Use them when you need orchestration patterns (routing, decomposition, synthesis) but don't need the full autonomy of agents.

---

## Key Concepts

### What Is a Query Engine?

**Definition:** A query engine takes a natural language query, retrieves relevant context, synthesizes a response, and returns it. It's the orchestration layer that connects retrieval â†’ synthesis â†’ response.

**Architecture:**
```
Query â†’ Query Engine â†’ [Retriever â†’ Nodes] â†’ [Response Synthesizer â†’ Final Response]
```

**The Three Core Abstractions:**

1. **Retriever:** Fetches relevant nodes (documents/chunks) from an index
2. **Response Synthesizer:** Combines nodes + query â†’ generates answer using LLM
3. **Query Engine:** Orchestrates the flow (runs retriever, feeds to synthesizer)

**Key Insight:** Query engines are stateless. Each query is independent. For multi-turn conversations, use Chat Engines instead.

### Query Engine vs. Chat Engine vs. Agent

**Query Engine:**
- Stateless (no conversation history)
- Deterministic retrieval + synthesis
- Single query â†’ single response
- **When to use:** One-shot Q&A over documents

**Chat Engine:**
- Stateful (maintains conversation history)
- Deterministic retrieval + synthesis
- Multi-turn conversation
- **When to use:** Conversational interface over documents

**Agent:**
- Stateful or stateless
- Agentic (LLM decides which tools to call, when)
- Multi-step reasoning with tool use
- **When to use:** Complex tasks requiring planning, tool orchestration, decisions

**Our Focus:** Query engines. They're the foundation. Chat engines add state. Agents add autonomy.

### Response Synthesis Modes

The response synthesizer determines how retrieved nodes are combined into a final answer. This is critical for quality/latency/cost tradeoffs.

**Available Modes:**

1. **compact (default):** Concatenate chunks to fit context window, refine across chunks
   - **Pros:** Fewer LLM calls than refine, good quality
   - **Cons:** Can still be slow with many nodes
   - **When:** Default starting point for most use cases

2. **refine:** Iterative refinementâ€”use first node for initial answer, refine with each subsequent node
   - **Pros:** High quality, considers all context
   - **Cons:** N LLM calls for N nodes (slow, expensive)
   - **When:** High-stakes answers where quality > speed

3. **tree_summarize:** Build tree of summaries, recursively combine upward
   - **Pros:** Good for summarization, parallelizable
   - **Cons:** More complex, not always better than compact
   - **When:** Summarization tasks, long documents

4. **simple_summarize:** Truncate all nodes to fit single prompt
   - **Pros:** Fast, single LLM call
   - **Cons:** Loses detail through truncation
   - **When:** Quick summaries, not critical answers

5. **accumulate:** Query each node separately, return all responses
   - **Pros:** Parallel execution, captures diverse perspectives
   - **Cons:** No synthesis, user gets raw list
   - **When:** Comparing different viewpoints, exploratory search

6. **no_text:** Run retriever only, don't synthesize
   - **Pros:** Fastest, useful for debugging
   - **Cons:** No answer, just retrieved nodes
   - **When:** Testing retrieval quality, debugging

**Production Tradeoffs:**

| Mode | LLM Calls | Latency | Quality | Cost | Use Case |
|------|-----------|---------|---------|------|----------|
| simple_summarize | 1 | Lowest | Low | Lowest | Quick summaries |
| compact | 2-5 | Medium | High | Medium | Default choice |
| tree_summarize | 3-10 | High | High | High | Summarization |
| refine | N | Highest | Highest | Highest | Critical answers |
| accumulate | N (parallel) | Medium | Medium | High | Multiple perspectives |

**Key Insight:** Start with `compact`. It's the right tradeoff for 80% of use cases. Only optimize if you have proven latency/quality issues.

---

## Implementation Patterns

### Pattern 1: Basic Vector Store Query Engine

**The Foundation:** Simple RAG using LlamaIndex abstractions.

```python
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Settings
)
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

# Configure settings
Settings.llm = OpenAI(model="gpt-4", temperature=0)
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

# Load documents
documents = SimpleDirectoryReader("./data").load_data()

# Create index
index = VectorStoreIndex.from_documents(documents)

# Create query engine (uses default settings)
query_engine = index.as_query_engine()

# Query
response = query_engine.query("What are the key findings?")
print(response)

# Access source nodes for citations
for node in response.source_nodes:
    print(f"Score: {node.score:.4f}")
    print(f"Text: {node.text[:200]}...")
    print(f"Metadata: {node.metadata}")
```

**Key Insight:** `as_query_engine()` creates a `RetrieverQueryEngine` with sensible defaults. It's simple but not configurable. For production, use explicit configuration.

### Pattern 2: Configured Query Engine with Response Modes

**Production Setup:** Explicit configuration for full control.

```python
from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import ResponseMode, get_response_synthesizer
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.postprocessor import SimilarityPostprocessor

# Build index
index = VectorStoreIndex.from_documents(documents)

# Configure retriever
retriever = VectorIndexRetriever(
    index=index,
    similarity_top_k=10,  # Retrieve top 10 candidates
)

# Configure response synthesizer
response_synthesizer = get_response_synthesizer(
    response_mode=ResponseMode.COMPACT,  # Explicit mode
    use_async=True,  # Enable async LLM calls
    streaming=False,  # Set True for streaming responses
)

# Add postprocessors (filter, rerank)
node_postprocessors = [
    SimilarityPostprocessor(similarity_cutoff=0.7)  # Filter low-relevance nodes
]

# Build query engine
query_engine = RetrieverQueryEngine(
    retriever=retriever,
    response_synthesizer=response_synthesizer,
    node_postprocessors=node_postprocessors,
)

# Query
response = query_engine.query("What are the main conclusions?")
print(response)
```

**Key Insight:** Production query engines need explicit configuration. Don't rely on defaultsâ€”they're not optimized for your use case.

### Pattern 3: Streaming Responses

**Real-Time UX:** Stream tokens as they're generated.

```python
from llama_index.core import VectorStoreIndex
from llama_index.core.response_synthesizers import ResponseMode, get_response_synthesizer

# Build index
index = VectorStoreIndex.from_documents(documents)

# Create streaming query engine
streaming_synthesizer = get_response_synthesizer(
    response_mode=ResponseMode.COMPACT,
    streaming=True,  # Enable streaming
)

query_engine = index.as_query_engine(
    response_synthesizer=streaming_synthesizer
)

# Query with streaming
streaming_response = query_engine.query("Explain the findings in detail")

# Print tokens as they arrive
for text in streaming_response.response_gen:
    print(text, end="", flush=True)
```

**Key Insight:** Streaming improves perceived latency. User sees progress immediately. Critical for long responses.

### Pattern 4: Async Query Execution

**High Throughput:** Process multiple queries concurrently.

```python
import asyncio
from llama_index.core import VectorStoreIndex
from llama_index.core.response_synthesizers import get_response_synthesizer

# Build index
index = VectorStoreIndex.from_documents(documents)

# Create async-enabled query engine
async_synthesizer = get_response_synthesizer(
    use_async=True  # Enable async execution
)

query_engine = index.as_query_engine(
    response_synthesizer=async_synthesizer
)

# Async query function
async def async_query(query_str: str):
    response = await query_engine.aquery(query_str)
    return response

# Process multiple queries concurrently
async def process_batch(queries: list[str]):
    tasks = [async_query(q) for q in queries]
    responses = await asyncio.gather(*tasks)
    return responses

# Run
queries = [
    "What are the key findings?",
    "What methodology was used?",
    "What are the limitations?"
]

responses = asyncio.run(process_batch(queries))
for q, r in zip(queries, responses):
    print(f"Q: {q}")
    print(f"A: {r}\n")
```

**Key Insight:** Async execution is critical for batch processing. Don't process queries sequentiallyâ€”you're leaving performance on the table.

### Pattern 5: Router Query Engine - Multi-Index Routing

**The Problem:** You have multiple indexes (docs, code, FAQs), and queries should route to the right one(s).

**Solution:** Router query engine with selector strategies.

```python
from llama_index.core import VectorStoreIndex, SummaryIndex, SimpleDirectoryReader
from llama_index.core.query_engine import RouterQueryEngine
from llama_index.core.selectors import LLMSingleSelector, LLMMultiSelector
from llama_index.core.tools import QueryEngineTool

# Load different document sets
docs_technical = SimpleDirectoryReader("./technical_docs").load_data()
docs_marketing = SimpleDirectoryReader("./marketing_content").load_data()
docs_faqs = SimpleDirectoryReader("./faqs").load_data()

# Create separate indexes
vector_index_technical = VectorStoreIndex.from_documents(docs_technical)
vector_index_marketing = VectorStoreIndex.from_documents(docs_marketing)
summary_index_faqs = SummaryIndex.from_documents(docs_faqs)

# Create query engines for each index
technical_engine = vector_index_technical.as_query_engine(
    similarity_top_k=5
)
marketing_engine = vector_index_marketing.as_query_engine(
    similarity_top_k=5
)
faq_engine = summary_index_faqs.as_query_engine()

# Wrap in QueryEngineTool with descriptions
# CRITICAL: Descriptions guide the router's decision
technical_tool = QueryEngineTool.from_defaults(
    query_engine=technical_engine,
    description=(
        "Useful for retrieving technical documentation, API references, "
        "architecture details, and implementation guides. Use for queries "
        "about 'how to implement', 'API endpoints', 'technical specifications'."
    )
)

marketing_tool = QueryEngineTool.from_defaults(
    query_engine=marketing_engine,
    description=(
        "Useful for marketing content, product descriptions, use cases, "
        "customer success stories, and value propositions. Use for queries "
        "about 'product benefits', 'use cases', 'customer examples'."
    )
)

faq_tool = QueryEngineTool.from_defaults(
    query_engine=faq_engine,
    description=(
        "Useful for frequently asked questions, common issues, troubleshooting, "
        "and quick answers. Use for queries like 'how do I...', 'what is...', "
        "'can I...', 'does it support...'."
    )
)

# Create router with single selector (routes to ONE index)
router_query_engine = RouterQueryEngine(
    selector=LLMSingleSelector.from_defaults(),
    query_engine_tools=[
        technical_tool,
        marketing_tool,
        faq_tool,
    ],
    verbose=True  # See routing decisions
)

# Query - router decides which index to use
response = router_query_engine.query(
    "How do I authenticate API requests?"
)
print(response)

# Router output (verbose=True):
# selections=[SingleSelection(index=0, reason='Technical API question requires technical docs')]
```

**Key Insight:** Router quality depends on tool descriptions. Invest time writing clear, distinctive descriptions that capture when each index should be used.

### Pattern 6: Router with Multi-Selector (Query Multiple Indexes)

**The Problem:** Some queries require information from multiple indexes.

**Solution:** Multi-selector routes to multiple indexes, aggregates responses.

```python
from llama_index.core.query_engine import RouterQueryEngine
from llama_index.core.selectors import LLMMultiSelector
from llama_index.core.tools import QueryEngineTool

# Create router with MULTI selector
multi_router_engine = RouterQueryEngine(
    selector=LLMMultiSelector.from_defaults(),  # Can select multiple tools
    query_engine_tools=[
        technical_tool,
        marketing_tool,
        faq_tool,
    ],
    verbose=True
)

# Query that needs multiple indexes
response = multi_router_engine.query(
    "Compare the technical architecture with customer use cases"
)
print(response)

# Router output:
# selections=[
#     SingleSelection(index=0, reason='Need technical architecture details'),
#     SingleSelection(index=1, reason='Need customer use cases')
# ]
# Synthesizes responses from both indexes
```

**Key Insight:** Multi-selector is powerful but expensive (queries multiple indexes, then synthesizes). Use when cross-index synthesis is genuinely needed, not as default.

### Pattern 7: Sub-Question Query Engine - Complex Query Decomposition

**The Problem:** Complex questions like "Compare X and Y" require breaking into sub-questions.

**Solution:** Sub-question engine decomposes query, executes sub-questions, synthesizes.

```python
from llama_index.core import VectorStoreIndex
from llama_index.core.query_engine import SubQuestionQueryEngine
from llama_index.core.tools import QueryEngineTool
from llama_index.core.question_gen import LLMQuestionGenerator

# Load documents for two topics
docs_uber = SimpleDirectoryReader("./uber_10k").load_data()
docs_lyft = SimpleDirectoryReader("./lyft_10k").load_data()

# Create separate indexes
uber_index = VectorStoreIndex.from_documents(docs_uber)
lyft_index = VectorStoreIndex.from_documents(docs_lyft)

# Create query engines
uber_engine = uber_index.as_query_engine(similarity_top_k=5)
lyft_engine = lyft_index.as_query_engine(similarity_top_k=5)

# Wrap in tools with SPECIFIC descriptions
uber_tool = QueryEngineTool.from_defaults(
    query_engine=uber_engine,
    description="Provides detailed information about Uber's financials, operations, and strategy from their 2021 10-K filing"
)

lyft_tool = QueryEngineTool.from_defaults(
    query_engine=lyft_engine,
    description="Provides detailed information about Lyft's financials, operations, and strategy from their 2021 10-K filing"
)

# Create sub-question query engine
sub_question_engine = SubQuestionQueryEngine.from_defaults(
    query_engine_tools=[uber_tool, lyft_tool],
    use_async=True,  # Execute sub-questions in parallel
    verbose=True  # See sub-question generation
)

# Complex query gets decomposed
response = sub_question_engine.query(
    "Compare and contrast the revenue growth strategies of Uber and Lyft"
)

# Sub-questions generated:
# 1. What is Uber's revenue growth strategy? (uber_tool)
# 2. What is Lyft's revenue growth strategy? (lyft_tool)
# Then synthesizes comparison

print(response)
```

**Key Insight:** Sub-question engines shine on comparative/analytical queries. The question generator decides decompositionâ€”trust but verify with `verbose=True`.

### Pattern 8: Custom Query Engine

**When to Build Custom:** When built-in engines don't fit your orchestration pattern.

```python
from llama_index.core.query_engine import BaseQueryEngine
from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.core.response import Response
from typing import List

class CustomFilteredQueryEngine(BaseQueryEngine):
    """Query engine with custom pre-filtering logic."""
    
    def __init__(self, base_engine, user_access_level: int):
        self.base_engine = base_engine
        self.user_access_level = user_access_level
        super().__init__(callback_manager=base_engine.callback_manager)
    
    def _query(self, query_bundle: QueryBundle) -> Response:
        """Custom query logic with access control."""
        
        # Step 1: Get base retrieval
        response = self.base_engine.query(query_bundle)
        
        # Step 2: Filter nodes by access level
        filtered_nodes = [
            node for node in response.source_nodes
            if node.metadata.get('access_level', 0) <= self.user_access_level
        ]
        
        # Step 3: If no nodes pass filter, return access denied
        if not filtered_nodes:
            return Response(
                response="Access denied: No results match your permission level",
                source_nodes=[]
            )
        
        # Step 4: Re-synthesize with filtered nodes
        from llama_index.core.response_synthesizers import get_response_synthesizer
        synthesizer = get_response_synthesizer()
        
        filtered_response = synthesizer.synthesize(
            query_str=query_bundle.query_str,
            nodes=filtered_nodes
        )
        
        return filtered_response
    
    async def _aquery(self, query_bundle: QueryBundle) -> Response:
        """Async version."""
        # Implement async version if needed
        return self._query(query_bundle)

# Usage
base_engine = index.as_query_engine()
user_engine = CustomFilteredQueryEngine(
    base_engine=base_engine,
    user_access_level=2  # User has level 2 access
)

response = user_engine.query("What are the confidential findings?")
# Only returns findings where metadata['access_level'] <= 2
```

**Key Insight:** Custom query engines extend `BaseQueryEngine` and override `_query()`. Use when you need custom orchestration, filtering, or routing logic that built-in engines don't provide.

### Pattern 9: Query Transformations - HyDE

**The Problem:** User queries are often suboptimal for semantic search.

**Solution:** HyDE (Hypothetical Document Embeddings) generates hypothetical answer, embeds that instead of query.

```python
from llama_index.core import VectorStoreIndex
from llama_index.core.indices.query.query_transform import HyDEQueryTransform
from llama_index.core.query_engine import TransformQueryEngine

# Build index
index = VectorStoreIndex.from_documents(documents)
base_engine = index.as_query_engine()

# Create HyDE transformer
hyde = HyDEQueryTransform(include_original=True)

# Wrap query engine with transformation
hyde_engine = TransformQueryEngine(
    base_engine,
    query_transform=hyde
)

# Query transformation happens automatically
query = "What did the author work on after college?"

# Without HyDE - embeds query directly
response_no_hyde = base_engine.query(query)

# With HyDE - generates hypothetical answer, embeds that
# Hypothetical: "After college, the author worked as a software engineer..."
response_hyde = hyde_engine.query(query)

print("Without HyDE:", response_no_hyde)
print("\nWith HyDE:", response_hyde)
```

**Key Insight:** HyDE helps with vague queries but adds latency (LLM call to generate hypothetical document). Use when retrieval quality is poor, not by default.

### Pattern 10: Multi-Step Query Engine

**The Problem:** Some queries need iterative refinement (answer â†’ followup â†’ answer â†’ ...).

**Solution:** Multi-step query engine with query decomposition.

```python
from llama_index.core import VectorStoreIndex
from llama_index.core.query_engine import MultiStepQueryEngine
from llama_index.core.indices.query.query_transform import (
    StepDecomposeQueryTransform
)

# Build index
index = VectorStoreIndex.from_documents(documents)
base_engine = index.as_query_engine()

# Create step decompose transformer
step_decompose = StepDecomposeQueryTransform(
    llm=Settings.llm,
    verbose=True  # See step-by-step reasoning
)

# Create multi-step engine
multi_step_engine = MultiStepQueryEngine(
    query_engine=base_engine,
    query_transform=step_decompose,
    num_steps=3,  # Maximum number of refinement steps
)

# Complex query gets iteratively refined
response = multi_step_engine.query(
    "What were the main outcomes of the project and how did they impact subsequent work?"
)

# Step 1: "What were the main outcomes?"
# Step 2: "How did these outcomes impact subsequent work?"
# Step 3: Synthesize final answer

print(response)
```

**Key Insight:** Multi-step engines are powerful but expensive (multiple retrieval+synthesis rounds). Use for truly complex queries, not simple Q&A.

### Pattern 11: Metadata Filtering in Queries

**Production Essential:** Filter by date, category, access level, source.

```python
from llama_index.core import VectorStoreIndex
from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter

# Build index with metadata
documents = []
for file_path in file_paths:
    doc = SimpleDirectoryReader(input_files=[file_path]).load_data()[0]
    # Add metadata
    doc.metadata = {
        "category": "technical",
        "date": "2024-11-01",
        "access_level": 2,
        "source": file_path
    }
    documents.append(doc)

index = VectorStoreIndex.from_documents(documents)

# Query with metadata filters
filters = MetadataFilters(
    filters=[
        ExactMatchFilter(key="category", value="technical"),
        ExactMatchFilter(key="access_level", value=2)
    ]
)

query_engine = index.as_query_engine(
    filters=filters,
    similarity_top_k=5
)

response = query_engine.query("What are the API specifications?")
# Only searches documents where category="technical" AND access_level=2
```

**Key Insight:** Metadata filtering is non-negotiable for production. Semantic search + structured filtering = powerful system.

### Pattern 12: Response with Citations

**Production Essential:** Users need to verify claims.

```python
from llama_index.core import VectorStoreIndex

# Build index
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine(
    similarity_top_k=3
)

# Query
response = query_engine.query("What are the key findings?")

# Extract sources with scores
print("Answer:", response.response)
print("\nSources:")
for i, node in enumerate(response.source_nodes, 1):
    print(f"\n[{i}] Score: {node.score:.4f}")
    print(f"Text: {node.text[:200]}...")
    print(f"Metadata: {node.metadata}")
    
    # Generate citation
    source_file = node.metadata.get('file_path', 'Unknown')
    page = node.metadata.get('page_label', 'Unknown')
    print(f"Citation: {source_file}, page {page}")
```

**Key Insight:** Always return source nodes. Users need to verify claims. Build citation generation into your response formatting.

### Pattern 13: Reranking for Improved Accuracy

**The Problem:** Bi-encoder retrieval (fast) has lower precision than cross-encoder reranking (slow).

**Solution:** Two-stage retrieval: fast retrieval â†’ slow reranking.

```python
from llama_index.core import VectorStoreIndex
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.retrievers import VectorIndexRetriever

# Build index
index = VectorStoreIndex.from_documents(documents)

# Stage 1: Fast retrieval (top 50)
retriever = VectorIndexRetriever(
    index=index,
    similarity_top_k=50  # Retrieve many candidates
)

# Stage 2: Slow reranking (top 3)
reranker = SentenceTransformerRerank(
    model="cross-encoder/ms-marco-MiniLM-L-2-v2",
    top_n=3  # Rerank to top 3
)

# Create query engine with reranking
query_engine = index.as_query_engine(
    node_postprocessors=[reranker],
    similarity_top_k=50  # Initial retrieval
)

response = query_engine.query("What are the conclusions?")
# Retrieves 50, reranks to 3, synthesizes from top 3
```

**Key Insight:** Reranking trades latency for accuracy. Use for high-stakes queries (legal, medical, financial) where accuracy > speed.

### Pattern 14: Cost Tracking and Optimization

**Production Essential:** Track token usage and costs.

```python
from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.callbacks import CallbackManager, TokenCountingHandler
from llama_index.llms.openai import OpenAI
import tiktoken

# Setup token counter
token_counter = TokenCountingHandler(
    tokenizer=tiktoken.encoding_for_model("gpt-4").encode
)

# Configure callback manager
Settings.callback_manager = CallbackManager([token_counter])
Settings.llm = OpenAI(model="gpt-4", temperature=0)

# Build index and query
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()

# Reset counter
token_counter.reset_counts()

# Query
response = query_engine.query("What are the findings?")

# Get token usage
print(f"Prompt tokens: {token_counter.prompt_llm_token_count}")
print(f"Completion tokens: {token_counter.completion_llm_token_count}")
print(f"Total tokens: {token_counter.total_llm_token_count}")

# Calculate cost (GPT-4 pricing)
prompt_cost = (token_counter.prompt_llm_token_count / 1000) * 0.03
completion_cost = (token_counter.completion_llm_token_count / 1000) * 0.06
total_cost = prompt_cost + completion_cost

print(f"Estimated cost: ${total_cost:.4f}")
```

**Key Insight:** Token tracking is essential for cost optimization. Track per-query costs, identify expensive patterns, optimize response modes.

### Pattern 15: Error Handling and Fallbacks

**Production Essential:** Graceful degradation when things fail.

```python
from llama_index.core import VectorStoreIndex
from llama_index.core.schema import QueryBundle
from llama_index.core.response import Response
import logging

class RobustQueryEngine:
    """Query engine with fallback strategies."""
    
    def __init__(self, primary_engine, fallback_engine=None):
        self.primary_engine = primary_engine
        self.fallback_engine = fallback_engine
        self.logger = logging.getLogger(__name__)
    
    def query(self, query_str: str) -> Response:
        """Query with error handling and fallback."""
        
        try:
            # Try primary engine
            response = self.primary_engine.query(query_str)
            
            # Check if response is empty or low quality
            if not response.source_nodes:
                self.logger.warning(f"No results for query: {query_str}")
                if self.fallback_engine:
                    return self._fallback_query(query_str)
                return self._no_results_response(query_str)
            
            # Check similarity scores
            avg_score = sum(n.score for n in response.source_nodes) / len(response.source_nodes)
            if avg_score < 0.5:  # Threshold for low quality
                self.logger.warning(f"Low quality results (avg score: {avg_score:.2f})")
                if self.fallback_engine:
                    return self._fallback_query(query_str)
            
            return response
            
        except Exception as e:
            self.logger.error(f"Query failed: {e}")
            if self.fallback_engine:
                return self._fallback_query(query_str)
            return self._error_response(query_str, e)
    
    def _fallback_query(self, query_str: str) -> Response:
        """Execute fallback query."""
        try:
            self.logger.info("Attempting fallback query engine")
            return self.fallback_engine.query(query_str)
        except Exception as e:
            self.logger.error(f"Fallback failed: {e}")
            return self._error_response(query_str, e)
    
    def _no_results_response(self, query_str: str) -> Response:
        """Return response when no results found."""
        return Response(
            response="I couldn't find relevant information to answer that question. Please try rephrasing or asking something else.",
            source_nodes=[]
        )
    
    def _error_response(self, query_str: str, error: Exception) -> Response:
        """Return response when error occurs."""
        return Response(
            response=f"An error occurred while processing your query. Please try again or contact support.",
            source_nodes=[]
        )

# Usage
primary_engine = index.as_query_engine(similarity_top_k=5)
fallback_engine = index.as_query_engine(
    similarity_top_k=10,  # Cast wider net
    response_mode="simple_summarize"  # Faster fallback
)

robust_engine = RobustQueryEngine(primary_engine, fallback_engine)
response = robust_engine.query("ambiguous query")
```

**Key Insight:** Production systems fail. Build fallbacks, log failures, degrade gracefully. Don't crash on bad queries.

### Pattern 16: Batch Query Processing

**High Throughput:** Process many queries efficiently.

```python
import asyncio
from llama_index.core import VectorStoreIndex
from llama_index.core.response_synthesizers import get_response_synthesizer
from typing import List, Dict
import time

class BatchQueryProcessor:
    """Efficient batch query processing with async."""
    
    def __init__(self, index: VectorStoreIndex, batch_size: int = 10):
        self.batch_size = batch_size
        
        # Create async query engine
        synthesizer = get_response_synthesizer(use_async=True)
        self.query_engine = index.as_query_engine(
            response_synthesizer=synthesizer
        )
    
    async def _process_query(self, query_id: str, query: str) -> Dict:
        """Process single query async."""
        try:
            response = await self.query_engine.aquery(query)
            return {
                "query_id": query_id,
                "query": query,
                "response": str(response),
                "success": True,
                "error": None
            }
        except Exception as e:
            return {
                "query_id": query_id,
                "query": query,
                "response": None,
                "success": False,
                "error": str(e)
            }
    
    async def process_batch(self, queries: Dict[str, str]) -> List[Dict]:
        """Process batch of queries concurrently."""
        tasks = [
            self._process_query(qid, q)
            for qid, q in queries.items()
        ]
        return await asyncio.gather(*tasks)
    
    def process_queries(self, queries: Dict[str, str]) -> List[Dict]:
        """Synchronous wrapper for batch processing."""
        return asyncio.run(self.process_batch(queries))

# Usage
index = VectorStoreIndex.from_documents(documents)
processor = BatchQueryProcessor(index, batch_size=10)

# Process 100 queries
queries = {
    f"q_{i}": f"Query {i} text"
    for i in range(100)
}

start = time.time()
results = processor.process_queries(queries)
elapsed = time.time() - start

print(f"Processed {len(results)} queries in {elapsed:.2f}s")
print(f"Rate: {len(results)/elapsed:.2f} queries/sec")

# Analyze results
successful = sum(1 for r in results if r['success'])
print(f"Success rate: {successful/len(results)*100:.1f}%")
```

**Key Insight:** Async execution is critical for throughput. Process queries concurrently, not sequentially.

### Pattern 17: Production Query Engine Class

**Complete Production Setup:** Everything together.

```python
from llama_index.core import VectorStoreIndex, Settings, StorageContext
from llama_index.core.query_engine import RouterQueryEngine
from llama_index.core.response_synthesizers import get_response_synthesizer, ResponseMode
from llama_index.core.callbacks import CallbackManager, TokenCountingHandler
from llama_index.core.postprocessor import SimilarityPostprocessor, SentenceTransformerRerank
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from typing import Optional, Dict, Any
import logging
import tiktoken

class ProductionQueryEngine:
    """
    Production-grade query engine with:
    - Multiple indexes with routing
    - Cost tracking
    - Error handling
    - Metadata filtering
    - Reranking
    - Streaming support
    - Async execution
    """
    
    def __init__(
        self,
        indexes: Dict[str, VectorStoreIndex],
        index_descriptions: Dict[str, str],
        model: str = "gpt-4",
        response_mode: str = "compact",
        enable_reranking: bool = True,
        enable_cost_tracking: bool = True,
    ):
        self.indexes = indexes
        self.logger = logging.getLogger(__name__)
        
        # Setup token tracking
        if enable_cost_tracking:
            self.token_counter = TokenCountingHandler(
                tokenizer=tiktoken.encoding_for_model(model).encode
            )
            Settings.callback_manager = CallbackManager([self.token_counter])
        
        # Configure LLM and embeddings
        Settings.llm = OpenAI(model=model, temperature=0)
        Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
        
        # Build router query engine
        self.query_engine = self._build_router_engine(
            indexes,
            index_descriptions,
            response_mode,
            enable_reranking
        )
    
    def _build_router_engine(
        self,
        indexes: Dict[str, VectorStoreIndex],
        descriptions: Dict[str, str],
        response_mode: str,
        enable_reranking: bool
    ):
        """Build router query engine from multiple indexes."""
        from llama_index.core.selectors import LLMSingleSelector
        from llama_index.core.tools import QueryEngineTool
        
        # Create query engines for each index
        tools = []
        for name, index in indexes.items():
            # Configure postprocessors
            postprocessors = [
                SimilarityPostprocessor(similarity_cutoff=0.7)
            ]
            if enable_reranking:
                postprocessors.append(
                    SentenceTransformerRerank(
                        model="cross-encoder/ms-marco-MiniLM-L-2-v2",
                        top_n=3
                    )
                )
            
            # Create query engine
            engine = index.as_query_engine(
                similarity_top_k=50 if enable_reranking else 5,
                response_mode=response_mode,
                node_postprocessors=postprocessors
            )
            
            # Wrap in tool
            tool = QueryEngineTool.from_defaults(
                query_engine=engine,
                description=descriptions[name]
            )
            tools.append(tool)
        
        # Create router
        return RouterQueryEngine(
            selector=LLMSingleSelector.from_defaults(),
            query_engine_tools=tools,
            verbose=True
        )
    
    def query(
        self,
        query_str: str,
        metadata_filters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute query with full error handling.
        
        Returns dict with:
        - response: Answer text
        - sources: List of source nodes
        - cost: Estimated cost in USD
        - tokens: Token usage
        """
        try:
            # Reset token counter
            if hasattr(self, 'token_counter'):
                self.token_counter.reset_counts()
            
            # Execute query
            response = self.query_engine.query(query_str)
            
            # Calculate cost
            cost_info = self._calculate_cost() if hasattr(self, 'token_counter') else {}
            
            return {
                "success": True,
                "response": str(response),
                "sources": [
                    {
                        "text": node.text[:200],
                        "score": node.score,
                        "metadata": node.metadata
                    }
                    for node in response.source_nodes
                ],
                **cost_info
            }
            
        except Exception as e:
            self.logger.error(f"Query failed: {e}", exc_info=True)
            return {
                "success": False,
                "response": "An error occurred processing your query.",
                "error": str(e),
                "sources": []
            }
    
    async def aquery(
        self,
        query_str: str,
        metadata_filters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Async version of query."""
        # Similar implementation with await
        pass
    
    def _calculate_cost(self) -> Dict[str, Any]:
        """Calculate query cost from token usage."""
        # GPT-4 pricing (adjust as needed)
        PROMPT_COST_PER_1K = 0.03
        COMPLETION_COST_PER_1K = 0.06
        
        prompt_cost = (self.token_counter.prompt_llm_token_count / 1000) * PROMPT_COST_PER_1K
        completion_cost = (self.token_counter.completion_llm_token_count / 1000) * COMPLETION_COST_PER_1K
        
        return {
            "tokens": {
                "prompt": self.token_counter.prompt_llm_token_count,
                "completion": self.token_counter.completion_llm_token_count,
                "total": self.token_counter.total_llm_token_count
            },
            "cost_usd": prompt_cost + completion_cost
        }

# Usage
indexes = {
    "technical": technical_index,
    "marketing": marketing_index,
    "faqs": faq_index
}

descriptions = {
    "technical": "Technical documentation and API references",
    "marketing": "Product marketing and use cases",
    "faqs": "Frequently asked questions"
}

engine = ProductionQueryEngine(
    indexes=indexes,
    index_descriptions=descriptions,
    enable_reranking=True,
    enable_cost_tracking=True
)

result = engine.query("How do I authenticate API requests?")
print(f"Answer: {result['response']}")
print(f"Cost: ${result['cost_usd']:.4f}")
print(f"Tokens: {result['tokens']['total']}")
```

**Key Insight:** Production query engines bundle all the pieces: routing, reranking, cost tracking, error handling, metadata filtering. Build once, use everywhere.

---

## Common Pitfalls

### 1. Using Default Configurations in Production

**Problem:** `index.as_query_engine()` uses defaults not optimized for your use case.

**Example of Failure:**
```python
# BAD: Default everything
query_engine = index.as_query_engine()
# - Default similarity_top_k=2 (too few nodes)
# - Default response_mode=compact (maybe wrong for your use case)
# - No reranking
# - No metadata filtering
# - No cost tracking
```

**Solution:**
```python
# GOOD: Explicit configuration
from llama_index.core.postprocessor import SimilarityPostprocessor

query_engine = index.as_query_engine(
    similarity_top_k=10,  # Retrieve enough candidates
    response_mode=ResponseMode.COMPACT,  # Explicit choice
    node_postprocessors=[
        SimilarityPostprocessor(similarity_cutoff=0.7)
    ]
)
```

**Why:** Defaults are starting points, not production settings. Every parameter affects quality/latency/cost. Optimize for your use case.

### 2. Poor Router Tool Descriptions

**Problem:** Router quality depends entirely on tool descriptions. Vague descriptions â†’ wrong routing.

**Example of Failure:**
```python
# BAD: Vague descriptions
technical_tool = QueryEngineTool.from_defaults(
    query_engine=technical_engine,
    description="Technical information"  # Too vague!
)

marketing_tool = QueryEngineTool.from_defaults(
    query_engine=marketing_engine,
    description="Marketing information"  # Overlaps!
)

# Result: Router can't distinguish, routes randomly
```

**Solution:**
```python
# GOOD: Specific, distinctive descriptions
technical_tool = QueryEngineTool.from_defaults(
    query_engine=technical_engine,
    description=(
        "Provides detailed technical documentation including: "
        "API endpoints, authentication methods, request/response formats, "
        "error codes, rate limits, SDKs, and implementation examples. "
        "Use for queries containing: 'how to implement', 'API', 'code', "
        "'authentication', 'endpoint', 'SDK', 'technical specs'."
    )
)

marketing_tool = QueryEngineTool.from_defaults(
    query_engine=marketing_engine,
    description=(
        "Provides marketing and business content including: "
        "product benefits, use cases, customer success stories, "
        "competitive comparisons, pricing, and ROI examples. "
        "Use for queries containing: 'benefits', 'use case', 'customer', "
        "'pricing', 'why use', 'ROI', 'advantages'."
    )
)
```

**Why:** LLM makes routing decisions based on descriptions. Invest time writing clear, specific descriptions with example query patterns.

### 3. Not Using Metadata Filtering

**Problem:** Semantic search alone returns irrelevant results (wrong date, wrong category, wrong access level).

**Example of Failure:**
```python
# BAD: No filtering
query_engine = index.as_query_engine()
response = query_engine.query("Latest API changes")
# Returns old docs because they're semantically similar
```

**Solution:**
```python
# GOOD: Filter by date
from datetime import datetime, timedelta
from llama_index.core.vector_stores import MetadataFilters, MetadataFilter

recent_date = datetime.now() - timedelta(days=30)

filters = MetadataFilters(
    filters=[
        MetadataFilter(
            key="date",
            value=recent_date.isoformat(),
            operator=">"  # Date greater than 30 days ago
        )
    ]
)

query_engine = index.as_query_engine(filters=filters)
response = query_engine.query("Latest API changes")
# Only searches recent docs
```

**Why:** Semantic + structured filtering = production system. Use metadata for date, category, access control, source filtering.

### 4. Ignoring Response Mode Tradeoffs

**Problem:** Using wrong response mode for the task.

**Example of Failure:**
```python
# BAD: Using refine for simple Q&A
query_engine = index.as_query_engine(
    response_mode="refine",  # N LLM calls, high latency
    similarity_top_k=10
)
response = query_engine.query("What is X?")
# 10 LLM calls for simple question = slow and expensive
```

**Solution:**
```python
# GOOD: Match mode to task

# Simple Q&A â†’ compact (default)
qa_engine = index.as_query_engine(
    response_mode="compact"
)

# Summarization â†’ tree_summarize
summary_engine = index.as_query_engine(
    response_mode="tree_summarize"
)

# Critical accuracy â†’ refine (accept cost/latency)
critical_engine = index.as_query_engine(
    response_mode="refine"
)
```

**Why:** Response modes have different latency/cost/quality profiles. Use compact for most cases. Only use expensive modes when justified.

### 5. Not Handling Low-Quality Retrievals

**Problem:** Query engine always returns answer, even with irrelevant results.

**Example of Failure:**
```python
# BAD: No quality check
query_engine = index.as_query_engine()
response = query_engine.query("Completely off-topic question")
# LLM tries to answer from irrelevant nodes â†’ hallucination
```

**Solution:**
```python
# GOOD: Check similarity scores, return "no answer" if low quality
from llama_index.core.postprocessor import SimilarityPostprocessor

query_engine = index.as_query_engine(
    node_postprocessors=[
        SimilarityPostprocessor(similarity_cutoff=0.7)
    ]
)

response = query_engine.query("Off-topic question")

# Check if any nodes passed filter
if not response.source_nodes:
    print("I don't have information to answer that question.")
else:
    print(response)
```

**Why:** Irrelevant context â†’ hallucinated answers. Filter low-quality retrievals, return honest "I don't know" when appropriate.

### 6. Not Caching Query Engines

**Problem:** Rebuilding query engines on every request.

**Example of Failure:**
```python
# BAD: Rebuild on every request
@app.get("/query")
def query_endpoint(q: str):
    index = VectorStoreIndex.from_documents(documents)  # SLOW!
    query_engine = index.as_query_engine()  # Rebuild every time
    response = query_engine.query(q)
    return {"response": str(response)}
```

**Solution:**
```python
# GOOD: Build once, reuse
from functools import lru_cache

@lru_cache(maxsize=1)
def get_query_engine():
    """Build and cache query engine."""
    index = VectorStoreIndex.from_documents(documents)
    return index.as_query_engine()

@app.get("/query")
def query_endpoint(q: str):
    query_engine = get_query_engine()  # Cached
    response = query_engine.query(q)
    return {"response": str(response)}
```

**Why:** Query engine construction is expensive. Build once, cache, reuse across requests.

### 7. Forgetting Async for Batch Processing

**Problem:** Processing queries sequentially instead of concurrently.

**Example of Failure:**
```python
# BAD: Sequential processing
query_engine = index.as_query_engine()

responses = []
for query in queries:  # Processes one at a time
    response = query_engine.query(query)
    responses.append(response)

# 100 queries * 2s each = 200s total
```

**Solution:**
```python
# GOOD: Concurrent processing
import asyncio

query_engine = index.as_query_engine(
    response_synthesizer=get_response_synthesizer(use_async=True)
)

async def process_queries(queries):
    tasks = [query_engine.aquery(q) for q in queries]
    return await asyncio.gather(*tasks)

responses = asyncio.run(process_queries(queries))
# 100 queries in ~10s with concurrency
```

**Why:** Async execution is critical for throughput. Use for batch processing, high-volume applications.

### 8. Not Tracking Costs

**Problem:** No visibility into token usage and costs.

**Example of Failure:**
```python
# BAD: No cost tracking
query_engine = index.as_query_engine()
response = query_engine.query(query)
# No idea what this cost
```

**Solution:**
```python
# GOOD: Track every query
from llama_index.core.callbacks import TokenCountingHandler
import tiktoken

token_counter = TokenCountingHandler(
    tokenizer=tiktoken.encoding_for_model("gpt-4").encode
)
Settings.callback_manager = CallbackManager([token_counter])

query_engine = index.as_query_engine()

token_counter.reset_counts()
response = query_engine.query(query)

cost = (token_counter.total_llm_token_count / 1000) * 0.03  # Adjust pricing
print(f"Cost: ${cost:.4f}")
```

**Why:** Can't optimize what you don't measure. Track costs, identify expensive patterns, optimize.

### 9. Using Sub-Question Engine for Simple Queries

**Problem:** Complex engines for simple tasks.

**Example of Failure:**
```python
# BAD: Sub-question engine for "What is X?"
sub_question_engine = SubQuestionQueryEngine.from_defaults(
    query_engine_tools=[tool1, tool2, tool3]
)

response = sub_question_engine.query("What is the API key?")
# Generates sub-questions for simple lookup = overkill
```

**Solution:**
```python
# GOOD: Route by complexity
def get_query_engine(query: str):
    # Simple queries â†’ basic engine
    if len(query.split()) < 10 and "compare" not in query.lower():
        return basic_engine
    
    # Complex queries â†’ sub-question engine
    return sub_question_engine

response = get_query_engine(query).query(query)
```

**Why:** Advanced engines add latency and cost. Use complexity-appropriate engines.

### 10. Not Testing Retrieval Quality

**Problem:** Assuming retrieval works without validation.

**Example of Failure:**
```python
# BAD: No retrieval quality checks
query_engine = index.as_query_engine()
# How do you know it's retrieving relevant docs?
```

**Solution:**
```python
# GOOD: Test and monitor retrieval
def evaluate_retrieval(query_engine, test_queries):
    """Evaluate retrieval quality."""
    results = []
    
    for query, expected_docs in test_queries.items():
        response = query_engine.query(query)
        
        retrieved_ids = {node.id_ for node in response.source_nodes}
        expected_ids = set(expected_docs)
        
        # Calculate recall
        recall = len(retrieved_ids & expected_ids) / len(expected_ids)
        results.append({
            "query": query,
            "recall": recall,
            "retrieved": len(retrieved_ids)
        })
    
    avg_recall = sum(r["recall"] for r in results) / len(results)
    return avg_recall, results

# Test
test_queries = {
    "What is the API authentication method?": ["doc_123", "doc_456"],
    "How do I handle errors?": ["doc_789"],
}

avg_recall, details = evaluate_retrieval(query_engine, test_queries)
print(f"Average recall: {avg_recall:.2%}")
```

**Why:** Can't improve what you don't measure. Test retrieval with ground truth, monitor quality metrics.

---

## Integration Points

### 1. Prompt Routing

**Use Case:** Route queries to the right handler based on intent.

**Pattern:** Router query engine as routing layer.

```python
from llama_index.core import VectorStoreIndex
from llama_index.core.query_engine import RouterQueryEngine
from llama_index.core.tools import QueryEngineTool
from llama_index.core.selectors import LLMSingleSelector

# Create indexes for different intents
general_qa_index = VectorStoreIndex.from_documents(general_docs)
technical_index = VectorStoreIndex.from_documents(technical_docs)
support_index = VectorStoreIndex.from_documents(support_docs)

# Wrap in tools with intent descriptions
general_tool = QueryEngineTool.from_defaults(
    query_engine=general_qa_index.as_query_engine(),
    description="General questions about products, features, and company information"
)

technical_tool = QueryEngineTool.from_defaults(
    query_engine=technical_index.as_query_engine(),
    description="Technical implementation questions, API usage, code examples"
)

support_tool = QueryEngineTool.from_defaults(
    query_engine=support_index.as_query_engine(),
    description="Troubleshooting, error resolution, account issues"
)

# Router engine for intent-based routing
router_engine = RouterQueryEngine(
    selector=LLMSingleSelector.from_defaults(),
    query_engine_tools=[general_tool, technical_tool, support_tool]
)

# Query gets routed to appropriate handler
response = router_engine.query("How do I reset my password?")
# Routes to support_tool

response = router_engine.query("Show me authentication code example")
# Routes to technical_tool
```

**Key Insight:** Router query engines provide intelligent routing without manual intent classification. Tool descriptions guide routing decisions.

### 2. Query Writing

**Use Case:** Retrieve SQL examples and schema documentation to guide query generation.

**Pattern:** Sub-question engine for complex SQL generation.

```python
from llama_index.core import VectorStoreIndex
from llama_index.core.query_engine import SubQuestionQueryEngine
from llama_index.core.tools import QueryEngineTool

# Index for SQL examples
sql_examples_index = VectorStoreIndex.from_documents(sql_example_docs)

# Index for schema documentation
schema_index = VectorStoreIndex.from_documents(schema_docs)

# Tools for sub-question engine
examples_tool = QueryEngineTool.from_defaults(
    query_engine=sql_examples_index.as_query_engine(),
    description="SQL query examples for similar tasks and patterns"
)

schema_tool = QueryEngineTool.from_defaults(
    query_engine=schema_index.as_query_engine(),
    description="Database schema documentation, table structures, and relationships"
)

# Sub-question engine breaks complex query generation into steps
sub_question_engine = SubQuestionQueryEngine.from_defaults(
    query_engine_tools=[examples_tool, schema_tool],
    use_async=True
)

# Complex SQL generation request
response = sub_question_engine.query(
    "Generate SQL to find users who made purchases above $1000 in the last month"
)

# Sub-questions:
# 1. What tables contain user and purchase information? (schema_tool)
# 2. What are example queries for filtering by date and amount? (examples_tool)
# Synthesizes final SQL from both sources
```

**Key Insight:** Sub-question engines excel at multi-faceted tasks like SQL generation that require schema knowledge + example patterns.

### 3. Data Processing

**Use Case:** Retrieve transformation examples from historical pipelines.

**Pattern:** Vector retrieval for similar transformations.

```python
from llama_index.core import VectorStoreIndex

# Index historical data transformation examples
transformation_docs = [
    {"task": "CSV to JSON", "code": "...", "notes": "..."},
    {"task": "Data normalization", "code": "...", "notes": "..."},
    {"task": "Date parsing", "code": "...", "notes": "..."},
]

transformation_index = VectorStoreIndex.from_documents(transformation_docs)

# Query engine for finding similar transformations
transformation_engine = transformation_index.as_query_engine(
    similarity_top_k=3,
    response_mode="accumulate"  # Get multiple examples
)

# Find similar transformation examples
response = transformation_engine.query(
    "How do I convert nested JSON to flat CSV format?"
)

# Returns top 3 similar transformation examples
print(response)
```

**Key Insight:** Accumulate response mode works well for example retrievalâ€”you want multiple relevant examples, not synthesized answer.

### 4. Tool Orchestration

**Use Case:** Route to appropriate tools, compose tool chains.

**Pattern:** Router engine for tool selection.

```python
from llama_index.core.query_engine import RouterQueryEngine
from llama_index.core.tools import QueryEngineTool

# Different tool indexes
api_docs_index = VectorStoreIndex.from_documents(api_docs)
code_examples_index = VectorStoreIndex.from_documents(code_examples)
troubleshooting_index = VectorStoreIndex.from_documents(troubleshooting_docs)

# Tool descriptions guide orchestration
api_tool = QueryEngineTool.from_defaults(
    query_engine=api_docs_index.as_query_engine(),
    description="API endpoint documentation, parameters, and responses"
)

examples_tool = QueryEngineTool.from_defaults(
    query_engine=code_examples_index.as_query_engine(),
    description="Working code examples and integration patterns"
)

troubleshooting_tool = QueryEngineTool.from_defaults(
    query_engine=troubleshooting_index.as_query_engine(),
    description="Error messages, debugging steps, and solutions"
)

# Multi-selector for tool composition
from llama_index.core.selectors import LLMMultiSelector

router_engine = RouterQueryEngine(
    selector=LLMMultiSelector.from_defaults(),
    query_engine_tools=[api_tool, examples_tool, troubleshooting_tool]
)

# Query may use multiple tools
response = router_engine.query(
    "Show me how to implement user authentication with error handling"
)
# Uses api_tool (auth endpoints) + examples_tool (code) + troubleshooting_tool (errors)
```

**Key Insight:** Multi-selector enables tool composition. Single query can retrieve from multiple sources, synthesize unified response.

### 5. Decision Support

**Use Case:** Multi-faceted analysis for decision-making.

**Pattern:** Sub-question engine for analytical breakdown.

```python
from llama_index.core import VectorStoreIndex
from llama_index.core.query_engine import SubQuestionQueryEngine
from llama_index.core.tools import QueryEngineTool

# Index past decisions and outcomes
decisions_index = VectorStoreIndex.from_documents(decision_history_docs)

# Index market/competitive data
market_index = VectorStoreIndex.from_documents(market_research_docs)

# Index technical feasibility assessments
technical_index = VectorStoreIndex.from_documents(technical_assessments)

# Tools for different decision perspectives
historical_tool = QueryEngineTool.from_defaults(
    query_engine=decisions_index.as_query_engine(),
    description="Past decisions, outcomes, and lessons learned"
)

market_tool = QueryEngineTool.from_defaults(
    query_engine=market_index.as_query_engine(),
    description="Market trends, competitive landscape, customer needs"
)

technical_tool = QueryEngineTool.from_defaults(
    query_engine=technical_index.as_query_engine(),
    description="Technical feasibility, resource requirements, implementation risks"
)

# Sub-question engine for multi-perspective analysis
decision_engine = SubQuestionQueryEngine.from_defaults(
    query_engine_tools=[historical_tool, market_tool, technical_tool],
    use_async=True
)

# Complex decision analysis
response = decision_engine.query(
    "Should we invest in building a mobile app? Analyze technical feasibility, market demand, and lessons from past product launches."
)

# Sub-questions:
# 1. What technical resources and timeline required for mobile app? (technical_tool)
# 2. What is market demand for mobile access to our product? (market_tool)
# 3. What lessons can we learn from past product launches? (historical_tool)
# Synthesizes comprehensive decision analysis
```

**Key Insight:** Sub-question engines excel at decision supportâ€”breaking complex decisions into analytical sub-questions, gathering diverse perspectives, synthesizing recommendations.

---

## Our Takeaways

### For agentic_ai_development

**1. Query Engines Are RAG Orchestration, Not RAG Replacement**

We built RAG from scratch (document 9). LlamaIndex query engines don't replace thatâ€”they orchestrate it. Use query engines when you need:
- Multi-index routing (different data sources)
- Complex query decomposition (sub-questions)
- Advanced synthesis patterns (tree summarize, refine)

Skip query engines for simple single-index RAG. Our custom implementation is lighter and more controllable.

**2. Router Query Engines Solve the Multi-Source Problem**

When you have multiple data sources (docs, code, FAQs, historical data), router engines provide intelligent routing without manual intent classification.

**Critical success factor:** Tool descriptions. Router quality depends entirely on clear, specific, distinctive descriptions. Invest time here.

**Production pattern:** Start with single selector (routes to ONE index). Only use multi-selector when cross-index synthesis is genuinely needed.

**3. Sub-Question Engines Excel at Complex Analytical Queries**

"Compare X and Y", "Analyze A considering B and C", "Explain D in context of E"â€”these naturally decompose into sub-questions.

**When to use:** Comparative analysis, multi-faceted decisions, queries requiring multiple perspectives.

**When NOT to use:** Simple lookups ("What is X?"), single-topic questions. Sub-question overhead isn't worth it.

**Production insight:** Sub-question engines work best with `use_async=True`. Parallel execution of sub-questions dramatically reduces latency.

**4. Response Synthesis Modes Have Clear Tradeoffs**

| Mode | When to Use | When NOT to Use |
|------|-------------|-----------------|
| **compact** | Default choice, 80% of use cases | When you need highest quality regardless of cost |
| **refine** | High-stakes answers (legal, medical, financial) | High-volume applications (too slow) |
| **tree_summarize** | Summarization tasks, long documents | Simple Q&A (overkill) |
| **simple_summarize** | Quick summaries, low-stakes | Critical accuracy needs (truncates) |
| **accumulate** | Multiple perspectives, example retrieval | When you need synthesized answer |

**Production default:** Start with compact. Measure quality. Only switch if you have proven issues.

**5. Query Transformations Improve Retrieval But Add Latency**

**HyDE (Hypothetical Document Embeddings):**
- Generates hypothetical answer, embeds that instead of query
- Helps with vague/ambiguous queries
- Adds latency (extra LLM call)
- **Use when:** Retrieval quality is poor with direct query embedding
- **Skip when:** Queries are already well-formed

**Multi-step query transformations:**
- Iterative refinement (answer â†’ followup â†’ answer)
- Powerful but expensive (multiple retrieval+synthesis rounds)
- **Use when:** Complex queries need iterative exploration
- **Skip when:** Single retrieval round suffices

**Production approach:** Measure retrieval quality without transformations first. Only add transformations if you have proven quality issues.

**6. Metadata Filtering Is Non-Negotiable for Production**

Semantic search alone is insufficient. Production systems need:
- Date filtering ("recent changes", "as of 2024")
- Category filtering ("technical docs only")
- Access control (user permission level)
- Source filtering ("official documentation only")

**Pattern:** Every document gets metadata dictionary. Every query can filter by metadata.

**Implementation:** LlamaIndex supports metadata filters natively. Use them.

**7. Cost Tracking Reveals Optimization Opportunities**

Track tokens and costs for every query. Patterns emerge:
- Which response modes are most expensive
- Which queries generate highest token usage
- Where reranking adds value vs. cost
- Opportunities to cache, batch, or optimize

**Production pattern:** Log token usage with every query. Build cost dashboard. Review weekly. Optimize based on data.

**8. Async Execution Is Critical for Throughput**

Sequential query processing = leaving 10-50x performance on the table.

**Patterns requiring async:**
- Batch query processing
- Sub-question engines (parallel sub-questions)
- High-volume applications
- Response synthesis with multiple LLM calls

**Implementation:** Set `use_async=True` on response synthesizer. Use `aquery()` instead of `query()`. Process with `asyncio.gather()`.

**Production insight:** Async doesn't just improve throughputâ€”it improves user experience (faster responses) and reduces costs (better resource utilization).

**9. Reranking Trades Latency for Accuracy**

**Two-stage architecture:**
1. Stage 1: Bi-encoder retrieval (fast, ~70% accuracy) â†’ top 50
2. Stage 2: Cross-encoder reranking (slow, ~85% accuracy) â†’ top 3

**Latency impact:** +200-500ms for reranking step.

**When to rerank:**
- High-stakes applications (legal, medical, financial)
- Poor retrieval quality with bi-encoder alone
- Users willing to wait for accuracy

**When NOT to rerank:**
- Latency-sensitive applications
- High-volume query processing
- Good retrieval quality without reranking

**Production pattern:** A/B test with and without reranking. Measure quality improvement vs. latency cost. Only deploy if justified.

**10. Error Handling and Fallbacks Prevent Production Failures**

Query engines fail in multiple ways:
- No relevant results found
- Low-quality retrievals (low similarity scores)
- LLM errors (rate limits, timeouts)
- Index unavailable

**Production pattern:** Build fallback strategies:
1. Primary query engine (optimized for quality)
2. Fallback query engine (cast wider net, faster synthesis)
3. Error response (graceful degradation)

**Implementation:** Wrap query engines in error handling class. Log failures. Degrade gracefully.

**11. Don't Build Custom Query Engines Unless Necessary**

LlamaIndex provides rich set of query engines. Build custom only when:
- You need orchestration logic not provided (e.g., custom routing, filtering)
- You're integrating with external systems query engines don't support
- You have very specific synthesis requirements

**Before building custom:**
- Check if existing engines can be composed to solve problem
- Consider if router/sub-question engines with good descriptions solve it
- Evaluate if custom postprocessors/transformations suffice

**Building custom:** Extend `BaseQueryEngine`, override `_query()`. Keep simple.

**12. Query Engines vs. Chat Engines vs. Agents: Know the Boundaries**

**Query Engine:**
- Stateless (each query independent)
- Deterministic retrieval + synthesis
- Fast, predictable, controllable
- **Use when:** Single-shot Q&A, no conversation context needed

**Chat Engine:**
- Stateful (maintains conversation history)
- Deterministic retrieval + synthesis
- Handles context window management
- **Use when:** Multi-turn conversations, follow-up questions

**Agent:**
- Stateful or stateless
- Agentic (LLM decides tool use)
- Flexible but unpredictable
- **Use when:** Complex tasks, tool orchestration, planning

**Our approach:** Start with query engines (simplest). Add chat engines for conversations. Use agents only when autonomy is needed.

**13. Vector Database Choice Matters Less Than Query Engine Configuration**

All modern vector DBs (Pinecone, Weaviate, ChromaDB, Qdrant) are fast enough for production. Performance bottlenecks are typically:
- Poor response synthesis mode choice
- Missing metadata filtering
- No reranking when needed
- Inefficient batch processing

**Production focus:** Optimize query engine configuration before worrying about vector DB performance.

**Vector DB selection criteria:**
1. Operational burden (managed vs. self-hosted)
2. Scale (document count, query volume)
3. Cost (managed services vs. infrastructure costs)
4. Integration (native LlamaIndex support)

**14. Test Retrieval Quality, Don't Assume It Works**

Can't improve what you don't measure. Build evaluation pipeline:
1. Ground truth test queries with known relevant documents
2. Calculate recall@k (% of relevant docs retrieved)
3. Measure average similarity scores
4. Track query latency p50/p95/p99

**Production pattern:** Automated testing in CI/CD. Manual review of sample queries weekly. Quality dashboard.

**15. When LlamaIndex Helps vs. When It Hurts**

**LlamaIndex helps when:**
- Multiple indexes require intelligent routing
- Complex queries need decomposition
- You need proven synthesis patterns (tree summarize, refine)
- Team lacks time to build orchestration from scratch

**LlamaIndex hurts when:**
- Simple single-index RAG (adds unnecessary abstraction)
- You need fine-grained control over every step
- Debugging LlamaIndex internals wastes more time than building custom
- Performance requirements demand bare-metal optimization

**Our approach:** Use LlamaIndex for routing and complex synthesis. Use our custom RAG for simple cases. Evaluate based on complexity, not dogma.

---

## Implementation Checklist

### Phase 1: Basic Setup (Week 1)

**Day 1-2: Installation and Configuration**
- [ ] Install LlamaIndex: `pip install llama-index`
- [ ] Install vector store: `pip install llama-index-vector-stores-pinecone` (or Weaviate, ChromaDB)
- [ ] Configure LLM: OpenAI API key in `.env`
- [ ] Configure embeddings: `text-embedding-3-small` as default
- [ ] Test basic query engine on sample documents

**Day 3-4: Single Index Query Engine**
- [ ] Load documents with `SimpleDirectoryReader`
- [ ] Build `VectorStoreIndex` from documents
- [ ] Create query engine with `as_query_engine()`
- [ ] Test queries, inspect source nodes
- [ ] Configure `similarity_top_k` based on retrieval quality
- [ ] Add `SimilarityPostprocessor` with cutoff threshold

**Day 5-7: Response Synthesis**
- [ ] Test different response modes (compact, refine, tree_summarize)
- [ ] Measure latency and quality for each mode
- [ ] Choose default mode based on use case
- [ ] Configure streaming for long responses
- [ ] Add cost tracking with `TokenCountingHandler`
- [ ] Document synthesis mode choices

**Success Criteria:** Single index query engine working with good retrieval quality, cost tracking enabled.

---

### Phase 2: Advanced Features (Week 2)

**Day 8-9: Metadata Filtering**
- [ ] Add metadata to documents (date, category, access_level, source)
- [ ] Test metadata filters with `MetadataFilters`
- [ ] Implement access control filtering
- [ ] Add date-range filtering for recent documents
- [ ] Test filter combinations

**Day 10-11: Router Query Engine**
- [ ] Create multiple indexes for different content types
- [ ] Write clear, specific tool descriptions
- [ ] Build router with `LLMSingleSelector`
- [ ] Test routing decisions with `verbose=True`
- [ ] Refine descriptions based on routing errors
- [ ] Add multi-selector for cross-index queries

**Day 12-14: Sub-Question Query Engine**
- [ ] Identify queries that benefit from decomposition
- [ ] Create sub-question engine with multiple tools
- [ ] Enable async execution (`use_async=True`)
- [ ] Test on comparative queries ("Compare X and Y")
- [ ] Monitor sub-question generation quality
- [ ] Tune tool descriptions for better decomposition

**Success Criteria:** Router working with good routing accuracy. Sub-question engine decomposes complex queries correctly.

---

### Phase 3: Production Hardening (Week 3)

**Day 15-16: Error Handling**
- [ ] Wrap query engines in error handling
- [ ] Build fallback query engines
- [ ] Handle empty results gracefully
- [ ] Log all failures with context
- [ ] Add retry logic for transient failures
- [ ] Test failure modes

**Day 17-18: Performance Optimization**
- [ ] Add reranking postprocessor
- [ ] Measure reranking impact on latency/quality
- [ ] Implement batch query processing with async
- [ ] Cache query engines (don't rebuild)
- [ ] Optimize `similarity_top_k` for reranking
- [ ] Profile and optimize slow queries

**Day 19-21: Monitoring and Testing**
- [ ] Build retrieval quality evaluation
- [ ] Create test suite with ground truth queries
- [ ] Calculate recall@k for test queries
- [ ] Track average similarity scores
- [ ] Monitor query latency (p50, p95, p99)
- [ ] Set up cost dashboard
- [ ] Document quality metrics and targets

**Success Criteria:** Production-ready query engine with error handling, monitoring, and quality metrics.

---

## Testing Strategy

### Unit Tests

Test individual components in isolation.

```python
import pytest
from llama_index.core import VectorStoreIndex, Document

@pytest.fixture
def sample_index():
    """Create sample index for testing."""
    docs = [
        Document(text="Python is a programming language", metadata={"category": "tech"}),
        Document(text="Marketing strategies for startups", metadata={"category": "business"}),
    ]
    return VectorStoreIndex.from_documents(docs)

def test_query_engine_returns_response(sample_index):
    """Test that query engine returns valid response."""
    engine = sample_index.as_query_engine()
    response = engine.query("What is Python?")
    
    assert response is not None
    assert len(str(response)) > 0
    assert len(response.source_nodes) > 0

def test_metadata_filtering(sample_index):
    """Test that metadata filtering works."""
    from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter
    
    filters = MetadataFilters(
        filters=[ExactMatchFilter(key="category", value="tech")]
    )
    
    engine = sample_index.as_query_engine(filters=filters)
    response = engine.query("programming")
    
    # Should only return tech documents
    for node in response.source_nodes:
        assert node.metadata["category"] == "tech"

def test_similarity_threshold():
    """Test that similarity postprocessor filters low-quality results."""
    from llama_index.core.postprocessor import SimilarityPostprocessor
    
    postprocessor = SimilarityPostprocessor(similarity_cutoff=0.8)
    
    # Create mock nodes
    from llama_index.core.schema import NodeWithScore, TextNode
    nodes = [
        NodeWithScore(node=TextNode(text="relevant"), score=0.9),
        NodeWithScore(node=TextNode(text="less relevant"), score=0.6),
    ]
    
    filtered = postprocessor.postprocess_nodes(nodes)
    
    assert len(filtered) == 1
    assert filtered[0].score >= 0.8
```

### Integration Tests

Test end-to-end query engine workflows.

```python
def test_router_query_engine():
    """Test router query engine routing decisions."""
    from llama_index.core import VectorStoreIndex
    from llama_index.core.query_engine import RouterQueryEngine
    from llama_index.core.selectors import LLMSingleSelector
    from llama_index.core.tools import QueryEngineTool
    
    # Create indexes
    tech_docs = [Document(text="API authentication methods")]
    marketing_docs = [Document(text="Product benefits and use cases")]
    
    tech_index = VectorStoreIndex.from_documents(tech_docs)
    marketing_index = VectorStoreIndex.from_documents(marketing_docs)
    
    # Create tools
    tech_tool = QueryEngineTool.from_defaults(
        query_engine=tech_index.as_query_engine(),
        description="Technical documentation and API references"
    )
    
    marketing_tool = QueryEngineTool.from_defaults(
        query_engine=marketing_index.as_query_engine(),
        description="Marketing content and product information"
    )
    
    # Create router
    router = RouterQueryEngine(
        selector=LLMSingleSelector.from_defaults(),
        query_engine_tools=[tech_tool, marketing_tool]
    )
    
    # Test technical query routes to tech tool
    response = router.query("How do I authenticate API requests?")
    # Manual inspection: should route to tech_tool
    
    assert response is not None

def test_sub_question_query_engine():
    """Test sub-question decomposition."""
    from llama_index.core.query_engine import SubQuestionQueryEngine
    
    # Create indexes for two topics
    uber_docs = [Document(text="Uber financial data")]
    lyft_docs = [Document(text="Lyft financial data")]
    
    uber_index = VectorStoreIndex.from_documents(uber_docs)
    lyft_index = VectorStoreIndex.from_documents(lyft_docs)
    
    # Create tools
    from llama_index.core.tools import QueryEngineTool
    
    uber_tool = QueryEngineTool.from_defaults(
        query_engine=uber_index.as_query_engine(),
        description="Uber financial information"
    )
    
    lyft_tool = QueryEngineTool.from_defaults(
        query_engine=lyft_index.as_query_engine(),
        description="Lyft financial information"
    )
    
    # Create sub-question engine
    sub_engine = SubQuestionQueryEngine.from_defaults(
        query_engine_tools=[uber_tool, lyft_tool],
        verbose=True
    )
    
    # Test comparative query
    response = sub_engine.query("Compare Uber and Lyft revenue")
    # Should generate sub-questions for each company
    
    assert response is not None

def test_async_batch_processing():
    """Test async query execution."""
    import asyncio
    from llama_index.core import VectorStoreIndex
    from llama_index.core.response_synthesizers import get_response_synthesizer
    
    docs = [Document(text="Sample document")]
    index = VectorStoreIndex.from_documents(docs)
    
    synthesizer = get_response_synthesizer(use_async=True)
    engine = index.as_query_engine(response_synthesizer=synthesizer)
    
    async def process_queries():
        queries = ["Query 1", "Query 2", "Query 3"]
        tasks = [engine.aquery(q) for q in queries]
        return await asyncio.gather(*tasks)
    
    responses = asyncio.run(process_queries())
    
    assert len(responses) == 3
    assert all(r is not None for r in responses)
```

### Performance Tests

Test latency and throughput under load.

```python
import time
import pytest

def test_query_latency(sample_index):
    """Test that query latency is within acceptable range."""
    engine = sample_index.as_query_engine(
        response_mode="compact"
    )
    
    # Warm up
    engine.query("test query")
    
    # Measure latency
    start = time.time()
    response = engine.query("What is the main topic?")
    latency = time.time() - start
    
    # Should be under 5s for simple query with compact mode
    assert latency < 5.0

def test_batch_throughput():
    """Test batch query throughput."""
    import asyncio
    from llama_index.core.response_synthesizers import get_response_synthesizer
    
    docs = [Document(text=f"Document {i}") for i in range(10)]
    index = VectorStoreIndex.from_documents(docs)
    
    synthesizer = get_response_synthesizer(use_async=True)
    engine = index.as_query_engine(response_synthesizer=synthesizer)
    
    queries = [f"Query {i}" for i in range(100)]
    
    async def process():
        tasks = [engine.aquery(q) for q in queries]
        return await asyncio.gather(*tasks)
    
    start = time.time()
    responses = asyncio.run(process())
    elapsed = time.time() - start
    
    throughput = len(responses) / elapsed
    
    # Should achieve >5 queries/sec with async
    assert throughput > 5.0

@pytest.mark.parametrize("response_mode", [
    "compact",
    "refine",
    "tree_summarize",
    "simple_summarize"
])
def test_response_mode_latency(sample_index, response_mode):
    """Compare latency across response modes."""
    engine = sample_index.as_query_engine(
        response_mode=response_mode,
        similarity_top_k=5
    )
    
    start = time.time()
    response = engine.query("What is the main topic?")
    latency = time.time() - start
    
    print(f"{response_mode} latency: {latency:.2f}s")
    
    # Log results for comparison
    assert response is not None
```

### Evaluation Metrics

Test retrieval and synthesis quality.

```python
def test_retrieval_recall():
    """Test retrieval recall on ground truth queries."""
    # Create index with known documents
    docs = [
        Document(text="Python programming", doc_id="python_1"),
        Document(text="JavaScript development", doc_id="js_1"),
        Document(text="Machine learning basics", doc_id="ml_1"),
    ]
    index = VectorStoreIndex.from_documents(docs)
    engine = index.as_query_engine(similarity_top_k=2)
    
    # Ground truth: query -> relevant doc_ids
    test_cases = {
        "python programming language": {"python_1"},
        "javascript for web dev": {"js_1"},
    }
    
    recalls = []
    for query, relevant_ids in test_cases.items():
        response = engine.query(query)
        retrieved_ids = {node.node.id_ for node in response.source_nodes}
        
        recall = len(retrieved_ids & relevant_ids) / len(relevant_ids)
        recalls.append(recall)
    
    avg_recall = sum(recalls) / len(recalls)
    
    # Target: >80% recall
    assert avg_recall > 0.8

def test_answer_correctness():
    """Test that answers are factually correct."""
    docs = [
        Document(text="The capital of France is Paris"),
        Document(text="Python was created by Guido van Rossum"),
    ]
    index = VectorStoreIndex.from_documents(docs)
    engine = index.as_query_engine()
    
    response = engine.query("What is the capital of France?")
    
    # Check if answer contains expected fact
    assert "Paris" in str(response)
```

---

## Comparison to Alternatives

### LlamaIndex Query Engines vs. LangChain Chains

**LlamaIndex Query Engines:**
- Focused on RAG and document retrieval
- Built-in response synthesis modes
- Native support for multiple index types
- Strong router and sub-question engines
- Simpler API for RAG use cases

**LangChain Chains:**
- More general-purpose (not just RAG)
- Flexible chain composition (LCEL)
- Larger ecosystem of integrations
- Better for complex multi-step workflows beyond RAG
- Steeper learning curve

**When to use LlamaIndex:**
- Primary use case is RAG/document Q&A
- Need router or sub-question patterns
- Team wants simpler API
- Focus on response synthesis quality

**When to use LangChain:**
- Building complex workflows beyond RAG
- Need LCEL for dynamic chain composition
- Want broader tool ecosystem
- Already invested in LangChain

**Our approach:** Use LlamaIndex for RAG, LangChain for agent orchestration. They complement each other.

---

### Query Engines vs. Chat Engines

**Query Engine:**
- Stateless (each query independent)
- No conversation history
- Faster (no context management overhead)
- Simpler implementation
- **Use case:** One-shot Q&A, batch processing, when queries are independent

**Chat Engine:**
- Stateful (maintains conversation history)
- Handles follow-up questions
- Context window management
- Supports multi-turn conversations
- **Use case:** Conversational UI, when queries reference previous questions

**Implementation difference:**
```python
# Query Engine (stateless)
query_engine = index.as_query_engine()
response1 = query_engine.query("What is X?")
response2 = query_engine.query("What is Y?")  # Independent

# Chat Engine (stateful)
chat_engine = index.as_chat_engine()
response1 = chat_engine.chat("What is X?")
response2 = chat_engine.chat("And what about Y?")  # Uses context from response1
```

**Production consideration:** Query engines are simpler and faster for independent queries. Only use chat engines when conversation context is genuinely needed.

---

### Query Engines vs. Agents

**Query Engine:**
- Deterministic (always retrieves, then synthesizes)
- Fast and predictable
- No decision-making about tool use
- Lower cost (no agentic overhead)
- **Use case:** When you know what to do (retrieve + synthesize)

**Agent:**
- Agentic (LLM decides which tools to call, when)
- Flexible but unpredictable
- Multi-step reasoning and planning
- Higher cost (multiple LLM calls for decisions)
- **Use case:** When task requires planning, tool selection, iterative refinement

**The spectrum:**
```
Query Engine â†’ Router Engine â†’ Sub-Question Engine â†’ Agent
(deterministic)                                    (agentic)
```

**When query engines suffice:**
- Task is always "retrieve + synthesize"
- No complex decision-making needed
- Latency and cost matter
- Predictability is important

**When agents are needed:**
- LLM must decide which tools to use
- Multi-step planning required
- Task varies significantly per query
- Flexibility > predictability

**Our approach:** Start with query engines (simplest). Graduate to agents only when autonomy is demonstrably needed.

---

### Different Response Synthesis Modes Compared

**Compact vs. Refine:**
- **Compact:** Concatenates chunks, fewer LLM calls, faster, good quality
- **Refine:** Iterative refinement, more LLM calls, slower, highest quality
- **Choose compact unless:** Proven quality issues require refine

**Compact vs. Tree Summarize:**
- **Compact:** General-purpose, works for Q&A and summarization
- **Tree Summarize:** Specialized for summarization, hierarchical synthesis
- **Choose tree_summarize when:** Task is explicitly summarization

**Simple Summarize vs. Compact:**
- **Simple Summarize:** Single LLM call, truncates to fit context, fast but loses detail
- **Compact:** Multiple calls if needed, preserves more context, higher quality
- **Choose simple_summarize when:** Quick summary acceptable, detail not critical

**Accumulate vs. Compact:**
- **Accumulate:** Separate answer per node, returns array, no synthesis
- **Compact:** Synthesizes unified answer from all nodes
- **Choose accumulate when:** Want multiple perspectives, not single answer

**Production default:** Start with compact. It's the right balance for 80% of use cases. Measure quality. Only switch if proven necessary.

---

### Router Strategies Compared

**LLMSingleSelector:**
- Routes to ONE index
- Faster (single decision)
- Lower cost (one query to one index)
- **Use when:** Query fits one data source

**LLMMultiSelector:**
- Routes to MULTIPLE indexes
- Slower (queries multiple, then aggregates)
- Higher cost (multiple queries + synthesis)
- **Use when:** Query genuinely needs cross-index synthesis

**PydanticSingleSelector:**
- Uses OpenAI function calling for selection
- More reliable than prompt-based selection
- Only works with models supporting function calling
- **Use when:** Using OpenAI models, want more reliable routing

**Production pattern:** Start with LLMSingleSelector. Only use multi-selector when cross-index synthesis is demonstrably valuable.

---

### Query Engines vs. Raw RAG

**Query Engine (LlamaIndex):**
- Abstraction over retrieval + synthesis
- Built-in patterns (routing, sub-questions, synthesis modes)
- Easier to use for common patterns
- Less control over internals
- **Overhead:** Framework abstractions

**Raw RAG (Custom Implementation):**
- Direct control over every step
- No framework overhead
- Optimizable to exact requirements
- More code to maintain
- **Complexity:** Build everything yourself

**When LlamaIndex query engines help:**
- Need routing between multiple indexes
- Complex queries benefit from sub-question decomposition
- Team lacks time to build orchestration
- Proven synthesis patterns suffice

**When custom RAG is better:**
- Simple single-index RAG
- Need fine-grained control for optimization
- Framework debugging costs more than building
- Have specific orchestration requirements LlamaIndex doesn't support

**Our approach:** Use LlamaIndex for routing and complex synthesis. Use custom RAG for simple cases. Choose based on complexity, not ideology.

---

### When Framework Is Overkill vs. When It Helps

**Framework is overkill when:**
- Simple single-index, single-query-type RAG
- Embedding + retrieve top-k + pass to LLM is sufficient
- Need every optimization for latency/cost
- Debugging framework is harder than building custom

**Framework helps when:**
- Multiple indexes need intelligent routing
- Queries vary in complexity (simple vs. comparative vs. analytical)
- Team doesn't want to build orchestration
- Benefits from proven synthesis patterns

**Cost of abstraction:**
- Framework overhead (latency)
- Less control over internals
- Debugging framework code
- Lock-in to framework patterns

**Benefit of abstraction:**
- Faster development
- Proven patterns
- Less code to maintain
- Community support

**Production decision matrix:**

| Use Case | Complexity | Team Size | Recommendation |
|----------|------------|-----------|----------------|
| Simple RAG | Low | Small | Custom |
| Multi-index routing | Medium | Small | LlamaIndex |
| Complex synthesis | High | Small | LlamaIndex |
| Simple RAG | Low | Large | Custom (more control) |
| Multi-index routing | Medium | Large | LlamaIndex (faster dev) |
| Complex synthesis | High | Large | LlamaIndex (proven patterns) |

**The pragmatic answer:** Use both. LlamaIndex for routing and complex synthesis. Custom for simple RAG. Evaluate per use case.

---

## Summary

**LlamaIndex query engines provide orchestration patterns over RAG:**

1. **Query engines = retrieval + synthesis orchestration** (not RAG replacement)
2. **Router engines** solve multi-index routing with LLM-based selector
3. **Sub-question engines** decompose complex queries into executable sub-questions
4. **Response synthesis modes** offer latency/quality/cost tradeoffs (compact, refine, tree_summarize, etc.)
5. **Query transformations** (HyDE, multi-step) improve retrieval but add latency
6. **Metadata filtering** is essential for production (date, category, access, source)
7. **Async execution** critical for throughput (batch processing, sub-questions)
8. **Reranking** trades latency for accuracy (use for high-stakes applications)
9. **Cost tracking** reveals optimization opportunities (monitor token usage)
10. **Query engines vs. chat engines vs. agents:** Know when each pattern fits
11. **Production requires:** error handling, fallbacks, monitoring, quality metrics
12. **LlamaIndex helps when:** routing, complex synthesis, proven patterns. Custom is better for simple RAG.

**The Bridge:** Query engines are the middle ground between raw RAG (too low-level) and agents (too autonomous). Use when you need orchestration (routing, decomposition, synthesis) without full agent autonomy.

**For Our Project:** Use LlamaIndex query engines for:
- **Prompt Routing:** Router engine for intent-based routing to handlers
- **Query Writing:** Sub-question engine for schema + examples â†’ SQL generation
- **Data Processing:** Vector retrieval for similar transformation examples
- **Tool Orchestration:** Router engine for tool selection and composition
- **Decision Support:** Sub-question engine for multi-perspective analysis

Use custom RAG for simple single-source retrieval. Choose based on complexity, not ideology.

**The Bottom Line:** Query engines aren't always the answerâ€”they're one tool in the toolkit. Use them when orchestration patterns (routing, decomposition, synthesis) add value. Skip them when simple RAG suffices. Measure, optimize, choose pragmatically.