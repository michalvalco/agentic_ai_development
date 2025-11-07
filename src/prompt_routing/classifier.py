"""
Intent classification for prompt routing.

This module implements the IntentClassifier that uses an LLM
to classify user queries into intent categories.

Design Principles:
- Structured output using Pydantic models
- Few-shot prompting for accuracy
- Confidence scoring for ambiguous queries
- Graceful fallback for classification failures

Reference:
- docs/PKB/anthropic_prompt_engineering.md - Few-shot prompting
- docs/PKB/pydantic_validation.md - Structured outputs
- docs/PKB/react_pattern.md - Reasoning patterns
"""

import json
import time
from typing import Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import ValidationError

from src.common.config import settings
from src.common.cost_tracker import cost_tracker
from src.common.exceptions import ClassificationError, InvalidResponseError
from src.common.logger import get_logger
from src.common.models import Context
from src.common.utils import retry_on_rate_limit, sanitize_prompt

from .models import (
    IntentClassification,
    IntentType,
    CLASSIFICATION_EXAMPLES,
)

logger = get_logger(__name__)


class IntentClassifier:
    """
    Classifies user intent using an LLM.

    The classifier uses few-shot prompting with examples to accurately
    determine the user's intent, confidence level, and extract parameters.

    Attributes:
        llm: Language model for classification
        confidence_threshold: Minimum confidence to proceed (default 0.7)
        max_prompt_length: Maximum prompt length to prevent context overflow
    """

    def __init__(
        self,
        llm: Optional[BaseChatModel] = None,
        confidence_threshold: float = 0.7,
        max_prompt_length: int = 2000
    ):
        """
        Initialize intent classifier.

        Args:
            llm: Language model (uses default from settings if not provided)
            confidence_threshold: Minimum confidence threshold
            max_prompt_length: Max prompt length in characters
        """
        self.llm = llm or self._create_default_llm()
        self.confidence_threshold = confidence_threshold
        self.max_prompt_length = max_prompt_length

        logger.info(
            "intent_classifier_initialized",
            model=getattr(self.llm, 'model_name', 'unknown'),
            threshold=confidence_threshold
        )

    def _create_default_llm(self) -> BaseChatModel:
        """Create default LLM from settings."""
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=settings.default_model,
            anthropic_api_key=settings.anthropic_api_key,
            max_tokens=1024,
            temperature=0.0  # Deterministic for classification
        )

    def _build_classification_prompt(
        self,
        query: str,
        context: Optional[Context] = None
    ) -> tuple[SystemMessage, HumanMessage]:
        """
        Build few-shot classification prompt.

        Args:
            query: User query to classify
            context: Optional context for classification

        Returns:
            Tuple of (system_message, human_message)
        """
        # Build few-shot examples
        examples_text = "\n\n".join([
            f"Query: {ex.query}\n"
            f"Intent: {ex.intent.value}\n"
            f"Parameters: {json.dumps(ex.parameters)}\n"
            f"Reasoning: {ex.reasoning}"
            for ex in CLASSIFICATION_EXAMPLES
        ])

        # Build available intents list
        intents_list = "\n".join([
            f"- {intent.value}: {self._get_intent_description(intent)}"
            for intent in IntentType
        ])

        system_prompt = f"""You are an expert intent classifier for an AI agent system.

Your job is to classify user queries into one of the following intent categories:

{intents_list}

For each query, you must:
1. Identify the primary intent
2. Assign a confidence score (0.0 to 1.0)
3. Extract relevant parameters from the query
4. Provide clear reasoning for your classification
5. Identify if the query is ambiguous and needs clarification

Here are examples of correct classifications:

{examples_text}

You MUST respond with valid JSON in this exact format:
{{
    "intent": "intent_name",
    "confidence": 0.95,
    "parameters": {{}},
    "reasoning": "explanation here",
    "requires_clarification": false,
    "suggested_clarifications": [],
    "alternative_intents": []
}}

Guidelines:
- Be conservative with confidence scores (0.7-0.8 for clear queries, lower for ambiguous)
- Set requires_clarification=true if the query is too vague
- Include alternative_intents if multiple interpretations are possible
- Extract specific parameters mentioned in the query
- Provide clear, actionable reasoning"""

        # Add context if provided
        context_info = ""
        if context and context.metadata:
            context_info = f"\n\nContext: {json.dumps(context.metadata)}"

        human_prompt = f"Query: {query}{context_info}"

        return (
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt)
        )

    def _get_intent_description(self, intent: IntentType) -> str:
        """Get human-readable description for an intent type."""
        descriptions = {
            IntentType.DATABASE_QUERY: "Query a database for specific data",
            IntentType.API_QUERY: "Call an external API to fetch information",
            IntentType.KNOWLEDGE_SEARCH: "Search documentation or knowledge base",
            IntentType.TOOL_EXECUTION: "Execute a specific tool or action",
            IntentType.WORKFLOW_ORCHESTRATION: "Multi-step workflow requiring tool chaining",
            IntentType.DATA_ANALYSIS: "Analyze data and provide insights",
            IntentType.DECISION_SUPPORT: "Complex decision requiring analysis",
            IntentType.COMPARISON: "Compare multiple options or items",
            IntentType.DIRECT_RESPONSE: "Simple query answerable directly",
            IntentType.CLARIFICATION_NEEDED: "Query is too ambiguous",
            IntentType.UNKNOWN: "Cannot determine intent",
        }
        return descriptions.get(intent, "Unknown intent type")

    @retry_on_rate_limit(max_attempts=3)
    async def classify(
        self,
        query: str,
        context: Optional[Context] = None
    ) -> IntentClassification:
        """
        Classify user intent from natural language query.

        Args:
            query: User's natural language input
            context: Optional context for classification

        Returns:
            IntentClassification with intent, confidence, and reasoning

        Raises:
            ClassificationError: If classification fails or returns invalid format
            RateLimitError: If API rate limit exceeded (will retry)

        Example:
            >>> classifier = IntentClassifier()
            >>> result = await classifier.classify("What were Q3 sales?")
            >>> print(result.intent)  # IntentType.DATABASE_QUERY
            >>> print(result.confidence)  # 0.95
        """
        start_time = time.time()

        try:
            # Sanitize and validate input
            query = sanitize_prompt(query, max_length=self.max_prompt_length)

            if not query.strip():
                raise ClassificationError(
                    "Empty query after sanitization",
                    context={"original_length": len(query)}
                )

            logger.info(
                "classifying_intent",
                query_length=len(query),
                context=context.request_id if context else None
            )

            # Build prompt
            system_msg, human_msg = self._build_classification_prompt(query, context)

            # Call LLM
            with cost_tracker.track("intent_classification"):
                response = await self.llm.ainvoke([system_msg, human_msg])

            # Extract content
            content = response.content.strip()

            # Handle markdown code blocks if present
            if content.startswith("```"):
                # Extract JSON from code block
                lines = content.split("\n")
                content = "\n".join([
                    line for line in lines
                    if not line.strip().startswith("```")
                ])

            # Parse JSON response
            try:
                response_data = json.loads(content)
            except json.JSONDecodeError as e:
                raise InvalidResponseError(
                    f"LLM returned invalid JSON: {e}",
                    response=content
                )

            # Validate with Pydantic
            try:
                classification = IntentClassification(**response_data)
            except ValidationError as e:
                raise ClassificationError(
                    f"Invalid classification format: {e}",
                    context={"response": response_data}
                )

            # Calculate latency
            latency_ms = (time.time() - start_time) * 1000

            # Track cost if available
            if hasattr(response, 'usage_metadata'):
                usage = response.usage_metadata
                cost_tracker.track_call(
                    model=getattr(self.llm, 'model_name', settings.default_model),
                    input_tokens=usage.get('input_tokens', 0),
                    output_tokens=usage.get('output_tokens', 0),
                    operation="intent_classification",
                    latency_ms=latency_ms
                )

            logger.info(
                "intent_classified",
                intent=classification.intent.value,
                confidence=classification.confidence,
                requires_clarification=classification.requires_clarification,
                latency_ms=latency_ms,
                context=context.request_id if context else None
            )

            return classification

        except (ClassificationError, InvalidResponseError):
            # Re-raise our custom errors
            raise

        except Exception as e:
            logger.error(
                "classification_failed",
                error=str(e),
                error_type=type(e).__name__,
                query=query[:100]
            )
            raise ClassificationError(
                f"Failed to classify intent: {e}",
                context={"query": query[:100], "error_type": type(e).__name__}
            )

    async def classify_batch(
        self,
        queries: list[str],
        context: Optional[Context] = None,
        concurrency: int = 5
    ) -> list[IntentClassification]:
        """
        Classify multiple queries in parallel.

        Args:
            queries: List of queries to classify
            context: Optional shared context
            concurrency: Maximum concurrent classifications

        Returns:
            List of IntentClassification results

        Example:
            >>> classifier = IntentClassifier()
            >>> results = await classifier.classify_batch([
            ...     "What were Q3 sales?",
            ...     "Show me EMEA customers"
            ... ])
        """
        from src.common.utils import gather_with_concurrency

        logger.info("classifying_batch", count=len(queries), concurrency=concurrency)

        classifications = await gather_with_concurrency(
            concurrency,
            *[self.classify(query, context) for query in queries],
            return_exceptions=True
        )

        # Convert exceptions to failed classifications
        results = []
        for i, result in enumerate(classifications):
            if isinstance(result, Exception):
                logger.error(
                    "batch_classification_failed",
                    query_index=i,
                    error=str(result)
                )
                # Create fallback classification
                results.append(IntentClassification(
                    intent=IntentType.UNKNOWN,
                    confidence=0.0,
                    reasoning=f"Classification failed: {result}",
                    requires_clarification=True,
                    parameters={}
                ))
            else:
                results.append(result)

        return results

    def get_confidence_threshold(self) -> float:
        """Get current confidence threshold."""
        return self.confidence_threshold

    def set_confidence_threshold(self, threshold: float) -> None:
        """
        Update confidence threshold.

        Args:
            threshold: New threshold (0.0 to 1.0)

        Raises:
            ValueError: If threshold is out of range
        """
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("Threshold must be between 0.0 and 1.0")

        self.confidence_threshold = threshold
        logger.info("confidence_threshold_updated", new_threshold=threshold)
