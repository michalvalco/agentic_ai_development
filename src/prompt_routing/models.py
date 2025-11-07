"""
Data models for prompt routing capability.

This module defines Pydantic models for intent classification,
routing decisions, and handler responses.

Reference:
- docs/ARCHITECTURE.md - Prompt Routing section
- docs/PKB/pydantic_validation.md - Validation patterns
- docs/PKB/anthropic_prompt_engineering.md - Classification patterns
"""

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

from src.common.models import BaseAgenticModel


# ============================================================================
# Intent Types
# ============================================================================

class IntentType(str, Enum):
    """
    Predefined intent categories for routing.

    Each intent type maps to a specific handler that knows
    how to process that type of request.
    """

    # Data retrieval intents
    DATABASE_QUERY = "database_query"
    API_QUERY = "api_query"
    KNOWLEDGE_SEARCH = "knowledge_search"

    # Action intents
    TOOL_EXECUTION = "tool_execution"
    WORKFLOW_ORCHESTRATION = "workflow_orchestration"

    # Analysis intents
    DATA_ANALYSIS = "data_analysis"
    DECISION_SUPPORT = "decision_support"
    COMPARISON = "comparison"

    # Communication intents
    DIRECT_RESPONSE = "direct_response"
    CLARIFICATION_NEEDED = "clarification_needed"

    # Unknown/fallback
    UNKNOWN = "unknown"


# ============================================================================
# Intent Classification Models
# ============================================================================

class IntentClassification(BaseAgenticModel):
    """
    Result of intent classification for a user query.

    This is the structured output from the IntentClassifier,
    providing not just the intent but also confidence and reasoning.

    Attributes:
        intent: Classified intent type
        confidence: Confidence score (0.0 to 1.0)
        parameters: Extracted parameters from the query
        reasoning: LLM's reasoning for this classification
        requires_clarification: Whether the query is ambiguous
        suggested_clarifications: Questions to ask user if clarification needed
        alternative_intents: Other possible intents with lower confidence
    """

    intent: IntentType = Field(
        description="Primary classified intent"
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence score for this classification"
    )
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Extracted parameters from the query"
    )
    reasoning: str = Field(
        description="Explanation of why this intent was chosen"
    )
    requires_clarification: bool = Field(
        default=False,
        description="Whether the query is too ambiguous"
    )
    suggested_clarifications: List[str] = Field(
        default_factory=list,
        description="Questions to ask user for clarification"
    )
    alternative_intents: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Other possible intents with their confidence scores"
    )

    @field_validator('confidence')
    @classmethod
    def validate_confidence(cls, v: float) -> float:
        """Ensure confidence is between 0 and 1."""
        if not 0.0 <= v <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")
        return v

    def should_proceed(self, threshold: float = 0.7) -> bool:
        """
        Check if confidence exceeds threshold for proceeding.

        Args:
            threshold: Minimum confidence required (default 0.7)

        Returns:
            True if confident enough to proceed
        """
        return self.confidence >= threshold and not self.requires_clarification

    def get_top_alternatives(self, n: int = 3) -> List[Dict[str, Any]]:
        """
        Get top N alternative intents.

        Args:
            n: Number of alternatives to return

        Returns:
            List of alternative intents sorted by confidence
        """
        return sorted(
            self.alternative_intents,
            key=lambda x: x.get('confidence', 0.0),
            reverse=True
        )[:n]


# ============================================================================
# Routing Models
# ============================================================================

class RoutingDecision(BaseAgenticModel):
    """
    Decision about how to route a classified query.

    After classification, this model describes which handler
    should process the query and with what configuration.

    Attributes:
        intent_classification: The classification result
        handler_name: Name of the handler to use
        handler_config: Configuration for the handler
        should_route: Whether to proceed with routing
        fallback_handler: Alternative handler if primary fails
    """

    intent_classification: IntentClassification
    handler_name: str = Field(
        description="Name of the registered handler"
    )
    handler_config: Dict[str, Any] = Field(
        default_factory=dict,
        description="Configuration to pass to the handler"
    )
    should_route: bool = Field(
        default=True,
        description="Whether to proceed with routing"
    )
    fallback_handler: Optional[str] = Field(
        default=None,
        description="Fallback handler if primary fails"
    )


# ============================================================================
# Handler Response Models
# ============================================================================

class HandlerResponse(BaseAgenticModel):
    """
    Response from a handler after processing a query.

    All handlers return this standardized format for consistency.

    Attributes:
        success: Whether handler succeeded
        result: Handler-specific result data
        error: Error message if failed
        next_action: Suggested next action (if applicable)
        requires_followup: Whether this needs additional steps
    """

    success: bool = Field(
        description="Whether the handler succeeded"
    )
    result: Any = Field(
        default=None,
        description="Handler result (structure varies by handler)"
    )
    error: Optional[str] = Field(
        default=None,
        description="Error message if handler failed"
    )
    next_action: Optional[str] = Field(
        default=None,
        description="Suggested next action"
    )
    requires_followup: bool = Field(
        default=False,
        description="Whether this requires additional processing"
    )


# ============================================================================
# Classification Prompt Template Data
# ============================================================================

class ClassificationExample(BaseModel):
    """
    Few-shot example for intent classification.

    Used in the classification prompt to guide the LLM.

    Attributes:
        query: Example user query
        intent: Correct intent for this query
        parameters: Parameters that should be extracted
        reasoning: Why this intent is correct
    """

    query: str
    intent: IntentType
    parameters: Dict[str, Any] = Field(default_factory=dict)
    reasoning: str


# Predefined examples for few-shot learning
CLASSIFICATION_EXAMPLES = [
    ClassificationExample(
        query="What were the total sales in Q3 2024?",
        intent=IntentType.DATABASE_QUERY,
        parameters={"timeframe": "Q3 2024", "metric": "sales"},
        reasoning="User wants data from a database about historical sales"
    ),
    ClassificationExample(
        query="Show me all customers in EMEA",
        intent=IntentType.DATABASE_QUERY,
        parameters={"entity": "customers", "filter": "region=EMEA"},
        reasoning="Direct request for customer data filtered by region"
    ),
    ClassificationExample(
        query="What's the weather in San Francisco?",
        intent=IntentType.API_QUERY,
        parameters={"api": "weather", "location": "San Francisco"},
        reasoning="Requires calling an external weather API"
    ),
    ClassificationExample(
        query="How do I reset my password?",
        intent=IntentType.KNOWLEDGE_SEARCH,
        parameters={"topic": "password_reset"},
        reasoning="User needs information from knowledge base/documentation"
    ),
    ClassificationExample(
        query="Send an email to the sales team",
        intent=IntentType.TOOL_EXECUTION,
        parameters={"tool": "email", "recipient": "sales_team"},
        reasoning="Action request requiring tool execution"
    ),
    ClassificationExample(
        query="Compare Enterprise vs Standard pricing",
        intent=IntentType.COMPARISON,
        parameters={"items": ["Enterprise", "Standard"], "aspect": "pricing"},
        reasoning="Explicit comparison request"
    ),
    ClassificationExample(
        query="Should we invest in cloud migration?",
        intent=IntentType.DECISION_SUPPORT,
        parameters={"decision": "cloud_migration"},
        reasoning="Complex decision requiring analysis and recommendations"
    ),
    ClassificationExample(
        query="Hello! How are you?",
        intent=IntentType.DIRECT_RESPONSE,
        parameters={},
        reasoning="Simple greeting requiring direct conversational response"
    ),
]
