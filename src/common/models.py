"""
Base Pydantic models for the agentic AI system.

This module defines common data models used across all capabilities,
following Pydantic v2 best practices for validation and serialization.

Design Principles:
- All models inherit from BaseAgenticModel for consistency
- Type hints are comprehensive and accurate
- Validation is explicit with clear error messages
- Models are immutable by default (frozen=True where appropriate)
- JSON serialization is properly configured

Reference: docs/PKB/pydantic_validation.md
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator, ConfigDict


# ============================================================================
# Base Models
# ============================================================================

class BaseAgenticModel(BaseModel):
    """
    Base model with common fields for all agentic models.

    All models in the system should inherit from this to ensure
    consistent timestamping, metadata tracking, and serialization.

    Attributes:
        id: Unique identifier for the model instance
        timestamp: ISO format timestamp of creation
        metadata: Additional flexible metadata
    """

    model_config = ConfigDict(
        # Allow arbitrary types for complex objects
        arbitrary_types_allowed=True,
        # Validate on assignment
        validate_assignment=True,
        # Use enum values in JSON
        use_enum_values=False,
        # Populate by field name
        populate_by_name=True,
        # JSON schema extra
        json_schema_extra={
            "examples": []
        }
    )

    id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique identifier for this instance"
    )
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Creation timestamp"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata for extensibility"
    )


# ============================================================================
# Context Management
# ============================================================================

class Context(BaseModel):
    """
    Execution context propagated through all operations.

    The Context carries information needed for tracing, logging,
    cost tracking, and user-specific data throughout the system.

    This is the primary mechanism for observability and debugging.

    Attributes:
        request_id: Unique ID for this request/operation
        user_id: Optional user identifier
        session_id: Optional session identifier
        trace_id: Distributed tracing ID
        parent_span_id: Parent span for nested operations
        metadata: Additional context-specific data
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    request_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique request identifier"
    )
    user_id: Optional[str] = Field(
        default=None,
        description="User identifier for multi-tenant scenarios"
    )
    session_id: Optional[str] = Field(
        default=None,
        description="Session identifier for stateful interactions"
    )
    trace_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Distributed tracing identifier"
    )
    parent_span_id: Optional[str] = Field(
        default=None,
        description="Parent span ID for nested operations"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional context metadata"
    )

    def create_child_context(self, operation: str) -> "Context":
        """
        Create a child context for nested operations.

        Args:
            operation: Name of the child operation

        Returns:
            New Context with current request_id as parent_span_id
        """
        return Context(
            request_id=str(uuid4()),
            user_id=self.user_id,
            session_id=self.session_id,
            trace_id=self.trace_id,
            parent_span_id=self.request_id,
            metadata={**self.metadata, "parent_operation": operation}
        )


# ============================================================================
# Token Tracking
# ============================================================================

class TokenCount(BaseModel):
    """
    Token usage for an LLM call.

    Attributes:
        input_tokens: Number of tokens in the input
        output_tokens: Number of tokens in the output
        total_tokens: Total tokens used
    """

    input_tokens: int = Field(ge=0, description="Input token count")
    output_tokens: int = Field(ge=0, description="Output token count")
    total_tokens: int = Field(ge=0, description="Total token count")

    @field_validator('total_tokens')
    @classmethod
    def validate_total(cls, v: int, info) -> int:
        """Ensure total equals sum of input and output."""
        if 'input_tokens' in info.data and 'output_tokens' in info.data:
            expected = info.data['input_tokens'] + info.data['output_tokens']
            if v != expected:
                # Auto-correct if provided total doesn't match
                return expected
        return v


class LLMCall(BaseModel):
    """
    Record of a single LLM API call.

    Attributes:
        model: Model identifier (e.g., "claude-sonnet-4")
        tokens: Token usage for this call
        cost_usd: Cost in USD
        latency_ms: Latency in milliseconds
        timestamp: When the call was made
        operation: What operation triggered this call
    """

    model: str = Field(description="LLM model identifier")
    tokens: TokenCount = Field(description="Token usage")
    cost_usd: float = Field(ge=0, description="Cost in USD")
    latency_ms: float = Field(ge=0, description="Latency in milliseconds")
    timestamp: datetime = Field(default_factory=datetime.now)
    operation: str = Field(description="Operation that triggered this call")


# ============================================================================
# Response Metadata
# ============================================================================

class ResponseMetadata(BaseModel):
    """
    Metadata included with all capability responses.

    This provides observability into the performance and cost of operations.

    Attributes:
        latency_ms: Total operation latency
        cost_usd: Total cost in USD
        tokens_used: Token usage (if applicable)
        model_used: Model identifier (if applicable)
        confidence: Confidence score (if applicable)
        trace_id: Trace identifier for debugging
        timestamp: Response timestamp
    """

    latency_ms: float = Field(ge=0, description="Operation latency")
    cost_usd: float = Field(ge=0, description="Operation cost")
    tokens_used: Optional[TokenCount] = Field(
        default=None,
        description="Token usage if LLM was used"
    )
    model_used: Optional[str] = Field(
        default=None,
        description="Model identifier if LLM was used"
    )
    confidence: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Confidence score if applicable"
    )
    trace_id: str = Field(description="Trace ID for debugging")
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Response timestamp"
    )


# ============================================================================
# Health Check
# ============================================================================

class HealthStatus(str, Enum):
    """Health status for capabilities and dependencies."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


class HealthCheck(BaseModel):
    """
    Health check response for a capability.

    Attributes:
        status: Overall health status
        checks: Individual check results
        timestamp: When the health check was performed
    """

    status: HealthStatus
    checks: Dict[str, bool] = Field(
        default_factory=dict,
        description="Individual health checks"
    )
    timestamp: datetime = Field(default_factory=datetime.now)
    message: Optional[str] = Field(
        default=None,
        description="Additional health check message"
    )


# ============================================================================
# Utility Functions
# ============================================================================
# (Removed create_success and create_failure - Result pattern no longer used)
