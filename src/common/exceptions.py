"""
Exception hierarchy for agentic AI system.

This module defines all custom exceptions used throughout the system,
providing a clear hierarchy and explicit error handling patterns.

Design Principles:
- All custom exceptions inherit from AgenticAIError
- Each exception includes a `recoverable` flag for error handling
- Context information is preserved in all exceptions
- Capability-specific errors are clearly separated
"""

from typing import Optional, Dict, Any


class AgenticAIError(Exception):
    """
    Base exception for all agentic AI errors.

    All custom exceptions in the system should inherit from this base class
    to enable consistent error handling and logging.

    Attributes:
        message: Human-readable error message
        recoverable: Whether the error can be recovered from with retry/fallback
        context: Additional context information about the error
    """

    def __init__(
        self,
        message: str,
        recoverable: bool = False,
        context: Optional[Dict[str, Any]] = None
    ):
        self.message = message
        self.recoverable = recoverable
        self.context = context or {}
        super().__init__(message)

    def __str__(self) -> str:
        context_str = f" | Context: {self.context}" if self.context else ""
        recoverable_str = " [RECOVERABLE]" if self.recoverable else ""
        return f"{self.message}{recoverable_str}{context_str}"


# ============================================================================
# Capability-Specific Errors
# ============================================================================

class RoutingError(AgenticAIError):
    """Raised when prompt routing fails."""
    pass


class ClassificationError(RoutingError):
    """Raised when intent classification fails."""
    pass


class HandlerNotFoundError(RoutingError):
    """Raised when no handler is registered for an intent."""
    pass


class QueryGenerationError(AgenticAIError):
    """Raised when query generation (SQL/API) fails."""
    pass


class SQLGenerationError(QueryGenerationError):
    """Raised when SQL query generation fails."""
    pass


class APIQueryBuilderError(QueryGenerationError):
    """Raised when API query construction fails."""
    pass


class SchemaError(QueryGenerationError):
    """Raised when schema loading or validation fails."""
    pass


class DataProcessingError(AgenticAIError):
    """Raised when data processing operations fail."""
    pass


class TransformationError(DataProcessingError):
    """Raised when data transformation fails."""
    pass


class ValidationError(DataProcessingError):
    """Raised when data validation fails."""
    pass


class EnrichmentError(DataProcessingError):
    """Raised when data enrichment fails."""
    pass


class ToolExecutionError(AgenticAIError):
    """Raised when tool execution fails."""
    pass


class ToolNotFoundError(ToolExecutionError):
    """Raised when a requested tool is not registered."""
    pass


class ToolTimeoutError(ToolExecutionError):
    """Raised when tool execution exceeds timeout."""

    def __init__(self, message: str, timeout_seconds: float, **kwargs):
        super().__init__(message, recoverable=True, **kwargs)
        self.timeout_seconds = timeout_seconds


class OrchestrationError(AgenticAIError):
    """Raised when tool orchestration fails."""
    pass


class DecisionAnalysisError(AgenticAIError):
    """Raised when decision analysis fails."""
    pass


class RecommendationError(DecisionAnalysisError):
    """Raised when generating recommendations fails."""
    pass


# ============================================================================
# Infrastructure Errors
# ============================================================================

class LLMProviderError(AgenticAIError):
    """Base class for LLM provider errors."""
    pass


class RateLimitError(LLMProviderError):
    """
    Raised when API rate limit is exceeded.

    This error is always recoverable with exponential backoff retry.
    """

    def __init__(
        self,
        message: str = "Rate limit exceeded",
        retry_after: Optional[int] = None,
        **kwargs
    ):
        super().__init__(message, recoverable=True, **kwargs)
        self.retry_after = retry_after


class AuthenticationError(LLMProviderError):
    """
    Raised when API authentication fails.

    This error is not recoverable without user intervention.
    """

    def __init__(self, message: str = "Authentication failed", **kwargs):
        super().__init__(message, recoverable=False, **kwargs)


class InvalidResponseError(LLMProviderError):
    """
    Raised when LLM returns an invalid or malformed response.

    This may be recoverable with a retry or alternative prompt.
    """

    def __init__(self, message: str, response: Optional[str] = None, **kwargs):
        super().__init__(message, recoverable=True, **kwargs)
        self.response = response


class ContextLengthError(LLMProviderError):
    """
    Raised when input exceeds model's context length.

    This is recoverable by truncating or summarizing the input.
    """

    def __init__(
        self,
        message: str,
        max_tokens: int,
        actual_tokens: int,
        **kwargs
    ):
        super().__init__(message, recoverable=True, **kwargs)
        self.max_tokens = max_tokens
        self.actual_tokens = actual_tokens


class ConfigurationError(AgenticAIError):
    """
    Raised when system configuration is invalid or missing.

    This error is not recoverable without fixing the configuration.
    """

    def __init__(self, message: str, **kwargs):
        super().__init__(message, recoverable=False, **kwargs)


class DependencyError(AgenticAIError):
    """
    Raised when a required dependency is missing or unavailable.

    This error is not recoverable without installing the dependency.
    """

    def __init__(self, message: str, dependency: str, **kwargs):
        super().__init__(message, recoverable=False, **kwargs)
        self.dependency = dependency


# ============================================================================
# Security Errors
# ============================================================================

class SecurityError(AgenticAIError):
    """Base class for security-related errors."""

    def __init__(self, message: str, **kwargs):
        super().__init__(message, recoverable=False, **kwargs)


class PromptInjectionError(SecurityError):
    """Raised when potential prompt injection is detected."""
    pass


class SQLInjectionError(SecurityError):
    """Raised when potential SQL injection is detected."""
    pass


class UnauthorizedOperationError(SecurityError):
    """Raised when an unauthorized operation is attempted."""
    pass


# ============================================================================
# Utility Functions
# ============================================================================

def is_recoverable(error: Exception) -> bool:
    """
    Check if an error is recoverable.

    Args:
        error: Exception to check

    Returns:
        True if error can be recovered from with retry/fallback

    Example:
        >>> try:
        ...     raise_some_error()
        ... except Exception as e:
        ...     if is_recoverable(e):
        ...         # Attempt retry
        ...         pass
    """
    if isinstance(error, AgenticAIError):
        return error.recoverable
    # Unknown errors are assumed non-recoverable
    return False


def get_error_context(error: Exception) -> Dict[str, Any]:
    """
    Extract context information from an error.

    Args:
        error: Exception to extract context from

    Returns:
        Dictionary containing error context

    Example:
        >>> try:
        ...     raise_some_error()
        ... except Exception as e:
        ...     context = get_error_context(e)
        ...     logger.error("Error occurred", extra=context)
    """
    if isinstance(error, AgenticAIError):
        return {
            "error_type": type(error).__name__,
            "message": error.message,
            "recoverable": error.recoverable,
            **error.context
        }
    return {
        "error_type": type(error).__name__,
        "message": str(error),
        "recoverable": False
    }
