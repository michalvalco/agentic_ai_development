"""
Structured logging for the agentic AI system.

This module configures structured logging using structlog, providing
consistent, searchable, and analyzable logs across all components.

Design Principles:
- Structured logging with key-value pairs
- Context propagation (request_id, trace_id, etc.)
- Different log levels for different environments
- JSON output for production, human-readable for development
- Performance-focused (no blocking I/O)

Usage:
    from src.common.logger import get_logger

    logger = get_logger(__name__)
    logger.info("query_generated",
                query_type="sql",
                tokens=150,
                cost_usd=0.002)
"""

import logging
import sys
from typing import Any, Dict, Optional

import structlog
from structlog.types import EventDict, WrappedLogger

from .config import settings


# ============================================================================
# Custom Processors
# ============================================================================

def add_app_context(
    logger: WrappedLogger,
    method_name: str,
    event_dict: EventDict
) -> EventDict:
    """
    Add application-level context to all log entries.

    Args:
        logger: The logger instance
        method_name: The logging method name (info, error, etc.)
        event_dict: The event dictionary

    Returns:
        Modified event dictionary
    """
    event_dict["app"] = "agentic_ai"
    event_dict["environment"] = settings.environment
    return event_dict


def add_request_context(
    logger: WrappedLogger,
    method_name: str,
    event_dict: EventDict
) -> EventDict:
    """
    Add request context if available.

    This processor looks for request_id and trace_id in the event dict
    and promotes them to top-level fields for easier searching.

    Args:
        logger: The logger instance
        method_name: The logging method name
        event_dict: The event dictionary

    Returns:
        Modified event dictionary
    """
    # Check if context is provided
    context = event_dict.pop("context", None)
    if context:
        if hasattr(context, "request_id"):
            event_dict["request_id"] = context.request_id
        if hasattr(context, "trace_id"):
            event_dict["trace_id"] = context.trace_id
        if hasattr(context, "user_id"):
            event_dict["user_id"] = context.user_id

    return event_dict


def censor_sensitive_data(
    logger: WrappedLogger,
    method_name: str,
    event_dict: EventDict
) -> EventDict:
    """
    Censor sensitive data from logs.

    Removes or masks fields that might contain sensitive information.

    Args:
        logger: The logger instance
        method_name: The logging method name
        event_dict: The event dictionary

    Returns:
        Modified event dictionary with sensitive data censored
    """
    sensitive_keys = [
        "api_key",
        "password",
        "secret",
        "token",
        "authorization",
        "anthropic_api_key",
        "openai_api_key",
    ]

    for key in sensitive_keys:
        if key in event_dict:
            event_dict[key] = "***REDACTED***"

    return event_dict


# ============================================================================
# Logger Configuration
# ============================================================================

def setup_logging(
    log_level: Optional[str] = None,
    json_output: Optional[bool] = None
) -> None:
    """
    Configure structured logging for the application.

    This should be called once at application startup.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        json_output: Whether to output JSON format (default: True in prod)

    Example:
        >>> setup_logging("INFO", json_output=False)
    """
    # Use settings if not explicitly provided
    if log_level is None:
        log_level = settings.log_level

    if json_output is None:
        json_output = settings.is_production()

    # Convert log level string to logging constant
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)

    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=numeric_level,
    )

    # Build processor chain
    processors = [
        # Add timestamp
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        # Add app context
        add_app_context,
        # Add request context
        add_request_context,
        # Censor sensitive data
        censor_sensitive_data,
        # Add stack info if exception
        structlog.processors.StackInfoRenderer(),
        # Format exceptions
        structlog.processors.format_exc_info,
        # Add timestamp
        structlog.processors.TimeStamper(fmt="iso"),
    ]

    # Choose output format based on environment
    if json_output:
        # JSON output for production (machine-readable)
        processors.append(structlog.processors.JSONRenderer())
    else:
        # Console output for development (human-readable)
        processors.append(structlog.dev.ConsoleRenderer())

    # Configure structlog
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """
    Get a structured logger instance.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Configured structlog logger

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("operation_complete",
        ...             operation="query_generation",
        ...             duration_ms=245)
    """
    return structlog.get_logger(name)


# ============================================================================
# Context Manager for Request Logging
# ============================================================================

class LogContext:
    """
    Context manager for adding context to all logs within a block.

    Example:
        >>> with LogContext(request_id="123", user_id="user_456"):
        ...     logger.info("processing_request")
        ...     # All logs within this block will include request_id and user_id
    """

    def __init__(self, **context_data):
        """
        Initialize log context.

        Args:
            **context_data: Key-value pairs to add to log context
        """
        self.context_data = context_data
        self.token = None

    def __enter__(self):
        """Enter context and bind context data."""
        self.token = structlog.contextvars.bind_contextvars(**self.context_data)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context and unbind context data."""
        structlog.contextvars.unbind_contextvars(*self.context_data.keys())


# ============================================================================
# Performance Logging Utilities
# ============================================================================

class LogPerformance:
    """
    Context manager to log operation performance.

    Automatically logs duration, and optionally cost and tokens.

    Example:
        >>> with LogPerformance("query_generation"):
        ...     result = generate_query(prompt)
        ...
        # Logs: query_generation_complete, duration_ms=245
    """

    def __init__(
        self,
        operation: str,
        logger: Optional[structlog.stdlib.BoundLogger] = None,
        **extra_context
    ):
        """
        Initialize performance logger.

        Args:
            operation: Operation name
            logger: Logger instance (creates one if not provided)
            **extra_context: Additional context to log
        """
        self.operation = operation
        self.logger = logger or get_logger("performance")
        self.extra_context = extra_context
        self.start_time = None

    def __enter__(self):
        """Start timing."""
        import time
        self.start_time = time.time()
        self.logger.debug(f"{self.operation}_started", **self.extra_context)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Log completion with duration."""
        import time
        duration_ms = (time.time() - self.start_time) * 1000

        if exc_type is not None:
            self.logger.error(
                f"{self.operation}_failed",
                duration_ms=duration_ms,
                error=str(exc_val),
                error_type=exc_type.__name__,
                **self.extra_context
            )
        else:
            self.logger.info(
                f"{self.operation}_complete",
                duration_ms=duration_ms,
                **self.extra_context
            )


# ============================================================================
# Initialize Logging
# ============================================================================

# Auto-configure logging on module import
setup_logging()

# Export commonly used logger
logger = get_logger("agentic_ai")
