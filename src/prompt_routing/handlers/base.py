"""
Base handler interface for prompt routing.

This module defines the abstract Handler interface that all
concrete handlers must implement.

Design Principles:
- Consistent interface across all handlers
- Context propagation for observability
- Typed inputs and outputs
- Health check support

Reference:
- docs/ARCHITECTURE.md - Handler interface design
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from src.common.logger import get_logger
from src.common.models import Context, HealthCheck, HealthStatus
from src.prompt_routing.models import HandlerResponse, IntentClassification

logger = get_logger(__name__)


class Handler(ABC):
    """
    Abstract base class for all intent handlers.

    Handlers process classified queries and return standardized responses.
    Each handler is responsible for a specific intent type.

    Attributes:
        name: Unique handler identifier
        supported_intents: List of intents this handler can process
    """

    def __init__(self, name: str):
        """
        Initialize handler.

        Args:
            name: Unique handler identifier
        """
        self.name = name
        logger.info("handler_initialized", handler_name=name)

    @abstractmethod
    async def handle(
        self,
        query: str,
        classification: IntentClassification,
        context: Optional[Context] = None,
        **kwargs
    ) -> HandlerResponse:
        """
        Handle a classified query.

        This is the main entry point for query processing.

        Args:
            query: Original user query
            classification: Intent classification result
            context: Execution context
            **kwargs: Additional handler-specific parameters

        Returns:
            HandlerResponse with result or error

        Example:
            >>> handler = MyHandler()
            >>> response = await handler.handle(
            ...     query="What were Q3 sales?",
            ...     classification=classification,
            ...     context=context
            ... )
            >>> print(response.success)
        """
        pass

    @abstractmethod
    def can_handle(self, intent_type: str) -> bool:
        """
        Check if this handler can process the given intent.

        Args:
            intent_type: Intent type to check

        Returns:
            True if handler supports this intent
        """
        pass

    def health_check(self) -> HealthCheck:
        """
        Check handler health.

        Override this to add handler-specific health checks
        (e.g., database connectivity, API availability).

        Returns:
            HealthCheck result
        """
        return HealthCheck(
            status=HealthStatus.HEALTHY,
            checks={"handler_initialized": True},
            message=f"Handler '{self.name}' is healthy"
        )

    def __repr__(self) -> str:
        """String representation of handler."""
        return f"{self.__class__.__name__}(name='{self.name}')"


class BaseHandler(Handler):
    """
    Base handler with common functionality.

    Provides common utility methods that concrete handlers can use.
    """

    def __init__(self, name: str, supported_intents: list[str]):
        """
        Initialize base handler.

        Args:
            name: Handler identifier
            supported_intents: List of supported intent types
        """
        super().__init__(name)
        self.supported_intents = supported_intents

    def can_handle(self, intent_type: str) -> bool:
        """Check if intent is supported."""
        return intent_type in self.supported_intents

    async def handle(
        self,
        query: str,
        classification: IntentClassification,
        context: Optional[Context] = None,
        **kwargs
    ) -> HandlerResponse:
        """
        Handle query with common error handling.

        Subclasses should override _handle_impl instead.
        """
        try:
            logger.info(
                "handler_processing",
                handler=self.name,
                intent=classification.intent.value,
                confidence=classification.confidence,
                context=context.request_id if context else None
            )

            # Call implementation
            result = await self._handle_impl(query, classification, context, **kwargs)

            logger.info(
                "handler_success",
                handler=self.name,
                context=context.request_id if context else None
            )

            return result

        except Exception as e:
            logger.error(
                "handler_error",
                handler=self.name,
                error=str(e),
                error_type=type(e).__name__,
                context=context.request_id if context else None
            )

            return HandlerResponse(
                success=False,
                result=None,
                error=f"Handler error: {e}",
                next_action="retry_or_fallback"
            )

    @abstractmethod
    async def _handle_impl(
        self,
        query: str,
        classification: IntentClassification,
        context: Optional[Context] = None,
        **kwargs
    ) -> HandlerResponse:
        """
        Implementation of query handling.

        Subclasses must implement this method.

        Args:
            query: Original user query
            classification: Intent classification result
            context: Execution context
            **kwargs: Additional parameters

        Returns:
            HandlerResponse with result
        """
        pass

    def _extract_parameter(
        self,
        classification: IntentClassification,
        param_name: str,
        default: Any = None
    ) -> Any:
        """
        Extract a parameter from classification.

        Args:
            classification: Classification result
            param_name: Parameter name to extract
            default: Default value if not found

        Returns:
            Parameter value or default
        """
        return classification.parameters.get(param_name, default)

    def _validate_parameters(
        self,
        classification: IntentClassification,
        required: list[str]
    ) -> tuple[bool, Optional[str]]:
        """
        Validate required parameters are present.

        Args:
            classification: Classification result
            required: List of required parameter names

        Returns:
            Tuple of (is_valid, error_message)
        """
        missing = []
        for param in required:
            if param not in classification.parameters:
                missing.append(param)

        if missing:
            return False, f"Missing required parameters: {', '.join(missing)}"

        return True, None
