"""
Router for directing queries to appropriate handlers.

This module implements the Router that classifies queries
and routes them to registered handlers based on intent.

Design Principles:
- Centralized routing logic
- Handler registry with fallbacks
- Confidence-based routing decisions
- Comprehensive logging and observability

Reference:
- docs/ARCHITECTURE.md - Routing flow
- docs/PKB/langchain_agents.md - Agent patterns
"""

import time
from typing import Dict, Optional

from src.common.exceptions import HandlerNotFoundError, RoutingError
from src.common.logger import get_logger
from src.common.models import Context

from .classifier import IntentClassifier
from .handlers.base import Handler
from .models import (
    HandlerResponse,
    IntentClassification,
    IntentType,
    RoutingDecision,
)

logger = get_logger(__name__)


class Router:
    """
    Routes classified queries to appropriate handlers.

    The router maintains a registry of handlers and uses the
    IntentClassifier to determine which handler should process each query.

    Attributes:
        classifier: Intent classifier instance
        handlers: Registry of intent -> handler mappings
        default_handler: Fallback handler for unknown intents
        confidence_threshold: Minimum confidence for routing
    """

    def __init__(
        self,
        classifier: Optional[IntentClassifier] = None,
        confidence_threshold: float = 0.7
    ):
        """
        Initialize router.

        Args:
            classifier: Intent classifier (creates default if not provided)
            confidence_threshold: Minimum confidence to proceed with routing
        """
        self.classifier = classifier or IntentClassifier()
        self.handlers: Dict[str, Handler] = {}
        self.default_handler: Optional[Handler] = None
        self.confidence_threshold = confidence_threshold

        logger.info(
            "router_initialized",
            threshold=confidence_threshold
        )

    def register_handler(
        self,
        intent_type: IntentType,
        handler: Handler
    ) -> None:
        """
        Register a handler for an intent type.

        Args:
            intent_type: Intent type this handler processes
            handler: Handler instance

        Example:
            >>> router = Router()
            >>> router.register_handler(
            ...     IntentType.DATABASE_QUERY,
            ...     DatabaseQueryHandler()
            ... )
        """
        intent_key = intent_type.value
        self.handlers[intent_key] = handler

        logger.info(
            "handler_registered",
            intent=intent_key,
            handler_name=handler.name
        )

    def register_default_handler(self, handler: Handler) -> None:
        """
        Register a default handler for unknown/low-confidence intents.

        Args:
            handler: Default handler instance
        """
        self.default_handler = handler
        logger.info("default_handler_registered", handler_name=handler.name)

    def get_handler(self, intent_type: IntentType) -> Optional[Handler]:
        """
        Get handler for an intent type.

        Args:
            intent_type: Intent type

        Returns:
            Handler if registered, None otherwise
        """
        return self.handlers.get(intent_type.value)

    def _make_routing_decision(
        self,
        classification: IntentClassification
    ) -> RoutingDecision:
        """
        Make routing decision based on classification.

        Args:
            classification: Intent classification result

        Returns:
            RoutingDecision with handler selection
        """
        intent_key = classification.intent.value

        # Check if handler is registered
        if intent_key not in self.handlers:
            logger.warning(
                "no_handler_registered",
                intent=intent_key,
                using_default=self.default_handler is not None
            )

            if self.default_handler:
                return RoutingDecision(
                    intent_classification=classification,
                    handler_name=self.default_handler.name,
                    handler_config={},
                    should_route=True
                )
            else:
                return RoutingDecision(
                    intent_classification=classification,
                    handler_name="none",
                    handler_config={},
                    should_route=False
                )

        # Check confidence threshold
        if classification.confidence < self.confidence_threshold:
            logger.warning(
                "low_confidence_classification",
                intent=intent_key,
                confidence=classification.confidence,
                threshold=self.confidence_threshold
            )

            # Use default handler for low confidence
            if self.default_handler:
                return RoutingDecision(
                    intent_classification=classification,
                    handler_name=self.default_handler.name,
                    handler_config={},
                    should_route=True,
                    fallback_handler=self.handlers[intent_key].name
                )

        # Check if clarification is needed
        if classification.requires_clarification:
            logger.info(
                "clarification_required",
                intent=intent_key,
                suggested_clarifications=classification.suggested_clarifications
            )

            # Route to clarification handler if available,
            # otherwise use default
            if IntentType.CLARIFICATION_NEEDED.value in self.handlers:
                handler_name = self.handlers[IntentType.CLARIFICATION_NEEDED.value].name
            elif self.default_handler:
                handler_name = self.default_handler.name
            else:
                return RoutingDecision(
                    intent_classification=classification,
                    handler_name="none",
                    handler_config={},
                    should_route=False
                )

            return RoutingDecision(
                intent_classification=classification,
                handler_name=handler_name,
                handler_config={
                    "clarifications": classification.suggested_clarifications
                },
                should_route=True
            )

        # Normal routing to primary handler
        handler = self.handlers[intent_key]
        return RoutingDecision(
            intent_classification=classification,
            handler_name=handler.name,
            handler_config=classification.parameters,
            should_route=True
        )

    async def route(
        self,
        query: str,
        context: Optional[Context] = None
    ) -> HandlerResponse:
        """
        Route a query to the appropriate handler.

        This is the main entry point for query processing.

        Args:
            query: User query to route
            context: Execution context

        Returns:
            HandlerResponse from the selected handler

        Raises:
            RoutingError: If routing fails
            HandlerNotFoundError: If no suitable handler is found

        Example:
            >>> router = Router()
            >>> # ... register handlers ...
            >>> response = await router.route("What were Q3 sales?")
            >>> print(response.success)
        """
        start_time = time.time()

        try:
            logger.info(
                "routing_query",
                query_length=len(query),
                context=context.request_id if context else None
            )

            # Classify intent
            classification = await self.classifier.classify(query, context)

            logger.info(
                "intent_classified_for_routing",
                intent=classification.intent.value,
                confidence=classification.confidence,
                context=context.request_id if context else None
            )

            # Make routing decision
            decision = self._make_routing_decision(classification)

            if not decision.should_route:
                raise HandlerNotFoundError(
                    f"No handler available for intent: {classification.intent.value}",
                    context={
                        "intent": classification.intent.value,
                        "confidence": classification.confidence
                    }
                )

            # Get handler
            handler = None
            if decision.handler_name == self.default_handler.name if self.default_handler else False:
                handler = self.default_handler
            else:
                intent_key = classification.intent.value
                if intent_key in self.handlers:
                    handler = self.handlers[intent_key]

            if not handler:
                raise HandlerNotFoundError(
                    f"Handler '{decision.handler_name}' not found",
                    context={"handler_name": decision.handler_name}
                )

            # Route to handler
            logger.info(
                "routing_to_handler",
                handler_name=handler.name,
                intent=classification.intent.value,
                context=context.request_id if context else None
            )

            response = await handler.handle(
                query=query,
                classification=classification,
                context=context,
                **decision.handler_config
            )

            # Calculate total routing time
            latency_ms = (time.time() - start_time) * 1000

            logger.info(
                "routing_complete",
                success=response.success,
                handler=handler.name,
                latency_ms=latency_ms,
                context=context.request_id if context else None
            )

            return response

        except (HandlerNotFoundError, RoutingError):
            # Re-raise routing-specific errors
            raise

        except Exception as e:
            logger.error(
                "routing_failed",
                error=str(e),
                error_type=type(e).__name__,
                query=query[:100],
                context=context.request_id if context else None
            )

            raise RoutingError(
                f"Failed to route query: {e}",
                context={
                    "query": query[:100],
                    "error_type": type(e).__name__
                }
            )

    async def route_batch(
        self,
        queries: list[str],
        context: Optional[Context] = None,
        concurrency: int = 5
    ) -> list[HandlerResponse]:
        """
        Route multiple queries in parallel.

        Args:
            queries: List of queries to route
            context: Shared execution context
            concurrency: Maximum concurrent routings

        Returns:
            List of HandlerResponse results

        Example:
            >>> router = Router()
            >>> responses = await router.route_batch([
            ...     "What were Q3 sales?",
            ...     "Show me EMEA customers"
            ... ])
        """
        from src.common.utils import gather_with_concurrency

        logger.info("routing_batch", count=len(queries), concurrency=concurrency)

        responses = await gather_with_concurrency(
            concurrency,
            *[self.route(query, context) for query in queries],
            return_exceptions=True
        )

        # Convert exceptions to error responses
        results = []
        for i, result in enumerate(responses):
            if isinstance(result, Exception):
                logger.error(
                    "batch_routing_failed",
                    query_index=i,
                    error=str(result)
                )
                results.append(HandlerResponse(
                    success=False,
                    result=None,
                    error=f"Routing failed: {result}"
                ))
            else:
                results.append(result)

        return results

    def list_handlers(self) -> Dict[str, str]:
        """
        Get list of registered handlers.

        Returns:
            Dictionary mapping intent types to handler names
        """
        return {
            intent: handler.name
            for intent, handler in self.handlers.items()
        }

    def health_check(self) -> Dict[str, any]:
        """
        Check router and all handler health.

        Returns:
            Dictionary with health status
        """
        handler_health = {}
        for intent, handler in self.handlers.items():
            health = handler.health_check()
            handler_health[intent] = {
                "status": health.status.value,
                "message": health.message
            }

        return {
            "router": "healthy",
            "handlers": handler_health,
            "handler_count": len(self.handlers),
            "has_default_handler": self.default_handler is not None
        }
