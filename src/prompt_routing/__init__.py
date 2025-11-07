"""
Prompt Routing capability.

This capability classifies user intent and routes queries to
appropriate handlers for processing.

Main Components:
- IntentClassifier: Classifies user queries using LLM
- Router: Routes classified queries to registered handlers
- Handlers: Process specific intent types

Example:
    >>> from src.prompt_routing import Router, IntentClassifier
    >>> from src.prompt_routing.handlers import DirectResponseHandler
    >>>
    >>> # Create router with handlers
    >>> router = Router()
    >>> router.register_handler(
    ...     IntentType.DIRECT_RESPONSE,
    ...     DirectResponseHandler()
    ... )
    >>>
    >>> # Route a query
    >>> response = await router.route("Hello, how are you?")
    >>> print(response.result['response'])
"""

from .classifier import IntentClassifier
from .router import Router
from .models import (
    IntentType,
    IntentClassification,
    RoutingDecision,
    HandlerResponse,
    ClassificationExample,
)

# Import handlers for convenience
from .handlers import (
    Handler,
    BaseHandler,
    DirectResponseHandler,
    DatabaseQueryHandler,
    KnowledgeSearchHandler,
)

__all__ = [
    # Core classes
    "IntentClassifier",
    "Router",

    # Models
    "IntentType",
    "IntentClassification",
    "RoutingDecision",
    "HandlerResponse",
    "ClassificationExample",

    # Handlers
    "Handler",
    "BaseHandler",
    "DirectResponseHandler",
    "DatabaseQueryHandler",
    "KnowledgeSearchHandler",
]
