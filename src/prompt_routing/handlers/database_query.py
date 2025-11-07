"""
Database query handler (placeholder).

This handler will integrate with the Query Writing capability
to process database queries. For now, it's a placeholder that
acknowledges the intent and prepares for future implementation.
"""

from typing import Optional

from src.common.logger import get_logger
from src.common.models import Context
from src.prompt_routing.handlers.base import BaseHandler
from src.prompt_routing.models import HandlerResponse, IntentClassification, IntentType

logger = get_logger(__name__)


class DatabaseQueryHandler(BaseHandler):
    """
    Handler for database query intents.

    This is a placeholder that will integrate with Query Writing
    capability once it's implemented (Day 3).

    For now, it extracts parameters and returns a structured response
    indicating what database query would be generated.
    """

    def __init__(self):
        """Initialize database query handler."""
        super().__init__(
            name="database_query",
            supported_intents=[IntentType.DATABASE_QUERY.value]
        )

    async def _handle_impl(
        self,
        query: str,
        classification: IntentClassification,
        context: Optional[Context] = None,
        **kwargs
    ) -> HandlerResponse:
        """
        Handle database query intent.

        Args:
            query: User query
            classification: Intent classification
            context: Execution context
            **kwargs: Additional parameters

        Returns:
            HandlerResponse indicating database query is needed
        """
        # Extract relevant parameters
        query_type = self._extract_parameter(classification, "query_type", "select")
        entities = self._extract_parameter(classification, "entities", [])
        filters = self._extract_parameter(classification, "filter", {})
        timeframe = self._extract_parameter(classification, "timeframe")

        logger.info(
            "database_query_intent_processed",
            query_type=query_type,
            entities=entities,
            has_filters=len(filters) > 0,
            context=context.request_id if context else None
        )

        # For now, return structured information about the query
        result = {
            "type": "database_query",
            "query_type": query_type,
            "entities": entities,
            "filters": filters,
            "timeframe": timeframe,
            "original_query": query,
            "message": (
                "Database query intent recognized. "
                "Will be processed by Query Writing capability."
            )
        }

        return HandlerResponse(
            success=True,
            result=result,
            next_action="generate_sql",  # Will be handled by Query Writing
            requires_followup=True
        )
