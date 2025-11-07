"""
Unit tests for prompt routing capability.

These tests use mocked LLMs to test logic without making real API calls.
"""

import pytest
from unittest.mock import AsyncMock, Mock, patch

from src.common.exceptions import ClassificationError
from src.common.models import Context
from src.prompt_routing import (
    IntentClassifier,
    Router,
    IntentType,
    DirectResponseHandler,
    DatabaseQueryHandler,
)


# ============================================================================
# IntentClassifier Tests
# ============================================================================

class TestIntentClassifier:
    """Unit tests for IntentClassifier."""

    @pytest.mark.asyncio
    async def test_classify_database_query(self, mock_llm):
        """Test classification of database query intent."""
        # Configure mock to return database query classification
        mock_llm.ainvoke = AsyncMock(return_value=Mock(
            content="""{
                "intent": "database_query",
                "confidence": 0.95,
                "parameters": {
                    "query_type": "select",
                    "entities": ["sales"],
                    "timeframe": "Q3 2024"
                },
                "reasoning": "User is requesting sales data from Q3 2024",
                "requires_clarification": false,
                "suggested_clarifications": [],
                "alternative_intents": []
            }""",
            usage_metadata={
                "input_tokens": 100,
                "output_tokens": 50
            }
        ))

        classifier = IntentClassifier(llm=mock_llm)
        result = await classifier.classify("What were the total sales in Q3 2024?")

        assert result.intent == IntentType.DATABASE_QUERY
        assert result.confidence == 0.95
        assert result.parameters["timeframe"] == "Q3 2024"
        assert not result.requires_clarification

    @pytest.mark.asyncio
    async def test_classify_with_low_confidence(self, mock_llm):
        """Test classification with low confidence score."""
        mock_llm.ainvoke = AsyncMock(return_value=Mock(
            content="""{
                "intent": "unknown",
                "confidence": 0.4,
                "parameters": {},
                "reasoning": "Query is ambiguous and unclear",
                "requires_clarification": true,
                "suggested_clarifications": ["Could you be more specific?"],
                "alternative_intents": []
            }""",
            usage_metadata={"input_tokens": 100, "output_tokens": 50}
        ))

        classifier = IntentClassifier(llm=mock_llm)
        result = await classifier.classify("Do the thing")

        assert result.confidence == 0.4
        assert result.requires_clarification
        assert len(result.suggested_clarifications) > 0

    @pytest.mark.asyncio
    async def test_classify_with_context(self, mock_llm, test_context):
        """Test classification with context propagation."""
        mock_llm.ainvoke = AsyncMock(return_value=Mock(
            content="""{
                "intent": "direct_response",
                "confidence": 0.9,
                "parameters": {},
                "reasoning": "Simple greeting",
                "requires_clarification": false,
                "suggested_clarifications": [],
                "alternative_intents": []
            }""",
            usage_metadata={"input_tokens": 100, "output_tokens": 50}
        ))

        classifier = IntentClassifier(llm=mock_llm)
        result = await classifier.classify("Hello!", context=test_context)

        assert result.intent == IntentType.DIRECT_RESPONSE
        # Verify LLM was called
        assert mock_llm.ainvoke.called

    @pytest.mark.asyncio
    async def test_classify_handles_invalid_json(self, mock_llm_invalid_response):
        """Test graceful handling of invalid JSON response."""
        classifier = IntentClassifier(llm=mock_llm_invalid_response)

        with pytest.raises(ClassificationError):
            await classifier.classify("Test query")

    @pytest.mark.asyncio
    async def test_classify_batch(self, mock_llm):
        """Test batch classification."""
        mock_llm.ainvoke = AsyncMock(return_value=Mock(
            content="""{
                "intent": "database_query",
                "confidence": 0.9,
                "parameters": {},
                "reasoning": "Database query",
                "requires_clarification": false,
                "suggested_clarifications": [],
                "alternative_intents": []
            }""",
            usage_metadata={"input_tokens": 100, "output_tokens": 50}
        ))

        classifier = IntentClassifier(llm=mock_llm)
        queries = [
            "What were Q3 sales?",
            "Show me customers in EMEA"
        ]

        results = await classifier.classify_batch(queries, concurrency=2)

        assert len(results) == 2
        assert all(isinstance(r.intent, IntentType) for r in results)


# ============================================================================
# Router Tests
# ============================================================================

class TestRouter:
    """Unit tests for Router."""

    @pytest.mark.asyncio
    async def test_route_to_registered_handler(self, mock_llm):
        """Test routing to a registered handler."""
        # Setup classifier mock
        mock_llm.ainvoke = AsyncMock(return_value=Mock(
            content="""{
                "intent": "direct_response",
                "confidence": 0.95,
                "parameters": {},
                "reasoning": "Greeting",
                "requires_clarification": false,
                "suggested_clarifications": [],
                "alternative_intents": []
            }""",
            usage_metadata={"input_tokens": 100, "output_tokens": 50}
        ))

        classifier = IntentClassifier(llm=mock_llm)
        router = Router(classifier=classifier)

        # Register handler
        handler = DirectResponseHandler(llm=mock_llm)
        router.register_handler(IntentType.DIRECT_RESPONSE, handler)

        # Route query
        response = await router.route("Hello!")

        assert response.success
        assert response.result is not None

    @pytest.mark.asyncio
    async def test_route_with_low_confidence(self, mock_llm):
        """Test routing with low confidence classification."""
        mock_llm.ainvoke = AsyncMock(return_value=Mock(
            content="""{
                "intent": "database_query",
                "confidence": 0.5,
                "parameters": {},
                "reasoning": "Uncertain",
                "requires_clarification": false,
                "suggested_clarifications": [],
                "alternative_intents": []
            }""",
            usage_metadata={"input_tokens": 100, "output_tokens": 50}
        ))

        classifier = IntentClassifier(llm=mock_llm, confidence_threshold=0.7)
        router = Router(classifier=classifier, confidence_threshold=0.7)

        # Register default handler
        default_handler = DirectResponseHandler(llm=mock_llm)
        router.register_default_handler(default_handler)

        # Should route to default handler due to low confidence
        response = await router.route("Ambiguous query")

        assert response.success

    @pytest.mark.asyncio
    async def test_route_to_multiple_handlers(self, mock_llm):
        """Test router with multiple registered handlers."""
        # Mock classifier to return different intents
        responses = [
            Mock(
                content="""{
                    "intent": "direct_response",
                    "confidence": 0.9,
                    "parameters": {},
                    "reasoning": "Greeting",
                    "requires_clarification": false,
                    "suggested_clarifications": [],
                    "alternative_intents": []
                }""",
                usage_metadata={"input_tokens": 100, "output_tokens": 50}
            ),
            Mock(
                content="""{
                    "intent": "database_query",
                    "confidence": 0.9,
                    "parameters": {},
                    "reasoning": "Data request",
                    "requires_clarification": false,
                    "suggested_clarifications": [],
                    "alternative_intents": []
                }""",
                usage_metadata={"input_tokens": 100, "output_tokens": 50}
            )
        ]

        mock_llm.ainvoke = AsyncMock(side_effect=responses)

        classifier = IntentClassifier(llm=mock_llm)
        router = Router(classifier=classifier)

        # Register multiple handlers
        router.register_handler(IntentType.DIRECT_RESPONSE, DirectResponseHandler(llm=mock_llm))
        router.register_handler(IntentType.DATABASE_QUERY, DatabaseQueryHandler())

        # Route different queries
        response1 = await router.route("Hello!")
        response2 = await router.route("Show me sales data")

        assert response1.success
        assert response2.success

    def test_list_handlers(self):
        """Test listing registered handlers."""
        router = Router()
        router.register_handler(IntentType.DIRECT_RESPONSE, DirectResponseHandler())
        router.register_handler(IntentType.DATABASE_QUERY, DatabaseQueryHandler())

        handlers = router.list_handlers()

        assert len(handlers) == 2
        assert IntentType.DIRECT_RESPONSE.value in handlers
        assert IntentType.DATABASE_QUERY.value in handlers

    def test_health_check(self):
        """Test router health check."""
        router = Router()
        router.register_handler(IntentType.DIRECT_RESPONSE, DirectResponseHandler())

        health = router.health_check()

        assert health["router"] == "healthy"
        assert len(health["handlers"]) == 1


# ============================================================================
# Handler Tests
# ============================================================================

class TestDirectResponseHandler:
    """Unit tests for DirectResponseHandler."""

    @pytest.mark.asyncio
    async def test_handle_simple_greeting(self, mock_llm):
        """Test handling a simple greeting."""
        mock_llm.ainvoke = AsyncMock(return_value=Mock(
            content="Hello! I'm doing well, thank you. How can I help you today?"
        ))

        handler = DirectResponseHandler(llm=mock_llm)

        from src.prompt_routing.models import IntentClassification
        classification = IntentClassification(
            intent=IntentType.DIRECT_RESPONSE,
            confidence=0.95,
            parameters={},
            reasoning="Simple greeting"
        )

        response = await handler.handle("Hello!", classification)

        assert response.success
        assert "response" in response.result
        assert len(response.result["response"]) > 0

    @pytest.mark.asyncio
    async def test_handle_clarification_request(self, mock_llm):
        """Test handling a query that needs clarification."""
        handler = DirectResponseHandler(llm=mock_llm)

        from src.prompt_routing.models import IntentClassification
        classification = IntentClassification(
            intent=IntentType.CLARIFICATION_NEEDED,
            confidence=0.6,
            parameters={},
            reasoning="Query is ambiguous",
            requires_clarification=True,
            suggested_clarifications=[
                "What specific data are you looking for?",
                "Which time period?"
            ]
        )

        response = await handler.handle("Show me the data", classification)

        assert response.success
        assert response.requires_followup
        assert "clarifications" in response.result


class TestDatabaseQueryHandler:
    """Unit tests for DatabaseQueryHandler."""

    @pytest.mark.asyncio
    async def test_handle_database_query(self):
        """Test handling a database query intent."""
        handler = DatabaseQueryHandler()

        from src.prompt_routing.models import IntentClassification
        classification = IntentClassification(
            intent=IntentType.DATABASE_QUERY,
            confidence=0.95,
            parameters={
                "query_type": "select",
                "entities": ["customers"],
                "filter": "region=EMEA"
            },
            reasoning="User wants customer data"
        )

        response = await handler.handle("Show me EMEA customers", classification)

        assert response.success
        assert response.result["type"] == "database_query"
        assert "entities" in response.result
        assert response.requires_followup
