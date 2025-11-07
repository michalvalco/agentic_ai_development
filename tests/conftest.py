"""
Pytest configuration and fixtures for agentic AI testing.

This module provides reusable test fixtures for all test suites,
including mocked LLMs, sample data, and integration test setup.

Design Principles:
- Fixtures are composable and reusable
- Clear separation between unit tests (mocked) and integration tests (real APIs)
- Fixtures provide realistic test data
- Easy to extend with new fixtures
"""

import asyncio
import os
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest
from pydantic import BaseModel

# Import common modules
from src.common.config import Settings
from src.common.models import Context, TokenCount, LLMCall
from src.common.cost_tracker import CostTracker


# ============================================================================
# Pytest Configuration
# ============================================================================

def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "unit: Unit tests (fast, mocked)"
    )
    config.addinivalue_line(
        "markers", "integration: Integration tests (slow, real APIs)"
    )
    config.addinivalue_line(
        "markers", "llm_integration: Tests that make real LLM API calls"
    )
    config.addinivalue_line(
        "markers", "slow: Slow-running tests"
    )


# ============================================================================
# Test Configuration Fixtures
# ============================================================================

@pytest.fixture
def test_settings() -> Settings:
    """
    Provide test settings with safe defaults.

    Returns test-safe configuration that won't hit real APIs
    unless explicitly requested in integration tests.
    """
    return Settings(
        anthropic_api_key="test-key-123",
        openai_api_key="test-key-456",
        default_model="claude-sonnet-4-20250514",
        environment="development",
        log_level="DEBUG",
        enable_cost_tracking=True,
        enable_langsmith=False,
        enable_phoenix=False,
    )


@pytest.fixture
def test_context() -> Context:
    """
    Provide a test execution context.

    Returns:
        Context instance for testing
    """
    return Context(
        request_id="test-request-123",
        user_id="test-user",
        session_id="test-session",
        trace_id="test-trace-123",
        metadata={"test": True}
    )


@pytest.fixture
def cost_tracker() -> CostTracker:
    """
    Provide a fresh cost tracker for each test.

    Returns:
        New CostTracker instance
    """
    return CostTracker()


# ============================================================================
# Mock LLM Fixtures
# ============================================================================

@pytest.fixture
def mock_llm_response() -> str:
    """
    Provide a sample LLM response for testing.

    Returns:
        Sample JSON response from LLM
    """
    return """{
        "intent": "database_query",
        "confidence": 0.95,
        "parameters": {
            "query_type": "select",
            "entities": ["customers", "sales"]
        },
        "reasoning": "The user is asking for sales data which requires a database query"
    }"""


@pytest.fixture
def mock_llm(mock_llm_response):
    """
    Provide a mocked LLM that returns predefined responses.

    This is for unit tests where we don't want to hit real APIs.

    Returns:
        Mocked LLM object with ainvoke and invoke methods
    """
    llm = AsyncMock()

    # Mock response object
    response = Mock()
    response.content = mock_llm_response
    response.usage_metadata = {
        "input_tokens": 100,
        "output_tokens": 50,
        "total_tokens": 150
    }

    # Set up async and sync invoke
    llm.ainvoke = AsyncMock(return_value=response)
    llm.invoke = Mock(return_value=response)

    # Add model name
    llm.model_name = "claude-sonnet-4-20250514"

    return llm


@pytest.fixture
def mock_llm_with_failure():
    """
    Provide a mocked LLM that fails with rate limit error.

    Useful for testing retry logic and error handling.

    Returns:
        Mocked LLM that raises RateLimitError
    """
    from src.common.exceptions import RateLimitError

    llm = AsyncMock()
    llm.ainvoke = AsyncMock(side_effect=RateLimitError("Rate limit exceeded"))
    llm.invoke = Mock(side_effect=RateLimitError("Rate limit exceeded"))

    return llm


@pytest.fixture
def mock_llm_invalid_response():
    """
    Provide a mocked LLM that returns invalid JSON.

    Useful for testing validation and error handling.

    Returns:
        Mocked LLM with malformed response
    """
    llm = AsyncMock()

    response = Mock()
    response.content = "This is not valid JSON {{"  # Malformed JSON

    llm.ainvoke = AsyncMock(return_value=response)
    llm.invoke = Mock(return_value=response)

    return llm


# ============================================================================
# Real LLM Fixtures (for Integration Tests)
# ============================================================================

@pytest.fixture
def real_anthropic_llm():
    """
    Provide a real Anthropic LLM for integration tests.

    Only use in tests marked with @pytest.mark.llm_integration.

    Returns:
        Real ChatAnthropic instance

    Raises:
        pytest.skip: If API key is not configured
    """
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key or api_key == "your-key-here":
        pytest.skip("ANTHROPIC_API_KEY not configured for integration tests")

    from langchain_anthropic import ChatAnthropic

    return ChatAnthropic(
        model="claude-sonnet-4-20250514",
        anthropic_api_key=api_key,
        max_tokens=1024,
        temperature=0.0
    )


@pytest.fixture
def real_openai_llm():
    """
    Provide a real OpenAI LLM for integration tests.

    Only use in tests marked with @pytest.mark.llm_integration.

    Returns:
        Real ChatOpenAI instance

    Raises:
        pytest.skip: If API key is not configured
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or api_key == "your-key-here":
        pytest.skip("OPENAI_API_KEY not configured for integration tests")

    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model="gpt-4o-mini",
        openai_api_key=api_key,
        max_tokens=1024,
        temperature=0.0
    )


# ============================================================================
# Sample Data Fixtures
# ============================================================================

@pytest.fixture
def sample_prompts() -> List[str]:
    """
    Provide sample user prompts for testing.

    Returns:
        List of realistic user prompts
    """
    return [
        "What were the total sales for Q3 2024?",
        "Show me all customers in the EMEA region",
        "How do I reset my password?",
        "Compare pricing plans for enterprise vs. standard",
        "What's the weather in San Francisco?",
        "Generate a sales report for last month",
        "Explain the difference between async and sync programming",
    ]


@pytest.fixture
def sample_database_schema() -> Dict[str, Any]:
    """
    Provide sample database schema for testing.

    Returns:
        Dictionary representing a database schema
    """
    return {
        "tables": {
            "customers": {
                "columns": [
                    {"name": "id", "type": "INTEGER", "primary_key": True},
                    {"name": "name", "type": "VARCHAR(255)"},
                    {"name": "email", "type": "VARCHAR(255)"},
                    {"name": "region", "type": "VARCHAR(50)"},
                    {"name": "created_at", "type": "TIMESTAMP"},
                ],
                "indexes": ["email", "region"]
            },
            "orders": {
                "columns": [
                    {"name": "id", "type": "INTEGER", "primary_key": True},
                    {"name": "customer_id", "type": "INTEGER"},
                    {"name": "amount", "type": "DECIMAL(10,2)"},
                    {"name": "status", "type": "VARCHAR(50)"},
                    {"name": "order_date", "type": "TIMESTAMP"},
                ],
                "foreign_keys": [
                    {"column": "customer_id", "references": "customers.id"}
                ]
            },
            "products": {
                "columns": [
                    {"name": "id", "type": "INTEGER", "primary_key": True},
                    {"name": "name", "type": "VARCHAR(255)"},
                    {"name": "price", "type": "DECIMAL(10,2)"},
                    {"name": "category", "type": "VARCHAR(100)"},
                ],
            }
        }
    }


@pytest.fixture
def sample_api_spec() -> Dict[str, Any]:
    """
    Provide sample OpenAPI specification for testing.

    Returns:
        Dictionary representing an API specification
    """
    return {
        "openapi": "3.0.0",
        "info": {"title": "Test API", "version": "1.0.0"},
        "paths": {
            "/users": {
                "get": {
                    "summary": "List users",
                    "parameters": [
                        {"name": "page", "in": "query", "schema": {"type": "integer"}},
                        {"name": "limit", "in": "query", "schema": {"type": "integer"}},
                    ]
                }
            },
            "/users/{id}": {
                "get": {
                    "summary": "Get user by ID",
                    "parameters": [
                        {"name": "id", "in": "path", "required": True}
                    ]
                }
            }
        }
    }


@pytest.fixture
def sample_data_records() -> List[Dict[str, Any]]:
    """
    Provide sample data records for testing.

    Returns:
        List of sample records
    """
    return [
        {
            "id": 1,
            "name": "John Doe",
            "email": "john@example.com",
            "region": "NORTH_AMERICA",
            "amount": 1250.50
        },
        {
            "id": 2,
            "name": "Jane Smith",
            "email": "jane@example.com",
            "region": "EMEA",
            "amount": 2340.75
        },
        {
            "id": 3,
            "name": "Bob Johnson",
            "email": "bob@example.com",
            "region": "APAC",
            "amount": 890.25
        },
    ]


# ============================================================================
# Async Test Support
# ============================================================================

@pytest.fixture
def event_loop():
    """
    Provide event loop for async tests.

    Returns:
        New event loop for each test
    """
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# ============================================================================
# Cleanup Fixtures
# ============================================================================

@pytest.fixture(autouse=True)
def reset_cost_tracker():
    """
    Automatically reset cost tracker before each test.

    This prevents test pollution where one test's costs
    affect another test's assertions.
    """
    from src.common.cost_tracker import cost_tracker
    cost_tracker.reset()
    yield
    cost_tracker.reset()


@pytest.fixture(autouse=True)
def reset_logging():
    """
    Reset logging configuration before each test.

    Prevents logging pollution between tests.
    """
    from src.common.logger import setup_logging
    setup_logging(log_level="INFO", json_output=False)
    yield


# ============================================================================
# Parametrized Test Helpers
# ============================================================================

def get_test_models() -> List[str]:
    """
    Get list of models to test against.

    Returns:
        List of model identifiers
    """
    return [
        "claude-sonnet-4-20250514",
        "gpt-4o-mini",
    ]


@pytest.fixture(params=get_test_models())
def model_name(request):
    """
    Parametrized fixture for testing multiple models.

    Use with: @pytest.mark.parametrize("model_name", get_test_models())
    """
    return request.param
