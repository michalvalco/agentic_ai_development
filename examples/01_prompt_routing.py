"""
Example: Prompt Routing Capability

This example demonstrates the prompt routing capability, which:
1. Classifies user intent using an LLM
2. Routes queries to appropriate handlers based on intent
3. Handles various query types with confidence scoring

Usage:
    python examples/01_prompt_routing.py

Requirements:
    - ANTHROPIC_API_KEY environment variable set
    - Dependencies installed: pip install -r requirements.txt
"""

import asyncio
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from src.common.config import settings
from src.common.cost_tracker import cost_tracker
from src.common.logger import get_logger, setup_logging
from src.common.models import Context
from src.prompt_routing import (
    Router,
    IntentClassifier,
    IntentType,
    DirectResponseHandler,
    DatabaseQueryHandler,
    KnowledgeSearchHandler,
)

# Setup logging
setup_logging(log_level="INFO", json_output=False)
logger = get_logger(__name__)


async def main():
    """
    Main example function demonstrating prompt routing.
    """
    print("\n" + "=" * 70)
    print("Prompt Routing Example")
    print("=" * 70 + "\n")

    # Verify API key is configured
    if not os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_API_KEY") == "your-key-here":
        print("❌ ANTHROPIC_API_KEY not configured!")
        print("Please set your API key in the .env file\n")
        return

    # Create execution context
    context = Context(
        request_id="example-001",
        user_id="example-user",
        metadata={"example": "prompt_routing"}
    )

    print("📋 Setting up routing system...\n")

    # Initialize router with classifier
    classifier = IntentClassifier()
    router = Router(classifier=classifier)

    # Register handlers for different intent types
    router.register_handler(
        IntentType.DIRECT_RESPONSE,
        DirectResponseHandler()
    )
    router.register_handler(
        IntentType.DATABASE_QUERY,
        DatabaseQueryHandler()
    )
    router.register_handler(
        IntentType.KNOWLEDGE_SEARCH,
        KnowledgeSearchHandler()
    )

    # Register a default handler for unknown/low-confidence intents
    router.register_default_handler(DirectResponseHandler())

    print(f"✅ Router configured with {len(router.list_handlers())} handlers\n")
    print("Registered handlers:")
    for intent, handler_name in router.list_handlers().items():
        print(f"  - {intent}: {handler_name}")
    print()

    # Test queries demonstrating different intents
    test_queries = [
        ("Hello! How are you today?", "Direct conversational response"),
        ("What were the total sales for Q3 2024?", "Database query"),
        ("Show me all customers in the EMEA region", "Database query with filter"),
        ("How do I reset my password?", "Knowledge search"),
        ("What's the difference between async and sync?", "Knowledge search"),
        ("Do the thing", "Ambiguous query requiring clarification"),
    ]

    print("=" * 70)
    print("Testing Queries")
    print("=" * 70 + "\n")

    for i, (query, description) in enumerate(test_queries, 1):
        print(f"\n{'─' * 70}")
        print(f"Query #{i}: {description}")
        print(f"{'─' * 70}")
        print(f"Query: \"{query}\"\n")

        try:
            # Route the query
            response = await router.route(query, context)

            # Display classification info (from metadata if available)
            print(f"✅ Successfully routed")
            print(f"Success: {response.success}")

            if response.success:
                result_type = response.result.get('type', 'unknown')
                print(f"Result type: {result_type}")

                if result_type == "direct":
                    print(f"\n💬 Response: {response.result['response']}")
                elif result_type == "clarification":
                    print(f"\n❓ Clarification needed:")
                    print(f"Response: {response.result['response']}")
                elif result_type == "database_query":
                    print(f"\n🗄️  Database Query Info:")
                    print(f"Entities: {response.result.get('entities', [])}")
                    print(f"Filters: {response.result.get('filters', {})}")
                    print(f"Timeframe: {response.result.get('timeframe', 'N/A')}")
                    print(f"Message: {response.result.get('message')}")
                elif result_type == "knowledge_search":
                    print(f"\n📚 Knowledge Search Info:")
                    print(f"Topic: {response.result.get('topic', 'N/A')}")
                    print(f"Keywords: {response.result.get('keywords', [])}")
                    print(f"Message: {response.result.get('message')}")

                if response.requires_followup:
                    print(f"\n⚠️  Requires followup: {response.next_action}")
            else:
                print(f"\n❌ Error: {response.error}")

        except Exception as e:
            print(f"❌ Error routing query: {e}")
            logger.error("query_routing_failed", query=query, error=str(e))

        # Small delay between queries
        await asyncio.sleep(0.5)

    # Display cost report
    print("\n" + "=" * 70)
    print("Cost Report")
    print("=" * 70 + "\n")

    report = cost_tracker.get_report()
    print(f"Total Cost: ${report['total_cost_usd']:.4f}")
    print(f"Total Calls: {report['total_calls']}")
    print(f"Total Tokens: {report['total_tokens']:,}")

    if report['operations']:
        print(f"\nBy Operation:")
        for op in report['operations'][:5]:  # Top 5
            print(f"  {op['operation']}: ${op['total_cost_usd']:.4f} ({op['num_calls']} calls)")

    print("\n" + "=" * 70)
    print("Example Complete!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())
