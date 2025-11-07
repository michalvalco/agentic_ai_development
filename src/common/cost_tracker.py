"""
LLM cost tracking for the agentic AI system.

This module provides comprehensive cost tracking for all LLM API calls,
enabling budget monitoring and cost optimization.

Design Principles:
- Track every LLM call with model, tokens, and cost
- Provide real-time cost visibility
- Support both context manager and decorator patterns
- Generate detailed cost reports
- Thread-safe for concurrent operations
- Load pricing from external config (pricing_config.yaml)

Usage:
    # Context manager
    with cost_tracker.track("intent_classification"):
        result = await llm.ainvoke(prompt)

    # Get report
    report = cost_tracker.get_report()
    print(f"Total cost: ${report['total_cost_usd']:.4f}")
"""

import asyncio
import time
from contextlib import contextmanager, asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from functools import wraps
from pathlib import Path
from threading import Lock
from typing import Any, Callable, Dict, List, Optional

import yaml

from .models import LLMCall, TokenCount


# ============================================================================
# Pricing Configuration Loader
# ============================================================================

def load_pricing_config() -> Dict[str, Dict[str, float]]:
    """
    Load LLM pricing from external YAML configuration.
    
    Returns:
        Dictionary mapping model names to pricing info
    
    Raises:
        FileNotFoundError: If pricing_config.yaml not found
        yaml.YAMLError: If config file is malformed
    """
    config_path = Path(__file__).parent.parent.parent / "pricing_config.yaml"
    
    if not config_path.exists():
        # Fallback to hardcoded pricing if config file missing
        return _get_fallback_pricing()
    
    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)
        
        # Convert to expected format
        pricing = {}
        for model_name, model_config in config.get('models', {}).items():
            pricing[model_name] = {
                "input": model_config['input_cost_per_1m_tokens'],
                "output": model_config['output_cost_per_1m_tokens']
            }
        
        return pricing
    except Exception as e:
        # Log warning and fall back to hardcoded pricing
        import warnings
        warnings.warn(
            f"Failed to load pricing_config.yaml: {e}. "
            "Using fallback pricing.",
            UserWarning
        )
        return _get_fallback_pricing()


def _get_fallback_pricing() -> Dict[str, Dict[str, float]]:
    """
    Fallback pricing if config file unavailable.
    
    Note: These prices may be outdated. Update pricing_config.yaml instead.
    """
    # Anthropic Claude pricing (per million tokens) - as of 2025-11-07
    anthropic = {
        "claude-3-opus-20240229": {"input": 15.00, "output": 75.00},
        "claude-3-sonnet-20240229": {"input": 3.00, "output": 15.00},
        "claude-3-5-sonnet-20240620": {"input": 3.00, "output": 15.00},
        "claude-sonnet-4-20250514": {"input": 3.00, "output": 15.00},
        "claude-3-haiku-20240307": {"input": 0.25, "output": 1.25},
    }
    
    # OpenAI pricing (per million tokens) - as of 2025-11-07
    openai = {
        "gpt-4o": {"input": 5.00, "output": 15.00},
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "gpt-4-turbo": {"input": 10.00, "output": 30.00},
        "gpt-4": {"input": 30.00, "output": 60.00},
        "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
    }
    
    return {**anthropic, **openai}


# Load pricing at module initialization
MODEL_PRICING = load_pricing_config()


# ============================================================================
# Cost Calculation
# ============================================================================

def calculate_cost(
    model: str,
    input_tokens: int,
    output_tokens: int
) -> float:
    """
    Calculate cost for an LLM call.

    Args:
        model: Model identifier
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens

    Returns:
        Cost in USD

    Example:
        >>> cost = calculate_cost("claude-sonnet-4-20250514", 1000, 500)
        >>> print(f"${cost:.4f}")
    """
    if model not in MODEL_PRICING:
        # Unknown model - estimate based on average pricing
        avg_input = 3.00
        avg_output = 15.00
        return (input_tokens * avg_input + output_tokens * avg_output) / 1_000_000

    pricing = MODEL_PRICING[model]
    input_cost = (input_tokens * pricing["input"]) / 1_000_000
    output_cost = (output_tokens * pricing["output"]) / 1_000_000

    return input_cost + output_cost


# ============================================================================
# Cost Tracker
# ============================================================================

@dataclass
class OperationCost:
    """Cost breakdown for a single operation."""

    operation: str
    calls: List[LLMCall] = field(default_factory=list)
    total_cost: float = 0.0
    total_tokens: int = 0
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None

    def add_call(self, call: LLMCall) -> None:
        """Add an LLM call to this operation."""
        self.calls.append(call)
        self.total_cost += call.cost_usd
        self.total_tokens += call.tokens.total_tokens

    def duration_ms(self) -> float:
        """Get operation duration in milliseconds."""
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds() * 1000
        return 0.0


class CostTracker:
    """
    Thread-safe LLM cost tracker.

    Tracks all LLM API calls with detailed cost breakdown by operation,
    model, and time period.

    Attributes:
        total_cost: Total cost across all operations
        total_calls: Total number of LLM calls
        operations: Cost breakdown by operation
    """

    def __init__(self):
        self.total_cost: float = 0.0
        self.total_calls: int = 0
        self.operations: Dict[str, OperationCost] = {}
        self._lock = Lock()
        self._current_operation: Optional[str] = None

    def track_call(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        operation: str = "unknown",
        latency_ms: float = 0.0
    ) -> LLMCall:
        """
        Track a single LLM call.

        Args:
            model: Model identifier
            input_tokens: Input token count
            output_tokens: Output token count
            operation: Operation name for grouping
            latency_ms: Call latency in milliseconds

        Returns:
            LLMCall record

        Example:
            >>> tracker = CostTracker()
            >>> call = tracker.track_call(
            ...     "claude-sonnet-4-20250514",
            ...     1000,
            ...     500,
            ...     "intent_classification"
            ... )
            >>> print(f"Cost: ${call.cost_usd:.4f}")
        """
        cost = calculate_cost(model, input_tokens, output_tokens)
        tokens = TokenCount(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens
        )

        call = LLMCall(
            model=model,
            tokens=tokens,
            cost_usd=cost,
            latency_ms=latency_ms,
            operation=operation
        )

        with self._lock:
            # Budget enforcement - check before adding cost
            from .config import settings
            from .exceptions import BudgetExceededError
            from .logger import get_logger

            logger = get_logger(__name__)

            if self.total_cost + cost > settings.max_daily_budget_usd:
                logger.error(
                    "budget_exceeded",
                    current_cost=self.total_cost,
                    attempted_cost=cost,
                    budget=settings.max_daily_budget_usd,
                    operation=operation
                )
                raise BudgetExceededError(
                    f"Daily budget exceeded: ${self.total_cost + cost:.4f} > "
                    f"${settings.max_daily_budget_usd}. Blocking operation: {operation}"
                )

            self.total_cost += cost
            self.total_calls += 1

            if operation not in self.operations:
                self.operations[operation] = OperationCost(operation=operation)

            self.operations[operation].add_call(call)

        return call

    @contextmanager
    def track(self, operation: str):
        """
        Context manager to track an operation.

        Args:
            operation: Operation name

        Yields:
            None

        Example:
            >>> tracker = CostTracker()
            >>> with tracker.track("query_generation"):
            ...     result = await llm.ainvoke(prompt)
        """
        start_time = time.time()

        # Set current operation for any calls made within this context
        with self._lock:
            if operation not in self.operations:
                self.operations[operation] = OperationCost(operation=operation)
            self._current_operation = operation

        try:
            yield
        finally:
            with self._lock:
                if operation in self.operations:
                    self.operations[operation].end_time = datetime.now()
                self._current_operation = None

    @asynccontextmanager
    async def track_async(self, operation: str):
        """
        Async context manager to track an operation.

        Args:
            operation: Operation name

        Yields:
            None

        Example:
            >>> tracker = CostTracker()
            >>> async with tracker.track_async("query_generation"):
            ...     result = await llm.ainvoke(prompt)
        """
        start_time = time.time()

        with self._lock:
            if operation not in self.operations:
                self.operations[operation] = OperationCost(operation=operation)
            self._current_operation = operation

        try:
            yield
        finally:
            with self._lock:
                if operation in self.operations:
                    self.operations[operation].end_time = datetime.now()
                self._current_operation = None

    def get_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive cost report.

        Returns:
            Dictionary containing cost breakdown

        Example:
            >>> tracker = CostTracker()
            >>> # ... make some LLM calls ...
            >>> report = tracker.get_report()
            >>> print(f"Total: ${report['total_cost_usd']:.2f}")
        """
        with self._lock:
            operations_summary = []
            for op_name, op_cost in self.operations.items():
                operations_summary.append({
                    "operation": op_name,
                    "total_cost_usd": op_cost.total_cost,
                    "total_tokens": op_cost.total_tokens,
                    "num_calls": len(op_cost.calls),
                    "duration_ms": op_cost.duration_ms(),
                    "avg_cost_per_call": (
                        op_cost.total_cost / len(op_cost.calls)
                        if op_cost.calls else 0.0
                    ),
                })

            # Group by model
            model_costs: Dict[str, float] = {}
            model_tokens: Dict[str, int] = {}
            for op_cost in self.operations.values():
                for call in op_cost.calls:
                    model_costs[call.model] = (
                        model_costs.get(call.model, 0.0) + call.cost_usd
                    )
                    model_tokens[call.model] = (
                        model_tokens.get(call.model, 0) + call.tokens.total_tokens
                    )

            return {
                "total_cost_usd": self.total_cost,
                "total_calls": self.total_calls,
                "total_tokens": sum(model_tokens.values()),
                "operations": sorted(
                    operations_summary,
                    key=lambda x: x["total_cost_usd"],
                    reverse=True
                ),
                "by_model": [
                    {
                        "model": model,
                        "cost_usd": cost,
                        "tokens": model_tokens[model]
                    }
                    for model, cost in sorted(
                        model_costs.items(),
                        key=lambda x: x[1],
                        reverse=True
                    )
                ],
            }

    def get_operation_cost(self, operation: str) -> float:
        """
        Get total cost for a specific operation.

        Args:
            operation: Operation name

        Returns:
            Total cost in USD
        """
        with self._lock:
            if operation in self.operations:
                return self.operations[operation].total_cost
            return 0.0

    def reset(self) -> None:
        """Reset all tracking data."""
        with self._lock:
            self.total_cost = 0.0
            self.total_calls = 0
            self.operations.clear()
            self._current_operation = None

    def print_report(self) -> None:
        """Print a formatted cost report to stdout."""
        report = self.get_report()

        print("\n" + "=" * 60)
        print("LLM COST REPORT")
        print("=" * 60)
        print(f"\nTotal Cost: ${report['total_cost_usd']:.4f}")
        print(f"Total Calls: {report['total_calls']}")
        print(f"Total Tokens: {report['total_tokens']:,}")

        if report['operations']:
            print("\n" + "-" * 60)
            print("BY OPERATION:")
            print("-" * 60)
            for op in report['operations']:
                print(f"\n{op['operation']}:")
                print(f"  Cost: ${op['total_cost_usd']:.4f}")
                print(f"  Calls: {op['num_calls']}")
                print(f"  Tokens: {op['total_tokens']:,}")
                print(f"  Avg/Call: ${op['avg_cost_per_call']:.4f}")

        if report['by_model']:
            print("\n" + "-" * 60)
            print("BY MODEL:")
            print("-" * 60)
            for model_data in report['by_model']:
                print(f"\n{model_data['model']}:")
                print(f"  Cost: ${model_data['cost_usd']:.4f}")
                print(f"  Tokens: {model_data['tokens']:,}")

        print("\n" + "=" * 60 + "\n")


# ============================================================================
# Singleton Instance
# ============================================================================

# Global cost tracker instance
cost_tracker = CostTracker()


# ============================================================================
# Decorator for Tracking Functions
# ============================================================================

def track_cost(operation: str):
    """
    Decorator to automatically track cost for a function.

    Args:
        operation: Operation name for tracking

    Example:
        >>> @track_cost("query_generation")
        ... async def generate_query(prompt: str):
        ...     return await llm.ainvoke(prompt)
    """
    def decorator(func: Callable):
        if asyncio.iscoroutinefunction(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                async with cost_tracker.track_async(operation):
                    return await func(*args, **kwargs)
            return async_wrapper
        else:
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                with cost_tracker.track(operation):
                    return func(*args, **kwargs)
            return sync_wrapper
    return decorator
