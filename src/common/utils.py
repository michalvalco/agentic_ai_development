"""
Common utilities for the agentic AI system.

This module provides reusable utility functions for retry logic,
async helpers, validation, and other common operations.

Design Principles:
- Composable and reusable across all capabilities
- Type-safe with comprehensive hints
- Well-tested and documented
- Performance-conscious
"""

import asyncio
import functools
import hashlib
import time
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
    after_log,
)

from .exceptions import RateLimitError, LLMProviderError, is_recoverable
from .logger import get_logger

logger = get_logger(__name__)

T = TypeVar('T')


# ============================================================================
# Retry Utilities
# ============================================================================

def retry_on_rate_limit(
    max_attempts: int = 3,
    min_wait: float = 2.0,
    max_wait: float = 60.0,
    multiplier: float = 2.0
):
    """
    Decorator to retry functions on rate limit errors with exponential backoff.

    Args:
        max_attempts: Maximum number of retry attempts
        min_wait: Minimum wait time between retries (seconds)
        max_wait: Maximum wait time between retries (seconds)
        multiplier: Exponential backoff multiplier

    Returns:
        Decorated function with retry logic

    Example:
        >>> @retry_on_rate_limit(max_attempts=3)
        ... async def call_llm(prompt: str):
        ...     return await llm.ainvoke(prompt)
    """
    return retry(
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(
            multiplier=multiplier,
            min=min_wait,
            max=max_wait
        ),
        retry=retry_if_exception_type(RateLimitError),
        before_sleep=before_sleep_log(logger, "WARNING"),
        after=after_log(logger, "INFO")
    )


def retry_on_recoverable_error(
    max_attempts: int = 3,
    min_wait: float = 1.0,
    max_wait: float = 30.0
):
    """
    Decorator to retry functions on any recoverable error.

    Uses the is_recoverable() function to determine if an error
    should trigger a retry.

    Args:
        max_attempts: Maximum number of retry attempts
        min_wait: Minimum wait time between retries (seconds)
        max_wait: Maximum wait time between retries (seconds)

    Returns:
        Decorated function with retry logic

    Example:
        >>> @retry_on_recoverable_error()
        ... async def fetch_data(url: str):
        ...     return await client.get(url)
    """
    def should_retry(exception: Exception) -> bool:
        """Check if exception is recoverable."""
        return is_recoverable(exception)

    return retry(
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(min=min_wait, max=max_wait),
        retry=retry_if_exception_type(Exception) & should_retry,
        before_sleep=before_sleep_log(logger, "WARNING")
    )


# ============================================================================
# Async Utilities
# ============================================================================

async def run_with_timeout(
    coro,
    timeout_seconds: float,
    error_message: str = "Operation timed out"
) -> Any:
    """
    Run an async coroutine with a timeout.

    Args:
        coro: Coroutine to run
        timeout_seconds: Timeout in seconds
        error_message: Error message if timeout occurs

    Returns:
        Result of the coroutine

    Raises:
        asyncio.TimeoutError: If operation exceeds timeout

    Example:
        >>> result = await run_with_timeout(
        ...     llm.ainvoke(prompt),
        ...     timeout_seconds=30.0
        ... )
    """
    try:
        return await asyncio.wait_for(coro, timeout=timeout_seconds)
    except asyncio.TimeoutError:
        logger.error("operation_timeout", timeout_seconds=timeout_seconds)
        raise asyncio.TimeoutError(error_message)


async def gather_with_concurrency(
    n: int,
    *coros,
    return_exceptions: bool = False
) -> List[Any]:
    """
    Run multiple coroutines with a concurrency limit.

    Useful for batch processing with rate limits.

    Args:
        n: Maximum number of concurrent coroutines
        *coros: Coroutines to run
        return_exceptions: Whether to return exceptions instead of raising

    Returns:
        List of results

    Example:
        >>> results = await gather_with_concurrency(
        ...     5,  # Max 5 concurrent requests
        ...     *[process_item(item) for item in items]
        ... )
    """
    semaphore = asyncio.Semaphore(n)

    async def with_semaphore(coro):
        async with semaphore:
            return await coro

    return await asyncio.gather(
        *[with_semaphore(coro) for coro in coros],
        return_exceptions=return_exceptions
    )


# ============================================================================
# Validation Utilities
# ============================================================================

def sanitize_prompt(prompt: str, max_length: Optional[int] = None) -> str:
    """
    Sanitize user prompt to prevent injection attacks.

    Removes or escapes potentially dangerous patterns.

    Args:
        prompt: User input prompt
        max_length: Maximum allowed length (truncate if exceeded)

    Returns:
        Sanitized prompt

    Example:
        >>> safe_prompt = sanitize_prompt(user_input, max_length=1000)
    """
    # Remove null bytes
    prompt = prompt.replace('\x00', '')

    # Remove excessive newlines (potential prompt injection)
    while '\n\n\n' in prompt:
        prompt = prompt.replace('\n\n\n', '\n\n')

    # Truncate if needed
    if max_length and len(prompt) > max_length:
        logger.warning("prompt_truncated", original_length=len(prompt), max_length=max_length)
        prompt = prompt[:max_length]

    return prompt.strip()


def validate_sql_query(query: str, allowed_tables: Optional[List[str]] = None) -> bool:
    """
    Validate SQL query for safety.

    Checks for dangerous operations and table access.

    Args:
        query: SQL query to validate
        allowed_tables: Whitelist of allowed tables (None = no restriction)

    Returns:
        True if query is safe, False otherwise

    Example:
        >>> is_safe = validate_sql_query(
        ...     "SELECT * FROM users WHERE id = ?",
        ...     allowed_tables=["users", "orders"]
        ... )
    """
    query_upper = query.upper()

    # Check for dangerous operations
    dangerous_operations = [
        'DROP',
        'DELETE',
        'TRUNCATE',
        'ALTER',
        'CREATE',
        'GRANT',
        'REVOKE',
        'EXEC',
        'EXECUTE',
    ]

    for op in dangerous_operations:
        if op in query_upper:
            logger.warning("dangerous_sql_operation", operation=op, query=query[:100])
            return False

    # Validate table access if whitelist provided
    if allowed_tables:
        # Simple check - could be more sophisticated
        query_tables = []
        for table in allowed_tables:
            if table.upper() in query_upper:
                query_tables.append(table)

        if not query_tables:
            logger.warning("unauthorized_table_access", query=query[:100])
            return False

    return True


# ============================================================================
# Hashing and Caching Utilities
# ============================================================================

def compute_hash(data: Union[str, bytes, Dict]) -> str:
    """
    Compute SHA256 hash of data.

    Useful for caching and deduplication.

    Args:
        data: Data to hash (string, bytes, or dict)

    Returns:
        Hex digest of hash

    Example:
        >>> cache_key = compute_hash(prompt)
    """
    if isinstance(data, dict):
        # Sort keys for consistent hashing
        import json
        data = json.dumps(data, sort_keys=True)

    if isinstance(data, str):
        data = data.encode('utf-8')

    return hashlib.sha256(data).hexdigest()


def memoize_async(ttl_seconds: Optional[float] = None):
    """
    Memoization decorator for async functions with optional TTL.

    Args:
        ttl_seconds: Time-to-live for cached results (None = infinite)

    Returns:
        Decorated function with memoization

    Example:
        >>> @memoize_async(ttl_seconds=60)
        ... async def expensive_computation(x: int):
        ...     await asyncio.sleep(1)
        ...     return x * 2
    """
    cache: Dict[str, tuple[Any, float]] = {}

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Create cache key from arguments
            key = compute_hash(f"{func.__name__}:{args}:{kwargs}")

            # Check cache
            if key in cache:
                result, timestamp = cache[key]
                if ttl_seconds is None or (time.time() - timestamp) < ttl_seconds:
                    logger.debug("cache_hit", function=func.__name__)
                    return result

            # Compute result
            result = await func(*args, **kwargs)

            # Store in cache
            cache[key] = (result, time.time())
            logger.debug("cache_miss", function=func.__name__)

            return result

        # Add cache clearing method
        wrapper.clear_cache = lambda: cache.clear()

        return wrapper

    return decorator


# ============================================================================
# Token Counting Utilities
# ============================================================================

def estimate_tokens(text: str, model: str = "claude") -> int:
    """
    Estimate token count for text.

    This is a rough estimate. For accurate counts, use tiktoken.

    Args:
        text: Text to count tokens for
        model: Model type (claude/gpt)

    Returns:
        Estimated token count

    Example:
        >>> tokens = estimate_tokens(prompt)
    """
    # Rough estimate: ~4 characters per token for English
    # Claude and GPT have similar tokenization
    return len(text) // 4


def truncate_to_tokens(
    text: str,
    max_tokens: int,
    model: str = "claude"
) -> str:
    """
    Truncate text to fit within token limit.

    Args:
        text: Text to truncate
        max_tokens: Maximum token count
        model: Model type

    Returns:
        Truncated text

    Example:
        >>> truncated = truncate_to_tokens(long_text, max_tokens=1000)
    """
    estimated = estimate_tokens(text, model)

    if estimated <= max_tokens:
        return text

    # Calculate how much to keep (with safety margin)
    keep_ratio = (max_tokens / estimated) * 0.95
    keep_chars = int(len(text) * keep_ratio)

    truncated = text[:keep_chars]
    logger.warning(
        "text_truncated",
        original_tokens=estimated,
        max_tokens=max_tokens,
        original_chars=len(text),
        kept_chars=keep_chars
    )

    return truncated


# ============================================================================
# Formatting Utilities
# ============================================================================

def format_cost(cost_usd: float) -> str:
    """
    Format cost in USD for display.

    Args:
        cost_usd: Cost in USD

    Returns:
        Formatted string

    Example:
        >>> print(format_cost(0.0123))
        $0.0123
    """
    return f"${cost_usd:.4f}"


def format_duration(duration_ms: float) -> str:
    """
    Format duration for display.

    Args:
        duration_ms: Duration in milliseconds

    Returns:
        Human-readable duration

    Example:
        >>> print(format_duration(1234.5))
        1.23s
    """
    if duration_ms < 1000:
        return f"{duration_ms:.0f}ms"
    else:
        return f"{duration_ms / 1000:.2f}s"


def format_tokens(token_count: int) -> str:
    """
    Format token count for display.

    Args:
        token_count: Number of tokens

    Returns:
        Formatted string with thousands separator

    Example:
        >>> print(format_tokens(12345))
        12,345 tokens
    """
    return f"{token_count:,} tokens"


# ============================================================================
# Batch Processing Utilities
# ============================================================================

def batch_items(items: List[T], batch_size: int) -> List[List[T]]:
    """
    Split items into batches.

    Args:
        items: List of items to batch
        batch_size: Size of each batch

    Returns:
        List of batches

    Example:
        >>> for batch in batch_items(items, batch_size=10):
        ...     process_batch(batch)
    """
    return [
        items[i:i + batch_size]
        for i in range(0, len(items), batch_size)
    ]


async def process_batches(
    items: List[T],
    processor: Callable[[T], Any],
    batch_size: int = 10,
    concurrency: int = 5
) -> List[Any]:
    """
    Process items in batches with concurrency control.

    Args:
        items: Items to process
        processor: Async function to process each item
        batch_size: Items per batch
        concurrency: Maximum concurrent batches

    Returns:
        List of results

    Example:
        >>> results = await process_batches(
        ...     items,
        ...     processor=classify_item,
        ...     batch_size=10,
        ...     concurrency=3
        ... )
    """
    batches = batch_items(items, batch_size)

    async def process_batch(batch: List[T]) -> List[Any]:
        return await asyncio.gather(*[processor(item) for item in batch])

    batch_results = await gather_with_concurrency(
        concurrency,
        *[process_batch(batch) for batch in batches]
    )

    # Flatten results
    return [result for batch_result in batch_results for result in batch_result]
