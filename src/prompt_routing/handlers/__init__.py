"""
Handlers for prompt routing.

This package contains all concrete handler implementations
for processing different intent types.
"""

from .base import Handler, BaseHandler
from .direct_response import DirectResponseHandler
from .database_query import DatabaseQueryHandler
from .knowledge_search import KnowledgeSearchHandler

__all__ = [
    "Handler",
    "BaseHandler",
    "DirectResponseHandler",
    "DatabaseQueryHandler",
    "KnowledgeSearchHandler",
]
