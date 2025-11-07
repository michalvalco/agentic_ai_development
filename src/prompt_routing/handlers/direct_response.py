"""
Direct response handler for simple queries.

This handler processes queries that can be answered directly
without needing database access, API calls, or complex analysis.

Use cases:
- Greetings and small talk
- Simple factual questions
- Status queries
- Help requests
"""

from typing import Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from src.common.config import settings
from src.common.cost_tracker import cost_tracker
from src.common.logger import get_logger
from src.common.models import Context
from src.prompt_routing.handlers.base import BaseHandler
from src.prompt_routing.models import HandlerResponse, IntentClassification, IntentType

logger = get_logger(__name__)


class DirectResponseHandler(BaseHandler):
    """
    Handler for queries that need direct conversational responses.

    This handler uses an LLM to generate appropriate responses for
    simple queries, greetings, and clarification requests.

    Attributes:
        llm: Language model for generating responses
    """

    def __init__(self, llm: Optional[BaseChatModel] = None):
        """
        Initialize direct response handler.

        Args:
            llm: Language model (uses default if not provided)
        """
        super().__init__(
            name="direct_response",
            supported_intents=[
                IntentType.DIRECT_RESPONSE.value,
                IntentType.CLARIFICATION_NEEDED.value,
                IntentType.UNKNOWN.value,
            ]
        )
        self.llm = llm or self._create_default_llm()

    def _create_default_llm(self) -> BaseChatModel:
        """Create default LLM from settings."""
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=settings.default_model,
            anthropic_api_key=settings.anthropic_api_key,
            max_tokens=512,  # Shorter for direct responses
            temperature=0.7  # More conversational
        )

    async def _handle_impl(
        self,
        query: str,
        classification: IntentClassification,
        context: Optional[Context] = None,
        **kwargs
    ) -> HandlerResponse:
        """
        Generate direct response to query.

        Args:
            query: User query
            classification: Intent classification
            context: Execution context
            **kwargs: Additional parameters

        Returns:
            HandlerResponse with generated response
        """
        try:
            # Check if this is a clarification request
            if classification.requires_clarification:
                return await self._handle_clarification(query, classification, context)

            # Generate direct response
            system_prompt = """You are a helpful AI assistant.

Provide a clear, concise, and friendly response to the user's query.
Keep your response brief (2-3 sentences unless more detail is needed).
Be conversational but professional."""

            with cost_tracker.track("direct_response"):
                response = await self.llm.ainvoke([
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=query)
                ])

            result_text = response.content.strip()

            logger.info(
                "direct_response_generated",
                response_length=len(result_text),
                context=context.request_id if context else None
            )

            return HandlerResponse(
                success=True,
                result={"response": result_text, "type": "direct"},
                next_action=None
            )

        except Exception as e:
            logger.error(
                "direct_response_failed",
                error=str(e),
                context=context.request_id if context else None
            )

            return HandlerResponse(
                success=False,
                result=None,
                error=f"Failed to generate response: {e}"
            )

    async def _handle_clarification(
        self,
        query: str,
        classification: IntentClassification,
        context: Optional[Context] = None
    ) -> HandlerResponse:
        """
        Handle queries that need clarification.

        Args:
            query: Original query
            classification: Classification with clarification needs
            context: Execution context

        Returns:
            HandlerResponse with clarification questions
        """
        clarifications = classification.suggested_clarifications

        if clarifications:
            # Use suggested clarifications
            response_text = (
                f"I need some clarification to help you better. "
                f"Could you please:\n\n"
            )
            response_text += "\n".join([
                f"- {clarification}"
                for clarification in clarifications
            ])
        else:
            # Generate clarification request
            response_text = (
                "I'm not sure I understand your request. "
                "Could you provide more details or rephrase your question?"
            )

        logger.info(
            "clarification_requested",
            num_clarifications=len(clarifications),
            context=context.request_id if context else None
        )

        return HandlerResponse(
            success=True,
            result={
                "response": response_text,
                "type": "clarification",
                "clarifications": clarifications
            },
            requires_followup=True,
            next_action="await_clarification"
        )
