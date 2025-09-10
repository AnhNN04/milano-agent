from typing import Any, Dict, List, Type

from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field, PrivateAttr

from ...domain.interfaces.chat_interface import BaseChat
from ...infra.utils.query_enhancer import QueryEnhancer
from ...shared.logging.logger import Logger
from .base import CustomBaseTool

logger = Logger.get_logger(__name__)


class ChatToolInput(BaseModel):
    """Input schema for ChatTool, compatible with LangChain's BaseMessage."""

    messages: List[BaseMessage] = Field(
        ...,
        description="List of messages for the LLM, typically including system and user messages.",
    )


class ChatToolOutput(BaseModel):
    """Output schema for ChatTool, including response and metadata."""

    response: str = Field(description="The LLM's response")


class ChatTool(CustomBaseTool):
    """Tool for interacting with an LLM to answer queries about the Vietnamese stock market."""

    name: str = "chat_llm"
    description: str = (
        "Interact with an LLM to answer queries about Vietnamese stocks, market news, and financial data"
    )
    args_schema: Type[BaseModel] = ChatToolInput

    _chat_provider: PrivateAttr = PrivateAttr()
    _query_enhancer: PrivateAttr = PrivateAttr()

    def __init__(self, chat_provider: BaseChat):
        super().__init__()
        self._chat_provider = chat_provider
        self._query_enhancer = QueryEnhancer()

    async def _execute_impl(
        self, messages: List[BaseMessage]
    ) -> Dict[str, Any]:
        """Execute LLM chat using the injected provider with structured messages."""

        # enhanced_messages = self._query_enhancer.enhance_messages(messages)
        enhanced_messages = messages
        try:
            logger.info(
                f"Using chat provider with messages: {enhanced_messages}"
            )
            chat_response = await self._chat_provider.chat(
                messages=enhanced_messages
            )
            sanitized_response = self._apply_content_filter(
                chat_response.get("response", "")
            )

            return ChatToolOutput(response=sanitized_response).model_dump()

        except Exception as e:
            logger.error(f"Chat execution failed: {str(e)}")

            return ChatToolOutput(
                response=""
            ).model_dump()  # Consistent return type

    def _apply_content_filter(self, text: str) -> str:
        """Apply basic content filtering to ensure safe and appropriate content."""

        sensitive_terms = ["inappropriate", "offensive", "harmful"]

        for term in sensitive_terms:
            if term.lower() in text.lower():
                return "Kết quả đã được lọc do chứa nội dung không phù hợp."

        return text

    def to_formatted_context(self, output: Dict[str, Any]) -> str:
        """Format chat results for LLM context."""

        return (
            f"LLM Response: {output.get('response', 'No response available')}"
        )
