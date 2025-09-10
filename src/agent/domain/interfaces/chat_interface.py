from abc import ABC, abstractmethod
from typing import Any, Dict, List

from langchain_core.messages import BaseMessage


class BaseChat(ABC):
    """Abstract interface for chat operations with LLMs in infrastructure layer."""

    @abstractmethod
    async def chat(self, messages: List[BaseMessage]) -> Dict[str, Any]:
        """
        Send a prompt to the LLM and return the response.

        Args:
            prompt: The input prompt to send to the LLM.

        Returns:
            Dict containing:
                - response: The generated text response
                - model: The model used for generation
                - usage: Token usage statistics
        """
        pass
