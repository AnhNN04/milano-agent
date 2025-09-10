from typing import Any, Dict, List

from langchain_core.exceptions import LangChainException
from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI

from ...domain.interfaces.chat_interface import BaseChat
from ...shared.logging.logger import Logger
from ...shared.settings.settings import settings

logger = Logger.get_logger(__name__)


class OpenAIChat(BaseChat):
    """
    Infrastructure-specific implementation of ChatProvider using LangChain's ChatOpenAI.
    """

    def __init__(self):
        # Initialize the LangChain chat client
        self.client = ChatOpenAI(
            api_key=settings.app.openai_api_key,
            model=settings.llm.openai_model,
            temperature=settings.llm.openai_temperature,
            max_tokens=settings.llm.openai_max_tokens,
            timeout=settings.llm.openai_timeout,
            max_retries=settings.llm.openai_max_retries,
            verbose=settings.llm.verbose,
            model_kwargs={
                "top_p": settings.llm.openai_top_p,
                "frequency_penalty": settings.llm.openai_frequency_penalty,
                "presence_penalty": settings.llm.openai_presence_penalty,
            },
        )
        self.streaming = getattr(settings.llm, "openai_streaming", False)
        self.callbacks = getattr(settings.llm, "openai_callbacks", [])

    async def chat(self, messages: List[BaseMessage]) -> Dict[str, Any]:
        """
        Sends a list of BaseMessages to OpenAI via LangChain.
        """
        try:
            if self.streaming:
                response_text = ""
                for chunk in self.client.stream(
                    messages, config={"callbacks": self.callbacks}
                ):
                    response_text += chunk.content
                response_content = response_text
                usage = {}
            else:
                response = await self.client.ainvoke(
                    messages, config={"callbacks": self.callbacks}
                )
                response_content = response.content
                usage = response.response_metadata.get(
                    "token_usage",
                    {
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0,
                    },
                )

            return {
                "response": response_content,
                "model": settings.llm.openai_model,
                "usage": {
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0),
                },
            }

        except LangChainException as e:
            logger.error(f"OpenAI chat failed: {str(e)}")
            return {}
        except Exception as e:
            logger.error(f"Unexpected error in OpenAI chat: {str(e)}")
            return {}
