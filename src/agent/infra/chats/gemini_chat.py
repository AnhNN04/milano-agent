from typing import Any, Dict, List

from langchain_core.exceptions import LangChainException
from langchain_core.messages import BaseMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from ...domain.interfaces.chat_interface import BaseChat
from ...shared.logging.logger import Logger
from ...shared.settings.settings import settings

logger = Logger.get_logger(__name__)


class GeminiChat(BaseChat):
    """
    Infrastructure-specific implementation of ChatProvider using LangChain's ChatGoogleGenerativeAI.
    """

    def __init__(self):
        # Initialize the LangChain chat client
        self.client = ChatGoogleGenerativeAI(
            api_key=settings.app.gemini_api_key,
            model=settings.llm.gemini_model,
            temperature=settings.llm.gemini_temperature,
            max_output_tokens=settings.llm.gemini_max_tokens,
            top_p=settings.llm.gemini_top_p,
            top_k=settings.llm.gemini_top_k,
            timeout=settings.llm.gemini_timeout,
            max_retries=settings.llm.gemini_max_retries,
            verbose=settings.llm.verbose,
        )
        self.streaming = getattr(settings.llm, "gemini_streaming", False)
        self.callbacks = getattr(settings.llm, "gemini_callbacks", [])

    async def chat(self, messages: List[BaseMessage]) -> Dict[str, Any]:
        """
        Sends a list of BaseMessages to Gemini via LangChain.
        """
        try:
            if self.streaming:
                response_text = ""
                for chunk in self.client.stream(
                    messages, config={"callbacks": self.callbacks}
                ):
                    response_text += chunk.content
                response_content = response_text
            else:
                response = await self.client.ainvoke(
                    messages, config={"callbacks": self.callbacks}
                )
                response_content = response.content
            return {
                "response": response_content,
            }

        except LangChainException as e:
            logger.error(f"Gemini chat failed: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in Gemini chat: {str(e)}")
            raise
