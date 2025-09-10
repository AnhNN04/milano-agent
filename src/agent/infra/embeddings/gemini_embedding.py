from typing import List, Optional

from langchain_core.exceptions import LangChainException
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from ...domain.interfaces.embedding_interface import BaseEmbeddings
from ...shared.logging.logger import Logger
from ...shared.settings.settings import settings

logger = Logger.get_logger(__name__)


class GeminiEmbedding(BaseEmbeddings):
    """Google Gemini embeddings implementation using LangChain"""

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        """
        Initialize Gemini embeddings

        Args:
            model: Gemini embedding model name (optional, will use settings if not provided)
            api_key: Gemini API key (optional, will use settings if not provided)
        """

        self.model = model or settings.embeddings.model
        self.api_key = api_key or settings.app.gemini_api_key
        self.timeout_seconds = 5
        self.expected_dimensions = 768

        try:
            self.embeddings = GoogleGenerativeAIEmbeddings(
                model=self.model,
                google_api_key=self.api_key,
                task_type="retrieval_document",
            )
            logger.info(
                f"Initialized Gemini embeddings with model: {self.model}"
            )
        except Exception as e:
            logger.error(f"Failed to initialize Gemini embeddings: {e}")
            raise

    async def embed_text(self, text: str) -> List[float]:
        """Generate embedding for single text"""

        try:
            logger.debug(
                f"Generating embedding for single text of length: {len(text)}"
            )
            embedding = await self.embeddings.aembed_query(text)
            if not embedding:
                raise ValueError("No embeddings returned from model")
            return embedding
        except LangChainException as e:
            logger.error(f"Failed to generate embedding for text: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in embedding text: {e}")
            raise

    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts"""

        try:
            if not texts:
                return []

            logger.debug(f"Generating embeddings for {len(texts)} documents")
            max_batch_size = getattr(
                settings.embeddings, "cohere_max_batch_size", 96
            )
            all_embeddings = []

            for i in range(0, len(texts), max_batch_size):
                batch_texts = texts[i : i + max_batch_size]
                batch_embeddings = await self.embeddings.aembed_documents(
                    batch_texts
                )
                all_embeddings.extend(batch_embeddings)
                logger.debug(
                    f"Processed batch {i//max_batch_size + 1}, texts: {len(batch_texts)}"
                )

            return all_embeddings
        except LangChainException as e:
            logger.error(f"Failed to generate embeddings for documents: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in embedding documents: {e}")
            raise

    def get_model_info(self) -> dict:
        """Get information about the current model configuration"""

        return {
            "model": self.model,
            "max_batch_size": getattr(
                settings.embeddings, "cohere_max_batch_size", 96
            ),
            "max_input_length": 2048,
        }
