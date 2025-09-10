import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from typing import List

import torch
from langchain_core.exceptions import LangChainException
from langchain_huggingface import HuggingFaceEmbeddings

from ...domain.interfaces.embedding_interface import BaseEmbeddings
from ...shared.logging.logger import Logger

logger = Logger.get_logger(__name__)


class HfEmbedding(BaseEmbeddings):
    """Local embeddings implementation using HuggingFaceEmbeddings"""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        try:
            logger.info(f"Loading local embedding model: {model_name}")
            self.model_name = model_name
            self.timeout_seconds = 10
            self.expected_dimension = 384
            self.embeddings = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={
                    "device": "cuda" if torch.cuda.is_available() else "cpu"
                },
            )
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise

    async def embed_text(self, text: str) -> List[float]:
        """Generate embedding for single text"""

        try:
            embedding = await self.embeddings.aembed_query(text)
            return embedding
        except LangChainException as e:
            logger.error(f"Failed to generate embedding: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in embedding text: {e}")
            raise

    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts"""

        try:
            embeddings = await self.embeddings.aembed_documents(texts)
            return embeddings
        except LangChainException as e:
            logger.error(f"Failed to generate embeddings: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in embedding documents: {e}")
            raise
