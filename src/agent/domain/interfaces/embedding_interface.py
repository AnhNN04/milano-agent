from abc import ABC, abstractmethod
from typing import List


class BaseEmbeddings(ABC):
    """Base class for embeddings"""

    @abstractmethod
    async def embed_text(self, text: str) -> List[float]:
        """Generate embeddings for text"""
        pass

    @abstractmethod
    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts"""
        pass
