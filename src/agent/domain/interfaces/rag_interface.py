import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List


class BaseRag(ABC):
    """Abstract interface for RAG retrieval operations in domain layer."""

    @abstractmethod
    async def retrieve(self, query: str) -> List[Dict[str, Any]]:
        """
        Retrieve similar documents from the knowledge base.

        Args:
            query: The query string to search for.

        Returns:
            List of dictionaries containing document content and metadata.
        """
        pass
