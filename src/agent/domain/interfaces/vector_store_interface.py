# src/stock_assistant/infrastructure/vector_stores/base.py
from abc import ABC, abstractmethod
from typing import Any, Dict, List

from ...domain.entities.document import DocumentChunk


class BaseVectorStore(ABC):
    """Base class for vector store implementations"""

    @abstractmethod
    async def add_documents(self, documents: List[DocumentChunk]) -> bool:
        """
        Add documents to vector store.

        Args:
            documents: List of DocumentChunk objects to add.

        Returns:
            True if successful, False otherwise.
        """
        pass

    @abstractmethod
    async def delete_documents(self, document_ids: List[str]) -> bool:
        """
        Delete documents by IDs.

        Args:
            document_ids: List of document IDs to delete.

        Returns:
            True if successful, False otherwise.
        """
        pass

    @abstractmethod
    async def similarity_search(self, query: str) -> List[Dict[str, Any]]:
        """
        Perform similarity search.

        Args:
            query: The query string to search for.

        Returns:
            List of dictionaries containing document content and metadata.
        """
        pass
