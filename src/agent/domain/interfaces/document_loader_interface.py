from abc import ABC, abstractmethod
from typing import List

from ...domain.entities.document import DocumentChunk


class BaseDocumentLoader(ABC):
    """Base class for document loaders"""

    @abstractmethod
    async def load_and_chunk_document(
        self, source: str
    ) -> List[DocumentChunk]:
        """
        Load and chunk a single document from the given source.

        Args:
            source: Identifier for the document (e.g., S3 key, file path).

        Returns:
            List of DocumentChunk objects containing document content and metadata.
        """
        pass

    @abstractmethod
    async def load_all_documents(self) -> List[DocumentChunk]:
        """
        Load and chunk all available documents.

        Returns:
            List of DocumentChunk objects from all documents.
        """
        pass
