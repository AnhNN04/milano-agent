from typing import Any, Dict, List

from ...domain.interfaces.rag_interface import BaseRag
from ...domain.interfaces.vector_store_interface import BaseVectorStore
from ...shared.logging.logger import Logger
from ...shared.settings.settings import settings

logger = Logger.get_logger(__name__)


class QdrantRag(BaseRag):
    """Infrastructure-specific implementation of RagProvider using QdrantVectorStore."""

    def __init__(self, vector_store: BaseVectorStore):
        self.vector_store = vector_store
        self.max_results = getattr(settings.qdrant, "max_results", 5)

    async def retrieve(self, query: str) -> List[Dict[str, Any]]:
        """
        Retrieve similar documents using Qdrant vector store.

        Args:
            query: The query string to search for.

        Returns:
            List of dictionaries containing document content and metadata.
        """

        try:
            # Use vector store's similarity search
            results = await self.vector_store.similarity_search(query)

            logger.info(f"Retrieved {len(results)} documents for query")

            return results

        except Exception as e:
            logger.error(f"RAG retrieval failed: {str(e)}")
            return []
