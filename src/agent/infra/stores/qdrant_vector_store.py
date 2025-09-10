import asyncio
from typing import Any, Dict, List

from langchain_aws import BedrockEmbeddings
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

from ...domain.entities.document import DocumentChunk
from ...domain.interfaces.vector_store_interface import BaseVectorStore
from ...shared.exceptions.domain_exceptions import VectorStoreError
from ...shared.logging.logger import Logger
from ...shared.settings.settings import settings

logger = Logger.get_logger(__name__)


class QdrantVectorStoreDB(BaseVectorStore):
    """Qdrant implementation of vector store using LangChain."""

    def __init__(
        self,
    ):

        self.embedding = BedrockEmbeddings(
            model_id=settings.embeddings.cohere_model_id,
            region_name=settings.s3.aws_region,
            aws_access_key_id=settings.s3.aws_access_key_id,
            aws_secret_access_key=settings.s3.aws_secret_access_key,
            model_kwargs={"input_type": "search_document"},
        )

        # Khởi tạo QdrantClient để quản lý collection
        self._raw_client = QdrantClient(
            host=settings.qdrant.host,
            port=settings.qdrant.port,
            # url=settings.qdrant.url,
            # api_key=settings.qdrant.api_key,
            timeout=getattr(settings.qdrant, "timeout_seconds", 5),
        )

        # Kiểm tra và tạo collection nếu chưa tồn tại
        collection_name = settings.qdrant.collection_name
        try:
            self._raw_client.get_collection(collection_name)
            logger.info(f"Collection {collection_name} already exists.")
        except Exception:
            logger.info(
                f"Creating new collection {collection_name} with vector_size {settings.qdrant.vector_size}."
            )
            self._raw_client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=settings.qdrant.vector_size,  # Thiết lập vector_size
                    distance=Distance.COSINE,
                ),
            )

        # Khởi tạo QdrantVectorStore
        self.client = QdrantVectorStore(
            client=self._raw_client,
            collection_name=collection_name,
            embedding=self.embedding,
            distance=Distance.COSINE,
        )
        self.collection_name = settings.qdrant.collection_name
        self.timeout_seconds = getattr(settings.qdrant, "timeout_seconds", 5)
        self.search_k = getattr(settings.qdrant, "search_k", 5)
        self.filter_conditions = getattr(
            settings.qdrant, "filter_conditions", None
        )

    async def add_documents(self, documents: List[DocumentChunk]) -> bool:
        """
        Add document chunks to Qdrant using LangChain.

        Args:
            documents: List of DocumentChunk objects to add.

        Returns:
            True if successful, False otherwise.
        """
        try:
            if not documents:
                logger.warning("No documents to add")
                return True

            # Convert DocumentChunk to LangChain Document
            langchain_docs = [
                Document(
                    page_content=doc.content,
                    metadata={
                        "source": doc.metadata.source,
                        "title": doc.metadata.title,
                        "document_type": doc.metadata.document_type,
                        "chunk_index": doc.metadata.chunk_index,
                        "tags": doc.metadata.tags,
                        "language": doc.metadata.language,
                    },
                )
                for doc in documents
            ]

            # Add documents using LangChain
            await asyncio.to_thread(self.client.add_documents, langchain_docs)
            logger.info(
                f"Successfully added {len(documents)} documents to Qdrant"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to add documents to Qdrant: {str(e)}")
            raise VectorStoreError(f"Failed to add documents: {str(e)}")

    async def delete_documents(self, document_ids: List[str]) -> bool:
        """
        Delete documents by IDs.

        Args:
            document_ids: List of document IDs to delete.

        Returns:
            True if successful, False otherwise.
        """
        try:
            # LangChain QdrantVectorStore uses document IDs in metadata
            await asyncio.to_thread(
                self.client.delete,
                filter={
                    "must": [
                        {"key": "source", "match": {"value": doc_id}}
                        for doc_id in document_ids
                    ]
                },
            )
            logger.info(f"Deleted {len(document_ids)} documents from Qdrant")
            return True

        except Exception as e:
            logger.error(f"Failed to delete documents: {str(e)}")
            raise VectorStoreError(f"Failed to delete documents: {str(e)}")

    async def similarity_search(self, query: str) -> List[Dict[str, Any]]:
        """
        Perform similarity search using LangChain.

        Args:
            query: The query string to search for.

        Returns:
            List of dictionaries containing document content and metadata.
        """
        try:
            # Perform search using LangChain
            results = await asyncio.to_thread(
                self.client.similarity_search_with_score,
                query=query,
                k=self.search_k,
                filter=self.filter_conditions,
            )

            # Convert results to expected format
            formatted_results = [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "score": score,
                }
                for doc, score in results
            ]

            logger.info(f"Found {len(formatted_results)} results for query")
            return formatted_results

        except Exception as e:
            logger.error(f"Similarity search failed: {str(e)}")
            raise VectorStoreError(f"Similarity search failed: {str(e)}")

    async def close(self) -> None:
        """
        Close Qdrant client connection.
        """
        try:
            await asyncio.to_thread(self._raw_client.close)
            logger.info("Qdrant client connection closed successfully.")
        except Exception as e:
            logger.error(f"Failed to close Qdrant client: {str(e)}")
            raise VectorStoreError(f"Failed to close Qdrant client: {str(e)}")
