import time
from typing import Any, Dict, List, Optional

from ...domain.entities.document import DocumentChunk
from ...shared.exceptions.domain_exceptions import DocumentProcessingError
from ...shared.logging.logger import Logger

logger = Logger.get_logger(__name__)


class DocumentProcessingService:
    """Service for processing documents from S3 to vector store."""

    def __init__(self, document_loader, vector_store):
        """
        Args:
            document_loader: Implementation of BaseDocumentLoader
            vector_store: Implementation of BaseVectorStore
        """
        self.document_loader = document_loader
        self.vector_store = vector_store

    async def process_documents(
        self, s3_keys: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Process documents from S3 and add to vector store."""
        start_time = time.time()

        try:
            if not s3_keys:
                all_chunks = await self._process_all_documents()
            else:
                all_chunks = await self._process_specific_documents(s3_keys)

            if not all_chunks:
                logger.warning("No document chunks were processed")
                return self._create_result(0, 0, [], time.time() - start_time)

            # Add to vector store
            success = await self.vector_store.add_documents(all_chunks)
            if not success:
                raise DocumentProcessingError(
                    "Failed to add documents to vector store"
                )

            processed_docs = len(
                set(chunk.metadata.source for chunk in all_chunks)
            )
            logger.info(
                f"Successfully processed {processed_docs} documents with {len(all_chunks)} chunks"
            )

            return self._create_result(
                processed_docs, len(all_chunks), [], time.time() - start_time
            )

        except Exception as e:
            logger.error(f"Document processing failed: {str(e)}")
            raise DocumentProcessingError(
                f"Document processing failed: {str(e)}"
            )

    async def _process_all_documents(self) -> List[DocumentChunk]:
        """Process all available documents."""
        return await self.document_loader.load_all_documents()

    async def _process_specific_documents(
        self, s3_keys: List[str]
    ) -> List[DocumentChunk]:
        """Process specific documents by S3 keys."""
        all_chunks = []

        for s3_key in s3_keys:
            try:
                chunks = await self.document_loader.load_and_chunk_document(
                    s3_key
                )
                all_chunks.extend(chunks)
            except Exception as e:
                logger.error(f"Failed to process document {s3_key}: {str(e)}")
                continue

        return all_chunks

    def _create_result(
        self,
        processed_docs: int,
        total_chunks: int,
        failed_docs: List[str],
        processing_time: float,
    ) -> Dict[str, Any]:
        """Create standardized result dictionary."""
        return {
            "processed_documents": processed_docs,
            "total_chunks": total_chunks,
            "failed_documents": failed_docs,
            "processing_time": processing_time,
        }

    async def list_available_documents(
        self, prefix: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """List all available documents."""
        try:
            return await self.document_loader.list_documents(prefix)
        except Exception as e:
            logger.error(f"Failed to list documents: {str(e)}")
            raise DocumentProcessingError(
                f"Failed to list documents: {str(e)}"
            )
