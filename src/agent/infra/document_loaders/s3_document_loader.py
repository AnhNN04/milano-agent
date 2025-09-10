import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional

import boto3
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
)
from langchain_aws import BedrockEmbeddings
from langchain_community.document_loaders import (
    S3DirectoryLoader,
    S3FileLoader,
)
from langchain_experimental.text_splitter import SemanticChunker

from ...domain.entities.document import DocumentChunk, DocumentMetadata
from ...domain.interfaces.document_loader_interface import BaseDocumentLoader
from ...shared.exceptions.domain_exceptions import DocumentProcessingError
from ...shared.logging.logger import Logger
from ...shared.settings.settings import settings

logger = Logger.get_logger(__name__)


class S3DocumentLoader(BaseDocumentLoader):
    """Load and process documents from S3 with advanced chunking strategies using LangChain."""

    def __init__(self):
        self.bucket_name = settings.s3.bucket_name
        self.documents_prefix = settings.s3.documents_prefix
        self.supported_extensions = {".pdf", ".docx", ".txt", ".doc"}
        self.s3_client = boto3.client(
            "s3",
            aws_access_key_id=settings.s3.aws_access_key_id,
            aws_secret_access_key=settings.s3.aws_secret_access_key,
            region_name=settings.s3.aws_region,
        )
        self.timeout_seconds = getattr(settings.s3, "timeout_seconds", 10)
        self.chunk_strategies = self._initialize_chunk_strategies()

    def _initialize_chunk_strategies(self) -> Dict[str, Any]:
        """Initialize chunking strategies using LangChain."""

        chunk_size = settings.embeddings.chunk_size
        chunk_overlap = settings.embeddings.chunk_overlap

        strategies = {
            "recursive": RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len,
                separators=["\n\n", "\n", ". ", " ", ""],
            ),
            "token": TokenTextSplitter(
                chunk_size=int(chunk_size * 0.8),
                chunk_overlap=int(chunk_overlap * 0.8),
                encoding_name="cl100k_base",
            ),
        }

        logger.info("RecursiveCharacterTextSplitter initialized")
        logger.info("TokenTextSplitter initialized")

        try:
            embeddings = BedrockEmbeddings(
                model_id=settings.embeddings.cohere_model_id,
                region_name=settings.s3.aws_region,
                aws_access_key_id=settings.s3.aws_access_key_id,
                aws_secret_access_key=settings.s3.aws_secret_access_key,
                model_kwargs={"input_type": "search_document"},
            )
            strategies["semantic"] = SemanticChunker(
                embeddings,
                breakpoint_threshold_type="percentile",
                breakpoint_threshold_amount=75,
                min_chunk_size=int(chunk_size * 0.5),
            )
            logger.info(
                "Semantic splitter initialized with Bedrock Cohere embeddings"
            )
        except Exception as e:
            logger.warning(f"Semantic splitter initialization failed: {e}")

        return strategies

    async def load_and_chunk_document(
        self, source: str
    ) -> List[DocumentChunk]:
        """
        Load and chunk a single document from S3 using all chunking strategies.

        Args:
            source: S3 key of the document.

        Returns:
            List of DocumentChunk objects from all chunking strategies.
        """
        if not self._is_supported_document(source):
            raise DocumentProcessingError(
                f"Unsupported file type for {source}"
            )

        try:
            loader = S3FileLoader(
                bucket=self.bucket_name,
                key=source,
                aws_access_key_id=settings.s3.aws_access_key_id,
                aws_secret_access_key=settings.s3.aws_secret_access_key,
                region_name=settings.s3.aws_region,
            )
            documents = await asyncio.to_thread(loader.load)

            if not documents:
                raise DocumentProcessingError(
                    f"No content loaded from {source}"
                )

            # Clean text
            for doc in documents:
                doc.page_content = self._clean_text(doc.page_content)

            # Apply all chunking strategies
            result = []
            # --- Trong load_and_chunk_document ---
            for strategy_name, strategy in self.chunk_strategies.items():
                try:
                    chunks = strategy.split_documents(documents)
                    for i, chunk in enumerate(chunks):

                        # Tính toán start và end char
                        start_char = chunk.metadata.get("start_index", 0)
                        end_char = start_char + len(chunk.page_content)

                        metadata = DocumentMetadata(
                            source=source,
                            title=self._extract_title_from_path(source),
                            document_type=self._extract_document_type_from_path(
                                source
                            ),
                            tags=self._extract_tags_from_path(source)
                            + [f"chunk_strategy:{strategy_name}"],
                            language="vi",
                            chunk_index=f"{source}_{strategy_name}_{i}",
                            start_char=start_char,
                            end_char=end_char,
                        )

                        result.append(
                            DocumentChunk(
                                content=chunk.page_content[:2047],
                                metadata=metadata,
                            )
                        )

                    logger.info(
                        f"Applied {strategy_name} chunking strategy to {source}, produced {len(chunks)} chunks"
                    )
                except Exception as e:
                    logger.warning(
                        f"Failed to apply {strategy_name} chunking strategy to {source}: {str(e)}"
                    )
                    continue

            if not result:
                raise DocumentProcessingError(
                    f"No chunks produced for {source} with any strategy"
                )

            return result

        except Exception as e:
            logger.error(
                f"Failed to load and chunk document {source}: {str(e)}"
            )
            raise DocumentProcessingError(
                f"Failed to load and chunk document: {str(e)}"
            )

    async def load_all_documents(self) -> List[DocumentChunk]:
        """
        Load and chunk all supported documents from S3 using all chunking strategies.

        Returns:
            List of DocumentChunk objects from all documents and chunking strategies.
        """
        try:
            loader = S3DirectoryLoader(
                bucket=self.bucket_name,
                prefix=self.documents_prefix,
                aws_access_key_id=settings.s3.aws_access_key_id,
                aws_secret_access_key=settings.s3.aws_secret_access_key,
                region_name=settings.s3.aws_region,
                file_filter=lambda x: self._is_supported_document(x),
            )
            documents = await asyncio.to_thread(loader.load)

            if not documents:
                logger.warning(
                    f"No documents found in S3 bucket {self.bucket_name} with prefix {self.documents_prefix}"
                )
                return []

            # Clean text
            for doc in documents:
                doc.page_content = self._clean_text(doc.page_content)

            # Apply all chunking strategies
            result = []
            # --- Trong load_all_documents ---
            for strategy_name, strategy in self.chunk_strategies.items():
                try:
                    chunks = strategy.split_documents(documents)
                    for i, chunk in enumerate(chunks):

                        start_char = chunk.metadata.get("start_index", 0)
                        end_char = start_char + len(chunk.page_content)

                        source = chunk.metadata.get("source", "")
                        metadata = DocumentMetadata(
                            source=source,
                            title=self._extract_title_from_path(source),
                            document_type=self._extract_document_type_from_path(
                                source
                            ),
                            tags=self._extract_tags_from_path(source)
                            + [f"chunk_strategy:{strategy_name}"],
                            language="vi",
                            chunk_index=f"{source}_{strategy_name}_{i}",
                            start_char=start_char,
                            end_char=end_char,
                        )

                        result.append(
                            DocumentChunk(
                                content=chunk.page_content[:2047],
                                metadata=metadata,
                            )
                        )

                    logger.info(
                        f"Applied {strategy_name} chunking strategy to all documents, produced {len(chunks)} chunks"
                    )
                except Exception as e:
                    logger.warning(
                        f"Failed to apply {strategy_name} chunking strategy to documents: {str(e)}"
                    )
                    continue

            if not result:
                logger.warning(
                    f"No chunks produced for any documents with any strategy"
                )
                return []

            return result

        except Exception as e:
            logger.error(f"Failed to load all documents: {str(e)}")
            raise DocumentProcessingError(
                f"Failed to load all documents: {str(e)}"
            )

    async def list_documents(
        self, prefix: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        List all supported documents in S3 bucket with optional prefix.
        Args:
            prefix: Optional prefix to filter documents (e.g., folder path).
        Returns:
            List of dictionaries containing document metadata (key, size, last_modified, content_type, etag).
        """
        try:
            # Use documents_prefix if no prefix is provided
            effective_prefix = (
                prefix if prefix is not None else self.documents_prefix
            )
            if not effective_prefix.endswith("/"):
                effective_prefix += "/"

            # Initialize pagination
            documents = []
            continuation_token = None

            while True:
                # Prepare parameters for list_objects_v2
                params = {
                    "Bucket": self.bucket_name,
                    "Prefix": effective_prefix,
                }
                if continuation_token:
                    params["ContinuationToken"] = continuation_token

                # List objects using boto3
                response = await asyncio.to_thread(
                    self.s3_client.list_objects_v2, **params
                )

                # Process objects
                if "Contents" in response:
                    for obj in response["Contents"]:
                        key = obj["Key"]
                        if self._is_supported_document(key):
                            documents.append(
                                {
                                    "key": key,
                                    "size": obj["Size"],
                                    "last_modified": obj["LastModified"],
                                    "content_type": key.split(".")[-1],
                                    "etag": obj["ETag"].strip('"'),
                                }
                            )

                # Check for pagination
                if response.get("IsTruncated", False):
                    continuation_token = response.get("NextContinuationToken")
                else:
                    break

            if not documents:
                logger.info(
                    f"No supported documents found in S3 bucket {self.bucket_name} with prefix {effective_prefix}"
                )
                return []

            logger.info(
                f"Found {len(documents)} supported documents in S3 bucket {self.bucket_name} with prefix {effective_prefix}"
            )
            return documents
        except Exception as e:
            logger.error(
                f"Failed to list documents with prefix {prefix}: {str(e)}"
            )
            raise DocumentProcessingError(
                f"Failed to list documents: {str(e)}"
            )

    def _clean_text(self, text: str) -> str:
        """Clean and normalize text."""

        import re

        text = re.sub(r"\n\s*\n", "\n\n", text)
        text = re.sub(r" +", " ", text)
        text = re.sub(r"\t+", " ", text)
        text = re.sub(r"(.)\1{20,}", "", text)
        return text.strip()

    def _extract_title_from_path(self, s3_key: str) -> str:
        """Extract a readable title from S3 key."""

        filename = Path(s3_key).stem
        title = filename.replace("_", " ").replace("-", " ")
        return " ".join(word.capitalize() for word in title.split())

    def _extract_document_type_from_path(self, s3_key: str) -> str:
        """Extract a document type from S3 key."""
        return s3_key.split(".")[-1]

    def _extract_tags_from_path(self, s3_key: str) -> List[str]:
        """Extract tags from S3 path structure."""

        tags = []
        path_parts = s3_key.split("/")
        for part in path_parts[:-1]:
            if part and part != self.documents_prefix.rstrip("/"):
                tags.append(part.replace("_", " ").replace("-", " ").title())
        ext = Path(s3_key).suffix.lower()
        if ext:
            tags.append(ext[1:].upper())
        return tags

    def _is_supported_document(self, key: str) -> bool:
        """Check if document type is supported."""

        if key.endswith("/"):
            return False
        return Path(key).suffix.lower() in self.supported_extensions

    async def get_document_info(self, s3_key: str) -> Dict[str, Any]:
        """Get information about a specific document."""

        try:
            response = await asyncio.to_thread(
                self.s3_client.head_object, Bucket=self.bucket_name, Key=s3_key
            )
            return {
                "key": s3_key,
                "size": response["ContentLength"],
                "last_modified": response["LastModified"],
                "content_type": response.get("ContentType", "unknown"),
                "etag": response["ETag"].strip('"'),
            }
        except Exception as e:
            logger.error(f"Failed to get document info for {s3_key}: {str(e)}")
            raise DocumentProcessingError(
                f"Failed to get document info: {str(e)}"
            )

    async def check_document_exists(self, s3_key: str) -> bool:
        """Check if document exists in S3."""

        try:
            await asyncio.to_thread(
                self.s3_client.head_object, Bucket=self.bucket_name, Key=s3_key
            )
            return True
        except self.s3_client.exceptions.NoSuchKey:
            return False
        except Exception as e:
            logger.error(
                f"Error checking document existence {s3_key}: {str(e)}"
            )
            return False
