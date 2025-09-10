from typing import List, Optional

from langchain_aws import BedrockEmbeddings
from langchain_core.exceptions import LangChainException

from ...domain.interfaces.embedding_interface import BaseEmbeddings
from ...shared.logging.logger import Logger
from ...shared.settings.settings import settings

logger = Logger.get_logger(__name__)


class CohereV3Embedding(BaseEmbeddings):
    """AWS Bedrock Cohere v3 multilingual embeddings implementation using LangChain"""

    def __init__(
        self,
        model_id: Optional[str] = None,
        input_type: str = "search_document",
        embedding_type: str = "float",
    ):
        """
        Initialize Cohere v3 embeddings with AWS Bedrock via LangChain

        Args:
            model_id: Bedrock model ID for Cohere v3 (optional, will use settings if not provided)
            input_type: Type of input text (search_document, search_query, classification, clustering)
            embedding_type: Type of embedding output (float, int8, uint8, binary, ubinary)
        """

        self.model_id = model_id or settings.embeddings.cohere_model_id
        self.input_type = input_type or settings.embeddings.cohere_input_type
        self.embedding_type = (
            embedding_type or settings.embeddings.cohere_embedding_type
        )
        self.max_batch_size = settings.embeddings.cohere_max_batch_size

        try:
            # Initialize LangChain BedrockEmbeddings
            self.embeddings = BedrockEmbeddings(
                model_id=self.model_id,
                region_name=settings.s3.aws_region,
                credentials_profile_name=None,  # Use provided credentials
                aws_access_key_id=settings.s3.aws_access_key_id,
                aws_secret_access_key=settings.s3.aws_secret_access_key,
                model_kwargs={"input_type": self.input_type},
            )
            logger.info(
                f"Initialized Bedrock client with model: {self.model_id}"
            )
        except Exception as e:
            logger.error(f"Failed to initialize Bedrock client: {e}")
            raise

    async def embed_text(self, text: str) -> List[float]:
        """Generate embedding for single text with token validation"""

        try:
            validated_text = self._validate_token_limit(text)
            embeddings = await self.embeddings.aembed_query(validated_text)
            if not embeddings:
                raise ValueError("No embeddings returned from model")
            return embeddings
        except LangChainException as e:
            logger.error(f"Failed to generate embedding for text: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in embedding text: {e}")
            return []

    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts with token validation"""

        try:
            if not texts:
                return []

            validated_texts = [
                self._validate_token_limit(text) for text in texts
            ]
            max_batch_size = self.max_batch_size
            all_embeddings = []

            for i in range(0, len(validated_texts), max_batch_size):
                batch_texts = validated_texts[i : i + max_batch_size]
                batch_embeddings = await self.embeddings.aembed_documents(
                    batch_texts
                )
                all_embeddings.extend(batch_embeddings)

            return all_embeddings
        except LangChainException as e:
            logger.error(f"Failed to generate embeddings for documents: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error in embedding documents: {e}")
            return []

    def set_input_type(self, input_type: str):
        """Change input type for different use cases"""

        valid_types = [
            "search_document",
            "search_query",
            "classification",
            "clustering",
        ]
        if input_type not in valid_types:
            raise ValueError(
                f"Invalid input_type. Must be one of: {valid_types}"
            )
        self.input_type = input_type
        self.embeddings.model_kwargs["input_type"] = input_type
        logger.info(f"Input type changed to: {input_type}")

    def set_embedding_type(self, embedding_type: str):
        """Change embedding type"""

        valid_types = ["float", "int8", "uint8", "binary", "ubinary"]
        if embedding_type not in valid_types:
            raise ValueError(
                f"Invalid embedding_type. Must be one of: {valid_types}"
            )
        self.embedding_type = embedding_type
        logger.info(f"Embedding type changed to: {embedding_type}")

    def get_model_info(self) -> dict:
        """Get information about the current model configuration"""

        return {
            "model_id": self.model_id,
            "input_type": self.input_type,
            "embedding_type": self.embedding_type,
            "max_batch_size": self.max_batch_size,
            "max_input_length": 512_000,
        }

    def _validate_token_limit(self, text: str) -> str:
        """Ensure text doesn't exceed Cohere token limit"""

        max_chars = 2000 * 1.5
        if len(text) <= max_chars:
            return text
        truncated = text[: int(max_chars)]
        for delimiter in [". ", ".\n", "! ", "? ", "\n\n", "\n"]:
            last_pos = truncated.rfind(delimiter)
            if last_pos > len(truncated) * 0.8:
                return truncated[: last_pos + len(delimiter)]
        return truncated
