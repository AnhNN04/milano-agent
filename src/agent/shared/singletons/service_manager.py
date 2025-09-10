import asyncio
import time
from typing import Any, Dict, Optional

from ...domain.interfaces.chat_interface import BaseChat
from ...domain.interfaces.embedding_interface import BaseEmbeddings
from ...domain.interfaces.search_interface import BaseWebSearch
from ...domain.interfaces.stock_analysis_interface import BaseStockAnalysis
from ...domain.interfaces.stock_data_interface import BaseStockData
from ...domain.interfaces.vector_store_interface import BaseVectorStore
from ...infra.chats.gemini_chat import GeminiChat
from ...infra.chats.openai_chat import OpenAIChat
from ...infra.embeddings.cohere_multilingual_v3_embedding import (
    CohereV3Embedding,
)
from ...infra.embeddings.gemini_embedding import GeminiEmbedding
from ...infra.embeddings.huggingface_embedding import HfEmbedding
from ...infra.providers.tavily_search_provider import TavilyWebSearch
from ...infra.providers.vnstock_analysis_provider import VnStockAnalysis
from ...infra.providers.vnstock_data_provider import VnStockData
from ...infra.stores.qdrant_vector_store import QdrantVectorStoreDB
from ...shared.logging.logger import Logger
from ...shared.settings.settings import settings

logger = Logger.get_logger(__name__)


class ServiceManager:
    """Singleton manager for heavy resources with eager initialization"""

    _instance: Optional["ServiceManager"] = None
    _lock = asyncio.Lock()

    @classmethod
    async def get_instance(cls) -> "ServiceManager":
        """Get or create the singleton instance with thread safety"""
        async with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
                await cls._instance._initialize_services()
                logger.info("ServiceManager singleton instance created")
            else:
                logger.debug(
                    "Returning existing ServiceManager singleton instance"
                )
            return cls._instance

    @classmethod
    def reset_instance(cls):
        """Reset singleton instance (for testing)"""
        cls._instance = None
        logger.info("ServiceManager singleton instance reset")

    def __init__(self):
        """Initialize ServiceManager with instance variables"""
        if ServiceManager._instance is not None:
            logger.warning(
                "Attempt to create another ServiceManager instance - using singleton"
            )
            return
        self._embeddings: Dict[str, BaseEmbeddings] = {}
        self._chat_providers: Dict[str, BaseChat] = {}
        self._vector_store: Optional[BaseVectorStore] = None
        self._search_provider: Optional[BaseWebSearch] = None
        self._stock_data_provider: Optional[BaseStockData] = None
        self._stock_analysis_provider: Optional[BaseStockAnalysis] = None
        self._stats = {
            "embeddings_initialized": False,
            "chat_providers_initialized": False,
            "vector_store_initialized": False,
            "search_provider_initialized": False,
            "stock_providers_initialized": False,
            "initialization_time": 0.0,
        }

    async def _initialize_services(self):
        """Initialize all services eagerly during startup"""
        start_time = time.time()
        logger.info("Starting eager initialization of all services")

        try:
            # Initialize embeddings
            default_embedding_provider = getattr(
                settings.embeddings, "default_provider", "cohere"
            )
            self._embeddings = {
                default_embedding_provider: {
                    "cohere": CohereV3Embedding,
                    "gemini": GeminiEmbedding,
                    "hf": HfEmbedding,
                }[default_embedding_provider]()
            }
            self._stats["embeddings_initialized"] = True
            logger.info(
                f"Default embedding provider {default_embedding_provider} initialized"
            )

            # Initialize vector store
            self._vector_store = QdrantVectorStoreDB()
            self._stats["vector_store_initialized"] = True
            logger.info("QdrantVectorStoreDB initialized")

            # Initialize chat providers
            default_chat_provider = getattr(
                settings.llm, "default_provider", "openai"
            )
            self._chat_providers = {
                default_chat_provider: {
                    "openai": OpenAIChat,
                    "gemini": GeminiChat,
                }[default_chat_provider]()
            }
            self._stats["chat_providers_initialized"] = True
            logger.info(
                f"Default chat provider {default_chat_provider} initialized"
            )

            # Initialize search provider
            self._search_provider = TavilyWebSearch()
            self._stats["search_provider_initialized"] = True
            logger.info("TavilyWebSearch initialized")

            # Initialize stock providers
            self._stock_data_provider = VnStockData()
            self._stock_analysis_provider = VnStockAnalysis()
            self._stats["stock_providers_initialized"] = True
            logger.info(
                "Stock providers (VnStockData, VnStockAnalysis) initialized"
            )

            self._stats["initialization_time"] = time.time() - start_time
            logger.info(
                f"All services initialized in {self._stats['initialization_time']:.2f} seconds"
            )

        except Exception as e:
            logger.error(
                f"Failed to initialize services: {str(e)}", exc_info=True
            )
            raise RuntimeError(f"Service initialization failed: {str(e)}")

    async def get_embeddings(
        self, provider_name: Optional[str] = None
    ) -> Dict[str, BaseEmbeddings]:
        """Get cached embeddings providers"""
        if not self._stats["embeddings_initialized"]:
            logger.error("Embedding providers not initialized")
            raise RuntimeError("Embedding providers not initialized")
        if provider_name and provider_name not in self._embeddings:
            logger.error(f"Embedding provider {provider_name} not available")
            raise ValueError(
                f"Embedding provider {provider_name} not available"
            )
        logger.debug("Returning embedding providers")
        return self._embeddings

    async def get_default_embedding(self) -> BaseEmbeddings:
        """Get default embedding provider"""
        provider_name = getattr(
            settings.embeddings, "default_provider", "cohere"
        )
        if provider_name not in self._embeddings:
            logger.error(
                f"Default embedding provider {provider_name} not initialized"
            )
            raise RuntimeError(
                f"Default embedding provider {provider_name} not initialized"
            )
        logger.debug(f"Returning default embedding provider: {provider_name}")
        return self._embeddings[provider_name]

    async def get_vector_store(self) -> BaseVectorStore:
        """Get cached vector store"""
        if not self._stats["vector_store_initialized"]:
            logger.error("Vector store not initialized")
            raise RuntimeError("Vector store not initialized")
        logger.debug("Returning vector store")
        return self._vector_store

    async def get_chat_providers(
        self, provider_name: Optional[str] = None
    ) -> Dict[str, BaseChat]:
        """Get cached chat providers"""
        if not self._stats["chat_providers_initialized"]:
            logger.error("Chat providers not initialized")
            raise RuntimeError("Chat providers not initialized")
        if provider_name and provider_name not in self._chat_providers:
            logger.error(f"Chat provider {provider_name} not available")
            raise ValueError(f"Chat provider {provider_name} not available")
        logger.debug("Returning chat providers")
        return self._chat_providers

    async def get_default_chat_provider(self) -> BaseChat:
        """Get default chat provider"""
        provider_name = getattr(settings.llm, "default_provider", "openai")
        if provider_name not in self._chat_providers:
            logger.error(
                f"Default chat provider {provider_name} not initialized"
            )
            raise RuntimeError(
                f"Default chat provider {provider_name} not initialized"
            )
        logger.debug(f"Returning default chat provider: {provider_name}")
        return self._chat_providers[provider_name]

    async def get_search_provider(self) -> BaseWebSearch:
        """Get cached search provider"""
        if not self._stats["search_provider_initialized"]:
            logger.error("Search provider not initialized")
            raise RuntimeError("Search provider not initialized")
        logger.debug("Returning search provider")
        return self._search_provider

    async def get_stock_data_provider(self) -> BaseStockData:
        """Get cached stock data provider"""
        if not self._stats["stock_providers_initialized"]:
            logger.error("Stock data provider not initialized")
            raise RuntimeError("Stock data provider not initialized")
        logger.debug("Returning stock data provider")
        return self._stock_data_provider

    async def get_stock_analysis_provider(self) -> BaseStockAnalysis:
        """Get cached stock analysis provider"""
        if not self._stats["stock_providers_initialized"]:
            logger.error("Stock analysis provider not initialized")
            raise RuntimeError("Stock analysis provider not initialized")
        logger.debug("Returning stock analysis provider")
        return self._stock_analysis_provider

    def get_stats(self) -> Dict[str, Any]:
        """Get service manager statistics"""
        logger.debug("Retrieving service initialization stats")
        return {
            "initialized_services": {
                "embeddings": self._stats["embeddings_initialized"],
                "chat_providers": self._stats["chat_providers_initialized"],
                "vector_store": self._stats["vector_store_initialized"],
                "search_provider": self._stats["search_provider_initialized"],
                "stock_providers": self._stats["stock_providers_initialized"],
            },
            "cached_embeddings": len(self._embeddings),
            "cached_chat_providers": len(self._chat_providers),
            "has_vector_store": self._vector_store is not None,
            "has_search_provider": self._search_provider is not None,
            "has_stock_providers": all(
                [
                    self._stock_data_provider is not None,
                    self._stock_analysis_provider is not None,
                ]
            ),
            "initialization_time": self._stats["initialization_time"],
        }

    async def close(self):
        """Clean up resources when shutting down"""
        logger.info("Closing ServiceManager resources")
        try:
            if self._vector_store:
                await self._vector_store.close()
                logger.info("QdrantVectorStoreDB connection closed")
            self._embeddings = {}
            self._chat_providers = {}
            self._vector_store = None
            self._search_provider = None
            self._stock_data_provider = None
            self._stock_analysis_provider = None
            self._stats.update(
                {
                    "embeddings_initialized": False,
                    "chat_providers_initialized": False,
                    "vector_store_initialized": False,
                    "search_provider_initialized": False,
                    "stock_providers_initialized": False,
                    "initialization_time": 0.0,
                }
            )
            logger.info("ServiceManager resources cleaned up successfully")
        except Exception as e:
            logger.error(
                f"Error during ServiceManager cleanup: {str(e)}", exc_info=True
            )
