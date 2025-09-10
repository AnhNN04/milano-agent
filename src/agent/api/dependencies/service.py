import hashlib
import time
from typing import Dict, Optional

import redis.asyncio as redis
from fastapi import Depends

from ...application.services.document_processing_service import (
    DocumentProcessingService,
)
from ...application.services.stock_analysis_service import StockAnalysisService
from ...domain.agents.react_agent import StockReActAgent
from ...domain.interfaces.chat_interface import BaseChat
from ...domain.interfaces.document_loader_interface import BaseDocumentLoader
from ...domain.interfaces.embedding_interface import BaseEmbeddings
from ...domain.interfaces.rag_interface import BaseRag
from ...domain.interfaces.search_interface import BaseWebSearch
from ...domain.interfaces.stock_analysis_interface import BaseStockAnalysis
from ...domain.interfaces.stock_data_interface import BaseStockData
from ...domain.interfaces.vector_store_interface import BaseVectorStore
from ...domain.tools.base import CustomBaseTool
from ...domain.tools.chat_tool import ChatTool
from ...domain.tools.fundamental_analysis_tool import FundamentalAnalysisTool
from ...domain.tools.industry_analysis_tool import IndustryAnalysisTool
from ...domain.tools.peers_comparison_tool import PeersComparisonTool
from ...domain.tools.rag_tool import RAGTool
from ...domain.tools.stock_data_tool import StockPriceTool
from ...domain.tools.tavily_search_tool import TavilySearchTool
from ...infra.document_loaders.s3_document_loader import S3DocumentLoader
from ...infra.providers.rag_provider import QdrantRag
from ...shared.logging.logger import Logger
from ...shared.session.redis_session_manager import (
    LangChainRedisSessionManager,
)
from ...shared.settings.settings import settings
from ...shared.singletons.service_manager import ServiceManager

logger = Logger.get_logger(__name__)

# Global instances
_session_manager: Optional[LangChainRedisSessionManager] = None
_service_manager: Optional[ServiceManager] = None


async def initialize_services():
    """Initialize global services eagerly during startup"""
    global _session_manager, _service_manager
    start_time = time.time()
    logger.info("Starting eager initialization of global services")

    try:
        # Check Redis connection
        try:
            redis_client = redis.Redis(
                host=settings.redis.host,
                port=settings.redis.port,
                decode_responses=True,
                db=settings.redis.db,
                max_connections=int(getattr(settings.redis, "pool_size", 20)),
                socket_timeout=getattr(settings.redis, "socket_timeout", 5),
                socket_connect_timeout=getattr(
                    settings.redis, "connection_timeout", 5
                ),
                health_check_interval=30,
            )
            await redis_client.ping()
            logger.info("Redis connection test successful")
            await redis_client.close()
        except Exception as e:
            logger.error(
                f"Redis connection test failed: {str(e)}", exc_info=True
            )
            raise RuntimeError(f"Failed to connect to Redis: {str(e)}")

        # Initialize session manager
        _session_manager = LangChainRedisSessionManager(
            default_ttl=getattr(settings.redis, "default_ttl", 3600)
        )
        await _session_manager.start()
        logger.info("LangChainRedisSessionManager initialized and started")

        # Initialize service manager
        _service_manager = await ServiceManager.get_instance()
        logger.info("ServiceManager initialized")

        logger.info(
            f"Global services initialized in {time.time() - start_time:.2f} seconds"
        )
    except Exception as e:
        logger.error(
            f"Failed to initialize global services: {str(e)}", exc_info=True
        )
        raise RuntimeError(f"Global services initialization failed: {str(e)}")


async def cleanup_services():
    """Clean up global services during shutdown"""
    global _session_manager, _service_manager
    logger.info("Cleaning up global services")

    try:
        if _service_manager:
            await _service_manager.close()
            logger.info("ServiceManager resources cleaned up")
        if _session_manager:
            await _session_manager.stop()
            logger.info("LangChainRedisSessionManager closed")
        _session_manager = None
        _service_manager = None
        logger.info("Global services cleanup completed")
    except Exception as e:
        logger.error(
            f"Error during global services cleanup: {str(e)}", exc_info=True
        )


async def get_session_manager() -> LangChainRedisSessionManager:
    """Get global session manager singleton"""
    if _session_manager is None:
        logger.error(
            "Session manager not initialized. Ensure initialize_services was called during startup."
        )
        raise RuntimeError(
            "Session manager not initialized. Please check Redis connection and startup logs."
        )
    logger.debug("Returning session manager instance")
    return _session_manager


async def get_service_manager() -> ServiceManager:
    """Get global service manager singleton"""
    if _service_manager is None:
        logger.error(
            "Service manager not initialized. Ensure initialize_services was called during startup."
        )
        raise RuntimeError(
            "Service manager not initialized. Please check startup logs."
        )
    logger.debug("Returning service manager instance")
    return _service_manager


async def get_embeddings() -> Dict[str, BaseEmbeddings]:
    """Get cached embeddings providers."""
    service_manager = await get_service_manager()
    return await service_manager.get_embeddings()


async def get_default_embedding() -> BaseEmbeddings:
    """Get cached default embedding provider."""
    service_manager = await get_service_manager()
    return await service_manager.get_default_embedding()


async def get_vector_store() -> BaseVectorStore:
    """Get cached vector store."""
    service_manager = await get_service_manager()
    return await service_manager.get_vector_store()


async def get_chat_providers() -> Dict[str, BaseChat]:
    """Get cached chat providers."""
    service_manager = await get_service_manager()
    return await service_manager.get_chat_providers()


async def get_chat_provider() -> BaseChat:
    """Get cached default chat provider."""
    service_manager = await get_service_manager()
    return await service_manager.get_default_chat_provider()


async def get_web_search_retriever() -> BaseWebSearch:
    """Get cached web search provider."""
    service_manager = await get_service_manager()
    return await service_manager.get_search_provider()


async def get_stock_data_provider() -> BaseStockData:
    """Get cached stock data provider."""
    service_manager = await get_service_manager()
    return await service_manager.get_stock_data_provider()


async def get_stock_analysis_provider() -> BaseStockAnalysis:
    """Get cached stock analysis provider."""
    service_manager = await get_service_manager()
    return await service_manager.get_stock_analysis_provider()


async def get_rag_retriever(
    vector_store: BaseVectorStore = Depends(get_vector_store),
) -> BaseRag:
    """Get RAG retriever with injected cached vector store."""
    return QdrantRag(vector_store=vector_store)


def get_document_loader() -> BaseDocumentLoader:
    """Get document loader instance (lightweight, no need to cache)."""
    return S3DocumentLoader()


async def get_tools(
    rag_retriever: BaseRag = Depends(get_rag_retriever),
    web_search_retriever: BaseWebSearch = Depends(get_web_search_retriever),
    stock_data_provider: BaseStockData = Depends(get_stock_data_provider),
    stock_analysis_provider: BaseStockAnalysis = Depends(
        get_stock_analysis_provider
    ),
    chat_provider: BaseChat = Depends(get_chat_provider),
) -> Dict[str, CustomBaseTool]:
    """Create tools dict with cached dependency injection."""
    return {
        "rag_knowledge": RAGTool(rag_retriever=rag_retriever),
        "tavily_search": TavilySearchTool(
            web_search_retriever=web_search_retriever
        ),
        "stock_price": StockPriceTool(stock_data_provider=stock_data_provider),
        "fundamental_analysis": FundamentalAnalysisTool(
            stock_analysis_provider=stock_analysis_provider
        ),
        "industry_analysis": IndustryAnalysisTool(
            stock_analysis_provider=stock_analysis_provider
        ),
        "peers_comparison": PeersComparisonTool(
            stock_analysis_provider=stock_analysis_provider
        ),
        "chat_llm": ChatTool(chat_provider=chat_provider),
    }


async def get_react_agent(
    chat_provider: BaseChat = Depends(get_chat_provider),
    tools: Dict[str, CustomBaseTool] = Depends(get_tools),
) -> StockReActAgent:
    """Get ReAct agent with injected cached tools."""
    return StockReActAgent(chat_provider=chat_provider, tools=tools)


async def get_stock_analysis_service(
    agent: StockReActAgent = Depends(get_react_agent),
    tools: Dict[str, CustomBaseTool] = Depends(get_tools),
    session_manager: LangChainRedisSessionManager = Depends(
        get_session_manager
    ),
) -> StockAnalysisService:
    """Get stock analysis service with cached dependencies."""
    return StockAnalysisService(
        agent=agent, tools=tools, session_manager=session_manager
    )


async def get_document_processing_service(
    document_loader: BaseDocumentLoader = Depends(get_document_loader),
    vector_store: BaseVectorStore = Depends(get_vector_store),
) -> DocumentProcessingService:
    """Get document processing service with cached dependencies."""
    return DocumentProcessingService(
        document_loader=document_loader, vector_store=vector_store
    )


def generate_query_hash(query: str, session_id: str = "") -> str:
    """Generate hash for query caching"""
    content = f"{query}:{session_id}"
    return hashlib.md5(content.encode()).hexdigest()
