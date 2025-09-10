import re
from typing import Any, Dict, List, Type

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import BaseModel, Field

from ...infra.providers.tavily_search_provider import BaseWebSearch
from ...infra.utils.query_enhancer import QueryEnhancer
from ...shared.logging.logger import Logger
from .base import CustomBaseTool

logger = Logger.get_logger(__name__)


class TavilySearchToolInput(BaseModel):
    """Input schema for TavilySearchTool."""

    query: str = Field(..., description="The query to search the web for.")


class TavilySearchToolOutput(BaseModel):
    """Output schema for TavilySearchTool."""

    search_results: List[Dict[str, Any]] = Field(
        description="Formatted search results"
    )
    sources: List[Dict[str, Any]] = Field(description="Metadata of sources")


class TavilySearchTool(CustomBaseTool):
    """Web search tool for retrieving current information about Vietnamese stocks, market news, and financial data."""

    name: str = "tavily_search"
    description: str = (
        "Search the web for current information about Vietnamese stocks, market news, and financial data"
    )
    args_schema: Type[BaseModel] = TavilySearchToolInput

    def __init__(self, web_search_retriever: BaseWebSearch):
        super().__init__()
        self._web_search_retriever = web_search_retriever
        self._query_enhancer = QueryEnhancer()

    async def _execute_impl(self, query: str) -> Dict[str, Any]:
        """Execute web search using the injected retriever."""

        enhanced_query = self._query_enhancer.enhance_query(query)
        try:
            # search_results: Liss[Dict[keys: url, title, content, raw_content, score], Any]
            search_results = await self._web_search_retriever.search(
                query=enhanced_query
            )

            formatted_results = self._format_search_results(search_results)
            sources = self._extract_sources(search_results)
            return TavilySearchToolOutput(
                search_results=formatted_results,
                sources=sources,
            ).model_dump()

        except Exception as e:
            logger.error(f"Tavily search failed: {str(e)}")
            return TavilySearchToolOutput(
                search_results=[], sources=[]
            ).model_dump()

    def _extract_sources(
        self, results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Extract source information from results."""
        return [
            {"url": result.get("url", ""), "score": result.get("score", 0)}
            for result in results
        ]

    def _format_search_results(
        self, results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Format raw search results into domain-specific structure."""
        return [
            {
                "title": result.get("title", ""),
                "content": result.get("content", ""),
            }
            for result in results
        ]

    def to_langchain_retriever(self) -> BaseRetriever:
        """Convert TavilySearchTool to a LangChain BaseRetriever."""

        class TavilySearchRetriever(BaseRetriever):
            tool: "TavilySearchTool" = self

            async def _aget_relevant_documents(
                self, query: str
            ) -> List[Document]:
                results = await self.tool._arun(query=query)
                return [
                    Document(
                        page_content=r["content"],
                        metadata={
                            "title": r["title"],
                            "url": r["url"],
                            "score": r["score"],
                            "published_date": r["published_date"],
                        },
                    )
                    for r in results["search_results"]
                ]

        return TavilySearchRetriever()
