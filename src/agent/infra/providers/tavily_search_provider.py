from typing import Any, Dict, List

from langchain_tavily import TavilySearch

from ...domain.interfaces.search_interface import BaseWebSearch
from ...shared.logging.logger import Logger
from ...shared.settings.settings import settings

logger = Logger.get_logger(__name__)


class TavilyWebSearch(BaseWebSearch):
    """Infrastructure-specific implementation of WebSearchRetriever using LangChain TavilySearch."""

    def __init__(self):
        try:
            self.client = TavilySearch(
                tavily_api_key=settings.app.tavily_api_key,
                max_results=getattr(settings.tavily, "max_results", 5),
                search_depth=getattr(
                    settings.tavily, "search_depth", "advanced"
                ),
                include_domains=getattr(
                    settings.tavily,
                    "include_domains",
                    [
                        "vnexpress.net",
                        "tuoitre.vn",
                        "thanhnien.vn",
                        "dantri.com.vn",
                        "vietnamnet.vn",
                        "laodong.vn",
                        "nld.com.vn",
                        "cafef.vn",
                        "vneconomy.vn",
                        "vietstock.vn",
                        "ndh.vn",
                        "tinnhanhchungkhoan.vn",
                        "stockbiz.vn",
                        "cophieu68.vn",
                        "vietcombank.com.vn",
                        "sbv.gov.vn",
                        "tinhte.vn",
                        "voz.vn",
                        "webtretho.com",
                        "chinhphu.vn",
                        "mpi.gov.vn",
                        "gso.gov.vn",
                    ],
                ),
                include_raw_content=True,
            )
            self.timeout_seconds = getattr(
                settings.tavily, "timeout_seconds", 5
            )
            logger.info("Initialized TavilyWebSearch successfully")
        except Exception as e:
            logger.error(f"Failed to initialize TavilyWebSearch: {str(e)}")
            raise

    async def search(self, query: str) -> List[Dict[str, Any]]:
        """
        Perform a web search using Tavily API via LangChain.

        Args:
            query: The query string to search for.

        Returns:
            List of dictionaries containing search results (title, url, content, score, etc.).
        """
        try:
            # Use LangChain TavilySearch
            raw_results = await self.client.ainvoke(query)
            # results: dict with the following keys:
            # query: content fetched into Tavily
            # follow_up_questions
            # answer
            # images
            # results: list of returned information dict = [{},{},...]
            # response_time
            # request_id

            # Extract list of results from raw_results:
            results = raw_results.get("results", "")

            # Format results to match expected output
            formatted_results = [
                {
                    "content": result.get("content", ""),
                    "title": result.get("title", ""),
                    "url": result.get("url", ""),
                    "score": result.get("score", 0.0),
                }
                for result in results
            ]

            logger.info(
                f"Retrieved {len(formatted_results)} search results for query: {query}"
            )
            return formatted_results

        except Exception as e:
            logger.error(f"Tavily search failed for query '{query}': {str(e)}")
            return []
