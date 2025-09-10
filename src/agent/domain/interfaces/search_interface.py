from abc import ABC, abstractmethod
from typing import Any, Dict, List


class BaseWebSearch(ABC):
    """Abstract interface for web search operations in domain layer."""

    @abstractmethod
    async def search(self, query: str) -> List[Dict[str, Any]]:
        """
        Perform a web search and return results.

        Args:
            query: The query string to search for.

        Returns:
            List of dictionaries containing search results (title, url, content, score, etc.).
        """
        pass
