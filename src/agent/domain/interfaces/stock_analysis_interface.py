from abc import ABC, abstractmethod
from typing import Any, Dict, List


class BaseStockAnalysis(ABC):
    """Abstract interface for stock analysis operations."""

    @abstractmethod
    async def get_fundamental_ratios(
        self, symbols: List[str]
    ) -> Dict[str, Any]:
        """
        Get fundamental ratios for given symbols.

        Args:
            symbols: List of stock symbols (e.g., ["VCB", "VNM"])

        Returns:
            Dictionary with symbol as key and data containing:
                - timestamp: Data fetch time (str, ISO format)
                - status: "success" or "no_data" or "error" (str)
                - data: Dict with quarter as key and metrics (pe, pb, roe, etc.) as values
                - error: Error message if applicable (str, optional)
        """
        pass

    @abstractmethod
    async def get_industry_analysis(self, symbol: str) -> Dict[str, Any]:
        """
        Get analysis of peer companies in the same industry.

        Args:
            symbol: Stock symbol (e.g., "VCB")

        Returns:
            Dictionary containing:
                - status: "success" or "no_data" or "error" (str)
                - error: Error message if applicable (str, optional)
                - ticker: Dict with indicators (Vốn hóa, P/E, etc.) as values
        """
        pass

    @abstractmethod
    async def get_peers_comparison(self, symbols: List[str]) -> Dict[str, Any]:
        """
        Get peer comparison and ranking for a list of symbols.

        Args:
            symbols: List of stock symbols (e.g., ["VCB", "VNM"])

        Returns:
            Dictionary containing:
                - status: "success" or "no_data" or "error" (str)
                - error: Error message if applicable (str, optional)
                - ticker: Dict with indicators (Vốn hóa, P/E, etc.) as values
        """
        pass
