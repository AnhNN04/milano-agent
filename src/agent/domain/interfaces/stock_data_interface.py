from abc import ABC, abstractmethod
from typing import Any, Dict, List


class BaseStockData(ABC):
    """Abstract interface for stock data retrieval operations in domain layer."""

    @abstractmethod
    async def get_realtime_data(self, symbols: List[str]) -> Dict[str, Any]:
        """
        Get realtime stock data for given symbols.

        Args:
            symbols: List of stock symbols.

        Returns:
            Dict containing realtime data for all symbols.
        """
        pass

    @abstractmethod
    async def get_historical_data(
        self, symbols: List[str], start_date: str, end_date: str
    ) -> Dict[str, Any]:
        """
        Get historical stock data for given symbols.

        Args:
            symbols: List of stock symbols.
            start_date: Start date in YYYY-MM-DD format.
            end_date: End date in YYYY-MM-DD format.

        Returns:
            Dict containing historical data for all symbols.
        """
        pass
