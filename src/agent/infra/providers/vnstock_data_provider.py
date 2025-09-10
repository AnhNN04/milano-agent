import asyncio
from datetime import datetime
from typing import Any, Dict, List

import vnstock as vns

from ...domain.interfaces.stock_data_interface import BaseStockData
from ...shared.logging.logger import Logger
from ...shared.settings.settings import settings

logger = Logger.get_logger(__name__)


class VnStockData(BaseStockData):
    """Infrastructure-specific implementation of StockDataProvider using vnstock library."""

    def __init__(self):
        """Initialize VnStock Data Provider."""
        self.interval = getattr(settings.vnstock, "interval", "1D")

    async def get_realtime_data(self, symbols: List[str]) -> Dict[str, Any]:
        """
        Get realtime stock data for given symbols using vnstock intraday data.

        Args:
            symbols: List of stock symbols.

        Returns:
            Dict containing realtime data for all symbols.
        """

        results = {}
        try:
            for symbol in symbols:
                try:
                    data = await asyncio.to_thread(
                        vns.stock_intraday_data,
                        symbol=symbol,
                        page_size=1,
                        page=0,
                        investor_segment=True,
                    )

                    if data is not None and not data.empty:
                        row = data.iloc[0].to_dict()
                        results[symbol] = {
                            "time": datetime.now().strftime(
                                "%Y-%m-%d %H:%M:%S"
                            ),
                            "order_type": row.get("orderType"),
                            "investor_type": row.get("investorType"),
                            "volume": row.get("volume", 0),
                            "average_price": row.get("averagePrice", 0),
                            "order_count": row.get("orderCount", 0),
                            "prev_price_change": row.get("prevPriceChange", 0),
                        }
                    else:
                        results[symbol] = {
                            "time": datetime.now().strftime(
                                "%Y-%m-%d %H:%M:%S"
                            ),
                            "order_type": None,
                            "investor_type": None,
                            "volume": None,
                            "average_price": None,
                            "order_count": None,
                            "prev_price_change": None,
                        }

                except Exception as e:
                    logger.error(
                        f"Failed to fetch realtime data for {symbol}: {str(e)}"
                    )
                    raise

            return results

        except Exception as e:
            logger.error(f"Realtime data retrieval failed: {str(e)}")
            raise

    async def get_historical_data(
        self, symbols: List[str], start_date: str, end_date: str
    ) -> Dict[str, Any]:
        """
        Get historical stock data for given symbols using vnstock.

        Args:
            symbols: List of stock symbols.
            start_date: Start date in YYYY-MM-DD format.
            end_date: End date in YYYY-MM-DD format.

        Returns:
            Dict containing historical data for all symbols.
        """

        results = {}
        try:
            for symbol in symbols:
                try:
                    data = await asyncio.to_thread(
                        vns.stock_historical_data,
                        symbol=symbol,
                        start_date=start_date,
                        end_date=end_date,
                        resolution=self.interval,
                    )

                    if data is not None and not data.empty:

                        formatted_data = [
                            {
                                k: v.strftime("%Y-%m-%d") if k == "time" else v
                                for k, v in item.items()
                                if k != "ticker"
                            }
                            for item in data.to_dict("records")
                        ]
                        results[symbol] = formatted_data
                    else:
                        results[symbol] = {}

                except Exception as e:
                    logger.error(
                        f"Failed to fetch historical data for {symbol}: {str(e)}"
                    )
                    raise
            return results

        except Exception as e:
            logger.error(f"Historical data retrieval failed: {str(e)}")
            raise
