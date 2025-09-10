from datetime import datetime
from typing import Any, Dict, List, Optional, Type, Union

from pydantic import BaseModel, Field, PrivateAttr

from ...infra.providers.vnstock_data_provider import BaseStockData
from ...shared.logging.logger import Logger
from .base import CustomBaseTool

logger = Logger.get_logger(__name__)


class StockPriceToolInput(BaseModel):
    """Input schema for StockPriceTool."""

    symbols: Union[str, List[str]] = Field(
        ...,
        description="The stock symbol(s) for price retrieval (e.g., 'FPT', ['HPG', 'VCB']).",
    )
    data_type: str = Field(
        "realtime",
        description="Type of data to retrieve: 'realtime' or 'historical'.",
    )
    start_date: Optional[str] = Field(
        None,
        description="Start date for historical data in YYYY-MM-DD format.",
    )
    end_date: Optional[str] = Field(
        None, description="End date for historical data in YYYY-MM-DD format."
    )


class StockPriceToolOutput(BaseModel):
    """Output schema for StockPriceTool."""

    results: Dict[str, Any] = Field(description="Stock price data")


class StockPriceTool(CustomBaseTool):
    """Stock price tool for retrieving realtime and historical stock data."""

    name: str = "stock_price"
    description: str = (
        "Retrieve realtime or historical stock price data for Vietnamese stocks"
    )
    args_schema: Type[BaseModel] = StockPriceToolInput

    _stock_data_provider: PrivateAttr = PrivateAttr()

    def __init__(self, stock_data_provider: BaseStockData):
        super().__init__()
        self._stock_data_provider = stock_data_provider

    async def _execute_impl(
        self,
        symbols: Union[str, List[str]],
        data_type: str = "realtime",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Execute stock price retrieval using the injected data provider."""
        try:
            # Normalize symbols
            normalized_symbols = self._normalize_symbols(symbols)

            # Validate input parameters
            self._validate_parameters(
                data_type, normalized_symbols, start_date, end_date
            )

            # Route to appropriate data retrieval method
            if data_type == "realtime":
                raw_data = await self._stock_data_provider.get_realtime_data(
                    normalized_symbols
                )
            elif data_type == "historical":
                raw_data = await self._stock_data_provider.get_historical_data(
                    normalized_symbols, start_date, end_date
                )
            else:
                raise ValueError(f"Invalid data_type: {data_type}")

            # Extract results
            tool_results = raw_data

            return StockPriceToolOutput(results=tool_results).model_dump()

        except Exception as e:
            logger.error(f"Stock price retrieval failed: {str(e)}")
            return StockPriceToolOutput(results={}).model_dump()

    def _normalize_symbols(self, symbols: Union[str, List[str]]) -> List[str]:
        """Normalize symbols to a list of uppercase strings."""

        if isinstance(symbols, str):
            symbols = [s.strip().upper() for s in symbols.split(",")]

        return [s.strip().upper() for s in symbols]

    def _validate_parameters(
        self,
        data_type: str,
        symbols: List[str],
        start_date: Optional[str],
        end_date: Optional[str],
    ) -> None:
        """Validate input parameters."""

        if not symbols:
            raise ValueError("symbols are required for stock price retrieval")
        if data_type not in ["realtime", "historical"]:
            raise ValueError("data_type must be 'realtime' or 'historical'")
        if data_type == "historical":
            if not start_date or not end_date:
                raise ValueError(
                    "start_date and end_date are required for historical data"
                )
            try:
                datetime.strptime(start_date, "%Y-%m-%d")
                datetime.strptime(end_date, "%Y-%m-%d")
            except ValueError:
                raise ValueError("Dates must be in YYYY-MM-DD format")
