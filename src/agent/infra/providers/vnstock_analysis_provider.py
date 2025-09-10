import asyncio
import functools
from typing import Any, Dict, List, Optional

import pandas as pd

from ...domain.interfaces.stock_analysis_interface import BaseStockAnalysis
from ...shared.logging.logger import Logger

logger = Logger.get_logger(__name__)

try:
    import vnstock as vns
except ImportError:
    logger.error(
        "vnstock library is not installed. Please install it with: pip install vnstock"
    )
    raise ImportError("vnstock library is required")


class VnStockAnalysis(BaseStockAnalysis):
    """Infrastructure-specific implementation of StockAnalysisProvider using vnstock library.

    This provider isolates the concrete vnstock implementation from the domain layer,
    adhering to Domain-Driven Design by depending on domain abstractions.
    """

    def __init__(self):
        """Initialize VnStock Analysis Provider."""
        self._initialized = False

    async def get_fundamental_ratios(
        self, symbols: List[str]
    ) -> Dict[str, Any]:
        """Get fundamental ratios for given symbols using vnstock."""

        try:
            logger.info(f"Fetching fundamental ratios for symbols: {symbols}")
            results = {}
            for symbol in symbols:
                try:
                    ratios_data = await self._run_sync_function(
                        vns.financial_ratio,
                        symbol=symbol,
                        report_range="quarterly",
                        is_all=False,
                    )
                    if ratios_data is not None and not ratios_data.empty:
                        formatted_ratios = self._format_financial_ratios(
                            ratios_data, symbol
                        )
                        results[symbol] = formatted_ratios
                    else:
                        logger.warning(
                            f"No fundamental data found for symbol: {symbol}"
                        )
                        results[symbol] = self._create_empty_fundamental_data(
                            symbol
                        )
                except Exception as e:
                    logger.error(
                        f"Failed to fetch fundamental ratios for {symbol}: {str(e)}"
                    )
                    results[symbol] = self._create_error_fundamental_data(
                        symbol, str(e)
                    )
            return results
        except Exception as e:
            logger.error(f"Fundamental ratios retrieval failed: {str(e)}")
            raise

    async def get_peers_comparison(self, symbols: List[str]) -> Dict[str, Any]:
        """Get peer comparison and ranking for a list of symbols using vnstock.
        Trả về cả data (raw formatted data) và insights (công ty nào cao nhất cho mỗi chỉ số).
        """
        try:
            logger.info(f"Fetching peer comparison for symbols: {symbols}")
            symbol_string = ", ".join(symbols)
            comparison_data = await self._run_sync_function(
                vns.stock_ls_analysis, symbol_string, lang="vi"
            )

            if comparison_data is not None and not comparison_data.empty:
                # Chuẩn hóa data gốc
                formatted_data = self._format_stock_ls_analysis(
                    comparison_data
                )

                # Tạo insights
                insights = []
                indicators = [idx for idx in comparison_data.index]

                for indicator in indicators:
                    try:
                        series = comparison_data.loc[indicator].dropna()
                        if not series.empty:
                            max_ticker = series.idxmax()
                            max_value = series[max_ticker]
                            insights.append(
                                f"Chỉ số {indicator} cao nhất là: {round(max_value, 2) if isinstance(max_value, (int, float)) else max_value} với mã {max_ticker}"
                            )
                    except Exception as e:
                        logger.warning(
                            f"Lỗi khi tạo insight cho chỉ số {indicator}: {str(e)}"
                        )

                return {"data": formatted_data, "insights": insights}
            else:
                logger.warning(
                    f"No comparison data found for symbols: {symbols}"
                )
                return {"data": {}, "insights": []}

        except Exception as e:
            logger.error(
                f"Failed to fetch peer comparison for {symbols}: {str(e)}"
            )
            return {"status": "error", "error": str(e)}

    async def get_industry_analysis(self, symbol: str) -> Dict[str, Any]:
        """Get analysis of peer companies in the same industry as the given symbol using vnstock.
        Trả về cả data (raw formatted data) và insights (công ty nào cao nhất cho mỗi chỉ số).
        """
        try:
            logger.info(f"Fetching industry analysis for symbol: {symbol}")
            industry_data = await self._run_sync_function(
                vns.industry_analysis, symbol=symbol, lang="vi"
            )

            if industry_data is not None and not industry_data.empty:
                # Chuẩn hóa data gốc
                formatted_data = self._format_stock_ls_analysis(industry_data)

                # Tạo insights
                insights = []
                # Lấy danh sách các chỉ số
                indicators = [idx for idx in industry_data.index]

                for indicator in indicators:
                    try:
                        # Lấy series của chỉ số
                        series = industry_data.loc[indicator]

                        # Bỏ NaN để tránh lỗi
                        series = series.dropna()

                        if not series.empty:
                            max_ticker = series.idxmax()
                            max_value = series[max_ticker]
                            insights.append(
                                f"Chỉ số {indicator} cao nhất là: {round(max_value, 2) if isinstance(max_value, (int, float)) else max_value} với mã {max_ticker}"
                            )
                    except Exception as e:
                        logger.warning(
                            f"Lỗi khi tạo insight cho chỉ số {indicator}: {str(e)}"
                        )

                return {"data": formatted_data, "insights": insights}
            else:
                logger.warning(
                    f"No industry analysis data found for symbol: {symbol}"
                )
                raise

        except Exception as e:
            logger.error(
                f"Failed to fetch industry analysis for {symbol}: {str(e)}"
            )
            raise

    def _format_financial_ratios(
        self, ratios_data: Any, symbol: str
    ) -> Dict[str, Any]:
        """Format financial ratios from vnstock's financial_ratio output into a nested dict structure."""

        try:
            # Lấy danh sách các cột (quý-năm), loại bỏ các cột không phải dữ liệu nếu có
            quarter_columns = [
                col
                for col in ratios_data.columns
                if col not in ["ticker", "quarter", "year"]
            ]

            # Khởi tạo dict kết quả cho mã chứng khoán
            formatted_data = {}

            # Lặp qua từng cột (quý-năm)
            for column in quarter_columns:
                quarter_data = {}
                # Lấy dữ liệu của cột hiện tại, sử dụng index làm key
                column_data = ratios_data[column].to_dict()
                quarter_data["pe"] = self._safe_get_numeric(
                    column_data, ["priceToEarning", "PE"]
                )
                quarter_data["pb"] = self._safe_get_numeric(
                    column_data, ["priceToBook", "PB"]
                )
                quarter_data["roe"] = self._safe_get_numeric(
                    column_data, ["roe", "returnOnEquity", "ROE"]
                )
                quarter_data["eps"] = self._safe_get_numeric(
                    column_data, ["earningPerShare", "EPS"]
                )
                quarter_data["market_cap"] = None
                quarter_data["revenue"] = self._safe_get_numeric(
                    column_data, ["totalRevenue"]
                )
                quarter_data["profit"] = self._safe_get_numeric(
                    column_data, ["netIncome"]
                )
                quarter_data["debt_ratio"] = self._safe_get_numeric(
                    column_data, ["debtOnEquity", "equityOnLiability"]
                )
                quarter_data["current_ratio"] = self._safe_get_numeric(
                    column_data, ["currentRatio"]
                )
                quarter_data["quick_ratio"] = self._safe_get_numeric(
                    column_data, ["quickRatio"]
                )
                quarter_data["gross_margin"] = self._safe_get_numeric(
                    column_data, ["grossMargin"]
                )
                quarter_data["net_margin"] = self._safe_get_numeric(
                    column_data, ["postTaxOnToi", "netMargin"]
                )
                quarter_data["roa"] = self._safe_get_numeric(
                    column_data, ["roa", "returnOnAssets", "ROA"]
                )

                # Thêm dữ liệu của quý-năm vào dict chính
                formatted_data[column] = quarter_data

            return formatted_data
        except Exception as e:
            logger.error(
                f"Failed to format fundamental ratios for {symbol}: {str(e)}"
            )
            return self._create_error_fundamental_data(symbol, str(e))

    def _format_stock_ls_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Formats the stock_ls_analysis DataFrame to a dict with ticker as key and indicators as values."""

        # Lấy danh sách các chỉ số (index) từ 'Vốn hóa (tỷ)' trở đi
        indicators = [
            idx for idx in df.index
        ]  # if idx in column_mapping.values()]

        # Chuẩn bị dict kết quả
        result = {}

        # Lặp qua từng cột (mã chứng khoán)
        for ticker in df.columns:
            ticker_data = {}
            for indicator in indicators:
                value = df.loc[indicator, ticker]
                if pd.notna(value):
                    if indicator == "Vốn hóa (tỷ)":
                        ticker_data[indicator] = (
                            int((value / 1)) if value > 0 else 0
                        )
                    elif isinstance(value, (int, float)):
                        ticker_data[indicator] = round(value, 2)
                    else:
                        ticker_data[indicator] = value
            result[ticker] = ticker_data

        return result

    def _safe_get_numeric(
        self, data: Dict[str, Any], keys: List[str]
    ) -> Optional[float]:
        """Safely get numeric value from data using multiple possible keys."""

        for key in keys:
            if key in data:
                value = data[key]
                if pd.notna(value):
                    try:
                        return float(value)
                    except (ValueError, TypeError):
                        continue
        return None

    async def _run_sync_function(self, func, *args, **kwargs):
        """Run synchronous vnstock functions in async context."""

        loop = asyncio.get_event_loop()
        bound_func = functools.partial(func, *args, **kwargs)
        return await loop.run_in_executor(None, bound_func)

    def _create_empty_fundamental_data(self) -> Dict[str, Any]:
        """Create empty fundamental data structure."""

        return {}

    def _create_error_fundamental_data(self, error: str) -> Dict[str, Any]:
        """Create error fundamental data structure."""

        return {}
