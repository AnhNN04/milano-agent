from typing import Any, ClassVar, Dict, List, Type, Union

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field, PrivateAttr

from ...infra.providers.vnstock_analysis_provider import BaseStockAnalysis
from ...shared.logging.logger import Logger
from .base import CustomBaseTool

logger = Logger.get_logger(__name__)


class FundamentalAnalysisToolInput(BaseModel):
    """Input schema for FundamentalAnalysisTool."""

    symbols: Union[str, List[str]] = Field(
        ...,
        description="Stock symbol(s) for analysis (e.g., 'FPT', ['HPG', 'VCB']).",
    )


class FundamentalAnalysisToolOutput(BaseModel):
    """Output schema for FundamentalAnalysisTool."""

    data: Dict[str, Any] = Field(description="Analysis table and rankings")


class FundamentalAnalysisTool(CustomBaseTool):
    """Tool for fundamental analysis of stocks."""

    name: str = "fundamental_analysis"
    description: str = "Analyze stocks using fundamental financial ratios"

    args_schema: Type[BaseModel] = FundamentalAnalysisToolInput

    _provider: PrivateAttr = PrivateAttr()

    SUPPORTED_METRICS: ClassVar[List[str]] = [
        "pe",
        "pb",
        "roe",
        "eps",
        "market_cap",
        "revenue",
        "profit",
        "debt_ratio",
        "current_ratio",
        "quick_ratio",
        "gross_margin",
        "net_margin",
        "roa",
    ]

    def __init__(self, stock_analysis_provider: BaseStockAnalysis):
        super().__init__()
        self._provider = stock_analysis_provider

    async def _execute_impl(
        self, symbols: Union[str, List[str]]
    ) -> Dict[str, Any]:
        """Execute fundamental analysis."""
        try:
            self._validate_parameters(symbols)
            normalized_symbols = self._normalize_symbols(symbols)

            # Fetch fundamental data
            fundamental_data = await self._provider.get_fundamental_ratios(
                normalized_symbols
            )

            return FundamentalAnalysisToolOutput(
                data=fundamental_data
            ).model_dump()

        except Exception as e:
            logger.error(f"Fundamental analysis failed: {str(e)}")
            return FundamentalAnalysisToolOutput(data={}).model_dump()

    def _validate_parameters(self, symbols: Union[str, List[str]]) -> None:
        """Validate input parameters."""

        if isinstance(symbols, str):
            symbols = [symbols]
        if not isinstance(symbols, list) or not symbols:
            raise ValueError("symbols must be a non-empty list or string")

    def _normalize_symbols(self, symbols: Union[str, List[str]]) -> List[str]:
        """Normalize symbols to a list of uppercase strings."""

        if isinstance(symbols, str):
            symbols = [s.strip().upper() for s in symbols.split(",")]

        return [s.strip().upper() for s in symbols]

    def to_formatted_context(self, output: Dict[str, Any]) -> str:
        """Format analysis results for LLM context."""

        prompt_template = ChatPromptTemplate.from_template(
            "Fundamental analysis for Vietnamese stocks:\n{data}"
        )
        analysis_table = output.get("data", {}).get("analysis_table", {})
        insights = output.get("data", {}).get("insights", [])

        formatted_data = []
        for symbol, data in analysis_table.items():
            quarter = data.get("quarter", "N/A")
            formatted_data.append(f"Stock: {symbol}, Quarter: {quarter}")
            metrics = data.get("metrics", {})
            for metric, value in metrics.items():
                if value != "N/A" and value is not None:
                    formatted_data.append(
                        f"  {metric.upper()}: {float(value):.2f}"
                    )

        formatted_data.extend(["Insights:"] + insights)

        return prompt_template.format(
            data=(
                "\n".join(formatted_data)
                if formatted_data
                else "No data available"
            )
        )
