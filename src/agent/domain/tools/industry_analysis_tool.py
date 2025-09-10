from typing import Any, Dict, Type

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field, PrivateAttr

from ...infra.providers.vnstock_analysis_provider import BaseStockAnalysis
from ...shared.logging.logger import Logger
from .base import CustomBaseTool

logger = Logger.get_logger(__name__)


class IndustryAnalysisToolInput(BaseModel):
    """Input schema for IndustryAnalysisTool."""

    symbol: str = Field(
        ..., description="The stock symbol for analysis (e.g., 'VCB')."
    )


class IndustryAnalysisToolOutput(BaseModel):
    """Output schema for IndustryAnalysisTool."""

    data: Dict[str, Any] = Field(
        description="Analysis table, rankings, and industry info"
    )


class IndustryAnalysisTool(CustomBaseTool):
    """Tool for industry analysis of a stock against its sector peers."""

    name: str = "industry_analysis"
    description: str = "Analyze a stock's performance against industry peers"
    args_schema: Type[BaseModel] = IndustryAnalysisToolInput

    _provider: PrivateAttr = PrivateAttr()

    def __init__(self, stock_analysis_provider: BaseStockAnalysis):
        super().__init__()
        self._provider = stock_analysis_provider

    async def _execute_impl(self, symbol: str) -> Dict[str, Any]:
        """Execute industry analysis."""
        try:
            self._validate_parameters(symbol)
            normalized_symbol = symbol.upper().strip()

            # Get industry data to find peers
            industry_data = await self._provider.get_industry_analysis(
                normalized_symbol
            )

            return IndustryAnalysisToolOutput(data=industry_data).model_dump()

        except Exception as e:
            logger.error(f"Industry analysis failed: {str(e)}")
            return IndustryAnalysisToolOutput(data={}).model_dump()

    def _validate_parameters(self, symbol: str) -> None:
        """Validate input parameters."""
        if not symbol or not isinstance(symbol, str):
            raise ValueError("industry_analysis requires a valid symbol")

    def to_formatted_context(self, output: Dict[str, Any]) -> str:
        """Format industry analysis results for LLM context."""
        prompt_template = ChatPromptTemplate.from_template(
            "Industry analysis for Vietnamese stock {symbol}:\n{data}"
        )
        analysis_table = output.get("data", {}).get("analysis_table", {})
        insights = output.get("data", {}).get("insights", [])
        symbol = (
            output.get("data", {})
            .get("analysis_table", {})
            .get("symbol", "N/A")
        )

        formatted_data = []
        for symbol, data in analysis_table.items():
            quarter = data.get("quarter", "N/A")
            industry_name = data.get("industry_info", {}).get("name", "ngành")
            formatted_data.append(
                f"Stock: {symbol}, Quarter: {quarter}, Industry: {industry_name}"
            )
            metrics = data.get("metrics", {})
            for metric, value in metrics.items():
                if value != "N/A" and value is not None:
                    formatted_data.append(
                        f"  {metric.upper()}: {float(value):.2f}"
                    )

        formatted_data.extend(["Insights:"] + insights)
        return prompt_template.format(
            symbol=symbol,
            data=(
                "\n".join(formatted_data)
                if formatted_data
                else "No data available"
            ),
        )
