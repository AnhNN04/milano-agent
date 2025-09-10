from typing import Any, Dict, List, Type, Union

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field, PrivateAttr

from ...infra.providers.vnstock_analysis_provider import BaseStockAnalysis
from ...shared.logging.logger import Logger
from .base import CustomBaseTool

logger = Logger.get_logger(__name__)


class PeersComparisonToolInput(BaseModel):
    """Input schema for PeersComparisonTool."""

    symbols: Union[str, List[str]] = Field(
        ...,
        description="List of stock symbols for comparison (e.g., ['HPG', 'NKG', 'HSG']).",
    )


class PeersComparisonToolOutput(BaseModel):
    """Output schema for PeersComparisonTool."""

    data: Dict[str, Any] = Field(
        description="Comparison table, rankings, and peers data"
    )


class PeersComparisonTool(CustomBaseTool):
    """Tool for comparing stocks with their market peers."""

    name: str = "peers_comparison"
    description: str = (
        "Compare stocks with their market peers using financial ratios"
    )
    args_schema: Type[BaseModel] = PeersComparisonToolInput

    _provider: PrivateAttr = PrivateAttr()

    def __init__(self, stock_analysis_provider: BaseStockAnalysis):
        super().__init__()
        self._provider = stock_analysis_provider

    async def _execute_impl(
        self,
        symbols: Union[str, List[str]],
    ) -> Dict[str, Any]:
        """Execute peers comparison analysis."""
        try:
            # Validate and normalize inputs
            self._validate_parameters(symbols)
            normalized_symbols = self._normalize_symbols(symbols)

            # Get both fundamental and peers data
            peers_data = await self._provider.get_peers_comparison(
                normalized_symbols
            )

            return PeersComparisonToolOutput(data=peers_data).model_dump()

        except Exception as e:
            logger.error(f"Peers comparison failed: {str(e)}")
            return PeersComparisonToolOutput(data={}).model_dump()

    def _validate_parameters(self, symbols: Union[str, List[str]]) -> None:
        """Validate input parameters."""
        if isinstance(symbols, str):
            symbols = [s.strip() for s in symbols.split(",")]
        if not isinstance(symbols, list) or len(symbols) < 2:
            raise ValueError(
                f"peers_comparison requires at least two symbols: {symbols}"
            )

    def _normalize_symbols(self, symbols: Union[str, List[str]]) -> List[str]:
        """Normalize symbols to a list of uppercase strings."""
        if isinstance(symbols, str):
            symbols = [s.strip().upper() for s in symbols.split(",")]
        return [s.strip().upper() for s in symbols]

    def to_formatted_context(self, output: Dict[str, Any]) -> str:
        """Format peers comparison results for LLM context."""
        prompt_template = ChatPromptTemplate.from_template(
            "Peers comparison for Vietnamese stocks:\n{data}"
        )
        comparison_table = output.get("data", {}).get("comparison_table", {})
        insights = output.get("data", {}).get("insights", [])

        formatted_data = []
        for symbol, data in comparison_table.items():
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
