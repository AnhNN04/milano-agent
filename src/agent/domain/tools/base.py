import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Type

from langchain.tools import StructuredTool
from pydantic import BaseModel, Field

from ..entities.context import ToolResult


class ToolState(BaseModel):
    """State object to track tool execution context for LangGraph integration."""

    tool_name: str = Field(description="Name of the tool")
    execution_time: float = Field(description="Time taken for execution")
    context: Dict[str, Any] = Field(
        default_factory=dict, description="Additional context for graph state"
    )


class CustomBaseTool(StructuredTool, ABC):
    """
    Abstract base class for all tools in the stock assistant system,
    inheriting from LangChain's StructuredTool for enhanced schema support.
    Integrates with LangGraph for state management and workflow orchestration.
    """

    args_schema: Optional[Type[BaseModel]] = None
    return_direct: bool = False
    state: Optional[ToolState] = None  # State for LangGraph integration

    async def _arun(self, *args, **kwargs) -> ToolResult:
        """Main execution logic, wrapping results in ToolResult and updating state."""

        start_time = time.time()

        try:
            result_data = await self._execute_impl(
                **kwargs
            )  # Expected to return dict
            end_time = time.time()
            execution_time = end_time - start_time

            # Update state for LangGraph
            self.state = ToolState(
                tool_name=self.name,
                execution_time=execution_time,
                context={
                    "market": "VN"
                },  # Example: Add Vietnam market context
            )

            return ToolResult(
                status="success",
                data=result_data,
                metadata=self.state.model_dump(),  # Serialize state for LangGraph
            )

        except Exception as e:
            end_time = time.time()
            execution_time = end_time - start_time

            # Update state with error information
            self.state = ToolState(
                tool_name=self.name,
                execution_time=execution_time,
                context={"market": "VN"},
                error=str(e),
            )

            return ToolResult(
                status="error", data={}, metadata=self.state.model_dump()
            )

    @abstractmethod
    async def _execute_impl(self, **kwargs) -> Any:
        """
        Abstract method to be implemented by child classes.
        Must return a dictionary which will be wrapped in ToolResult.
        """

        pass

    def to_langgraph_node(self) -> Any:
        """
        Convert tool to a LangGraph node for workflow orchestration.
        Returns configuration for integration with LangGraph's StateGraph.
        """

        return {
            "name": self.name,
            "tool": self,
            "state_update": lambda result: {
                "tool_state": self.state.model_dump() if self.state else {}
            },
        }
