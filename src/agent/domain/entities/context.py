from typing import Any, Dict, List, Optional

from langchain_core.messages import BaseMessage
from pydantic import BaseModel, ConfigDict, Field


class QueryContext(BaseModel):
    """Context for user queries"""

    query: str
    session_id: Optional[str] = None
    query_type: Optional[str] = Field(
        None,
        description="Type of query (price, analysis, news, conversational)",
    )
    metadata: Dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(arbitrary_types_allowed=True)


class ToolResult(BaseModel):
    """Result from tool execution"""

    status: str = Field(
        ..., description="Status of tool execution: 'success' or 'error'"
    )
    data: Any = Field(None, description="Output data from the tool execution")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Metadata about the execution"
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)


class AgentState(BaseModel):
    """State object for tracking ReAct agent workflow with JSON action format."""

    messages: List[BaseMessage] = Field(
        default_factory=list,
        description="Conversation history as BaseMessage objects",
    )
    tools_used: List[str] = Field(
        default_factory=list,
        description="List of tool names used in the workflow",
    )
    intermediate_results: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="JSON-formatted intermediate results from tools",
    )

    plan: str = Field(
        default="", description="Current plan for achieving the user's goal"
    )
    sub_goals: List[str] = Field(
        default_factory=list, description="Sub-goals to achieve the main goal"
    )

    reflection_notes: List[str] = Field(
        default_factory=list, description="Notes from reflection steps"
    )
    tool_output: Optional[ToolResult] = Field(
        default=None, description="Result of the most recent tool execution"
    )
    tool_error: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Error details from the most recent tool execution",
    )
    action_history: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="History of JSON actions taken, e.g., {'action': 'tool_name', 'input': {...}}",
    )
    final_answer: Optional[str] = Field(
        default=None, description="Final answer to the user's query"
    )
    reflection_decision: Optional[str] = Field(
        default="continue",
        description="Decision from reflection node: 'continue', 'retry', or 'end'",
    )
    current_step: int = Field(
        default=0, description="Current step in the workflow"
    )
    max_steps: int = Field(
        default=10, description="Maximum allowed steps in the workflow"
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def to_dict(self) -> Dict[str, Any]:
        """Convert AgentState to dictionary for serialization."""
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentState":
        """Create AgentState from dictionary."""
        return cls(**data)
