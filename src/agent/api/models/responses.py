from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class ToolResultResponse(BaseModel):
    """Enhanced response model for tool execution results"""

    tool_name: str = Field(..., description="Name of the executed tool")
    success: bool = Field(
        ..., description="Whether tool execution was successful"
    )
    execution_time: float = Field(
        ..., description="Tool execution time in seconds"
    )
    error_message: Optional[str] = Field(
        None, description="Error message if execution failed"
    )
    data: Optional[Dict[str, Any]] = Field(
        None,
        description="Tool result data in JSON format per tool specification",
    )

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        json_schema_extra={
            "example": {
                "tool_name": "stock_price",
                "success": True,
                "execution_time": 1.23,
                "error_message": None,
                "data": {
                    "VCB": {
                        "price": 95500,
                        "change_percent": 1.17,
                        "volume": 2500000,
                    }
                },
            }
        },
    )


class AgentResponse(BaseModel):
    """Enhanced response model for agent queries with comprehensive metadata"""

    answer: str = Field(..., description="Final answer from the agent")
    success: bool = Field(
        ..., description="Whether the query was processed successfully"
    )
    steps: int = Field(..., description="Number of processing steps taken")
    tools_used: List[str] = Field(
        ..., description="List of tools used during processing"
    )
    intermediate_results: List[Dict[str, Any]] = Field(
        ...,
        description="JSON-formatted intermediate results from tool executions, including llm_output, observation, tool_name, and tool_result",
    )
    action_history: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="History of JSON actions taken, e.g., {'action': 'tool_name', 'input': {...}}",
    )
    session_id: Optional[str] = Field(None, description="Session identifier")
    timestamp: datetime = Field(
        default_factory=datetime.now, description="Response timestamp"
    )
    cached: Optional[bool] = Field(
        False, description="Whether result was served from cache"
    )
    query_hash: Optional[str] = Field(
        None, description="Hash of the original query for caching"
    )
    risk_warning: Optional[str] = Field(
        default="Thông tin chỉ mang tính tham khảo, không phải lời khuyên đầu tư. Quyết định đầu tư cần dựa trên phân tích cá nhân và khả năng chấp nhận rủi ro.",
        description="Mandatory risk warning for investment-related responses",
    )
    metadata: Optional[Dict[str, Any]] = Field(
        None, description="Additional processing metadata"
    )
    execution_time: Optional[float] = Field(
        None, description="Total execution time in seconds"
    )

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        json_schema_extra={
            "example": {
                "answer": "Giá cổ phiếu VCB hôm nay là 95,500 VND, tăng 1,100 VND (+1.17%) so với phiên trước.",
                "success": True,
                "steps": 3,
                "tools_used": ["stock_price", "chat_llm"],
                "intermediate_results": [
                    {
                        "tool_name": "stock_price",
                        "llm_output": '{"action": "stock_price", "input": {"symbols": ["VCB"], "data_type": "realtime"}}',
                        "observation": 'Stock price data: {"VCB": {"price": 95500, "change_percent": 1.17, "volume": 2500000}}',
                        "tool_result": {
                            "success": True,
                            "data": {
                                "VCB": {
                                    "price": 95500,
                                    "change_percent": 1.17,
                                    "volume": 2500000,
                                }
                            },
                        },
                    }
                ],
                "action_history": [
                    {
                        "action": "stock_price",
                        "input": {"symbols": ["VCB"], "data_type": "realtime"},
                    }
                ],
                "session_id": "uuid-session-id",
                "timestamp": "2025-08-27T23:39:00Z",
                "cached": False,
                "query_hash": "abc123",
                "risk_warning": "Thông tin chỉ mang tính tham khảo, không phải lời khuyên đầu tư. Quyết định đầu tư cần dựa trên phân tích cá nhân và khả năng chấp nhận rủi ro.",
                "metadata": {
                    "workflow_state": {"total_messages": 4, "current_step": 3}
                },
                "execution_time": 2.5,
            }
        },
    )


class DocumentProcessResponse(BaseModel):
    """Enhanced response model for document processing"""

    success: bool = Field(
        ..., description="Whether document processing was successful"
    )
    processed_documents: int = Field(
        ..., description="Number of successfully processed documents"
    )
    total_chunks: int = Field(
        ..., description="Total number of text chunks created"
    )
    failed_documents: List[str] = Field(
        ..., description="List of documents that failed to process"
    )
    processing_time: float = Field(
        ..., description="Total processing time in seconds"
    )
    total_documents: Optional[int] = Field(
        None, description="Total number of documents attempted"
    )
    success_rate: Optional[float] = Field(
        None, description="Success rate as percentage"
    )
    average_chunks_per_doc: Optional[float] = Field(
        None, description="Average chunks per document"
    )

    def __init__(self, **data):
        super().__init__(**data)
        if (
            self.total_documents is None
            and hasattr(self, "processed_documents")
            and hasattr(self, "failed_documents")
        ):
            self.total_documents = self.processed_documents + len(
                self.failed_documents
            )
        if (
            self.success_rate is None
            and self.total_documents
            and self.total_documents > 0
        ):
            self.success_rate = (
                self.processed_documents / self.total_documents
            ) * 100
        if (
            self.average_chunks_per_doc is None
            and self.total_chunks
            and self.processed_documents
            and self.processed_documents > 0
        ):
            self.average_chunks_per_doc = (
                self.total_chunks / self.processed_documents
            )


class SessionResponse(BaseModel):
    """Response model for session operations"""

    session_id: str = Field(..., description="Session identifier")
    created_at: Optional[datetime] = Field(
        None, description="Session creation timestamp"
    )
    expires_at: Optional[datetime] = Field(
        None, description="Session expiration timestamp"
    )
    last_accessed: Optional[datetime] = Field(
        None, description="Last access timestamp"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        None, description="Session metadata"
    )
    message: Optional[str] = Field(None, description="Response message")


class SessionHistoryResponse(BaseModel):
    """Response model for session history"""

    session_id: str = Field(..., description="Session identifier")
    history: List[Dict[str, Any]] = Field(
        ..., description="Conversation history as BaseMessage-compatible JSON"
    )
    pagination: Dict[str, Any] = Field(
        ..., description="Pagination information"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "session_id": "uuid-session-id",
                "history": [
                    {
                        "type": "human",
                        "content": "Giá VCB hôm nay?",
                        "timestamp": "2025-08-27T23:39:00Z",
                    },
                    {
                        "type": "ai",
                        "content": "Giá VCB hôm nay là 95,500 VND...",
                        "timestamp": "2025-08-27T23:39:01Z",
                    },
                ],
                "pagination": {
                    "total": 10,
                    "limit": 50,
                    "offset": 0,
                    "has_more": False,
                    "next_offset": None,
                },
            }
        }
    )


class DocumentListResponse(BaseModel):
    """Response model for document listing"""

    documents: List[str] = Field(
        ..., description="List of document identifiers/paths"
    )
    total_count: int = Field(..., description="Total number of documents")
    prefix: Optional[str] = Field(None, description="Filter prefix used")
    limit: Optional[int] = Field(None, description="Limit applied")
    has_more: Optional[bool] = Field(
        None, description="Whether more documents are available"
    )
    next_prefix: Optional[str] = Field(
        None, description="Next prefix for pagination"
    )

    def __init__(self, **data):
        super().__init__(**data)
        if self.has_more is None and self.limit and self.total_count:
            self.has_more = self.total_count >= self.limit


class ErrorResponse(BaseModel):
    """Standard error response model"""

    error: str = Field(..., description="Error message")
    error_code: Optional[str] = Field(
        None, description="Error code for programmatic handling"
    )
    details: Optional[Dict[str, Any]] = Field(
        None,
        description="Additional error details, including JSON parsing errors",
    )
    timestamp: datetime = Field(
        default_factory=datetime.now, description="Error timestamp"
    )
    request_id: Optional[str] = Field(
        None, description="Request identifier for tracking"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "error": "Query processing failed: Invalid JSON action format",
                "error_code": "INVALID_JSON_ACTION",
                "details": {
                    "provided_action": "not_a_json_string",
                    "expected_format": '{"action": "tool_name", "input": {...}}',
                },
                "timestamp": "2025-08-27T23:39:00Z",
                "request_id": "req-123-456",
            }
        }
    )


class SystemStatsResponse(BaseModel):
    """Response model for system statistics"""

    uptime_seconds: float = Field(..., description="System uptime in seconds")
    total_requests: int = Field(..., description="Total requests processed")
    successful_requests: int = Field(
        ..., description="Number of successful requests"
    )
    failed_requests: int = Field(..., description="Number of failed requests")
    average_response_time: float = Field(
        ..., description="Average response time in seconds"
    )
    active_sessions: int = Field(..., description="Number of active sessions")
    total_conversations: int = Field(
        ..., description="Total conversation messages"
    )
    cache_hit_rate: Optional[float] = Field(
        None, description="Cache hit rate percentage"
    )
    memory_usage_mb: Optional[float] = Field(
        None, description="Memory usage in MB"
    )
    cpu_usage_percent: Optional[float] = Field(
        None, description="CPU usage percentage"
    )
    timestamp: datetime = Field(
        default_factory=datetime.now, description="Statistics timestamp"
    )
