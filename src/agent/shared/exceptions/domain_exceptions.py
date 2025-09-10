# src/stock_assistant/shared/exceptions/domain_exceptions.py
class StockAssistantException(Exception):
    """Base exception for Stock Assistant"""

    pass


class ToolExecutionError(StockAssistantException):
    """Raised when tool execution fails"""

    def __init__(self, tool_name: str, error_message: str):
        self.tool_name = tool_name
        self.error_message = error_message
        super().__init__(f"Tool '{tool_name}' failed: {error_message}")


class VectorStoreError(StockAssistantException):
    """Raised when vector store operations fail"""

    pass


class DocumentProcessingError(StockAssistantException):
    """Raised when document processing fails"""

    pass
