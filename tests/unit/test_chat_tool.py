import os
import sys

# Lấy đường dẫn project root (chỉnh tuỳ cấu trúc của bạn)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, PROJECT_ROOT)

import pytest
from unittest.mock import AsyncMock, MagicMock
from typing import Dict, Any, List

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

# Import the class to be tested
from src.agent.domain.tools.chat_tool import ChatTool, ChatToolInput
from src.agent.domain.interfaces.chat_interface import BaseChat
from src.agent.shared.settings.settings import settings
from src.agent.shared.logging.logger import Logger
from src.agent.domain.entities.context import ToolResult

# Mock the logger to prevent console output during tests
Logger.get_logger = MagicMock(return_value=MagicMock())

# Mock the settings module as well
settings.llm = MagicMock()
settings.llm.default_provider = "gemini"

# Define a fixture for a mock chat provider to simulate the LLM
@pytest.fixture
def mock_chat_provider():
    """Fixture for a mock BaseChat provider."""
    mock = AsyncMock(spec=BaseChat)
    mock.chat.return_value = {"response": "This is a test response.", "model": "test-model"}
    return mock

# Define a fixture for the ChatTool instance
@pytest.fixture
def chat_tool(mock_chat_provider):
    """Fixture for a ChatTool instance with a mocked provider."""
    return ChatTool(chat_provider=mock_chat_provider)

# Helper function to create a list of mock messages
def create_messages(user_query: str) -> List[BaseMessage]:
    """Helper to create a list of BaseMessage objects."""
    return [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content=user_query)
    ]

# --- Test Cases ---

@pytest.mark.asyncio
async def test_chat_tool_successful_execution(chat_tool, mock_chat_provider):
    """Test a successful execution of the ChatTool."""
    print("\n--- Running test_chat_tool_successful_execution ---")
    messages = create_messages("What is the current price of stock ABC?")
    
    # Execute the tool's private method
    result: Dict[str, Any] = await chat_tool._execute_impl(messages=messages)
    
    # Assertions
    assert "response" in result
    assert result["response"] == "This is a test response."
    mock_chat_provider.chat.assert_awaited_once_with(messages=messages)
    print("Test passed: Successful execution worked as expected.")

@pytest.mark.asyncio
async def test_chat_tool_with_error_during_execution(chat_tool, mock_chat_provider):
    """Test that the tool handles exceptions gracefully."""
    print("\n--- Running test_chat_tool_with_error_during_execution ---")
    messages = create_messages("Generate a long, complex report.")
    
    # Configure the mock to raise an exception
    mock_chat_provider.chat.side_effect = Exception("API call failed")
    
    # Execute the tool's private method
    result = await chat_tool._execute_impl(messages=messages)
    
    # Assertions
    assert "response" in result
    assert result["response"] == ""
    print("Test passed: Exception was handled, and an empty response was returned.")

@pytest.mark.asyncio
async def test_chat_tool_content_filter_triggered(chat_tool, mock_chat_provider):
    """Test the content filtering mechanism."""
    print("\n--- Running test_chat_tool_content_filter_triggered ---")
    sensitive_response = "This response contains inappropriate content."
    
    # Configure the mock to return a sensitive response
    mock_chat_provider.chat.return_value = {"response": sensitive_response}
    
    messages = create_messages("What should I do?")
    
    # Execute the tool's private method
    result = await chat_tool._execute_impl(messages=messages)
    
    # Assertions
    assert "response" in result
    assert result["response"] == "Kết quả đã được lọc do chứa nội dung không phù hợp."
    print("Test passed: Content filter successfully blocked the sensitive response.")

@pytest.mark.asyncio
async def test_chat_tool_with_empty_messages(chat_tool, mock_chat_provider):
    """Test the tool with an empty list of messages."""
    print("\n--- Running test_chat_tool_with_empty_messages ---")
    messages = []
    
    # The tool should still try to call the provider, which might handle this,
    # but the test checks if the execution itself doesn't crash.
    result = await chat_tool._execute_impl(messages=messages)
    
    # Assertions
    assert "response" in result
    assert result["response"] == "This is a test response."
    mock_chat_provider.chat.assert_awaited_once_with(messages=messages)
    print("Test passed: Empty messages list was handled without crashing.")

@pytest.mark.asyncio
async def test_chat_tool_to_langgraph_node_conversion(chat_tool):
    """Test the to_langgraph_node method for proper format."""
    print("\n--- Running test_chat_tool_to_langgraph_node_conversion ---")
    
    # The to_langgraph_node method should be callable and return a dictionary
    node_config = chat_tool.to_langgraph_node()
    
    # Assertions
    assert isinstance(node_config, dict)
    assert "name" in node_config
    assert "tool" in node_config
    assert "state_update" in node_config
    assert node_config["name"] == "chat_llm"
    print("Test passed: to_langgraph_node returns the correct structure.")

@pytest.mark.asyncio
async def test_chat_tool_arun_integration_success(chat_tool, mock_chat_provider):
    """Test the main _arun method with a successful execution."""
    print("\n--- Running test_chat_tool_arun_integration_success ---")
    messages = create_messages("How is the market today?")
    
    # Call the main _arun method
    tool_result: ToolResult = await chat_tool._arun(messages=messages)
    
    # Assertions
    assert isinstance(tool_result, ToolResult)
    assert tool_result.status == "success"
    assert "response" in tool_result.data
    assert tool_result.data["response"] == "This is a test response."
    assert "state" in tool_result.metadata
    assert tool_result.metadata["state"]["tool_name"] == "chat_llm"
    print("Test passed: _arun method successfully wrapped the result.")

@pytest.mark.asyncio
async def test_chat_tool_input_schema():
    """Test the Pydantic schema for ChatToolInput."""
    print("\n--- Running test_chat_tool_input_schema ---")
    
    # Validate the schema's fields
    schema = ChatToolInput.model_json_schema()
    
    assert "messages" in schema["properties"]
    assert schema["properties"]["messages"]["description"] == "List of messages for the LLM, typically including system and user messages."
    print("Test passed: ChatToolInput schema is correct.")

# Run the tests with `pytest` from the terminal.
# The 'print' statements are for demonstration and can be removed for cleaner output.