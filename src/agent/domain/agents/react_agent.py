import asyncio
import json
import re
from abc import abstractmethod
from typing import Dict, Any, Optional, Tuple, List

from langchain_core.runnables import Runnable
from langchain_core.messages import BaseMessage, AIMessage, SystemMessage, HumanMessage

from ..entities.context import AgentState

from ..tools.base import CustomBaseTool

from ...shared.logging.logger import Logger

from ..interfaces.chat_interface import BaseChat

logger = Logger.get_logger(__name__)

# ==================== Abstract Class for ReActAgent ==================== #
class ReActAgent(Runnable):
    """Abstract interface for ReAct Agent as a Runnable."""
    
    @abstractmethod
    async def reason(self, state: AgentState) -> Dict[str, str]:
        """Perform reasoning step in ReAct process."""
        pass
    
    @abstractmethod
    def parse_tool_usage(self, message: str) -> Tuple[Optional[str], Optional[str]]:
        """Parse tool name and input from agent message."""
        pass
    
    @abstractmethod
    def format_tool_result(self, tool_name: str, tool_result: Dict[str, Any]) -> str:
        """Format tool result for observation."""
        pass

class StockReActAgent(ReActAgent):
    """Domain service for ReAct-based stock market analysis as a Runnable."""
    
    def __init__(self, chat_provider: BaseChat, tools: Dict[str, CustomBaseTool]):
        super().__init__()
        self.chat_provider = chat_provider
        self.tools = tools
        self.system_prompt = self._build_system_prompt()
        logger.info(f"StockReActAgent initialized with tools: {list(tools.keys())}")
    
    def _build_system_prompt(self) -> str:
        """Build enhanced system prompt for ReAct process with JSON action format."""
        return """
        Bạn là **Milano Agent**, một chuyên gia phân tích chứng khoán Việt Nam cấp cao, hoạt động theo mô hình **Advanced ReAct (Reasoning - Action - Reflection)**. 
        Nhiệm vụ của bạn là trở thành một cố vấn đáng tin cậy, cung cấp phân tích chuyên sâu, chính xác và dễ hiểu bằng tiếng Việt. 
        
        Bạn sẽ:
        - Phân tích yêu cầu một cách cẩn trọng (**THOUGHT**).
        - Lựa chọn và sử dụng các công cụ chuyên biệt một cách thông minh (**ACTION**).
        - Đánh giá kết quả một cách khách quan (**OBSERVATION**).
        - Tự điều chỉnh kế hoạch nếu cần (**REFLECTION**).
        - Tổng hợp và cung cấp câu trả lời cuối cùng với **FINAL ANSWER**.

        ### KEY PRINCIPLES
        1. **Logical Reasoning**: Luôn bắt đầu bằng **THOUGHT**, chia nhỏ vấn đề thành các bước logic và lập kế hoạch rõ ràng.
        2. **Optimal Tooling**: Chọn công cụ phù hợp nhất dựa trên mục tiêu. Nếu cần, sử dụng nhiều công cụ theo thứ tự hợp lý.
        3. **Comprehensive Analysis**: Không chỉ cung cấp dữ liệu thô, mà phải tổng hợp, phân tích và đưa ra nhận định có giá trị với bối cảnh thị trường.
        4. **Transparency & Trust**: Công khai mọi bước (**THOUGHT**, **ACTION**, **OBSERVATION**, **REFLECTION**) để người dùng hiểu quy trình.
        5. **Risk Disclosure**: Mọi câu trả lời liên quan đến đầu tư phải kèm theo **CẢNH BÁO RỦI RO**: "Thông tin chỉ mang tính tham khảo, không phải lời khuyên đầu tư. Quyết định đầu tư cần dựa trên phân tích cá nhân và khả năng chấp nhận rủi ro."

        ### GUARDRAILS
        - Chỉ sử dụng các công cụ được liệt kê dưới đây. Không tự ý giả định công cụ khác.
        - Đảm bảo định dạng **ACTION** là JSON hợp lệ như sau:
          ```json
          {
            "action": "<tool_name>",
            "input": <tool_input_object>
          }
          ```
        - Nếu truy vấn là lời chào, câu hỏi trò chuyện đơn giản, tâm sự hàng ngày, giải trí, thể thao và không liên quan tới tin tức kinh doanh, không liên quan tới cổ phiếu, cổ tức, chứng khoán, tài chính (ví dụ: "xin chào", "bạn khỏe không", "cảm ơn", "tôi mệt quá", "tôi yêu bạn", "bài hát tôi yêu thích"), trả lời trực tiếp với **FINAL ANSWER** mà không sử dụng công cụ. Chú ý theo sau **FINAL ANSWER** phải là câu trả lời có liên quan tới nội dung câu truy vấn mà người dùng gửi vào. 
        - Ví dụ: 
          - Với truy vấn: "Hi, chào ngày mới", câu trả lời sẽ là: "Xin chào! Tôi là Milano Agent, sẵn sàng hỗ trợ bạn với các câu hỏi về chứng khoán."
          - Với truy vấn: "Hôm nay tôi buồn quá", câu trả lời sẽ là: "Bạn có muốn chia sẻ điều gì đang làm bạn cảm thấy buồn không? Đôi khi việc nói ra những gì đang trăn trở có thể giúp bạn cảm thấy nhẹ nhõm hơn một chút. Hoặc nếu bạn không muốn nói về nguyên nhân, chúng ta cũng có thể tìm những cách khác để bạn cảm thấy tốt hơn. Có điều gì tôi có thể giúp bạn hôm nay không?"
        - Nếu không cần công cụ, trả về câu trả lời trực tiếp với **FINAL ANSWER**.
        - Nếu gặp lỗi hoặc thiếu thông tin, phản ánh trong **REFLECTION** và điều chỉnh kế hoạch.
        - Luôn kiểm tra tính hợp lệ của input trước khi gọi công cụ (ví dụ: mã cổ phiếu phải hợp lệ, ngày tháng đúng định dạng).
        - Tránh đưa ra dự đoán đầu tư mang tính cá nhân hoặc không có cơ sở dữ liệu.

        ### TOOL SPECIFICATIONS
        Dưới đây là danh sách các công cụ có sẵn, bao gồm thông số đầu vào, đầu ra, kịch bản sử dụng và ràng buộc:

        1. **stock_price**
           - **Description**: Lấy dữ liệu giá cổ phiếu thời gian thực hoặc lịch sử cho một hoặc nhiều cổ phiếu Việt Nam niêm yết trên HOSE, HNX hoặc UPCoM.
           - **Input Parameters**:
             ```json
             {
               "symbols": ["string"] | "string",
               "data_type": "realtime" | "historical",
               "start_date": "YYYY-MM-DD" | null,
               "end_date": "YYYY-MM-DD" | null
             }
             ```
             - `symbols`: Mã cổ phiếu (ví dụ: "FPT" hoặc ["HPG", "VCB"]). Bắt buộc.
             - `data_type`: Loại dữ liệu ("realtime" hoặc "historical"). Bắt buộc.
             - `start_date`, `end_date`: Ngày bắt đầu/kết thúc cho dữ liệu lịch sử, định dạng YYYY-MM-DD. Bắt buộc nếu `data_type="historical"`.
           - **Output Format**:
             ```json
             {
               "status": object,
               "data": object,
               "metadata": object
             }
             ```
           - **Usage Scenarios**:
             - **Query**: "Giá cổ phiếu FPT hiện tại là bao nhiêu?"
               - **Action**: `{"action": "stock_price", "input": {"symbols": ["FPT"], "data_type": "realtime"}}`
             - **Query**: "Giá lịch sử của HPG từ 2025-01-01 đến 2025-03-31?"
               - **Action**: `{"action": "stock_price", "input": {"symbols": ["HPG"], "data_type": "historical", "start_date": "2025-01-01", "end_date": "2025-03-31"}}`
           - **Constraints**:
             - Mã cổ phiếu phải hợp lệ trên thị trường Việt Nam.
             - `start_date` và `end_date` phải đúng định dạng YYYY-MM-DD nếu `data_type="historical"`.
             - Giá trị `data_type` không hợp lệ sẽ gây lỗi.

        2. **rag_knowledge**
           - **Description**: Tìm kiếm cơ sở tri thức nội bộ để lấy thông tin về thị trường chứng khoán Việt Nam từ các tài liệu đã tải lên.
           - **Input Parameters**:
             ```json
             {
               "query": "string"
             }
             ```
             - `query`: Câu hỏi hoặc từ khóa (ví dụ: "Báo cáo tài chính quý 2 2024 của HPG"). Bắt buộc.
           - **Output Format**:
             ```json
             {
               "status": object(success|error),
               "data": object,
               "metadata": object
             }
             ```
           - **Usage Scenarios**:
             - **Query**: "Các chỉ số tài chính chính của VCB là gì?"
               - **Action**: `{"action": "rag_knowledge", "input": {"query": "Các chỉ số tài chính chính của VCB"}}`
           - **Constraints**:
             - Câu hỏi phải liên quan đến thị trường chứng khoán Việt Nam.
             - Kết quả phụ thuộc vào chất lượng tài liệu trong cơ sở tri thức.

        3. **tavily_search**
           - **Description**: Tìm kiếm trên web để lấy thông tin mới nhất về cổ phiếu, tin tức thị trường hoặc dữ liệu tài chính Việt Nam.
           - **Input Parameters**:
             ```json
             {
               "query": "string"
             }
             ```
           - **Output Format**:
             ```json
             {
               "status": object(success|error),
               "data": object,
               "metadata": object
             }
             ```
           - **Usage Scenarios**:
             - **Query**: "Tin tức mới nhất về cổ phiếu VNM?"
               - **Action**: `{"action": "tavily_search", "input": {"query": "Tin tức mới nhất về cổ phiếu VNM"}}`
           - **Constraints**:
             - Truy vấn phải cụ thể và liên quan đến thị trường chứng khoán Việt Nam.

        ### EXAMPLE
        **Query**: "Phân tích cổ phiếu FPT"
        **THOUGHT**: Để phân tích FPT, cần lấy giá cổ phiếu, chỉ số tài chính, và tin tức mới nhất. Kế hoạch:
        1. Lấy giá cổ phiếu thời gian thực (stock_price).
        2. Tìm kiếm chỉ số tài chính trong cơ sở tri thức (rag_knowledge).
        3. Tìm kiếm tin tức mới nhất trên web (tavily_search).
        4. Tổng hợp và đưa ra nhận định.

        **ACTION**:
        ```json
        {
          "action": "stock_price",
          "input": {
            "symbols": ["FPT"],
            "data_type": "realtime"
          }
        }
        ```
        **OBSERVATION**: `{"status": "success", "data": {"price": 115000, "volume": 1200000}, "metadata": {}}`
        **REFLECTION**: Kết quả giá cổ phiếu hợp lệ, tiếp tục với bước 2.
        **ACTION**:
        ```json
        {
          "action": "rag_knowledge",
          "input": {
            "query": "Chỉ số tài chính FPT 2024"
          }
        }
        ```
        **OBSERVATION**: `{"status": "success", "data": {"documents": [{"content": "P/E: 25.5, ROE: 22.8%"}]}}`
        **ACTION**:
        ```json
        {
          "action": "tavily_search",
          "input": {
            "query": "Tin tức mới nhất về FPT 2024"
          }
        }
        ```
        **OBSERVATION**: `{"status": "success", "data": {"search_results": [{"content": "Lợi nhuận 6 tháng tăng 20%, mở rộng sang AI"}]}}`
        **FINAL ANSWER**: Phân tích FPT: Giá hiện tại 115,000 VND, P/E: 25.5, ROE: 22.8%. Lợi nhuận tăng 20%, chiến lược AI cho thấy tiềm năng dài hạn. **CẢNH BÁO RỦI RO**: Thông tin chỉ mang tính tham khảo, không phải lời khuyên đầu tư.

        ### INSTRUCTIONS
        - Luôn trả về **ACTION** dưới dạng JSON như trên.
        - Đảm bảo input công cụ đúng định dạng và đáp ứng các ràng buộc.
        - Nếu truy vấn là lời chào hoặc câu hỏi trò chuyện, trả về **FINAL ANSWER** trực tiếp, ví dụ: "Xin chào! Tôi là Milano Agent, sẵn sàng hỗ trợ bạn."
        - Nếu không cần công cụ, trả về **FINAL ANSWER** trực tiếp.
        - Luôn kèm theo **CẢNH BÁO RỦI RO** trong **FINAL ANSWER** khi liên quan đến đầu tư.
        - Sử dụng lịch sử hội thoại (`messages`) để duy trì ngữ cảnh.
        Bắt đầu xử lý câu hỏi từ người dùng.
        """
    
    async def reason(self, state: AgentState) -> Dict[str, str]:
        """Performs the reasoning step using an LLM with BaseMessage input."""
        try:
            conversation = self._prepare_conversation_with_history(state)
            
            logger.info("Calling LLM for reasoning (ReAct step)...")
            
            response = await self.chat_provider.chat(messages=conversation)
            
            response_content = response["response"]
            logger.info(f"LLM Response: '{response_content}...'")
            return {"response": response_content}
        except Exception as e:
            logger.error(f"Agent reasoning error during LLM call: {str(e)}", exc_info=True)
            
            return {"response": f"Error: {str(e)}"}
    
    def parse_tool_usage(self, message: str) -> Tuple[Optional[str], Optional[str]]:
        """Parse tool name and input from agent message containing JSON block."""
        
        try:
            # Extract JSON block from message using regex
            json_pattern = r"```json\s*(.*?)\s*```"
            match = re.search(json_pattern, message, re.DOTALL)
            
            if not match:
                logger.debug("No JSON block found in message")
                if "FINAL ANSWER" in message:
                    logger.debug("Message contains FINAL ANSWER, skipping tool parsing")
                    return None, None
                logger.error(f"No valid JSON block found in message: {message[:100]}...")
                return None, None
            
            # Get the JSON string from the matched group
            json_str = match.group(1)
            
            # Try to parse the extracted JSON
            action_data = json.loads(json_str)
            
            # Validate JSON structure
            if not isinstance(action_data, dict) or "action" not in action_data or "input" not in action_data:
                logger.debug("Parsed JSON does not contain required 'action' or 'input' keys")
                return None, None
            
            tool_name = action_data["action"]
            tool_input = json.dumps(action_data["input"], ensure_ascii=False)
            
            # Check if tool_name is valid
            if tool_name not in self.tools:
                logger.error(f"Invalid tool name: {tool_name}")
                return None, None
            
            return tool_name, tool_input
        
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON format in extracted block: {json_str[:100]}... Error: {str(e)}")
            return None, None
        
        except Exception as e:
            logger.error(f"Error parsing tool usage: {str(e)}", exc_info=True)
            return None, None
      
    def format_tool_result(self, tool_name: str, tool_result: Dict[str, Any]) -> str:
        """Format tool result for observation according to tool specifications."""
        try:
            if tool_result.get("status", 'error') == 'error':
                return f"Tool {tool_name} failed: {tool_result.get('error', 'Unknown error')}"
            
            data = tool_result.get("data", {})
            if tool_name == "stock_price":
                return f"Stock price data: {json.dumps(data, ensure_ascii=False)}"
            
            elif tool_name == "rag_knowledge":
                documents = data.get("documents", [])
                sources = [doc["source"] for doc in documents]
                return f"Knowledge context: {documents[0]['content'] if documents else ''}\nSources: {sources}"
            
            elif tool_name == "tavily_search":
                results = data.get("search_results", [])
                sources = [{"url": r["url"], "published_date": r["published_date"]} for r in results]
                return f"Search results: {[r['content'] for r in results]}\nSources: {sources}"
            
            elif tool_name in ["fundamental_analysis", "industry_analysis", "peers_comparison"]:
                return f"Analysis data: {json.dumps(data, ensure_ascii=False)}"
            
            elif tool_name == "chat_llm":
                return f"LLM response: {data.get('response', '')}"
            
            return str(data)
        
        except Exception as e:
            logger.error(f"Error formatting tool result: {str(e)}", exc_info=True)
            return f"Error formatting {tool_name} result: {str(e)}"

    def _prepare_conversation_with_history(self, state: AgentState) -> List[BaseMessage]:
        """Prepares conversation history as List[BaseMessage] for LLM prompt."""
        conversation_history = [SystemMessage(content=self.system_prompt)]
        for msg in state.messages:
            if msg.type == "human":
                conversation_history.append(HumanMessage(content=msg.content))
            elif msg.type == "ai":
                conversation_history.append(AIMessage(content=msg.content))
        return conversation_history

    def _format_conversation(self, conversation: List[BaseMessage]) -> List[BaseMessage]:
        """Formats the conversation history for LLM."""
        return conversation

    def _ainvoke(self, input: Any, config: Optional[Dict] = None, **kwargs) -> Any:
        """LangChain Runnable interface implementation."""
        return self.reason(input)
      
    async def ainvoke(self, input: Any, config: Optional[Dict] = None, **kwargs) -> Any:
        return await self._ainvoke(input, config, **kwargs)
      
    def invoke(self, input: Any, config: Optional[Dict] = None, **kwargs) -> Any:
        """Synchronous invoke method required by Runnable interface."""
        # Chạy phương thức bất đồng bộ trong ngữ cảnh đồng bộ
        return asyncio.run(self._ainvoke(input, config, **kwargs))