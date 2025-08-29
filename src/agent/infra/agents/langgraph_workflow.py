import re
import time
import json
from typing import Dict, Any, Optional

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableLambda

from ...domain.entities.context import AgentState, ToolResult

from ...domain.agents.react_agent import StockReActAgent

from ...domain.tools.base import CustomBaseTool
from ...domain.tools.chat_tool import ChatToolInput
from ...domain.tools.fundamental_analysis_tool import FundamentalAnalysisToolInput
from ...domain.tools.industry_analysis_tool import IndustryAnalysisToolInput
from ...domain.tools.peers_comparison_tool import PeersComparisonToolInput
from ...domain.tools.rag_tool import RAGToolInput
from ...domain.tools.stock_data_tool import StockPriceToolInput
from ...domain.tools.tavily_search_tool import TavilySearchToolInput

from ...shared.logging.logger import Logger

logger = Logger.get_logger(__name__)


class ReActWorkflow:
    """Infrastructure layer for building and running LangGraph workflow with reasoning, action, reflection, and final output nodes."""

    # Ánh xạ tool_name với Pydantic InputModel
    TOOL_INPUT_MODELS = {
        "chat_llm": ChatToolInput,
        "fundamental_analysis": FundamentalAnalysisToolInput,
        "industry_analysis": IndustryAnalysisToolInput,
        "peers_comparison": PeersComparisonToolInput,
        "rag_knowledge": RAGToolInput,
        "stock_price": StockPriceToolInput,
        "tavily_search": TavilySearchToolInput,
    }


    def __init__(self, agent: StockReActAgent, tools: Dict[str, CustomBaseTool]):
        self.agent = agent
        self.tools = tools
        self.checkpointer = MemorySaver()
        self.workflow = self._build_workflow()
        logger.info("ReActWorkflow initialized with 4-node architecture")


    def _build_workflow(self) -> StateGraph:
        """Build LangGraph workflow with reasoning, action, reflection, and final output nodes."""
        workflow = StateGraph(AgentState)
        agent_runnable = RunnableLambda(self.agent.reason)

        workflow.add_node("reasoning", self._reasoning_node(agent_runnable))
        workflow.add_node("action", self._action_node)
        workflow.add_node("reflection", self._reflection_node)
        workflow.add_node("final_output", self._final_output_node)

        workflow.set_entry_point("reasoning")
        workflow.add_edge("reasoning", "action")
        workflow.add_edge("action", "reflection")
        workflow.add_conditional_edges(
            "reflection",
            self._should_continue,
            {
                "continue": "reasoning",
                "end": "final_output"
            }
        )
        workflow.add_edge("final_output", END)

        return workflow.compile(checkpointer=self.checkpointer)


    def _reasoning_node(self, agent_runnable: RunnableLambda):
        """Create reasoning node using agent as a Runnable."""
        async def node(state: AgentState) -> AgentState:
            node_start_time = time.time()
            logger.info(f"=== REASONING NODE START === Step: {state.current_step}")

            try:
                llm_response = await agent_runnable.ainvoke(state)
                llm_response_content = llm_response["response"] if isinstance(llm_response, dict) else llm_response

                updated_state = state.model_copy()
                updated_state.plan = updated_state.plan or "Generate plan to answer user query using available tools."
                updated_state.sub_goals = updated_state.sub_goals or ["Identify relevant tools", "Execute tools", "Synthesize results"]
                updated_state.messages.append(AIMessage(content=llm_response_content))

                # Check for simple conversational queries (e.g., greetings)
                if "FINAL ANSWER" in llm_response_content:
                    logger.info("Detected FINAL ANSWER in reasoning output, setting state for early termination")
                    final_answer_match = re.search(r"\*\*FINAL ANSWER\*\*:\s*(.*?)(?=\n|$)", llm_response_content, re.DOTALL)
                    if final_answer_match:
                        updated_state.final_answer = final_answer_match.group(1).strip()
                        updated_state.reflection_decision = "end"

                logger.info(
                    f"Reasoning completed - Plan: {updated_state.plan}..., "
                    f"Sub-goals: {updated_state.sub_goals}, "
                    f"Final answer set: {bool(updated_state.final_answer)}"
                )
                return updated_state

            except Exception as e:
                logger.error(f"Error in reasoning node: {str(e)}", exc_info=True)
                updated_state = state.model_copy()
                updated_state.messages.append(AIMessage(content=f"Error in reasoning: {str(e)}"))
                updated_state.current_step += 1
                updated_state.final_answer = f"Error: {str(e)}"
                updated_state.reflection_decision = "end"
                return updated_state

        return node


    async def _action_node(self, state: AgentState) -> AgentState:
        """Action node: Execute tools based on reasoning output."""
        node_start_time = time.time()
        logger.info(f"=== ACTION NODE START === Step: {state.current_step}")

        try:
            # Skip action node if final answer is already set
            if state.final_answer:
                logger.info("Final answer already set, skipping action node")
                action_state = state.model_copy()
                action_state.reflection_decision = "end"
                return action_state

            llm_response = state.messages[-1].content if state.messages else ""
            tool_name, tool_input = self.agent.parse_tool_usage(llm_response)

            if tool_name and tool_input:
                logger.info(f"Executing tool: {tool_name} with input: {tool_input}...")
                tool_result = await self._execute_tool_safely(tool_name, tool_input)
                
                formatted_result = self.agent.format_tool_result(tool_name, tool_result)

                updated_intermediate_results = state.intermediate_results + [
                    {"llm_output": llm_response, "observation": formatted_result, "tool_name": tool_name, "success": tool_result.get("status") == "success"}
                ]
                
                updated_tools_used = state.tools_used + [tool_name]

                action_state = state.model_copy()
                action_state.messages.append(AIMessage(content=f"Observation: {formatted_result}"))
                action_state.current_step += 1
                action_state.tools_used = updated_tools_used
                action_state.intermediate_results = updated_intermediate_results
                action_state.tool_output = ToolResult(**tool_result)
                
                return action_state
            
            else:
                logger.warning("No valid tool action found in reasoning output")
                updated_state = state.model_copy()
                updated_state.messages.append(AIMessage(content="No valid tool action found"))
                updated_state.current_step += 1
                updated_state.reflection_decision = "end" if state.final_answer else "continue"
                return updated_state

        except Exception as e:
            logger.error(f"Error in action node: {str(e)}", exc_info=True)
            updated_state = state.model_copy()
            updated_state.messages.append(AIMessage(content=f"Error in action: {str(e)}"))
            updated_state.current_step += 1
            updated_state.final_answer = f"Error: {str(e)}"
            updated_state.reflection_decision = "end"
            return updated_state


    async def _reflection_node(self, state: AgentState) -> AgentState:
        """Reflection node: Evaluate tool results and adjust plan using LLM."""
        node_start_time = time.time()
        logger.info(f"=== REFLECTION NODE START === Step: {state.current_step}")

        try:
            updated_state = state.model_copy()

            # Skip reflection if final answer is already set
            if state.final_answer:
                logger.info("Final answer already set, proceeding to final output")
                updated_state.reflection_decision = "end"
                updated_state.reflection_notes.append("Skipped reflection due to existing final answer")
                return updated_state

            # Prepare system prompt for LLM evaluation
            reflection_prompt = """
            Bạn là một trợ lý đánh giá dữ liệu thông minh. 
            Nhiệm vụ của bạn là phân tích câu hỏi người dùng và các kết quả trung gian từ các công cụ để quyết định liệu có đủ dữ liệu để trả lời câu hỏi, cần thử lại, hay cần tiếp tục thu thập thêm dữ liệu.

            ### HƯỚNG DẪN
            - Đánh giá câu hỏi gốc và các kết quả trung gian (`intermediate_results`).
            - Xác định xem dữ liệu có đủ để trả lời đầy đủ câu hỏi hay không:
              - Nếu câu hỏi yêu cầu thông tin cụ thể (ví dụ: giá cổ phiếu, chỉ số tài chính), kiểm tra xem tất cả dữ liệu cần thiết đã có chưa.
              - Nếu có lỗi trong kết quả công cụ, đề xuất thử lại với cách tiếp cận khác.
              - Nếu dữ liệu còn thiếu, đề xuất tiếp tục với các bước bổ sung.
            - Trả về quyết định trong định dạng JSON với các trường:
              - `decision`: "continue" (tiếp tục thu thập dữ liệu), "retry" (thử lại do lỗi), hoặc "end" (đủ dữ liệu để trả lời).
              - `reason`: Lý do cho quyết định, giải thích rõ ràng.
              - `updated_plan`: Kế hoạch cập nhật nếu cần thử lại hoặc tiếp tục (chuỗi rỗng nếu không cần).
            - Đảm bảo quyết định rõ ràng và phù hợp với ngữ cảnh.

            ### VÍ DỤ
            1. **Truy vấn**: "Giá cổ phiếu FPT hiện tại là bao nhiêu?"
               **Intermediate Results**: [
                 {"tool_name": "stock_price", "success": true, "observation": "Stock price data: {\"price\": 115000, \"volume\": 1200000}"}
               ]
               **Output**:
               ```json
               {
                 "decision": "end",
                 "reason": "Dữ liệu giá cổ phiếu FPT đã được cung cấp đầy đủ từ công cụ stock_price.",
                 "updated_plan": ""
               }
               ```

            2. **Truy vấn**: "Phân tích cổ phiếu FPT"
               **Intermediate Results**: [
                 {"tool_name": "stock_price", "success": true, "observation": "Stock price data: {\"price\": 115000}"},
                 {"tool_name": "rag_knowledge", "success": false, "observation": "Tool rag_knowledge failed: No relevant documents found"}
               ]
               **Output**:
               ```json
               {
                 "decision": "retry",
                 "reason": "Dữ liệu giá cổ phiếu đã có, nhưng công cụ rag_knowledge thất bại, cần thử lại để lấy chỉ số tài chính.",
                 "updated_plan": "Retry rag_knowledge with a more specific query or use tavily_search for financial data."
               }
               ```

            3. **Truy vấn**: "Phân tích cổ phiếu FPT"
               **Intermediate Results**: [
                 {"tool_name": "stock_price", "success": true, "observation": "Stock price data: {\"price\": 115000}"}
               ]
               **Output**:
               ```json
               {
                 "decision": "continue",
                 "reason": "Dữ liệu giá cổ phiếu đã có, nhưng cần thêm chỉ số tài chính và tin tức để phân tích đầy đủ.",
                 "updated_plan": "Use rag_knowledge for financial metrics and tavily_search for recent news."
               }
               ```

            ### DỮ LIỆU ĐẦU VÀO
            - **Câu hỏi gốc**: {query}
            - **Kết quả trung gian**: {intermediate_results}

            ### ĐỊNH DẠNG ĐẦU RA
            ```json
            {
              "decision": "continue|retry|end",
              "reason": "Lý do chi tiết cho quyết định",
              "updated_plan": "Kế hoạch cập nhật nếu cần tiếp tục hoặc thử lại"
            }
            ```
            """

            # Prepare query and intermediate results for LLM
            query = state.messages[0].content if state.messages else ""
            intermediate_results_str = json.dumps(state.intermediate_results, ensure_ascii=False)

            # Format the prompt with query and results
            formatted_prompt = reflection_prompt.format(
                query=query,
                intermediate_results=intermediate_results_str
            )

            # Call LLM for evaluation
            llm_response = await self.agent.chat_provider.chat(
                messages=[SystemMessage(content=formatted_prompt)]
            )
            llm_response_content = llm_response["response"] if isinstance(llm_response, dict) else llm_response

            # Parse LLM response
            try:
                evaluation = json.loads(llm_response_content)
                if not all(key in evaluation for key in ["decision", "reason", "updated_plan"]):
                    raise ValueError("Invalid LLM evaluation response format")
                if evaluation["decision"] not in ["continue", "retry", "end"]:
                    raise ValueError(f"Invalid decision: {evaluation['decision']}")
            except (json.JSONDecodeError, ValueError) as e:
                logger.error(f"Failed to parse LLM evaluation response: {str(e)}")
                evaluation = {
                    "decision": "end",
                    "reason": f"Failed to parse LLM evaluation: {str(e)}",
                    "updated_plan": ""
                }

            # Update state based on LLM evaluation
            reflection_note = f"LLM Evaluation: {evaluation['reason']}"
            updated_state.reflection_notes.append(reflection_note)
            updated_state.reflection_decision = evaluation["decision"]
            
            if evaluation["decision"] == "retry":
                updated_state.plan = evaluation["updated_plan"] or f"Retry with alternative approach: {state.plan}"
                updated_state.sub_goals.append(f"Resolve issue based on LLM evaluation: {evaluation['reason']}")
            elif evaluation["decision"] == "continue":
                updated_state.plan = evaluation["updated_plan"] or state.plan
                updated_state.sub_goals.append("Continue collecting data based on LLM evaluation")
            
            logger.info(f"Reflection decision: {updated_state.reflection_decision}, Note: {reflection_note}")
            return updated_state

        except Exception as e:
            logger.error(f"Error in reflection node: {str(e)}", exc_info=True)
            updated_state = state.model_copy()
            updated_state.messages.append(AIMessage(content=f"Error in reflection: {str(e)}"))
            updated_state.current_step += 1
            updated_state.final_answer = f"Error: {str(e)}"
            updated_state.reflection_decision = "end"
            updated_state.reflection_notes.append(f"Error during reflection: {str(e)}")
            return updated_state

    async def _final_output_node(self, state: AgentState) -> AgentState:
        """Final output node: Format and return final answer."""
        logger.info(f"=== FINAL OUTPUT NODE START === Step: {state.current_step}")

        try:
            final_answer = state.final_answer or self._generate_fallback_answer(state.messages, state.intermediate_results)
            updated_state = state.model_copy()
            updated_state.messages.append(AIMessage(content=final_answer))
            updated_state.final_answer = final_answer
            
            return updated_state
        except Exception as e:
            logger.error(f"Error in final output node: {str(e)}", exc_info=True)
            updated_state = state.model_copy()
            updated_state.messages.append(AIMessage(content=f"Error in final output: {str(e)}"))
            updated_state.final_answer = f"Error: {str(e)}"
            return updated_state

    def _should_continue(self, state: AgentState) -> str:
        """Decision logic for workflow continuation."""
        try:
            logger.info(
                f"Workflow decision - Step: {state.current_step}/{state.max_steps}, "
                f"Final answer: {'Yes' if state.final_answer else 'No'}, "
                f"Reflection decision: {state.reflection_decision}"
            )

            if state.final_answer or state.current_step >= state.max_steps:
                logger.info("Workflow ending: Final answer or max steps reached")
                return "end"

            if state.reflection_decision == "retry":
                logger.info("Workflow retrying due to reflection decision")
                return "continue"

            if state.reflection_decision == "end":
                logger.info("Workflow ending due to reflection decision")
                return "end"

            logger.info("Workflow continuing...")
            return "continue"

        except Exception as e:
            logger.error(f"Error in should_continue: {str(e)}", exc_info=True)
            return "end"

    async def run(self, query: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Run the workflow for a given query with session support."""
        start_time = time.time()
        logger.info(f"Starting workflow for query: '{query}...', session: {session_id or 'default'}")

        try:
            initial_state = AgentState(
                messages=[HumanMessage(content=query)],
                current_step=0,
                max_steps=10,
                tools_used=[],
                intermediate_results=[],
                plan="",
                sub_goals=[],
                reflection_notes=[],
                reflection_decision="continue"
            )

            result = await self.workflow.ainvoke(
                initial_state,
                config={"configurable": {"thread_id": session_id or "default"}}
            )
            final_state = AgentState(**result)

            workflow_time = time.time() - start_time
            
            final_result = {
                "answer": final_state.final_answer or self._generate_fallback_answer(final_state.messages, final_state.intermediate_results),
                "metadata": {
                    "success": True,
                    "steps": final_state.current_step,
                    "tools_used": final_state.tools_used,
                    "intermediate_results": final_state.intermediate_results,
                    "session_id": session_id,
                    "execution_time": workflow_time,
                    "plan": final_state.plan,
                    "sub_goals": final_state.sub_goals,
                    "reflection_notes": final_state.reflection_notes
                }
            }

            logger.info(
                f"=== WORKFLOW END === "
                f"Time: {workflow_time:.2f}s, "
                f"Steps: {final_result['metadata']['steps']}, "
                f"Tools used: {final_result['metadata']['tools_used']}, "
                f"Intermediate results: {len(final_result['metadata']['intermediate_results'])}"
            )
            return final_result

        except Exception as e:
            workflow_time = time.time() - start_time
            error_result = {
                "answer": f"Xin lỗi, đã xảy ra lỗi hệ thống khi xử lý câu hỏi của bạn: {str(e)}",
                "metadata": {
                    "success": False,
                    "steps": 0,
                    "tools_used": [],
                    "intermediate_results": [],
                    "error": str(e),
                    "session_id": session_id,
                    "execution_time": workflow_time,
                    "plan": "",
                    "sub_goals": [],
                    "reflection_notes": []
                }
            }
            logger.error(f"Workflow failed: {str(e)}", exc_info=True)
            return error_result

    async def _execute_tool_safely(self, tool_name: str, tool_input: str) -> Dict[str, Any]:
        """Execute tool with comprehensive error handling."""
        try:
            tool = self.tools[tool_name]
            
            # Chuyển đổi tool_input (chuỗi JSON) thành Pydantic model
            input_model_class = self.TOOL_INPUT_MODELS.get(tool_name)
            if not input_model_class:
                raise ValueError(f"No input model defined for tool: {tool_name}")
            
            input_data = json.loads(tool_input)
            input_model = input_model_class(**input_data)
            
            # Gọi _arun với các trường của Pydantic model
            result = await tool._arun(**input_model.dict())
            logger.info(f"Tool {tool_name} executed successfully: status={result.status}")
            
            return result.model_dump()  # Chuyển ToolResult thành dict để tương thích với luồng hiện tại

        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON input for tool {tool_name}: {str(e)}", exc_info=True)
            return {
                "error": f"Invalid JSON input: {str(e)}",
                "status": "error",
                "tool_name": tool_name,
                "input": tool_input
            }
        except ValueError as e:
            logger.error(f"Input validation failed for tool {tool_name}: {str(e)}", exc_info=True)
            return {
                "error": f"Input validation failed: {str(e)}",
                "status": "error",
                "tool_name": tool_name,
                "input": tool_input
            }
        except Exception as e:
            logger.error(f"Tool execution failed for {tool_name}: {str(e)}", exc_info=True)
            return {
                "error": str(e),
                "status": "error",
                "tool_name": tool_name,
                "input": tool_input
            }

    def _generate_fallback_answer(self, messages: list, intermediate_results: list = None) -> str:
        """Generate fallback answer from conversation messages and intermediate results."""
        if not messages:
            return "Tôi không thể xử lý câu hỏi của bạn lúc này."

        # Try to synthesize from intermediate results if available
        if intermediate_results:
            for result in reversed(intermediate_results):
                if result.get("success", False) and result.get("observation"):
                    return f"Based on available data: {result['observation'][:200]}..."

        for msg in reversed(messages):
            if (isinstance(msg, AIMessage) and
                msg.content and
                not msg.content.startswith("THOUGHT:") and
                not msg.content.startswith("ACTION:")):
                content = msg.content.strip()
                if len(content) > 10:
                    return content

        return "Tôi đã xử lý yêu cầu của bạn nhưng không thể đưa ra câu trả lời cụ thể. Vui lòng thử đặt câu hỏi khác."