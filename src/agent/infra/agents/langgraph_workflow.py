import json
import re
import time
from string import Template
from typing import Any, Dict, Optional

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableLambda
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from ...domain.agents.react_agent import StockReActAgent
from ...domain.entities.context import AgentState, ToolResult
from ...domain.tools.base import CustomBaseTool
from ...domain.tools.chat_tool import ChatToolInput
from ...domain.tools.fundamental_analysis_tool import (
    FundamentalAnalysisToolInput,
)
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

    def __init__(
        self, agent: StockReActAgent, tools: Dict[str, CustomBaseTool]
    ):
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
            {"continue": "reasoning", "end": "final_output"},
        )
        workflow.add_edge("final_output", END)

        return workflow.compile(checkpointer=self.checkpointer)

    def _reasoning_node(self, agent_runnable: RunnableLambda):
        """Create reasoning node using agent as a Runnable."""

        async def node(state: AgentState) -> AgentState:
            node_start_time = time.time()
            logger.info(
                f"=== REASONING NODE START === Step: {state.current_step}"
            )

            try:
                llm_response = await agent_runnable.ainvoke(state)
                llm_response_content = (
                    llm_response["response"]
                    if isinstance(llm_response, dict)
                    else llm_response
                )

                updated_state = state.model_copy()
                updated_state.plan = (
                    updated_state.plan
                    or "Generate plan to answer user query using available tools."
                )
                updated_state.sub_goals = updated_state.sub_goals or [
                    "Identify relevant tools",
                    "Execute tools",
                    "Synthesize results",
                ]
                updated_state.messages.append(
                    AIMessage(content=llm_response_content)
                )

                # Check for simple conversational queries (e.g., greetings)
                if "FINAL ANSWER" in llm_response_content:
                    logger.info(
                        "Detected FINAL ANSWER in reasoning output, setting state for early termination"
                    )
                    final_answer_match = re.search(
                        r"\*\*FINAL ANSWER\*\*:\s*(.*?)(?=\n|$)",
                        llm_response_content,
                        re.DOTALL,
                    )
                    if final_answer_match:
                        updated_state.final_answer = final_answer_match.group(
                            1
                        ).strip()
                        updated_state.reflection_decision = "end"

                logger.info(
                    f"Reasoning completed - Plan: {updated_state.plan}..., "
                    f"Sub-goals: {updated_state.sub_goals}, "
                    f"Final answer set: {bool(updated_state.final_answer)}"
                )
                return updated_state

            except Exception as e:
                logger.error(
                    f"Error in reasoning node: {str(e)}", exc_info=True
                )
                updated_state = state.model_copy()
                updated_state.messages.append(
                    AIMessage(content=f"Error in reasoning: {str(e)}")
                )
                updated_state.current_step += 1
                updated_state.final_answer = f"Error: {str(e)}"
                updated_state.reflection_decision = "end"
                return updated_state

        return node

    async def _action_node(self, state: AgentState) -> AgentState:
        node_start_time = time.time()
        logger.info(f"=== ACTION NODE START === Step: {state.current_step}")

        try:
            if state.final_answer:
                logger.info("Final answer already set, skipping action node")
                action_state = state.model_copy()
                action_state.reflection_decision = "end"
                return action_state

            llm_response = state.messages[-1].content if state.messages else ""
            tool_name, tool_input = self.agent.parse_tool_usage(llm_response)

            if tool_name and tool_input:
                # Đảm bảo tool_name là chuỗi
                if isinstance(tool_name, (list, tuple)):
                    tool_name = tool_name[0] if tool_name else None
                if not isinstance(tool_name, str):
                    raise ValueError(
                        f"Invalid tool_name type: {type(tool_name)}, value: {tool_name}"
                    )

                tool_result = await self._execute_tool_safely(
                    tool_name, tool_input
                )

                formatted_result = self.agent.format_tool_result(
                    tool_name, tool_result
                )

                updated_intermediate_results = state.intermediate_results + [
                    {
                        "llm_output": llm_response,
                        "observation": formatted_result,
                        "tool_name": tool_name,
                        "success": tool_result.get("status") == "success",
                    }
                ]

                updated_tools_used = state.tools_used + [tool_name]

                action_state = state.model_copy()
                action_state.messages.append(
                    AIMessage(content=f"Observation: {formatted_result}")
                )
                action_state.current_step += 1
                action_state.tools_used = updated_tools_used
                action_state.intermediate_results = (
                    updated_intermediate_results
                )
                action_state.tool_output = ToolResult(**tool_result)

                return action_state

            else:
                logger.warning(
                    "No valid tool action found in reasoning output"
                )
                updated_state = state.model_copy()
                updated_state.messages.append(
                    AIMessage(content="No valid tool action found")
                )
                updated_state.current_step += 1
                updated_state.reflection_decision = (
                    "end" if state.final_answer else "continue"
                )
                return updated_state

        except Exception as e:
            logger.error(f"Error in action node: {str(e)}", exc_info=True)
            updated_state = state.model_copy()
            updated_state.messages.append(
                AIMessage(content=f"Error in action: {str(e)}")
            )
            updated_state.current_step += 1
            updated_state.final_answer = f"Error: {str(e)}"
            updated_state.reflection_decision = "end"
            return updated_state

    async def _reflection_node(self, state: AgentState) -> AgentState:
        """Reflection node: Evaluate tool results and adjust plan using LLM."""
        node_start_time = time.time()
        logger.info(
            f"=== REFLECTION NODE START === Step: {state.current_step}"
        )

        try:
            updated_state = state.model_copy()

            # Skip reflection if final answer is already set
            if state.final_answer:
                logger.info(
                    "Final answer already set, proceeding to final output"
                )
                updated_state.reflection_decision = "end"
                updated_state.reflection_notes.append(
                    "Skipped reflection due to existing final answer"
                )
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
            - **Câu hỏi gốc**: $query
            - **Kết quả trung gian**: $intermediate_results

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
            intermediate_results_str = json.dumps(
                state.intermediate_results, ensure_ascii=False
            )

            # Format the prompt with query and results
            formatted_prompt = reflection_prompt.replace(
                "$query", query
            ).replace("$intermediate_results", intermediate_results_str)

            # Call LLM for evaluation
            messages = [HumanMessage(content=formatted_prompt)]

            llm_response = await self.agent.chat_provider.chat(
                messages=messages
            )

            llm_response_content = (
                llm_response["response"]
                if isinstance(llm_response, dict)
                else llm_response
            )

            # Clean the LLM response to remove markdown or extra characters
            llm_response_content = llm_response_content.strip()

            # Remove ```json and ``` markers if present
            llm_response_content = re.sub(
                r"^```json\s*|\s*```$",
                "",
                llm_response_content,
                flags=re.MULTILINE,
            )

            # Remove extra whitespace and newlines
            llm_response_content = " ".join(llm_response_content.split())

            # Parse LLM response
            try:
                evaluation = json.loads(llm_response_content)
                if not all(
                    key in evaluation
                    for key in ["decision", "reason", "updated_plan"]
                ):
                    raise ValueError("Invalid LLM evaluation response format")
                if evaluation["decision"] not in ["continue", "retry", "end"]:
                    raise ValueError(
                        f"Invalid decision: {evaluation['decision']}"
                    )
            except (json.JSONDecodeError, ValueError) as e:
                logger.error(
                    f"Failed to parse LLM evaluation response: {str(e)}"
                )
                evaluation = {
                    "decision": "end",
                    "reason": f"Failed to parse LLM evaluation: {str(e)}",
                    "updated_plan": "",
                }

            # Update state based on LLM evaluation
            reflection_note = f"LLM Evaluation: {evaluation['reason']}"
            updated_state.reflection_notes.append(reflection_note)
            updated_state.reflection_decision = evaluation["decision"]

            if evaluation["decision"] == "retry":
                updated_state.plan = (
                    evaluation["updated_plan"]
                    or f"Retry with alternative approach: {state.plan}"
                )
                updated_state.sub_goals.append(
                    f"Resolve issue based on LLM evaluation: {evaluation['reason']}"
                )
            elif evaluation["decision"] == "continue":
                updated_state.plan = evaluation["updated_plan"] or state.plan
                updated_state.sub_goals.append(
                    "Continue collecting data based on LLM evaluation"
                )

            logger.info(
                f"Reflection decision: {updated_state.reflection_decision}, Note: {reflection_note}"
            )
            return updated_state

        except Exception as e:
            logger.error(f"Error in reflection node: {str(e)}", exc_info=True)
            updated_state = state.model_copy()
            updated_state.messages.append(
                AIMessage(content=f"Error in reflection: {str(e)}")
            )
            updated_state.current_step += 1
            updated_state.final_answer = f"Error: {str(e)}"
            updated_state.reflection_decision = "end"
            updated_state.reflection_notes.append(
                f"Error during reflection: {str(e)}"
            )
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
                logger.info(
                    "Workflow ending: Final answer or max steps reached"
                )
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

    async def run(
        self, query: str, session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Run the workflow for a given query with session support."""
        start_time = time.time()
        logger.info(
            f"Starting workflow for query: '{query}...', session: {session_id or 'default'}"
        )

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
                reflection_decision="continue",
            )

            result = await self.workflow.ainvoke(
                initial_state,
                config={
                    "configurable": {"thread_id": session_id or "default"}
                },
            )
            final_state = AgentState(**result)

            workflow_time = time.time() - start_time

            final_result = {
                "answer": final_state.final_answer
                or self._generate_fallback_answer(
                    final_state.messages, final_state.intermediate_results
                ),
                "metadata": {
                    "success": True,
                    "steps": final_state.current_step,
                    "tools_used": final_state.tools_used,
                    "intermediate_results": final_state.intermediate_results,
                    "session_id": session_id,
                    "execution_time": workflow_time,
                    "plan": final_state.plan,
                    "sub_goals": final_state.sub_goals,
                    "reflection_notes": final_state.reflection_notes,
                },
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
                    "reflection_notes": [],
                },
            }
            logger.error(f"Workflow failed: {str(e)}", exc_info=True)
            return error_result

    async def _final_output_node(self, state: AgentState) -> AgentState:
        """Final output node: Format and return final answer using LLM for synthesis."""
        logger.info(
            f"=== FINAL OUTPUT NODE START === Step: {state.current_step}"
        )

        try:
            updated_state = state.model_copy()

            if state.final_answer:
                logger.info("Final answer already set, using it directly")
                updated_state.messages.append(
                    AIMessage(content=state.final_answer)
                )
                return updated_state

            # Synthesize final answer using LLM
            final_answer = await self._generate_final_answer(state)
            updated_state.messages.append(AIMessage(content=final_answer))
            updated_state.final_answer = final_answer

            return updated_state

        except Exception as e:
            logger.error(
                f"Error in final output node: {str(e)}", exc_info=True
            )
            updated_state = state.model_copy()
            updated_state.messages.append(
                AIMessage(content=f"Error in final output: {str(e)}")
            )
            updated_state.final_answer = f"Error: {str(e)}"
            return updated_state

    async def _generate_final_answer(self, state: AgentState) -> str:
        """Generate final answer by synthesizing messages and intermediate results using LLM."""
        logger.info("Generating final answer using LLM synthesis")

        try:
            # Prepare prompt for synthesizing final answer
            synthesis_prompt = Template(
                """
            Bạn là một trợ lý thông minh. Nhiệm vụ của bạn là tổng hợp thông tin từ câu hỏi gốc, các kết quả trung gian từ công cụ, và lịch sử hội thoại để tạo ra một câu trả lời hoàn chỉnh, tự nhiên và dễ hiểu.

            ### HƯỚNG DẪN
            - Đọc câu hỏi gốc (`query`) và các kết quả trung gian (`intermediate_results`).
            - Tổng hợp thông tin từ các kết quả trung gian để trả lời câu hỏi một cách chính xác và đầy đủ.
            - Nếu dữ liệu không đủ hoặc không liên quan, đưa ra câu trả lời chung chung nhưng lịch sự, giải thích rằng thông tin hiện tại chưa đầy đủ.
            - Câu trả lời phải:
              - Tự nhiên, mạch lạc, và dễ hiểu.
              - Trả lời trực tiếp câu hỏi của người dùng.
              - Không sử dụng thuật ngữ kỹ thuật phức tạp trừ khi cần thiết.
              - Viết bằng tiếng Việt, phù hợp với ngữ cảnh.

            ### DỮ LIỆU ĐẦU VÀO
            - **Câu hỏi gốc**: $query
            - **Kết quả trung gian**: $intermediate_results
            - **Lịch sử hội thoại**: $messages

            ### ĐỊNH DẠNG ĐẦU RA
            Một chuỗi văn bản chứa câu trả lời cuối cùng, không cần định dạng JSON.

            ### VÍ DỤ
            **Câu hỏi gốc**: "Cách thức hoạt động của thị trường chứng khoán là gì?"
            **Kết quả trung gian**: [
                {"tool_name": "rag_knowledge", "success": true, "observation": "Knowledge context: Thị trường chứng khoán là nơi các nhà đầu tư mua và bán cổ phiếu..."}
            ]
            **Lịch sử hội thoại**: [
                {"role": "user", "content": "Cách thức hoạt động của thị trường chứng khoán là gì?"},
                {"role": "assistant", "content": "Observation: Thị trường chứng khoán là nơi các nhà đầu tư mua và bán cổ phiếu..."}
            ]
            **Câu trả lời**:
            Thị trường chứng khoán là nơi các nhà đầu tư mua và bán cổ phiếu của các công ty niêm yết. Hoạt động chính bao gồm phát hành cổ phiếu, giao dịch qua sàn chứng khoán, và định giá dựa trên cung cầu. Các nhà đầu tư có thể kiếm lợi nhuận từ chênh lệch giá hoặc cổ tức.

            **Câu hỏi gốc**: "Giá cổ phiếu FPT hiện tại là bao nhiêu?"
            **Kết quả trung gian**: [
                {"tool_name": "stock_price", "success": true, "observation": "Stock price data: {\"price\": 115000, \"volume\": 1200000}"}
            ]
            **Câu trả lời**:
            Giá cổ phiếu FPT hiện tại là 115.000 VND.

            **Câu hỏi gốc**: "Cách thức hoạt động của thị trường chứng khoán là gì?"
            **Kết quả trung gian**: []
            **Câu trả lời**:
            Tôi xin lỗi, hiện tại tôi không có đủ thông tin để trả lời chi tiết câu hỏi của bạn về cách thức hoạt động của thị trường chứng khoán. Tuy nhiên, thị trường chứng khoán nói chung là nơi các nhà đầu tư giao dịch cổ phiếu và các chứng khoán khác. Nếu bạn cần thêm thông tin chi tiết, vui lòng hỏi lại hoặc cung cấp thêm ngữ cảnh!
            """
            )

            query = state.messages[0].content if state.messages else ""
            intermediate_results_str = json.dumps(
                state.intermediate_results, ensure_ascii=False
            )
            messages_str = json.dumps(
                [msg.dict() for msg in state.messages], ensure_ascii=False
            )

            if not query:
                logger.warning("No query found, returning default answer")
                return "Tôi không thể xử lý câu hỏi của bạn vì không có câu hỏi gốc. Vui lòng thử lại."

            formatted_prompt = synthesis_prompt.substitute(
                query=query,
                intermediate_results=intermediate_results_str,
                messages=messages_str,
            )

            logger.info(f"Synthesize prompt succeed.")
            # logger.info(f"Synthesis prompt: {formatted_prompt}")

            # Call LLM to synthesize final answer
            llm_response = await self.agent.chat_provider.chat(
                messages=[HumanMessage(content=formatted_prompt)]
            )
            final_answer = (
                llm_response["response"]
                if isinstance(llm_response, dict)
                else llm_response
            )

            # Clean the response to remove markdown or extra characters
            final_answer = final_answer.strip()
            final_answer = re.sub(
                r"^```.*?\n|\n```$", "", final_answer, flags=re.MULTILINE
            )

            logger.info(f"Synthesized final answer: {final_answer}")

            if not final_answer:
                logger.warning("LLM returned empty response, using fallback")
                return "Tôi không thể cung cấp câu trả lời chi tiết lúc này. Vui lòng thử lại hoặc cung cấp thêm thông tin."

            return final_answer

        except Exception as e:
            logger.error(f"Error in generating final answer: {str(e)}")
            return "Tôi xin lỗi, đã xảy ra lỗi khi xử lý câu hỏi của bạn. Vui lòng thử lại sau."

    async def _execute_tool_safely(
        self, tool_name: str, tool_input: str
    ) -> Dict[str, Any]:

        if not isinstance(tool_name, str):
            logger.error(
                f"Invalid tool_name type: {type(tool_name)}, value: {tool_name}"
            )
            return {
                "error": f"Invalid tool_name type: {type(tool_name)}",
                "status": "error",
                "tool_name": str(tool_name),
                "input": tool_input,
            }
        try:
            tool = self.tools[tool_name]
            input_model_class = self.TOOL_INPUT_MODELS.get(tool_name)
            if not input_model_class:
                raise ValueError(
                    f"No input model defined for tool: {tool_name}"
                )

            input_data = json.loads(tool_input)
            input_model = input_model_class(**input_data)

            result = await tool._arun(**input_model.dict())
            logger.info(
                f"Tool {tool_name} executed successfully: status={result.status}"
            )
            return result.model_dump()

        except json.JSONDecodeError as e:
            logger.error(
                f"Invalid JSON input for tool {tool_name}: {str(e)}",
                exc_info=True,
            )
            return {
                "error": f"Invalid JSON input: {str(e)}",
                "status": "error",
                "tool_name": tool_name,
                "input": tool_input,
            }
        except ValueError as e:
            logger.error(
                f"Input validation failed for tool {tool_name}: {str(e)}",
                exc_info=True,
            )
            return {
                "error": f"Input validation failed: {str(e)}",
                "status": "error",
                "tool_name": tool_name,
                "input": tool_input,
            }
        except Exception as e:
            logger.error(
                f"Tool execution failed for {tool_name}: {str(e)}",
                exc_info=True,
            )
            return {
                "error": str(e),
                "status": "error",
                "tool_name": tool_name,
                "input": tool_input,
            }
