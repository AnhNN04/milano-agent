import hashlib
from typing import Any, Dict, Optional

from langchain_core.messages import HumanMessage

from ...domain.agents.react_agent import StockReActAgent
from ...domain.tools.base import CustomBaseTool
from ...infra.agents.langgraph_workflow import ReActWorkflow
from ...shared.logging.logger import Logger
from ...shared.session.redis_session_manager import (
    LangChainRedisSessionManager,
)

logger = Logger.get_logger(__name__)


class StockAnalysisService:
    """Application service for stock market analysis using ReAct workflow."""

    def __init__(
        self,
        agent: StockReActAgent,
        tools: Dict[str, CustomBaseTool],
        session_manager: LangChainRedisSessionManager,
    ):
        self.agent = agent
        self.tools = tools
        self.session_manager = session_manager
        self.workflow_manager = ReActWorkflow(agent=agent, tools=tools)
        logger.info("StockAnalysisService initialized")

    async def analyze(
        self, query: str, session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Analyze stock market query with session and cache support."""

        effective_session_id = await self._ensure_session(session_id)
        logger.info(f"Using session: {effective_session_id}")

        query_hash = self._generate_query_hash(query)
        cached_result = await self.session_manager.get_cached_analysis(
            effective_session_id, query_hash, cache_ttl=300
        )

        if cached_result:
            logger.info(
                f"Cache HIT for query: '{query[:50]}...', session: {effective_session_id}"
            )
            cached_result["metadata"]["session_id"] = effective_session_id
            cached_result["metadata"]["cached"] = True
            return cached_result

        logger.info(
            f"Cache MISS - Starting fresh analysis for query: '{query[:50]}...', session: {effective_session_id}"
        )

        try:
            await self.session_manager.update_conversation(
                effective_session_id, HumanMessage(content=query)
            )

            logger.info(
                f"Starting workflow execution for session: {effective_session_id}"
            )
            final_result = await self.workflow_manager.run(
                query=query, session_id=effective_session_id
            )

            result = {
                "answer": final_result["answer"],
                "metadata": {
                    "success": final_result["metadata"]["success"],
                    "session_id": effective_session_id,
                    "steps": final_result["metadata"]["steps"],
                    "tools_used": final_result["metadata"]["tools_used"],
                    "intermediate_results": self._serialize_intermediate_results(
                        final_result["metadata"]["intermediate_results"]
                    ),
                    "cached": False,
                    "query_hash": query_hash,
                    "execution_time": final_result["metadata"][
                        "execution_time"
                    ],
                    "plan": final_result["metadata"]["plan"],
                    "sub_goals": final_result["metadata"]["sub_goals"],
                    "reflection_notes": final_result["metadata"][
                        "reflection_notes"
                    ],
                },
            }

            logger.info(
                f"Workflow completed for query: '{query[:50]}...', "
                f"Steps: {result['metadata']['steps']}, "
                f"Tools used: {result['metadata']['tools_used']}, "
                f"Intermediate results: {len(result['metadata']['intermediate_results'])}"
            )

            if result["metadata"]["success"]:
                await self.session_manager.cache_analysis(
                    effective_session_id, query_hash, result
                )
                logger.info(
                    f"Analysis cached for session: {effective_session_id}"
                )

            return result

        except Exception as e:
            logger.error(
                f"Analysis failed - Query: '{query[:50]}...', Session: {effective_session_id}, Error: {str(e)}",
                exc_info=True,
            )

            error_result = {
                "answer": f"Xin lỗi, đã xảy ra lỗi trong quá trình phân tích: {str(e)}",
                "metadata": {
                    "success": False,
                    "session_id": effective_session_id,
                    "steps": 0,
                    "tools_used": [],
                    "intermediate_results": [],
                    "cached": False,
                    "error": str(e),
                    "query_hash": query_hash,
                    "execution_time": 0.0,
                    "plan": "",
                    "sub_goals": [],
                    "reflection_notes": [],
                },
            }

            await self.session_manager.cache_analysis(
                effective_session_id, query_hash, error_result
            )

            return error_result

    async def _ensure_session(self, session_id: Optional[str]) -> str:
        """Ensure valid session exists and return effective session_id."""
        if session_id:
            session = await self.session_manager.get_session(session_id)
            if session:
                logger.info(f"Using existing session: {session_id}")
                return session_id
            else:
                logger.info(f"Creating session with provided ID: {session_id}")
                await self.session_manager.create_session_with_id(
                    session_id=session_id
                )
                return session_id
        else:
            new_session_id = await self.session_manager.create_session()
            logger.info(f"Created new session: {new_session_id}")
            return new_session_id

    def _serialize_intermediate_results(
        self, intermediate_results: list
    ) -> list:
        """Serialize intermediate results for JSON response."""
        if not intermediate_results:
            return []

        serialized = []
        for result in intermediate_results:
            if isinstance(result, dict):
                serialized.append(
                    {
                        "llm_output": result.get("llm_output", ""),
                        "observation": result.get("observation", ""),
                        "tool_name": result.get("tool_name", "unknown"),
                        "success": result.get("success", True),
                        "error_message": result.get("error_message", None),
                    }
                )
            else:
                serialized.append({"raw_result": str(result)})

        return serialized

    def _generate_query_hash(self, query: str) -> str:
        """Generate hash for query caching."""
        return hashlib.md5(query.encode()).hexdigest()

    async def get_session_history(self, session_id: str) -> Optional[list]:
        """Get conversation history for session."""
        session = await self.session_manager.get_session(session_id)
        return session.conversation_history if session else None
