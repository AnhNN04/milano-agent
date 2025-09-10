from typing import Any, Dict, List, Type

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import BaseModel, Field, PrivateAttr

from ...infra.providers.rag_provider import BaseRag
from ...shared.logging.logger import Logger
from .base import CustomBaseTool, ToolState

logger = Logger.get_logger(__name__)


class RAGToolInput(BaseModel):
    """Input schema for RAGTool."""

    query: str = Field(
        ..., description="The query to search the knowledge base for."
    )


class RAGToolOutput(BaseModel):
    """Output schema for RAGTool."""

    knowledge_context: str = Field(
        description="Formatted context from retrieved documents"
    )
    sources: List[Dict[str, Any]] = Field(
        description="Metadata of retrieved documents"
    )
    scores: List[float] = Field(
        description="Similarity scores of retrieved documents"
    )


class RAGTool(CustomBaseTool):
    """RAG (Retrieval Augmented Generation) tool for knowledge retrieval."""

    name: str = "rag_knowledge"
    description: str = (
        "Search internal knowledge base for Vietnamese stock market information from uploaded documents"
    )
    args_schema: Type[BaseModel] = RAGToolInput

    _rag_retriever: PrivateAttr = PrivateAttr()

    def __init__(self, rag_retriever: BaseRag):
        super().__init__()
        self._rag_retriever = rag_retriever

    async def _execute_impl(self, query: str) -> Dict[str, Any]:
        """Execute RAG search using the injected retriever."""

        try:
            # Retrieve similar documents
            search_results = await self._rag_retriever.retrieve(query=query)

            # Format results
            knowledge_context = self._format_knowledge_context(search_results)
            sources = self._extract_sources(search_results)
            scores = [result.get("score", 0) for result in search_results]

            # Update tool state for LangGraph
            self.state = ToolState(
                tool_name=self.name,
                execution_time=0.0,  # Will be updated in CustomBaseTool
                context={
                    "market": "VN",
                    "num_results": len(search_results),
                    "avg_score": sum(scores) / len(scores) if scores else 0.0,
                },
            )

            return RAGToolOutput(
                knowledge_context=knowledge_context,
                sources=sources,
                scores=scores,
            ).model_dump()

        except Exception as e:
            logger.error(f"RAG search failed: {str(e)}")
            return RAGToolOutput(
                knowledge_context="", sources=[], scores=[]
            ).model_dump()

    def _format_knowledge_context(self, results: List[Dict[str, Any]]) -> str:
        """Format search results into knowledge context for LLM."""

        if not results:
            return "Không tìm thấy thông tin liên quan trong cơ sở dữ liệu nội bộ."

        context_parts = []
        for i, result in enumerate(results, 1):
            content = result.get("content", "")
            source = result.get("metadata", {}).get("title", "Unknown")
            context_parts.append(f"Nguồn {i}: {source}\n{content}\n")
        return "\n---\n".join(context_parts)

    def _extract_sources(
        self, results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Extract source information from results."""

        return [result.get("metadata", {}) for result in results]

    def to_langchain_retriever(self) -> BaseRetriever:
        """Convert RAGTool to a LangChain BaseRetriever for integration with LLM chains."""

        class RAGToolRetriever(BaseRetriever):
            tool: "RAGTool" = self

            async def _aget_relevant_documents(
                self, query: str
            ) -> List[Document]:
                results = await self.tool._rag_retriever.retrieve(
                    query=query,
                    max_results=self.tool.args_schema.max_results.default,
                )
                filtered_results = [
                    r
                    for r in results
                    if r.get("score", 0)
                    >= self.tool.args_schema.min_score.default
                ]
                return [
                    Document(
                        page_content=r.get("content", ""),
                        metadata={
                            **r.get("metadata", {}),
                            "score": r.get("score", 0),
                        },
                    )
                    for r in filtered_results
                ]

        return RAGToolRetriever()
