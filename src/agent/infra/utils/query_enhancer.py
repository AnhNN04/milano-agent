import re
from typing import List

from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate

from ...shared.logging.logger import Logger

logger = Logger.get_logger(__name__)


class QueryEnhancer:
    """Utility to enhance queries and messages with Vietnamese stock market context."""

    def enhance_query(self, query: str) -> str:
        """Enhance a single query for Vietnamese stock market context."""

        vietnamese_market_terms = [
            "Vietnam stock",
            "Vietnamese stock market",
            "HOSE",
            "HNX",
            "UPCoM",
            "chứng khoán Việt Nam",
            "thị trường chứng khoán",
        ]
        query_lower = query.lower()
        has_vn_context = any(
            term.lower() in query_lower for term in vietnamese_market_terms
        )

        system_prompt = (
            "You are an expert in the Vietnamese stock market, including HOSE, HNX, and UPCoM. "
            "Refine the following query to focus on Vietnamese stock market information."
        )
        prompt_template = ChatPromptTemplate.from_messages(
            [("system", system_prompt), ("human", "Original query: {query}")]
        )

        try:
            formatted_query = prompt_template.format_messages(query=query)[
                1
            ].content
            if not has_vn_context:
                if re.search(r"\b[A-Z]{2,4}\b", query):
                    formatted_query += " Vietnam stock HOSE HNX"
                else:
                    formatted_query += " Vietnamese stock market"
            return formatted_query

        except Exception as e:
            logger.error(f"Query enhancement failed: {str(e)}")
            return query

    def enhance_messages(
        self, messages: List[BaseMessage]
    ) -> List[BaseMessage]:
        """Enhance messages with Vietnamese stock market context."""

        system_prompt = (
            "You are an expert in the Vietnamese stock market, including HOSE, HNX, and UPCoM. "
            "Provide accurate and relevant information about Vietnamese stocks, market news, and financial data."
        )
        prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                *[(m.type, m.content) for m in messages],
            ]
        )
        try:
            return prompt_template.format_messages()

        except Exception as e:
            logger.error(f"Message enhancement failed: {str(e)}")

            return messages
