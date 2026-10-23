# src/generation/base_llm.py
from typing import List, Dict, Any

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.common.config_manager import env_settings, config
from src.common.logger import logger_manager

logger = logger_manager.get_logger(__name__)


class BaseLLMGenerator:
    """Grounded LLM generation over retrieved financial chunks."""

    def __init__(self, provider: str = "groq"):
        if provider != "groq":
            raise ValueError(f"Provider '{provider}' is not supported yet.")

        self.llm = ChatGroq(
            api_key=env_settings.groq_api_key,
            model=config["llm"]["fast_agent"]["model"],
            temperature=config["llm"]["fast_agent"]["temperature"],
        )

        self.prompt_template = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are an expert Enterprise Financial AI Assistant analyzing annual and "
                    "sustainability reports of major Canadian banks.\n"
                    "Answer the user's question using STRICTLY and ONLY the provided context.\n\n"
                    "MANDATORY RULES:\n"
                    "1. Do not invent, assume, or extrapolate facts not supported by the context.\n"
                    "2. If the context does not contain enough information, reply exactly:\n"
                    "   'I don't have sufficient evidence in the available documents to answer this question.'\n"
                    "3. Cite sources using the format: [Bank Name — Section Title] for every major claim.\n"
                    "4. Maintain a professional, objective, analytical financial tone.\n\n"
                    "Context:\n{context}",
                ),
                ("human", "{question}"),
            ]
        )

    def _format_context(self, documents: List[Document]) -> str:
        blocks = []
        for i, doc in enumerate(documents):
            bank = doc.metadata.get("bank_name", "Unknown Bank")
            section = doc.metadata.get("section_title") or "General"
            year = doc.metadata.get("year", 2025)
            blocks.append(
                f"--- Source [{i + 1}] ---\n"
                f"Bank: {bank} | Year: {year} | Section: {section}\n"
                f"Content:\n{doc.page_content}"
            )
        return "\n\n".join(blocks)

    @retry(
        retry=retry_if_exception_type(Exception),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=5, max=30),
        reraise=True,
    )
    def _invoke(self, messages):
        return self.llm.invoke(messages)

    def generate(self, query: str, retrieved_docs: List[Document]) -> Dict[str, Any]:
        if not retrieved_docs:
            logger.warning("Generation requested with zero documents. Abstaining.")
            return {
                "answer": "I don't have sufficient evidence in the available documents to answer this question.",
                "sources": [],
            }

        logger.info(f"Generating for '{query[:80]}...' with {len(retrieved_docs)} chunks")

        context = self._format_context(retrieved_docs)
        messages = self.prompt_template.format_messages(context=context, question=query)
        response = self._invoke(messages)

        # Deduplicate sources by (bank, section)
        sources = []
        seen = set()
        for doc in retrieved_docs:
            key = (doc.metadata.get("bank_name"), doc.metadata.get("section_title"))
            if key in seen:
                continue
            seen.add(key)
            sources.append(
                {
                    "bank": doc.metadata.get("bank_name"),
                    "section": doc.metadata.get("section_title"),
                    "source_file": doc.metadata.get("source_file"),
                    "chunk_id": doc.metadata.get("chunk_id"),
                }
            )

        return {"answer": response.content, "sources": sources}
