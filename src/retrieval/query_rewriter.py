# src/retrieval/query_rewriter.py
"""
Query rewriting for financial documents.
Uses an LLM to reformulate user questions with technical financial terms.
"""

import json
from pathlib import Path

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

from src.common.config_manager import env_settings, config
from src.common.logger import logger_manager

logger = logger_manager.get_logger(__name__)


CACHE_PATH = Path("evaluation/cache/query_rewrites.json")


REWRITE_PROMPT = """You are a financial analyst assistant.
Rewrite the user's question using technical terms that would appear
in a bank's annual report (10-K, MD&A, financial statements).

Rules:
- Keep the SAME meaning and intent.
- ALWAYS reformulate, even if the question seems well-formed.
- Use terms like "total revenue", "net income", "provision for credit losses", "CET1 ratio", "total assets".
- Add context like "according to the annual report" or "for fiscal year 2025".
- If the question mentions a bank (TD, BMO, Scotiabank, RBC), keep it.
- Return ONLY the rewritten question. No explanation.

Original question: {question}
Rewritten question:"""


class QueryRewriter:
    """Rewrites user queries using an LLM for better retrieval."""

    def __init__(self):
        model_name = config["llm"]["fast_agent"]["model"]
        logger.info(f"Loading query rewriter LLM: {model_name}")

        self.llm = ChatGroq(
            model=model_name,
            temperature=0.0,
            api_key=env_settings.groq_api_key,
        )

        self.prompt = ChatPromptTemplate.from_template(REWRITE_PROMPT)

        # Charge le cache
        self.cache = {}
        if CACHE_PATH.exists():
            with open(CACHE_PATH, "r", encoding="utf-8") as f:
                self.cache = json.load(f)
            logger.info(f"Loaded cache with {len(self.cache)} entries")

    def _save_cache(self):
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(self.cache, f, indent=2, ensure_ascii=False)

    def rewrite(self, query: str) -> str:
        """Rewrite the query for better retrieval."""
        # Vérifie le cache
        if query in self.cache:
            logger.info(f"Cache hit for: '{query}'")
            return self.cache[query]

        try:
            chain = self.prompt | self.llm
            response = chain.invoke({"question": query})
            rewritten = response.content.strip()

            logger.info(f"Query rewritten: '{query}' → '{rewritten}'")

            # Sauvegarde dans le cache
            self.cache[query] = rewritten
            self._save_cache()

            return rewritten

        except Exception as e:
            logger.warning(f"Query rewriting failed: {e}. Using original query.")
            return query


# Singleton
_rewriter = None


def rewrite_query(query: str) -> str:
    """Public function to rewrite a query."""
    global _rewriter
    if _rewriter is None:
        _rewriter = QueryRewriter()
    return _rewriter.rewrite(query)
