# src/evaluation/rag_evaluation/retrieval_metrics.py
from typing import List

import numpy as np
from langchain_core.documents import Document


# Normalize bank names between the ground truth and Qdrant payloads
BANK_ALIASES = {
    "SCOTIABANK": "SCOTIA",
    "SCOTIA": "SCOTIA",
    "TD": "TD",
    "BMO": "BMO",
    "RBC": "RBC",
    "MULTI": "MULTI",
}


def _normalize_bank(bank: str) -> str:
    return BANK_ALIASES.get(bank.upper(), bank.upper())


# ---------------------------------------------------------------
# ID-based metrics (chunk_ids must match exactly)
# ---------------------------------------------------------------
def compute_recall(retrieved_ids: List[str], relevant_ids: List[str], k: int = 10) -> float:
    """Proportion of relevant chunks retrieved in the top-k."""
    if not relevant_ids:
        return 0.0
    retrieved_at_k = retrieved_ids[:k]
    hits = len(set(retrieved_at_k) & set(relevant_ids))
    return hits / len(relevant_ids)


def compute_precision(retrieved_ids: List[str], relevant_ids: List[str], k: int = 10) -> float:
    """Proportion of top-k retrieved chunks that are relevant."""
    retrieved_at_k = retrieved_ids[:k]
    if not retrieved_at_k:
        return 0.0
    hits = len(set(retrieved_at_k) & set(relevant_ids))
    return hits / len(retrieved_at_k)


def compute_mrr(retrieved_ids: List[str], relevant_ids: List[str]) -> float:
    """Reciprocal rank of the first relevant chunk."""
    for rank, doc_id in enumerate(retrieved_ids, 1):
        if doc_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def compute_ndcg(retrieved_ids: List[str], relevant_ids: List[str], k: int = 10) -> float:
    """Normalized Discounted Cumulative Gain at k."""
    retrieved_at_k = retrieved_ids[:k]

    dcg = sum(
        1.0 / np.log2(rank + 1)
        for rank, doc_id in enumerate(retrieved_at_k, 1)
        if doc_id in relevant_ids
    )

    idcg = sum(1.0 / np.log2(rank + 1) for rank in range(1, min(len(relevant_ids), k) + 1))

    return dcg / idcg if idcg > 0 else 0.0


# ---------------------------------------------------------------
# Semantic metrics (bank + keyword coverage, no chunk_id needed)
# ---------------------------------------------------------------
def _keyword_coverage(content: str, keywords: List[str]) -> int:
    """Number of keywords found in the content."""
    low = content.lower()
    return sum(1 for kw in keywords if kw.lower() in low)


def compute_semantic_hit(
    retrieved_docs: List[Document],
    expected_bank: str,
    keywords: List[str],
    k: int = 10,
) -> float:
    """Return 1.0 if any top-k chunk belongs to the right bank
    AND covers at least half of the keywords (min 2)."""
    expected = _normalize_bank(expected_bank)
    min_hits = max(2, len(keywords) // 2)

    for doc in retrieved_docs[:k]:
        bank_found = doc.metadata.get("bank_name", "").upper()
        if expected != "MULTI" and expected != bank_found:
            continue

        if _keyword_coverage(doc.page_content, keywords) >= min_hits:
            return 1.0
    return 0.0


def compute_semantic_mrr(
    retrieved_docs: List[Document],
    expected_bank: str,
    keywords: List[str],
) -> float:
    """Reciprocal rank of the first chunk that matches bank + keyword coverage."""
    expected = _normalize_bank(expected_bank)
    min_hits = max(2, len(keywords) // 2)

    for rank, doc in enumerate(retrieved_docs, 1):
        bank_found = doc.metadata.get("bank_name", "").upper()
        if expected != "MULTI" and expected != bank_found:
            continue

        if _keyword_coverage(doc.page_content, keywords) >= min_hits:
            return 1.0 / rank
    return 0.0
