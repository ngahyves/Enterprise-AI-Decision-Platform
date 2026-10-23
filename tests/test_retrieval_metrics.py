"""Pure tests for retrieval metrics and bank normalization.
 Tests the exact functions that produced
the R@10 / MRR / NDCG numbers in Results.txt.
"""

import pytest
from langchain_core.documents import Document

from src.evaluation.rag_evaluation.retrieval_metrics import (
    _normalize_bank,
    compute_recall,
    compute_precision,
    compute_mrr,
    compute_ndcg,
    compute_semantic_hit,
)


# ---------------------------------------------------------------
# Bank normalization — the bug that made Scotia score R@10=0.00
# ---------------------------------------------------------------
def test_scotiabank_alias_resolves_to_scotia():
    """GT uses 'SCOTIABANK', Qdrant stores 'SCOTIA' — aliases must bridge them."""
    assert _normalize_bank("SCOTIABANK") == "SCOTIA"
    assert _normalize_bank("scotiabank") == "SCOTIA"


def test_normalize_bank_is_idempotent_and_case_insensitive():
    """Canonical names must be stable under re-normalization."""
    assert _normalize_bank("SCOTIA") == "SCOTIA"
    assert _normalize_bank("scotia") == "SCOTIA"
    assert _normalize_bank("RBC") == "RBC"
    assert _normalize_bank("td") == "TD"


def test_normalize_bank_passes_through_unknown_banks():
    """Unknown banks fall back to uppercase (no crash, no silent alias)."""
    assert _normalize_bank("HSBC") == "HSBC"
    assert _normalize_bank("hsbc") == "HSBC"


# ---------------------------------------------------------------
# compute_recall
# ---------------------------------------------------------------
def test_recall_perfect_hit():
    """All relevant chunks retrieved in top-k → recall = 1.0."""
    assert compute_recall(["a", "b", "c"], ["a", "b"], k=10) == 1.0


def test_recall_partial_hit():
    """1 of 2 relevant chunks retrieved → recall = 0.5."""
    assert compute_recall(["a", "x", "y"], ["a", "b"], k=10) == 0.5


def test_recall_respects_k_cutoff():
    """A relevant chunk beyond position k must not count."""
    assert compute_recall(["x", "y", "a"], ["a"], k=2) == 0.0
    assert compute_recall(["x", "y", "a"], ["a"], k=3) == 1.0


def test_recall_empty_relevant_returns_zero():
    """No relevant chunks defined → recall is 0.0 (not a crash)."""
    assert compute_recall(["a", "b"], [], k=10) == 0.0


# ---------------------------------------------------------------
# compute_mrr
# ---------------------------------------------------------------
def test_mrr_first_position():
    """Relevant chunk at rank 1 → MRR = 1.0."""
    assert compute_mrr(["a", "x", "y"], ["a"]) == 1.0


def test_mrr_third_position():
    """Relevant chunk at rank 3 → MRR = 1/3."""
    assert compute_mrr(["x", "y", "a"], ["a"]) == pytest.approx(1 / 3)


def test_mrr_no_hit():
    """No relevant chunk retrieved → MRR = 0.0."""
    assert compute_mrr(["x", "y"], ["a"]) == 0.0


# ---------------------------------------------------------------
# compute_ndcg
# ---------------------------------------------------------------
def test_ndcg_perfect_ranking():
    """Relevant chunks at the very top → NDCG = 1.0."""
    assert compute_ndcg(["a", "b", "x"], ["a", "b"], k=10) == pytest.approx(1.0)


def test_ndcg_worse_than_perfect():
    """Relevant chunks ranked lower → NDCG strictly below 1.0."""
    assert compute_ndcg(["x", "a", "b"], ["a", "b"], k=10) < 1.0
