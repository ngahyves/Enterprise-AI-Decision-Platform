# src/retrieval/hybrid.py
import hashlib
from typing import List, Optional

from langchain_core.documents import Document
from qdrant_client.http import models as qm

from src.common.config_manager import config
from src.common.logger import logger_manager

logger = logger_manager.get_logger(__name__)


class HybridRetriever:
    """Dense + Sparse retrieval fused via Reciprocal Rank Fusion (RRF)."""

    def __init__(self, dense_retriever, sparse_retriever):
        self.dense = dense_retriever
        self.sparse = sparse_retriever
        self.rrf_k = config["retrieval"]["fusion"].get("rrf_k", 60)
        self.candidate_multiplier = config["retrieval"]["fusion"].get("candidate_multiplier", 2)

    def _bank_filter(self, bank: Optional[str]):
        if not bank or bank.upper() == "MULTI":
            return None
        return qm.Filter(
            must=[
                qm.FieldCondition(
                    key="metadata.bank_name",
                    match=qm.MatchValue(value=bank.upper()),
                )
            ]
        )

    def _get_unique_id(self, doc: Document) -> str:
        if doc.metadata.get("chunk_id"):
            return str(doc.metadata["chunk_id"])
        return hashlib.md5(doc.page_content.encode("utf-8")).hexdigest()

    def _rrf(self, dense_docs: List[Document], sparse_docs: List[Document]) -> List[Document]:
        scores = {}
        doc_map = {}

        def add_ranks(docs):
            for rank, doc in enumerate(docs):
                doc_id = self._get_unique_id(doc)
                doc_map.setdefault(doc_id, doc)
                scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (self.rrf_k + rank + 1)

        add_ranks(dense_docs)
        add_ranks(sparse_docs)

        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [doc_map[doc_id] for doc_id, _ in sorted_docs]

    def search(
        self,
        query: str,
        bank: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> List[Document]:
        """Hybrid search with optional bank filter applied on both branches."""
        k = top_k or config["retrieval"]["dense"]["top_k"]
        candidates = k * self.candidate_multiplier

        bank_filter = self._bank_filter(bank)

        logger.info(f"RRF Fusion: '{query[:60]}...' (k={k}, bank={bank})")

        # Call the underlying vector stores directly to pass the filter.
        # This avoids modifying DenseRetriever / SparseRetriever signatures.
        dense_docs = self.dense.vector_store.similarity_search(
            query, k=candidates, filter=bank_filter
        )
        sparse_docs = self.sparse.vector_store.similarity_search(
            query, k=candidates, filter=bank_filter
        )

        fused = self._rrf(dense_docs, sparse_docs)
        return fused[:k]


# ============================================================
# TEST UNITAIRE — 3 questions + impact du top-k
# ============================================================
if __name__ == "__main__":
    import json
    from src.retrieval.dense import DenseRetriever
    from src.retrieval.sparse import SparseRetriever
    from src.evaluation.rag_evaluation.retrieval_metrics import (
        compute_recall,
        compute_precision,
        compute_mrr,
        compute_ndcg,
    )

    # --- Charge le golden dataset ---
    with open("data/evaluation/test_set.json", "r", encoding="utf-8") as f:
        test_set = json.load(f)

    eval_set = [q for q in test_set if q.get("relevant_chunk_ids")][:3]

    # --- Initialise les deux retrievers ---
    dense = DenseRetriever()
    sparse = SparseRetriever()
    retriever = HybridRetriever(dense_retriever=dense, sparse_retriever=sparse)

    # ============================================================
    # PARTIE 1 : Détail par question (top_k=10)
    # ============================================================
    print("=" * 100)
    print(f"HYBRID RETRIEVAL — DÉTAIL SUR {len(eval_set)} QUESTIONS (top_k=10)")
    print("=" * 100)

    metrics = {
        "recall@5": [],
        "recall@10": [],
        "precision@10": [],
        "mrr": [],
        "ndcg@5": [],
        "ndcg@10": [],
    }

    for i, item in enumerate(eval_set):
        qid = item.get("question_id", f"Q{i + 1}")
        query = item["question"]
        relevant_ids = item["relevant_chunk_ids"]

        docs = retriever.search(query, top_k=10)
        retrieved_ids = [doc.metadata.get("chunk_id") for doc in docs]
        matches = set(retrieved_ids) & set(relevant_ids)

        print(f"\n[{i + 1}/{len(eval_set)}] {qid}")
        print(f"  Question      : {query}")
        print(f"  Relevant_ids  : {relevant_ids}")
        print(f"  Retrieved_ids : {retrieved_ids}")
        print(f"  Matches       : {matches} ({len(matches)}/{len(relevant_ids)})")

        r5 = compute_recall(retrieved_ids, relevant_ids, k=5)
        r10 = compute_recall(retrieved_ids, relevant_ids, k=10)
        p10 = compute_precision(retrieved_ids, relevant_ids, k=10)
        mrr = compute_mrr(retrieved_ids, relevant_ids)
        n5 = compute_ndcg(retrieved_ids, relevant_ids, k=5)
        n10 = compute_ndcg(retrieved_ids, relevant_ids, k=10)

        metrics["recall@5"].append(r5)
        metrics["recall@10"].append(r10)
        metrics["precision@10"].append(p10)
        metrics["mrr"].append(mrr)
        metrics["ndcg@5"].append(n5)
        metrics["ndcg@10"].append(n10)

        print(f"  recall@5      : {r5:.4f}")
        print(f"  recall@10     : {r10:.4f}")
        print(f"  precision@10  : {p10:.4f}")
        print(f"  mrr           : {mrr:.4f}")
        print(f"  ndcg@5        : {n5:.4f}")
        print(f"  ndcg@10       : {n10:.4f}")

    print()
    print("=" * 100)
    print("HYBRID RETRIEVAL — MOYENNES SUR 3 QUESTIONS (top_k=10)")
    print("=" * 100)
    for name, values in metrics.items():
        avg = sum(values) / len(values) if values else 0.0
        print(f"{name:<15}: {avg:.4f}")
    print("=" * 100)

    # ============================================================
    # PARTIE 2 : Impact du top-k
    # ============================================================
    K_VALUES = [5, 10, 20, 50, 100]

    print()
    print("=" * 100)
    print(f"HYBRID RETRIEVAL — IMPACT DU TOP-K SUR {len(eval_set)} QUESTIONS")
    print("=" * 100)

    for k in K_VALUES:
        recall_values = []
        mrr_values = []
        ndcg_values = []

        for item in eval_set:
            query = item["question"]
            relevant_ids = item["relevant_chunk_ids"]

            docs = retriever.search(query, top_k=k)
            retrieved_ids = [doc.metadata.get("chunk_id") for doc in docs]

            recall_values.append(compute_recall(retrieved_ids, relevant_ids, k=k))
            mrr_values.append(compute_mrr(retrieved_ids, relevant_ids))
            ndcg_values.append(compute_ndcg(retrieved_ids, relevant_ids, k=k))

        avg_recall = sum(recall_values) / len(recall_values)
        avg_mrr = sum(mrr_values) / len(mrr_values)
        avg_ndcg = sum(ndcg_values) / len(ndcg_values)

        print(
            f"k={k:<4} | recall@{k}={avg_recall:.4f} | mrr={avg_mrr:.4f} | ndcg@{k}={avg_ndcg:.4f}"
        )
