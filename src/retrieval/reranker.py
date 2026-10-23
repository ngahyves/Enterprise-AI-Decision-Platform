# src/retrieval/reranker.py
from typing import List, Optional

from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

from src.common.config_manager import config
from src.common.logger import logger_manager

logger = logger_manager.get_logger(__name__)


class Reranker:
    """Re-orders documents using a Cross-Encoder (HuggingFace)."""

    def __init__(self):
        model_name = config["retrieval"]["reranker"]["model_name"]
        logger.info(f"Loading reranker model: {model_name}...")
        self.model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_n: Optional[int] = None,
    ) -> List[Document]:
        """
        Re-rank documents using a Cross-Encoder.

        Args:
            query: User query.
            documents: List of documents to re-rank.
            top_n: Number of documents to keep.

        Returns:
            Re-ranked list of documents.
        """
        n = top_n or config["retrieval"]["reranker"]["top_n"]

        if not documents:
            return []

        logger.info(f"Reranking {len(documents)} documents (Top-N: {n})")

        pairs = [(query, doc.page_content) for doc in documents]
        scores = self.model.predict(pairs)

        ranked = sorted(
            zip(documents, scores),
            key=lambda x: x[1],
            reverse=True,
        )

        return [doc for doc, _ in ranked[:n]]


# # ============================================================
# TEST UNITAIRE — Comparaison Dense+Reranker vs Hybrid+Reranker
# ============================================================
if __name__ == "__main__":
    import json
    from src.retrieval.dense import DenseRetriever
    from src.retrieval.sparse import SparseRetriever
    from src.retrieval.hybrid import HybridRetriever
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

    # --- Initialise les briques ---
    dense = DenseRetriever()
    sparse = SparseRetriever()
    hybrid = HybridRetriever(dense_retriever=dense, sparse_retriever=sparse)
    reranker = Reranker()

    # Paramètres
    CANDIDATES = 100  # top-k du retriever avant reranking
    TOP_N = 10  # top-n final après reranking

    # ============================================================
    # Fonction générique d'évaluation
    # ============================================================
    def evaluate_pipeline(name, retrieve_fn):
        """Évalue un pipeline (retrieve_fn + reranker) sur les 3 questions."""
        print()
        print("=" * 100)
        print(f"{name} — CANDIDATES={CANDIDATES}, TOP_N={TOP_N}")
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

            # Étape 1 : Retrieve
            candidates = retrieve_fn(query, CANDIDATES)

            # Étape 2 : Rerank
            docs = reranker.rerank(query, candidates, top_n=TOP_N)

            retrieved_ids = [doc.metadata.get("chunk_id") for doc in docs]
            matches = set(retrieved_ids) & set(relevant_ids)

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

            print(f"\n[{i + 1}/{len(eval_set)}] {qid}")
            print(f"  Matches       : {matches} ({len(matches)}/{len(relevant_ids)})")
            print(f"  recall@10     : {r10:.4f} | mrr: {mrr:.4f} | ndcg@10: {n10:.4f}")

        # Moyennes
        print()
        print(f"--- MOYENNES : {name} ---")
        for name_metric, values in metrics.items():
            avg = sum(values) / len(values) if values else 0.0
            print(f"{name_metric:<15}: {avg:.4f}")

        return {k: sum(v) / len(v) for k, v in metrics.items()}

    # ============================================================
    # Évalue les deux pipelines
    # ============================================================
    results = {}

    results["dense_reranked"] = evaluate_pipeline(
        "DENSE + RERANKER",
        lambda q, k: dense.search(q, top_k=k),
    )

    results["hybrid_reranked"] = evaluate_pipeline(
        "HYBRID + RERANKER",
        lambda q, k: hybrid.search(q, top_k=k),
    )

    # ============================================================
    # Tableau comparatif final
    # ============================================================
    print()
    print("=" * 100)
    print("COMPARAISON FINALE")
    print("=" * 100)

    header = f"{'Metric':<15}" + "".join(f"{name:<22}" for name in results.keys())
    print(header)
    print("-" * 100)

    for metric in ["recall@5", "recall@10", "precision@10", "mrr", "ndcg@5", "ndcg@10"]:
        row = f"{metric:<15}"
        for name in results.keys():
            row += f"{results[name][metric]:<22.4f}"
        print(row)

    print("=" * 100)
