# src/retrieval/sparse.py
from typing import List, Optional
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore, FastEmbedSparse, RetrievalMode
from qdrant_client import QdrantClient

from src.common.config_manager import env_settings, config
from src.common.logger import logger_manager

logger = logger_manager.get_logger(__name__)


class SparseRetriever:
    """Performs pure lexical/sparse search using FastEmbed Qdrant/bm25 via Qdrant."""

    def __init__(self):
        self.collection_name = env_settings.qdrant_collection_name
        self.host = env_settings.qdrant_host
        self.port = env_settings.qdrant_port

        # Modèle sparse natif
        self.sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")

        # Connexion au QdrantVectorStore en mode SPARSE pur
        self.vector_store = QdrantVectorStore(
            client=QdrantClient(url=f"http://{self.host}:{self.port}"),
            collection_name=self.collection_name,
            embedding=None,  # Pas d'embedding dense ici
            sparse_embedding=self.sparse_embeddings,
            retrieval_mode=RetrievalMode.SPARSE,
            sparse_vector_name="sparse",
        )

    def search(self, query: str, top_k: Optional[int] = None) -> List[Document]:
        k = top_k or config["retrieval"].get("sparse", {}).get("top_k", 10)
        logger.info(f"Pure Sparse search for: '{query}' (Top-K: {k})")
        return self.vector_store.similarity_search(query, k=k)


# ============================================================
# TEST UNITAIRE — 3 questions du golden dataset + impact du top-k
# ============================================================
if __name__ == "__main__":
    import json
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

    # --- Initialise le retriever ---
    retriever = SparseRetriever()

    # ============================================================
    # PARTIE 1 : Détail par question (top_k=10)
    # ============================================================
    print("=" * 100)
    print(f"SPARSE RETRIEVAL — DÉTAIL SUR {len(eval_set)} QUESTIONS (top_k=10)")
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
    print("SPARSE RETRIEVAL — MOYENNES SUR 3 QUESTIONS (top_k=10)")
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
    print(f"SPARSE RETRIEVAL — IMPACT DU TOP-K SUR {len(eval_set)} QUESTIONS")
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
