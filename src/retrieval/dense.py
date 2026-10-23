# src/retrieval/dense.py
from typing import List, Optional
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from src.common.config_manager import env_settings, config
from src.common.logger import logger_manager

logger = logger_manager.get_logger(__name__)


class DenseRetriever:
    """Performs pure semantic vector search using Qdrant and Hugging Face embeddings."""

    def __init__(self):
        self.collection_name = env_settings.qdrant_collection_name
        self.host = env_settings.qdrant_host
        self.port = env_settings.qdrant_port

        # Loading the retriever model (ex: BGE-M3)
        embedding_model_name = config["retrieval"]["embedding"]["model_name"]
        logger.info(f"Loading Dense embedding model: {embedding_model_name}...")

        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model_name,
            model_kwargs={"device": "cpu"},
        )

        # Connexion to Vector Store Qdrant Dense mode
        self.vector_store = QdrantVectorStore(
            client=QdrantClient(url=f"http://{self.host}:{self.port}"),
            collection_name=self.collection_name,
            embedding=self.embeddings,
        )

    def search(self, query: str, top_k: Optional[int] = None) -> List[Document]:
        """Performs a pure dense semantic search.
        Args:
            query: User's query.
            top_k: Number of results to return.
        Returns:
            List of relevant LangChain documents.
        """
        k = top_k or config["retrieval"]["dense"]["top_k"]
        logger.info(f"Dense search for: '{query}' (Top-K: {k})")

        # Searching by COSINE similarity
        results = self.vector_store.similarity_search(query, k=k)
        return results


# ============================================================
# Unit test — 3 questions of the golden dataset
# ============================================================
if __name__ == "__main__":
    import json
    from src.evaluation.rag_evaluation.retrieval_metrics import (
        compute_recall,
        compute_precision,
        compute_mrr,
        compute_ndcg,
    )

    # ---Loading golden dataset ---
    with open("data/evaluation/test_set.json", "r", encoding="utf-8") as f:
        test_set = json.load(f)

    # Filter questions with chunk_IDS
    eval_set = [q for q in test_set if q.get("relevant_chunk_ids")]

    # Taking first 3 questions
    subset = eval_set[:3]

    print("=" * 100)
    print(f"DENSE RETRIEVAL — TEST SUR {len(subset)} QUESTIONS")
    print("=" * 100)

    # --- Initialize the  retriever ---
    retriever = DenseRetriever()

    # --- Accumulating IR metrics ---
    metrics = {
        "recall@5": [],
        "recall@10": [],
        "precision@10": [],
        "mrr": [],
        "ndcg@5": [],
        "ndcg@10": [],
    }

    # --- Loop on the 3 questions ---
    for i, item in enumerate(subset):
        qid = item.get("question_id", f"Q{i + 1}")
        query = item["question"]
        relevant_ids = item["relevant_chunk_ids"]

        # Retrieve
        docs = retriever.search(query, top_k=10)
        retrieved_ids = [doc.metadata.get("chunk_id") for doc in docs]
        matches = set(retrieved_ids) & set(relevant_ids)

        # Printing results
        print(f"\n[{i + 1}/{len(subset)}] {qid}")
        print(f"  Question      : {query}")
        print(f"  Relevant_ids  : {relevant_ids}")
        print(f"  Retrieved_ids : {retrieved_ids}")
        print(f"  Matches       : {matches} ({len(matches)}/{len(relevant_ids)})")

        # Computing metrics
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

    # --- Metrics means ---
    print()
    print("=" * 100)
    print("DENSE RETRIEVAL — Mean on 3 questions")
    print("=" * 100)
    for name, values in metrics.items():
        avg = sum(values) / len(values) if values else 0.0
        print(f"{name:<15}: {avg:.4f}")
    print("=" * 100)
