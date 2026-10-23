# src/evaluation/rag_evaluation/benchmark.py
"""Evaluate 4 retrieval architectures on the v3 ground truth.

  1. Dense (BGE-M3)
  2. Sparse (BM25)
  3. Hybrid (RRF fusion)
  4. Hybrid + Reranker (bge-reranker-v2-m3)

Run: python -m src.evaluation.rag_evaluation.benchmark_final
"""

import gc
import json

from sentence_transformers import CrossEncoder

from src.evaluation.rag_evaluation.retrieval_metrics import compute_mrr, compute_recall
from src.retrieval.dense import DenseRetriever
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.sparse import SparseRetriever

KS = [10, 20, 50]
CANDIDATES = 50
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
FAST_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
CASCADE_K = 10


def evaluate(retriever, eval_set, label, bank_aware, reranker=None, fast_reranker=None):
    print(f"\n{'=' * 60}")
    print(f"  {label}  ({len(eval_set)} questions)")
    print(f"{'=' * 60}")

    rows = []
    for item in eval_set:
        question = item["question"]
        bank = item.get("expected_bank", "").upper()
        if bank == "SCOTIABANK":
            bank = "SCOTIA"
        relevant = item.get("relevant_chunk_ids_v3", [])

        # 1. Retrieve
        try:
            if bank_aware and bank != "MULTI":
                docs = retriever.search(
                    question, bank=bank, top_k=CANDIDATES if reranker else max(KS)
                )
            else:
                docs = retriever.search(question, top_k=CANDIDATES if reranker else max(KS))
        except TypeError:
            docs = retriever.search(question, top_k=CANDIDATES if reranker else max(KS))

        # 2. Rerank if requested
        if reranker is not None:
            # Étape 1 — MiniLM : 50 → CASCADE_K (10)
            pairs = [(question, d.page_content) for d in docs]
            fast_scores = fast_reranker.predict(pairs, batch_size=32, show_progress_bar=False)
            top_cascade = [
                d for d, _ in sorted(zip(docs, fast_scores), key=lambda x: -x[1])[:CASCADE_K]
            ]
            # Étape 2 — BGE-large : 10 → max(KS)
            pairs_k = [(question, d.page_content) for d in top_cascade]
            scores = reranker.predict(pairs_k, batch_size=32, show_progress_bar=False)
            ranked = sorted(zip(top_cascade, scores), key=lambda x: -x[1])
            docs = [d for d, _ in ranked[: max(KS)]]

        retrieved = [d.metadata.get("chunk_id") for d in docs if d.metadata.get("chunk_id")]

        row = {"qid": item["question_id"]}
        for k in KS:
            row[f"recall@{k}"] = compute_recall(retrieved, relevant, k=k)
        row["mrr"] = compute_mrr(retrieved, relevant)
        rows.append(row)

    for k in KS:
        avg = sum(r[f"recall@{k}"] for r in rows) / len(rows)
        print(f"recall@{k:<4}: {avg:.4f}")
    print(f"mrr: {sum(r['mrr'] for r in rows) / len(rows):.4f}")
    return rows


def main():
    with open("data/evaluation/test_set_v2.json", encoding="utf-8") as f:
        test_set = json.load(f)

    eval_set = [q for q in test_set if q.get("relevant_chunk_ids_v3")]
    print(f"Evaluating on {len(eval_set)} questions")

    dense = DenseRetriever()
    sparse = SparseRetriever()
    hybrid = HybridRetriever(dense_retriever=dense, sparse_retriever=sparse)

    single_bank = [q for q in eval_set if q.get("expected_bank", "").upper() != "MULTI"]

    # --- 3 architectures without reranker ---
    evaluate(dense, single_bank, "1. DENSE", bank_aware=False)
    evaluate(sparse, single_bank, "2. SPARSE", bank_aware=False)
    evaluate(hybrid, eval_set, "3. HYBRID", bank_aware=True)

    # --- Free retriever memory ---
    gc.collect()

    print(f"\nLoading fast reranker: {FAST_RERANKER_MODEL}")
    fast_reranker = CrossEncoder(FAST_RERANKER_MODEL, max_length=256)
    print(f"Loading reranker: {RERANKER_MODEL}")
    reranker = CrossEncoder(RERANKER_MODEL, max_length=512)
    print("Rerankers ready")

    evaluate(
        hybrid,
        eval_set,
        "4. HYBRID + CASCADE",
        bank_aware=True,
        reranker=reranker,
        fast_reranker=fast_reranker,
    )


if __name__ == "__main__":
    main()
