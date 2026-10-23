# scripts/generate_answers.py
"""Run the full pipeline (retrieve + rerank + generate) on the 39 verified
questions, and save the results for DeepEval.

Output: results/generation_dataset.json
"""

import gc
import time
import json
from datetime import datetime
from pathlib import Path

from sentence_transformers import CrossEncoder

from src.generation.base_llm import BaseLLMGenerator
from src.retrieval.dense import DenseRetriever
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.sparse import SparseRetriever

CANDIDATES = 50
TOP_K = 5
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
RESULTS_DIR = Path("results")
TEST_SET = Path("data/evaluation/test_set_v2.json")


def main():
    with open(TEST_SET, encoding="utf-8") as f:
        test_set = json.load(f)

    eval_set = [q for q in test_set if q.get("relevant_chunk_ids_v3")]
    print(f"Generating answers for {len(eval_set)} questions")

    dense = DenseRetriever()
    sparse = SparseRetriever()
    hybrid = HybridRetriever(dense_retriever=dense, sparse_retriever=sparse)

    gc.collect()

    print(f"Loading reranker: {RERANKER_MODEL}")
    reranker = CrossEncoder(RERANKER_MODEL, max_length=512)

    generator = BaseLLMGenerator(provider="groq")

    dataset = []

    for i, item in enumerate(eval_set, 1):
        qid = item["question_id"]
        question = item["question"]
        reference = item.get("reference_answer", "")
        bank = item.get("expected_bank", "").upper()
        if bank == "SCOTIABANK":
            bank = "SCOTIA"

        # 1. Retrieve top-50
        try:
            if bank != "MULTI":
                candidates = hybrid.search(question, bank=bank, top_k=CANDIDATES)
            else:
                candidates = hybrid.search(question, top_k=CANDIDATES)
        except TypeError:
            candidates = hybrid.search(question, top_k=CANDIDATES)

        # 2. Rerank to top-5
        pairs = [(question, d.page_content) for d in candidates]
        scores = reranker.predict(pairs)
        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        top_docs = [d for d, _ in ranked[:TOP_K]]

        # 3. Throttle before LLM call (Groq TPM limit)
        if i > 1:
            time.sleep(15)

        # 4. Generate
        result = generator.generate(question, top_docs)

        dataset.append(
            {
                "question_id": qid,
                "input": question,
                "expected_bank": bank,
                "reference_answer": reference,
                "actual_output": result["answer"],
                "retrieval_context": [d.page_content for d in top_docs],
                "retrieved_chunk_ids": [d.metadata.get("chunk_id") for d in top_docs],
                "sources": result["sources"],
            }
        )

        print(f"[{i:>2}/{len(eval_set)}] {qid} — answer len={len(result['answer'])}")

    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"generation_dataset_{datetime.now():%Y%m%d_%H%M%S}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(
            {
                "timestamp": datetime.now().isoformat(),
                "n_questions": len(dataset),
                "top_k": TOP_K,
                "candidates": CANDIDATES,
                "dataset": dataset,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
