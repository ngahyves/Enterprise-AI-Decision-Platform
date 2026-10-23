# src/agents/nodes/retriever.py
from functools import lru_cache
import time

from qdrant_client.http import models as qm
from sentence_transformers import CrossEncoder

from src.agents.state import AgentState
from src.retrieval.dense import DenseRetriever
from src.retrieval.sparse import SparseRetriever
from src.retrieval.hybrid import HybridRetriever
from src.common.logger import logger_manager

logger = logger_manager.get_logger(__name__)

CANDIDATES = 50
TOP_K = 5
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"


@lru_cache(maxsize=1)
def _retriever() -> HybridRetriever:
    dense = DenseRetriever()
    sparse = SparseRetriever()
    return HybridRetriever(dense_retriever=dense, sparse_retriever=sparse)


@lru_cache(maxsize=1)
def _reranker() -> CrossEncoder:
    logger.info(f"Loading reranker: {RERANKER_MODEL}")
    return CrossEncoder(RERANKER_MODEL, max_length=512)


def retrieve_documents(state: AgentState) -> dict:
    banks = state.get("banks", [])
    objectives = state.get("sub_objectives", [])
    question = state["question"]

    retriever = _retriever()
    reranker = _reranker()

    chunks = {}
    errors = []

    for i, bank in enumerate(banks):
        query = objectives[i] if i < len(objectives) else question
        try:
            t0 = time.time()
            candidates = retriever.search(query, bank=bank, top_k=CANDIDATES)
            t1 = time.time()

            if candidates:
                pairs = [(query, d.page_content) for d in candidates]
                scores = reranker.predict(pairs, batch_size=32)
                ranked = sorted(zip(candidates, scores), key=lambda x: -x[1])
                docs = [d for d, _ in ranked[:TOP_K]]
            else:
                docs = []

            t2 = time.time()
            print(
                f"[TIMING][{bank}] hybrid={t1 - t0:.1f}s | "
                f"bge({len(candidates)} pairs)={t2 - t1:.1f}s"
            )

        except Exception as e:
            logger.exception(f"Retrieval+rerank failed for {bank}")
            chunks[bank] = []
            errors.append(f"{bank}: {e}")
            continue

        chunks[bank] = [
            {
                "chunk_id": d.metadata.get("chunk_id"),
                "content": d.page_content,
                "section": d.metadata.get("section_title"),
                "page": d.metadata.get("page_number"),
                "bank": bank,
            }
            for d in docs
        ]

        logger.info(f"{bank}: {len(docs)} chunks after rerank")

        if not docs:
            errors.append(f"No chunks found for {bank}")

    return {"chunks_by_bank": chunks, "errors": errors}
