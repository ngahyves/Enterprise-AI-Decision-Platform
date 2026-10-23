# Retrieval Evaluation — v3 Ground Truth


**Date:** 2026-10-06
**Questions:** 39 (manually verified)
**Ground truth:** relevant_chunk_ids_v3

| Architecture | R@10 | R@20 | R@50 | MRR |
|--------------|------|------|------|-----|
| Dense | 0.641 | 0.808 | 0.833 | 0.306 |
| Sparse | 0.385 | 0.474 | 0.654 | 0.247 |
| Hybrid | 0.705 | 0.808 | 0.910 | 0.261 |
| **Hybrid + Reranker** | **0.910** | **0.910** | **0.910** | **0.799** |

**Config:**
- Dense: BAAI/bge-m3 (1024d, cosine)
- Sparse: FastEmbed Qdrant/bm25
- Hybrid: RRF fusion (k=60)
- Reranker: BAAI/bge-reranker-v2-m3 (max_length=512)
- Top-50 candidates → top-10 after rerank