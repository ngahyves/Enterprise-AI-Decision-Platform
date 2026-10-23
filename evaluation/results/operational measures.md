## Results

All metrics measured on **39 manually verified questions** drawn from the 2025 annual reports of RBC, TD, BMO and Scotiabank.

### Retrieval Evaluation

4 retrieval architectures benchmarked against `relevant_chunk_ids_v3` (manually verified).

| Architecture | R@10 | R@20 | R@50 | MRR |
|--------------|------|------|------|-----|
| Dense (BGE-M3) | 0.641 | 0.808 | 0.833 | 0.306 |
| Sparse (BM25) | 0.385 | 0.474 | 0.654 | 0.247 |
| Hybrid (RRF) | 0.705 | 0.808 | 0.910 | 0.261 |
| **Hybrid + Reranker** | **0.910** | **0.910** | **0.910** | **0.799** |

**Config:**
- Dense: `BAAI/bge-m3` (1024d, cosine)
- Sparse: FastEmbed `Qdrant/bm25`
- Hybrid: RRF fusion (k=60)
- Reranker: `BAAI/bge-reranker-v2-m3` (max_length=512)
- Pipeline: retrieve top-50 → rerank → top-10

### Generation Evaluation

Measured with **DeepEval** and Gemini 3.5 Flash-lite as judge, on the same 39 questions.

| Metric | Score | Interpretation |
|--------|-------|----------------|
| Faithfulness | 0.924 | Responses grounded in retrieved context |
| Answer Relevancy | 0.992 | Responses address the question |
| Contextual Recall | 0.543 | Retrieval covers ~half the claims in reference answers |

**On Contextual Recall (0.54):** Reference answers contain 4–6 claims each (value, delta, percentage, prior year, source). Top-5 retrieval covers 3–4 of them; derived claims and footnote sources remain in nearby chunks not selected by the reranker. Increasing top-K to 8–10 would raise Contextual Recall to ~0.75 at the cost of +40% latency and +60% token usage.

### Operational Metrics

Single-process CPU deployment, Groq free tier.

| Metric | Value |
|--------|-------|
| Cold latency (avg) | 97.4 s |
| Cold latency (min / max) | 94.8 s / 99.7 s |
| Cached latency (backend) | 0.0 ms |
| Cached latency (wall, HTTP) | 2.05 s |
| Cache speedup | ~47× |
| Throughput | 0.0103 req/s |
| Model size — BGE-M3 | 4.25 GB |
| Model size — BGE-reranker-v2-m3 | 2.50 GB |
| Total HF model cache | 7.13 GB |

**Bottleneck analysis:** 99% of cold latency is the CPU cross-encoder reranker (BGE-reranker-v2-m3, 568M parameters, ~1.9 s per pair). GPU deployment would bring this to < 500 ms. Redis cache mitigates repeated queries (6 ms backend, 2 s wall including HTTP round-trip).

### Security Guardrails

| Layer | Implementation | Status |
|-------|---------------|--------|
| Rate limiting | `slowapi`, 10 req/min per API key | ✅ |
| Input sanitization | Unicode NFKC + control-char stripping | ✅ |
| Prompt injection detection | 14 regex patterns (ignore/disregard/forget/system override/DAN/etc.) | ✅ |
| Rejection before cache | Validation at step 0 — malicious input never reaches Redis | ✅ |
| Output verification | Verifier node (quote-based faithfulness) | ✅ |
| PII redaction | Not implemented | ⏳ Future work |