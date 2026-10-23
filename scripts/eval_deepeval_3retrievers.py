# scripts/eval_deepeval_3retrievers.py
"""DeepEval contextual recall + precision for 3 retrievers.

Rate-limit handling:
  - tenacity exponential backoff on the Groq judge (429 retries)
  - AsyncConfig(max_concurrent=2, throttle_value=6) to stay under free-tier

Run:
    python -m scripts.eval_deepeval_3retrievers
"""

import json
from datetime import datetime
from pathlib import Path

from deepeval import evaluate
from deepeval.evaluate.configs import AsyncConfig
from deepeval.metrics import ContextualPrecisionMetric, ContextualRecallMetric
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase
from groq import Groq
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.common.config_manager import env_settings
from src.common.logger import logger_manager
from src.retrieval.dense import DenseRetriever
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.sparse import SparseRetriever

logger = logger_manager.get_logger(__name__)

GROQ_MODEL = "openai/gpt-oss-20b"
TOP_K = 5
RESULTS_DIR = Path("results")


# ---------------------------------------------------------------
# Groq judge with retry
# ---------------------------------------------------------------
class GroqJudge(DeepEvalBaseLLM):
    def __init__(self, model_name=GROQ_MODEL):
        self.model_name = model_name
        self.client = Groq(api_key=env_settings.groq_api_key)

    def get_model_name(self):
        return self.model_name

    @retry(
        retry=retry_if_exception_type(Exception),
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=2, min=4, max=30),
        reraise=True,
    )
    def generate(self, prompt: str) -> str:
        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=1024,
        )
        return completion.choices[0].message.content or ""

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)


# ---------------------------------------------------------------
# Test case builder
# ---------------------------------------------------------------
def build_test_cases(retriever, items, bank_aware):
    cases = []
    for item in items:
        question = item["question"]
        expected = item.get("reference_answer", "")
        if not expected:
            continue

        bank = item.get("expected_bank", "").upper()
        if bank == "SCOTIABANK":
            bank = "SCOTIA"

        try:
            if bank_aware and bank and bank != "MULTI":
                docs = retriever.search(question, bank=bank, top_k=TOP_K)
            else:
                docs = retriever.search(question, top_k=TOP_K)
        except TypeError:
            docs = retriever.search(question, top_k=TOP_K)

        contexts = [d.page_content for d in docs]

        cases.append(
            LLMTestCase(
                input=question,
                actual_output="",
                expected_output=expected,
                retrieval_context=contexts,
            )
        )
    return cases


# ---------------------------------------------------------------
# Evaluate one retriever
# ---------------------------------------------------------------
def run_eval(retriever, items, label, bank_aware=True):
    print(f"\n{'=' * 70}")
    print(f"  {label}  ({len(items)} questions)")
    print(f"{'=' * 70}")

    cases = build_test_cases(retriever, items, bank_aware)
    print(f"Built {len(cases)} test cases")

    judge = GroqJudge()
    metrics = [
        ContextualRecallMetric(model=judge, threshold=0.5),
        ContextualPrecisionMetric(model=judge, threshold=0.5),
    ]

    result = evaluate(
        test_cases=cases,
        metrics=metrics,
        async_config=AsyncConfig(
            max_concurrent=2,
            throttle_value=6,
            ignore_errors=True,
        ),
    )

    # Aggregate
    scores = {"Contextual Recall": [], "Contextual Precision": []}
    for tr in result.test_results:
        for m in tr.metrics_data or []:
            if m.name in scores and m.score is not None:
                scores[m.name].append(m.score)

    summary = {k: sum(v) / len(v) if v else 0.0 for k, v in scores.items()}
    print(f"\n{label} — AVERAGES:")
    for k, v in summary.items():
        print(f"  {k}: {v:.4f}")

    return summary


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------
def main():
    with open("data/evaluation/test_set_v2.json", encoding="utf-8") as f:
        test_set = json.load(f)

    eval_set = [q for q in test_set if q.get("reference_answer")]
    print(f"Questions with reference answers: {len(eval_set)}")

    dense = DenseRetriever()
    sparse = SparseRetriever()
    hybrid = HybridRetriever(dense_retriever=dense, sparse_retriever=sparse)

    results = {}

    # Single-bank questions only for dense/sparse (no bank filter kwarg on them)
    single_bank = [q for q in eval_set if q.get("expected_bank", "").upper() != "MULTI"]

    results["DENSE"] = run_eval(dense, single_bank, "DENSE (single-bank only)", bank_aware=False)
    results["SPARSE"] = run_eval(sparse, single_bank, "SPARSE (single-bank only)", bank_aware=False)
    results["HYBRID"] = run_eval(hybrid, eval_set, "HYBRID (all questions)", bank_aware=True)

    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"deepeval_3retrievers_{datetime.now():%Y%m%d_%H%M%S}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(
            {
                "n_questions_dense_sparse": len(single_bank),
                "n_questions_hybrid": len(eval_set),
                "top_k": TOP_K,
                "results": results,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
