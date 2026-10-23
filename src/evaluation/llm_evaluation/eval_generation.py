# scripts/eval_generation.py
"""Evaluate generated answers with DeepEval + Gemini.

Metrics:
  - Faithfulness: is each claim supported by the retrieved context?
  - Answer Relevancy: does the answer address the question?
  - Contextual Recall: is the reference answer supported by the context?

Strategy:
  - One test case at a time (serial) to stay under Gemini's 15 RPM limit.
  - Exponential backoff on 429 with tenacity.
  - 15s sleep between test cases to reset the minute window.

Requirements:
  - pip install google-generativeai tenacity
  - $env:USE_GEMINI_MODEL = "1"
  - $env:GOOGLE_API_KEY = "your-key"

Run:
  python -m scripts.eval_generation
"""

import json
import os
import time
from datetime import datetime
from pathlib import Path

from deepeval import evaluate
from deepeval.evaluate.configs import AsyncConfig
from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualRecallMetric,
    FaithfulnessMetric,
)
from deepeval.models import GeminiModel
from deepeval.test_case import LLMTestCase
from google.genai.errors import ClientError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.common.logger import logger_manager

logger = logger_manager.get_logger(__name__)

# --- Configuration ---
JUDGE_MODEL = "gemini-3.5-flash-lite"
RESULTS_DIR = Path("results")

# --- Test mode ---
TEST_MODE = False
TEST_LIMIT = 2

# --- Rate-limit control ---
SLEEP_BETWEEN_CASES = 15  # seconds between two test cases


def build_test_cases(dataset: list[dict]) -> list[LLMTestCase]:
    """Convert generation rows into DeepEval test cases."""
    cases = []
    for row in dataset:
        if not row.get("actual_output") or not row.get("reference_answer"):
            continue
        cases.append(
            LLMTestCase(
                input=row["input"],
                actual_output=row["actual_output"],
                expected_output=row["reference_answer"],
                retrieval_context=row["retrieval_context"],
            )
        )
    return cases


@retry(
    retry=retry_if_exception_type(ClientError),
    stop=stop_after_attempt(6),
    wait=wait_exponential(multiplier=5, min=15, max=120),
    reraise=True,
)
def evaluate_one_case(case: LLMTestCase, metrics: list) -> dict:
    """Evaluate a single test case. Retries on 429 with exponential backoff.

    Returns a dict mapping metric name -> score (or None on failure).
    """
    result = evaluate(
        test_cases=[case],
        metrics=metrics,
        async_config=AsyncConfig(
            max_concurrent=1,
            throttle_value=0,
        ),
    )
    row = {}
    for tr in result.test_results:
        for m in tr.metrics_data or []:
            row[m.name] = m.score
    return row


def main():
    # 1. Find latest generation dataset
    candidates = sorted(RESULTS_DIR.glob("generation_dataset_*.json"))
    if not candidates:
        print("No generation dataset found. Run scripts.generate_answers first.")
        return
    dataset_path = candidates[-1]
    print(f"Using dataset: {dataset_path}")

    with open(dataset_path, encoding="utf-8") as f:
        payload = json.load(f)
    dataset = payload["dataset"]
    print(f"Questions with answers: {len(dataset)}")

    # 2. Build test cases
    cases = build_test_cases(dataset)
    if TEST_MODE:
        cases = cases[:TEST_LIMIT]
        print(f"TEST MODE — running on {len(cases)} questions only")
    else:
        print(f"Built {len(cases)} DeepEval test cases")

    # 3. Gemini judge
    judge = GeminiModel(
        model=JUDGE_MODEL,
        api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0.0,
    )

    metrics = [
        FaithfulnessMetric(
            model=judge,
            threshold=0.7,
            include_reason=False,
            truths_extraction_limit=10,
        ),
        AnswerRelevancyMetric(
            model=judge,
            threshold=0.7,
            include_reason=False,
        ),
        ContextualRecallMetric(
            model=judge,
            threshold=0.7,
            include_reason=False,
        ),
    ]

    print(f"\nEvaluating with judge: {JUDGE_MODEL}")
    print("Metrics: Faithfulness, Answer Relevancy, Contextual Recall")
    print(f"Mode: serial, {SLEEP_BETWEEN_CASES}s sleep between test cases\n")

    # 4. Serial evaluation with retry + sleep
    per_question = []

    for i, case in enumerate(cases, 1):
        qid = case.input[:60]
        try:
            scores = evaluate_one_case(case, metrics)
            per_question.append({"input": case.input, "scores": scores})
            print(
                f"[{i:>2}/{len(cases)}] ok  "
                f"F={scores.get('Faithfulness', 0):.2f}  "
                f"R={scores.get('Answer Relevancy', 0):.2f}  "
                f"C={scores.get('Contextual Recall', 0):.2f}"
            )
        except Exception as e:
            print(f"[{i:>2}/{len(cases)}] FAILED: {e}")
            per_question.append({"input": case.input, "scores": {}})

        if i < len(cases):
            time.sleep(SLEEP_BETWEEN_CASES)

    # 5. Aggregate
    buckets = {}
    for item in per_question:
        for name, score in item["scores"].items():
            if score is None:
                continue
            buckets.setdefault(name, []).append(score)

    scores = {name: sum(v) / len(v) for name, v in buckets.items()}

    print("\n" + "=" * 60)
    print("GENERATION EVALUATION — FINAL SCORES")
    print("=" * 60)
    for name, score in scores.items():
        print(f"  {name:<24}: {score:.4f}")
    print("=" * 60)

    # 6. Save
    out = RESULTS_DIR / f"generation_eval_{datetime.now():%Y%m%d_%H%M%S}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(
            {
                "timestamp": datetime.now().isoformat(),
                "judge_model": JUDGE_MODEL,
                "test_mode": TEST_MODE,
                "n_test_cases": len(cases),
                "scores": scores,
                "per_question": per_question,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
