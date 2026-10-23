"""Integrity tests for the evaluation ground truth.

Catches silent corruption: duplicate IDs, wrong counts, missing fields.
No API, no Docker — just loads the JSON file.
"""

import json
from pathlib import Path

import pytest

GT_CANDIDATES = [
    Path("data/evaluation/ground_truth.json"),  # first ground truth
    Path("data/evaluation/test_set_v2.json"),  # actual ground truth
]


@pytest.fixture(scope="module")
def ground_truth():
    for p in GT_CANDIDATES:
        if p.exists():
            with p.open(encoding="utf-8") as f:
                return json.load(f)
    pytest.fail(f"Ground truth not found. Looked in: {GT_CANDIDATES}")


def test_ground_truth_has_50_questions(ground_truth):
    """The corpus must contain exactly 50 questions."""
    assert len(ground_truth) == 50, f"Expected 50 questions, got {len(ground_truth)}"


def test_ground_truth_verified_count_is_39(ground_truth):
    """39 questions have a v3 (manually verified) chunk list."""
    verified = [q for q in ground_truth if q.get("relevant_chunk_ids_v3")]
    unverified_ids = [q["question_id"] for q in ground_truth if not q.get("relevant_chunk_ids_v3")]
    assert len(verified) == 39, (
        f"Expected 39 verified questions, got {len(verified)}. Unverified: {unverified_ids}"
    )


def test_ground_truth_has_no_duplicate_ids(ground_truth):
    """Question IDs must be unique (no copy-paste accidents)."""
    ids = [q["question_id"] for q in ground_truth]
    duplicates = {qid for qid in ids if ids.count(qid) > 1}
    assert not duplicates, f"Duplicate question IDs: {duplicates}"
