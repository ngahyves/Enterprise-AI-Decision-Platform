# scripts/find_chunks.py
"""Rebuild ground truth by matching expected answers to new chunks.

Strategy (from GraphRAG-Bench, arXiv 2604.17056):
  1. Substring matching on distinctive numeric values (including percentages)
  2. Fuzzy string matching on key phrases (TheFuzz token_set_ratio)
  3. Semantic fallback via BGE-M3 embeddings for paraphrased answers
"""

import json
import re
from pathlib import Path

from difflib import SequenceMatcher

# Optional — install with: pip install thefuzz
try:
    from thefuzz import fuzz

    HAS_FUZZ = True
except ImportError:
    HAS_FUZZ = False


CHUNKS_DIR = Path("data/processed/chunks")
GROUND_TRUTH = Path("data/evaluation/test_set.json")
OUTPUT = Path("data/evaluation/test_set_v2.json")

BANK_TO_FILE = {
    "TD": "TD_Annual_report_2025",
    "BMO": "BMO_Annual_Report_2025",
    "SCOTIA": "Scotia_Annual_Report_2025",
    "SCOTIABANK": "Scotia_Annual_Report_2025",
    "RBC": "RBC_Annual_Report_2025",
}

# Numbers: 21,522 or 21522 or 59.7 or $1,476,802
NUM_PATTERN = re.compile(r"\$?\d[\d,]*\.?\d*%?")
PERCENT_PATTERN = re.compile(r"\d+\.?\d*\s*%")


def extract_values(text: str) -> list[str]:
    """Extract distinctive values: large numbers + percentages."""
    if not text:
        return []

    values = []
    seen = set()

    # Percentages first (45%, 59.7%)
    for m in PERCENT_PATTERN.findall(text):
        v = m.replace(" ", "")
        if v not in seen:
            seen.add(v)
            values.append(v)

    # Numbers
    for raw in NUM_PATTERN.findall(text):
        v = raw.replace("$", "").replace(",", "").strip()
        if not v or v in seen:
            continue
        try:
            f = float(v.replace("%", ""))
        except ValueError:
            continue

        # Skip years
        if f == int(f) and 2020 <= f <= 2030:
            continue
        # Skip tiny numbers (< 100 for integers, < 5 for decimals)
        if f == int(f) and abs(f) < 100:
            continue
        if f != int(f) and abs(f) < 5:
            continue

        seen.add(v)
        values.append(v)

    # Sort by length (longer = more distinctive)
    values.sort(key=lambda x: -len(x))
    return values[:6]


def extract_phrases(text: str) -> list[str]:
    """Extract 2-3 word key phrases from the reference answer."""
    if not text:
        return []
    # Split on punctuation and take phrases of 2+ words
    clean = re.sub(r"[^\w\s]", " ", text.lower())
    words = clean.split()
    phrases = []
    for n in [3, 2]:
        for i in range(len(words) - n + 1):
            p = " ".join(words[i : i + n])
            if len(p) > 10:
                phrases.append(p)
    return phrases


def substring_match(values: list[str], chunks: list[dict]) -> list[str]:
    """Find chunks containing any of the values.
    Row chunks are prioritized; duplicates removed.
    """
    row_hits = []
    narrative_hits = []

    for c in chunks:
        content = c["page_content"].replace(",", "").replace("$", "")
        if any(v in content for v in values):
            cid = c["metadata"]["chunk_id"]
            if c["metadata"].get("chunk_type") == "row":
                row_hits.append(cid)
            else:
                narrative_hits.append(cid)

    # Dedup while preserving order
    seen = set()
    ordered = []
    for cid in row_hits + narrative_hits:
        if cid not in seen:
            seen.add(cid)
            ordered.append(cid)

    return ordered


def fuzzy_match(phrases: list[str], chunks: list[dict], threshold: int = 75) -> list[str]:
    """Find chunks whose content fuzzy-matches the phrases (TheFuzz)."""
    if not HAS_FUZZ or not phrases:
        return []

    hits = []
    for c in chunks:
        content_low = c["page_content"].lower()
        for p in phrases[:5]:  # limit to avoid O(n*m) blowup
            if fuzz.partial_ratio(p, content_low) >= threshold:
                hits.append(c["metadata"]["chunk_id"])
                break
    return hits


def semantic_fallback(answer: str, chunks: list[dict], threshold: float = 0.30) -> list[str]:
    """Cosine similarity between the reference answer and each chunk.

    Uses BGE-M3 — already loaded in the project. Falls back to empty
    if the embedding model cannot be loaded (e.g., in a script context).
    """
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np
    except ImportError:
        return []

    model = SentenceTransformer("BAAI/bge-m3", device="cpu")
    a_vec = model.encode([answer], normalize_embeddings=True)[0]

    scored = []
    for c in chunks:
        c_vec = model.encode([c["page_content"][:1500]], normalize_embeddings=True)[0]
        sim = float(np.dot(a_vec, c_vec))
        if sim >= threshold:
            scored.append((c["metadata"]["chunk_id"], sim))

    scored.sort(key=lambda x: -x[1])
    return [cid for cid, _ in scored[:5]]


def load_chunks(bank_file: str) -> list[dict]:
    path = CHUNKS_DIR / f"{bank_file}.jsonl"
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def find_relevant_chunks(q: dict, chunks: list[dict], bank: str) -> tuple[list[str], str]:
    """3-stage strategy. Returns (chunk_ids, method_used)."""
    answer = q.get("reference_answer", "")
    keywords = q.get("ground_truth_keywords", [])

    # Stage 1 — substring on distinctive values
    values = extract_values(answer)
    if values:
        hits = substring_match(values, chunks)
        if hits:
            return hits[:5], "substring"

    # Stage 2 — fuzzy on key phrases
    phrases = extract_phrases(answer)
    hits = fuzzy_match(phrases, chunks)
    if hits:
        return hits[:5], "fuzzy"

    # Stage 3 — semantic fallback
    if answer:
        hits = semantic_fallback(answer, chunks)
        if hits:
            return hits, "semantic"

    # Last resort — keyword match (original behaviour)
    if keywords:
        hits = [
            c["metadata"]["chunk_id"]
            for c in chunks
            if all(kw.lower() in c["page_content"].lower() for kw in keywords)
        ]
        if hits:
            return hits[:5], "keywords"

    return [], "failed"


def main():
    with open(GROUND_TRUTH, encoding="utf-8") as f:
        test_set = json.load(f)

    chunks_by_bank = {stem: load_chunks(stem) for stem in set(BANK_TO_FILE.values())}

    mapped = 0
    empty = 0
    method_counts = {"substring": 0, "fuzzy": 0, "semantic": 0, "keywords": 0, "failed": 0}

    for q in test_set:
        bank = q.get("expected_bank", "").upper()
        if bank == "MULTI" or bank not in BANK_TO_FILE:
            q["relevant_chunk_ids_v2"] = []
            q["mapping_method"] = "multi_skip"
            empty += 1
            continue

        chunks = chunks_by_bank[BANK_TO_FILE[bank]]
        new_ids, method = find_relevant_chunks(q, chunks, bank)
        q["relevant_chunk_ids_v2"] = new_ids
        q["mapping_method"] = method
        method_counts[method] += 1

        if new_ids:
            mapped += 1
        else:
            empty += 1
            print(f"  ⚠️  {q['question_id']} ({bank}) — no matching chunk")

    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(test_set, f, indent=2, ensure_ascii=False)

    print(f"\nMapped: {mapped} / {len(test_set)}")
    print(f"Empty : {empty}")
    print("\nMapping method breakdown:")
    for m, n in method_counts.items():
        print(f"  {m:<10}: {n}")
    print(f"\nSaved → {OUTPUT}")


if __name__ == "__main__":
    main()
