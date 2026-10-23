# scripts/build_training_data.py
"""
Combine les paires synthétiques avec le golden dataset.
"""

import json
from pathlib import Path


def build_training_data():
    # Charge les paires synthétiques
    synthetic_path = Path("data/training/synthetic_questions.jsonl")
    synthetic = []
    with open(synthetic_path, "r", encoding="utf-8") as f:
        for line in f:
            synthetic.append(json.loads(line))

    print(f"Paires synthétiques : {len(synthetic)}")

    # Charge le golden dataset
    with open("data/evaluation/test_set.json", "r", encoding="utf-8") as f:
        golden = json.load(f)

    # Extrait les paires du golden dataset (uniquement celles avec ground truth)
    golden_pairs = []
    for q in golden:
        if q.get("relevant_chunk_ids") and q.get("reference_answer"):
            golden_pairs.append(
                {
                    "question": q["question"],
                    "chunk_ids": q["relevant_chunk_ids"],
                    "reference_answer": q["reference_answer"],
                }
            )

    print(f"Paires golden       : {len(golden_pairs)}")

    # Sauvegarde combinée
    output_path = Path("data/training/train_pairs.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for pair in synthetic:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")
        for pair in golden_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    total = len(synthetic) + len(golden_pairs)
    print(f"\n✅ Total paires : {total}")
    print(f"📁 Sauvegardé dans : {output_path}")


if __name__ == "__main__":
    build_training_data()
