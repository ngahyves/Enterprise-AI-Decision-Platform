# scripts/clean_chunks.py
"""
Clean chunks before fine-tuning and reindexing.
"""

import json
import re
from pathlib import Path

from src.ingestion.linearizer import TableLinearizer


_linearizer = None


def get_linearizer():
    global _linearizer
    if _linearizer is None:
        _linearizer = TableLinearizer()
    return _linearizer


def separate_tables(content: str) -> str:
    """Add a blank line between two consecutive Markdown tables."""
    lines = content.split("\n")
    result = []
    table_count = 0

    for i, line in enumerate(lines):
        stripped = line.strip()

        if stripped.startswith("|") and i + 1 < len(lines):
            next_line = lines[i + 1].strip()
            is_separator = (
                next_line.startswith("|")
                and all(c in "|-: " for c in next_line)
                and "-" in next_line
            )
            if is_separator:
                if table_count > 0:
                    result.append("")
                table_count += 1

        result.append(line)

    return "\n".join(result)


def clean_chunk(chunk: dict) -> dict:
    """Clean a single chunk."""
    content = chunk["page_content"]
    metadata = chunk.get("metadata", {})
    section = metadata.get("section_title", "")

    if not section:
        bank = metadata.get("bank_name", "")
        year = metadata.get("year", "")
        section = f"{bank} {year} Annual Report"

    # 0. Separate consecutive tables
    content = separate_tables(content)

    # 1. Remove redundant prefix
    content = re.sub(
        r"^(TD|RBC|BMO|SCOTIA)\s+\d{4}\s+Annual Report\.\s+Section:\s*",
        "",
        content,
        flags=re.IGNORECASE,
    )

    # 2. Remove footnotes
    content = re.sub(r"\(\d+\)", "", content)

    # 3. Remove page references
    content = re.sub(r"\bpage\s+\d+\b", "", content, flags=re.IGNORECASE)
    content = re.sub(r"\bp\.\s*\d+\b", "", content, flags=re.IGNORECASE)

    # 4. Normalize whitespace
    content = re.sub(r"[ \t]+", " ", content)
    content = re.sub(r"\n{3,}", "\n\n", content)

    # 5. Normalize spaces around pipes
    content = re.sub(r"[ \t]*\|[ \t]*", " | ", content)

    # 6. Strip
    content = content.strip()

    # 6b. Remove leading spaces before pipes
    content = re.sub(r"^\s+\|", "|", content, flags=re.MULTILINE)

    # 7. Add section context
    if "|" in content and section:
        content = f"[Section: {section}]\n{content}"

    # 8. Linearize tables (replace tables with natural text)
    if "|" in content:
        linearizer = get_linearizer()
        content = linearizer.linearize(content)

    # 9. Filter out chunks that are too short
    if len(content) < 50:
        chunk["skip"] = True
    else:
        chunk["skip"] = False

    chunk["page_content_clean"] = content
    return chunk


def clean_all_chunks():
    """Clean all chunks."""
    input_path = Path("data/processed/all_chunks.jsonl")
    output_path = Path("data/processed/all_chunks_clean.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    chunks = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))

    print(f"Chunks to clean : {len(chunks)}")

    kept = 0
    skipped = 0

    with open(output_path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            chunk = clean_chunk(chunk)

            if chunk["skip"]:
                skipped += 1
                continue

            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
            kept += 1

    print(f"Chunks kept   : {kept}")
    print(f"Chunks skipped: {skipped}")
    print(f"Saved to      : {output_path}")


if __name__ == "__main__":
    clean_all_chunks()
