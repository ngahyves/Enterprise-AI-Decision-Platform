# scripts/extract_chunks.py
"""
Extrait tous les chunks de Qdrant dans un fichier JSONL.
"""

import json
from pathlib import Path
from qdrant_client import QdrantClient

from src.common.config_manager import env_settings


def extract_all_chunks():
    """Extrait tous les chunks de Qdrant."""

    # Connexion à Qdrant
    client = QdrantClient(
        host=env_settings.qdrant_host,
        port=env_settings.qdrant_port,
    )
    collection = env_settings.qdrant_collection_name

    print(f"Connexion à Qdrant : {collection}")

    # Scroll tous les points
    all_points = []
    offset = None
    batch_size = 100

    while True:
        points, offset = client.scroll(
            collection_name=collection,
            limit=batch_size,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        all_points.extend(points)
        print(f"  Récupérés: {len(all_points)} chunks", end="\r")

        if offset is None:
            break

    print(f"\n✅ Total chunks : {len(all_points)}")

    # Sauvegarde en JSONL
    output_path = Path("data/processed/all_chunks.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for p in all_points:
            meta = p.payload.get("metadata", {})
            record = {
                "chunk_id": meta.get("chunk_id"),
                "bank_name": meta.get("bank_name"),
                "section_title": meta.get("section_title"),
                "page_content": p.payload.get("page_content", ""),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"📁 Sauvegardé dans : {output_path}")


if __name__ == "__main__":
    extract_all_chunks()
