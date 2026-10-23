# src/ingestion/metadata.py

"""
Metadata attached to every chunk. Without it, the RAG cannot cite
its sources — which matters for a financial project.
"""

from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field


class DocumentMetadata(BaseModel):
    document_id: str = Field(..., description="Unique document ID (ex: rbc_2025)")
    bank_name: str = Field(..., description="Bank name (RBC, TD, BMO, SCOTIA)")
    year: int = Field(2025, description="Report year")
    source_file: str = Field(..., description="Path to the original PDF")


class ChunkMetadata(DocumentMetadata):
    chunk_id: str = Field(..., description="Unique chunk ID")
    section_title: Optional[str] = Field(None, description="Parent section title")
    page_number: Optional[int] = Field(None, ge=1, description="Page number")
    chunk_index: int = Field(0, description="Position within the document")
    chunk_type: Literal["narrative", "row"] = Field(
        "narrative", description="Chunk origin: prose or a table row"
    )


class ProcessedChunk(BaseModel):
    page_content: str = Field(..., min_length=1)
    metadata: ChunkMetadata


class MetadataExtractor:
    """Derive document metadata from a file path."""

    def extract_document_metadata(self, file_path: Path) -> DocumentMetadata:
        stem = file_path.stem
        parts = stem.split("_")
        bank = parts[0].upper() if parts else "UNKNOWN"
        year = int(parts[-1]) if parts[-1].isdigit() else 2025

        return DocumentMetadata(
            document_id=stem,
            bank_name=bank,
            year=year,
            source_file=str(file_path),
        )
