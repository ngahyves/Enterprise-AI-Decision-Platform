# src/ingestion/metadata.py

"""
Metadata are data associated with each chunk. This is the information that accompanies
the text but is not part of the text itself. Without metadata, our RAG cannot cite its
sources. This is a major problem for a financial project.
"""

from pydantic import BaseModel, Field
from typing import Optional
from pathlib import Path

class DocumentMetadata(BaseModel):
    document_id: str = Field(..., description="Unique document identifier (ex: rbc_2025)")
    bank_name: str = Field(..., description="Bank's name (RBC, TD, BMO, Scotiabank)")
    year: int = Field(2025, description="Report's year")
    source_file: str = Field(..., description="Path to the original PDF")

class ChunkMetadata(DocumentMetadata):
    chunk_id: str = Field(..., description="Chunk's Unique ID")
    section_title: Optional[str] = Field(None, description="Title of the parent Markdown section")
    page_number: Optional[int] = Field(None, ge=1, description="Page number")

class ProcessedChunk(BaseModel):
    page_content: str = Field(..., min_length=1, description="Chunk's text")
    metadata: ChunkMetadata

class MetadataExtractor:
    """Extracts document metadata from file paths."""
    def extract_document_metadata(self, file_path: Path) -> DocumentMetadata:
        """
        Extract document metadata from the file path.
        Args:
            file_path: Path to the document (e.g., "data/processed/markdown/rbc_2025.md").
        Returns:
            DocumentMetadata instance.
        """
        stem = file_path.stem  # ex: "rbc_2025"
        parts = stem.split("_")
        bank = parts[0].upper() if parts else "UNKNOWN"
        year = int(parts[-1]) if parts[-1].isdigit() else 2025

        return DocumentMetadata(
            document_id=stem,
            bank_name=bank,
            year=year,
            source_file=str(file_path),
        )