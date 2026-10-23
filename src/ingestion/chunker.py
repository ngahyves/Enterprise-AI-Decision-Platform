# src/ingestion/chunker.py
"""Chunk narrative prose from annual reports.

Tables are handled separately by TableChunker, which reads the structured
JSON produced by the parser. Any chunk dominated by markdown table rows
is filtered out here.
"""

from pathlib import Path
from typing import List

from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from src.common.config_manager import config
from src.common.logger import logger_manager
from src.ingestion.metadata import DocumentMetadata, ChunkMetadata, ProcessedChunk

logger = logger_manager.get_logger(__name__)


class FinancialChunker:
    """Split narrative markdown into chunks, skipping table-heavy sections."""

    def __init__(self):
        ingestion_config = config["ingestion"]
        self.chunk_size = ingestion_config["chunk_size"]
        self.chunk_overlap = ingestion_config["chunk_overlap"]

        logger.info(
            f"FinancialChunker initialized: "
            f"chunk_size={self.chunk_size}, overlap={self.chunk_overlap}"
        )

    def _is_table_dominated(self, content: str) -> bool:
        """True if more than half the non-empty lines are markdown table rows."""
        lines = [l for l in content.split("\n") if l.strip()]
        if not lines:
            return False
        table_lines = sum(1 for l in lines if l.strip().startswith("|"))
        return table_lines / len(lines) > 0.5

    def split_markdown(
        self,
        markdown_text: str,
        base_metadata: DocumentMetadata,
    ) -> List[ProcessedChunk]:
        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]
        markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
        header_splits = markdown_splitter.split_text(markdown_text)

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""],
        )
        splits = text_splitter.split_documents(header_splits)

        chunks = []
        skipped = 0

        for split in splits:
            content = split.page_content.strip()

            if len(content) < 40:
                continue

            if self._is_table_dominated(content):
                skipped += 1
                continue

            section_path = " > ".join(
                split.metadata.get(h, "")
                for h in ["Header 1", "Header 2", "Header 3"]
                if h in split.metadata
            )

            idx = len(chunks)
            meta = ChunkMetadata(
                document_id=base_metadata.document_id,
                bank_name=base_metadata.bank_name,
                year=base_metadata.year,
                source_file=base_metadata.source_file,
                chunk_id=f"{base_metadata.document_id}_narrative_{idx}",
                section_title=section_path or None,
                page_number=None,
                chunk_index=idx,
                chunk_type="narrative",
            )
            chunks.append(ProcessedChunk(page_content=content, metadata=meta))

        logger.info(
            f"{base_metadata.document_id}: {len(chunks)} narrative chunks "
            f"({skipped} table-heavy chunks skipped)"
        )
        return chunks

    def save_jsonl(self, chunks: List[ProcessedChunk], output_file: Path) -> None:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            for chunk in chunks:
                f.write(chunk.model_dump_json() + "\n")
        logger.info(f"{len(chunks)} chunks saved in {output_file}")
