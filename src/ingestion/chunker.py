# src/ingestion/chunker.py
from typing import List
from pathlib import Path
import json

from langchain_text_splitters import MarkdownHeaderTextSplitter,RecursiveCharacterTextSplitter

from src.common.logger import logger_manager
from src.common.config_manager import config
from src.ingestion.metadata import DocumentMetadata, ChunkMetadata, ProcessedChunk

logger = logger_manager.get_logger(__name__)


class FinancialChunker:
    """Break down the annual reports into manageable chunks for the RAG."""

    def __init__(self, chunk_size: int = None, chunk_overlap: int = None,chunk_strategy: str = None):
        # Read params from config.yaml
        ingestion_config = config["ingestion"]

        self.chunk_size = chunk_size or ingestion_config["chunk_size"] # 800
        self.chunk_overlap = chunk_overlap or ingestion_config["chunk_overlap"] # 100
        self.chunk_strategy = chunk_strategy or ingestion_config["chunk_strategy"]
        # chunk_strategy is documented in config.yaml but is not used written explicitely in code
        # (only one strategy is implemented: markdown_header_aware)

        logger.info(
            f"FinancialChunker initialized: strategy={self.chunk_strategy}, "
            f"chunk_size={self.chunk_size}, chunk_overlap={self.chunk_overlap}"
        )

    def split_markdown(self, markdown_text: str, base_metadata: DocumentMetadata) -> List[ProcessedChunk]:
        """
        Split the Markdown by headings and then by font size.
        Args:
            markdown_text: The document's Markdown content.
            base_metadata: The document's metadata (Pydantic).
        Returns:
            A list of ProcessedChunk objects.
        """
        # 1.Splitting by Markdown Headings
        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]
        markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on
        )
        header_splits = markdown_splitter.split_text(markdown_text)

        # 2. Secondary breakdown by size
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        final_splits = text_splitter.split_documents(header_splits)

        # 3. Building the ProcessedChunks
        processed_chunks = []
        for idx, split in enumerate(final_splits):
            section_path = " > ".join(
                split.metadata.get(h, "")
                for h in ["Header 1", "Header 2", "Header 3"]
                if h in split.metadata
            )
            meta = ChunkMetadata(
                document_id=base_metadata.document_id,
                bank_name=base_metadata.bank_name,
                year=base_metadata.year,
                source_file=base_metadata.source_file,
                chunk_id=f"{base_metadata.document_id}_chunk_{idx}",
                section_title=section_path if section_path else None,
                page_number=None,
                chunk_index=idx,
            )

            processed_chunks.append(
                ProcessedChunk(
                    page_content=split.page_content,
                    metadata=meta,
                )
            )

        logger.info(
            f"Document '{base_metadata.document_id}' broken in "
            f"{len(processed_chunks)} chunks."
        )
        return processed_chunks

    def save_jsonl(self, chunks: List[ProcessedChunk], output_file: Path) -> None:
        """
        Saves the chunks as JSONL.
        Args:
            chunks: List of ProcessedChunk objects.
            output_file: Path to the output file.
        """
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w", encoding="utf-8") as f:
            for chunk in chunks:
                f.write(chunk.model_dump_json() + "\n")

        logger.info(f"{len(chunks)} chunks saved in {output_file}")