# src/ingestion/ingest_pipeline.py
import asyncio
from pathlib import Path

from src.common.config_manager import env_settings, config
from src.common.logger import logger_manager
from src.ingestion.parser import FinancialReportParser
from src.ingestion.chunker import FinancialChunker
from src.ingestion.metadata import MetadataExtractor

logger = logger_manager.get_logger(__name__)


class IngestionPipeline:
    """Pipeline for ingesting financial reports: PDF → Markdown → Chunks JSONL."""

    def __init__(self):
        # 1. Paths from config
        self.raw_dir = Path(config["paths"]["data_raw"])
        self.markdown_dir = Path(config["paths"]["data_processed_markdown"])
        self.chunks_dir = Path(config["paths"]["data_processed_chunks"])

        # 2. Components
        self.parser = FinancialReportParser(api_key=env_settings.llama_cloud_api_key)
        self.chunker = FinancialChunker()  # Reads config automatically
        self.metadata_extractor = MetadataExtractor()

    async def run(self) -> None:
        """Run the full ingestion pipeline."""

        # 1. Check for PDFs
        pdf_files = list(self.raw_dir.glob("*.pdf"))
        if not pdf_files:
            logger.warning(
                f"No PDF found in {self.raw_dir}. "
                f"Place the annual reports here."
            )
            return

        # 2. Process each PDF
        for pdf_path in pdf_files:
            await self._process_pdf(pdf_path)

    async def _process_pdf(self, pdf_path: Path) -> None:
        """Process a single PDF: parse, chunk, save."""
        logger.info(f"Processing file: {pdf_path.name}")

        # 1. Parsing (PDF → Markdown)
        md_path = await self.parser.parse_pdf_to_markdown(
            pdf_path=pdf_path,
            output_dir=self.markdown_dir,
        )

        # 2. Extract document metadata
        doc_metadata = self.metadata_extractor.extract_document_metadata(md_path)

        # 3. Read Markdown
        markdown_content = md_path.read_text(encoding="utf-8")

        # 4. Chunking (Markdown → Chunks)
        chunks = self.chunker.split_markdown(markdown_content, doc_metadata)

        # 5. Save chunks as JSONL
        output_file = self.chunks_dir / f"{md_path.stem}.jsonl"
        self.chunker.save_jsonl(chunks, output_file)

        logger.info(
            f"{len(chunks)} chunks ready for {doc_metadata.bank_name} "
            f"({doc_metadata.year}) → {output_file}"
        )


if __name__ == "__main__":
    pipeline = IngestionPipeline()
    asyncio.run(pipeline.run())