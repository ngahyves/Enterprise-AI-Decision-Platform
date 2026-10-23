# src/ingestion/ingest_pipeline.py
"""PDF → markdown + structured JSON → combined chunks JSONL.

Two chunkers feed one index:
  - FinancialChunker  : narrative prose from the markdown
  - TableChunker      : one chunk per table row from the structured JSON

Both produce ProcessedChunk objects with a `chunk_type` field. The
combined file is what vector_store.py reads.
"""

import asyncio
import json
import re
from pathlib import Path

from src.common.config_manager import env_settings, config
from src.common.logger import logger_manager
from src.ingestion.parser import FinancialReportParser
from src.ingestion.chunker import FinancialChunker
from src.ingestion.chunker_tables import TableChunker
from src.ingestion.metadata import MetadataExtractor

logger = logger_manager.get_logger(__name__)


class IngestionPipeline:
    def __init__(self):
        self.raw_dir = Path(config["paths"]["data_raw"])
        self.chunks_dir = Path(config["paths"]["data_processed_chunks"])

        self.parser = FinancialReportParser(api_key=env_settings.llama_cloud_api_key)
        self.narrative_chunker = FinancialChunker()
        self.table_chunker = TableChunker()
        self.metadata_extractor = MetadataExtractor()

    async def run(self):
        pdfs = list(self.raw_dir.glob("*.pdf"))
        if not pdfs:
            logger.warning(f"No PDF in {self.raw_dir}")
            return
        for pdf in pdfs:
            await self._process_pdf(pdf)

    async def _process_pdf(self, pdf_path: Path):
        logger.info(f"Processing {pdf_path.name}")

        # 1. Parse → markdown + structured JSON
        paths = await self.parser.parse_pdf(pdf_path, force=False)
        md_path = paths["markdown"]
        json_path = paths["json"]

        # 2. Document-level metadata (bank, year, source)
        doc_metadata = self.metadata_extractor.extract_document_metadata(md_path)

        # 3. Narrative chunks from markdown
        markdown_text = md_path.read_text(encoding="utf-8")
        narrative = self.narrative_chunker.split_markdown(markdown_text, doc_metadata)
        narrative = [self._clean_narrative(c) for c in narrative]
        narrative = [c for c in narrative if len(c.page_content) >= 40]

        # 4. Row chunks from structured JSON
        pages = json.loads(json_path.read_text(encoding="utf-8"))
        rows = self.table_chunker.split_structured(pages, doc_metadata)

        # 5. Save combined JSONL
        all_chunks = narrative + rows
        out_file = self.chunks_dir / f"{md_path.stem}.jsonl"
        self._save_jsonl(all_chunks, out_file)

        logger.info(
            f"{md_path.stem}: {len(narrative)} narrative + {len(rows)} rows "
            f"= {len(all_chunks)} total → {out_file.name}"
        )

    @staticmethod
    def _clean_narrative(chunk):
        content = chunk.page_content

        # Drop LlamaParse artifacts (it echoes our prompt when a page is empty)
        NOISE = [
            "the content provided does not include",
            "the financial tables and data are not provided",
            "please provide the specific",
        ]
        if any(marker in content.lower() for marker in NOISE):
            chunk.page_content = ""
            return chunk

        content = re.sub(r"\(\d{1,2}\)", "", content)
        content = re.sub(r"\bpage\s+\d+\b", "", content, flags=re.I)
        content = re.sub(r"\bp\.\s*\d+\b", "", content, flags=re.I)
        content = re.sub(r"[ \t]+", " ", content)
        content = re.sub(r"\n{3,}", "\n\n", content)
        content = content.strip()

        parts = []
        if chunk.metadata.bank_name:
            parts.append(f"Bank: {chunk.metadata.bank_name}")
        if chunk.metadata.year:
            parts.append(f"Year: {chunk.metadata.year}")
        if chunk.metadata.section_title:
            parts.append(f"Section: {chunk.metadata.section_title}")

        if parts and not content.startswith("[Bank:"):
            content = f"[{' | '.join(parts)}]\n{content}"

        chunk.page_content = content
        return chunk

    @staticmethod
    def _save_jsonl(chunks, output_file: Path):
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            for c in chunks:
                f.write(c.model_dump_json() + "\n")
        logger.info(f"Saved {len(chunks)} chunks → {output_file}")


if __name__ == "__main__":
    pipeline = IngestionPipeline()
    asyncio.run(pipeline.run())
