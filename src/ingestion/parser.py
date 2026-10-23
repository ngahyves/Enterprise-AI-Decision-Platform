# src/ingestion/parser.py
"""
Extraction of PDF via LlamaParse with structured JSON output.

Produces two artifacts per PDF:
  - <stem>.md    : narrative content, saved to data_processed_markdown
  - <stem>.json  : structured elements with JSON tables, saved to data_processed_structured

Output paths come from config.yaml (section `paths`).
"""

import json
from pathlib import Path
from typing import Any
import asyncio

from llama_cloud_services import LlamaParse

from src.common.config_manager import env_settings, config
from src.common.logger import logger_manager

logger = logger_manager.get_logger(__name__)


class FinancialReportParser:
    """Parse annual reports into Markdown + structured JSON tables."""

    def __init__(self, api_key: str = None):
        self.api_key = api_key or env_settings.llama_cloud_api_key

        # Output directories — read from config.yaml
        self.markdown_dir = Path(config["paths"]["data_processed_markdown"])
        self.structured_dir = Path(config["paths"]["data_processed_structured"])

        self.parser = LlamaParse(
            api_key=self.api_key,
            result_type="json",
            verbose=False,
            system_prompt=config["ingestion"]["parsing_instruction"],
            premium_mode=False,
        )

    async def parse_pdf(self, pdf_path: Path, force: bool = False) -> dict[str, Path]:
        """
        Parse a PDF and return paths to both output artifacts.
        pdf_path: PDF file to parse.
        force: If True, re-parse even when outputs already exist.
        Returns:{'markdown': <path>, 'json': <path>}
        """
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        self.markdown_dir.mkdir(parents=True, exist_ok=True)
        self.structured_dir.mkdir(parents=True, exist_ok=True)

        md_file = self.markdown_dir / f"{pdf_path.stem}.md"
        json_file = self.structured_dir / f"{pdf_path.stem}.json"

        if md_file.exists() and json_file.exists() and not force:
            logger.info(f"Outputs already exist for {pdf_path.name}. Skipping.")
            return {"markdown": md_file, "json": json_file}

        logger.info(f"Parsing {pdf_path.name} with LlamaParse (JSON mode)")

        try:
            # JSON mode requires aget_json_result()
            json_result = await asyncio.to_thread(self.parser.get_json_result, str(pdf_path))

        except Exception as e:
            logger.error(f"Parsing failed for {pdf_path.name}: {e}")
            raise

        pages = self._extract_pages(json_result)

        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(pages, f, ensure_ascii=False, indent=2)

        md_content = "\n\n---\n\n".join(page["md"] for page in pages if page.get("md"))
        with open(md_file, "w", encoding="utf-8") as f:
            f.write(md_content)

        n_tables = sum(len(p.get("tables", [])) for p in pages)
        logger.info(
            f"Parsed {pdf_path.name}: {len(pages)} pages, {n_tables} tables "
            f"→ {md_file.name}, {json_file.name}"
        )

        return {"markdown": md_file, "json": json_file}

    def _extract_pages(self, json_result: list) -> list[dict[str, Any]]:
        """Flatten the LlamaParse JSON response into a list of normalized pages."""
        pages = []
        for doc in json_result:
            raw_pages = doc.get("pages") if isinstance(doc, dict) else None
            if not raw_pages:
                continue
            for page in raw_pages:
                pages.append(self._normalize_page(page))
        return pages

    def _normalize_page(self, page: dict) -> dict[str, Any]:
        """Turn one raw page dict into {page, md, text, tables}."""
        page_num = page.get("page") or page.get("page_number")
        md = page.get("md") or page.get("markdown") or ""
        text = page.get("text") or ""

        items = page.get("items") or page.get("elements") or page.get("children") or []
        tables = [item for item in items if self._is_table(item)]

        return {
            "page": page_num,
            "md": md,
            "text": text,
            "tables": tables,
        }

    @staticmethod
    def _is_table(item: dict) -> bool:
        """Return True if an item represents a table."""
        item_type = (item.get("type") or "").lower()
        return item_type in ("table", "table_row", "table_cell") or "rows" in item
