# src/ingestion/parser.py
"""
This module uses the official llama-parse package
to send the PDF and retrieve structured Markdown.
"""

from pathlib import Path
from llama_cloud_services import LlamaParse
from src.common.config_manager import env_settings, config
from src.common.logger import logger_manager

logger = logger_manager.get_logger(__name__)


class FinancialReportParser:
    """Extraction of PDF via LlamaParse."""

    def __init__(self, api_key: str = None):
        self.api_key = api_key or env_settings.llama_cloud_api_key

        # Read parameters from config.yaml
        ingestion_config = config["ingestion"]

        self.parser = LlamaParse(
            api_key=self.api_key,
            result_type="markdown",
            verbose=False,
            parsing_instruction=ingestion_config["parsing_instruction"],
        )

    async def parse_pdf_to_markdown(
        self, pdf_path: Path, output_dir: Path, force: bool = False
    ) -> Path:
        """
        Parse a PDF and return the result in markdown.
        Args:
            pdf_path: PDF's path.
            output_dir: output folder.
            force: If True, re-parse even if the file exists.
        Returns:
            Path to the generated Markdown file.
        """
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found : {pdf_path}")

        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"{pdf_path.stem}.md"

        # Skip if file already exists
        if output_file.exists() and not force:
            logger.info(f"Markdown already exists for {pdf_path.name}. Skipping.")
            return output_file

        logger.info(f"Starting parsing for: {pdf_path.name}")

        try:
            documents = await self.parser.aload_data(str(pdf_path))

            # Concatenate pages
            full_markdown = "\n\n---\n\n".join([doc.text for doc in documents])

            with open(output_file, "w", encoding="utf-8") as f:
                f.write(full_markdown)

            logger.info(
                f"Parsing successful. Saved to: {output_file} "
                f"({len(documents)} pages, {len(full_markdown)} characters)"
            )
            return output_file

        except Exception as e:
            logger.error(f"Error parsing {pdf_path.name}: {str(e)}")
            raise e