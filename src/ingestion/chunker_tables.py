# src/ingestion/chunker_tables.py
"""Row-level chunking for financial tables.

Each table row becomes its own chunk, with the bank name and section
title baked into the embedded text. This is the structure-aware
row-level approach recommended in the literature: a per-metric vector
instead of one vector averaging 20 metrics.
"""

from src.common.logger import logger_manager
from src.ingestion.metadata import DocumentMetadata, ChunkMetadata, ProcessedChunk

logger = logger_manager.get_logger(__name__)

MAX_HEADER_ROWS = 3


class TableChunker:
    def split_structured(self, pages, base_metadata):
        chunks = []
        skipped = 0

        for page in pages:
            page_num = page.get("page")
            section = page.get("section_title") or base_metadata.document_id

            for table in page.get("tables", []):
                if not self._usable(table):
                    skipped += 1
                    continue
                chunks += self._chunk_table(table, page_num, section, base_metadata)

        logger.info(
            f"{base_metadata.document_id}: {len(chunks)} row chunks, {skipped} tables skipped"
        )
        return chunks

    def _usable(self, table):
        rows = table.get("rows") or []
        return len(rows) >= 2

    def _split_header(self, rows):
        """Return (header_rows, data_rows).

        A row is still a header if its first cell is empty. The first row
        always counts as header even if the first cell has content.
        """
        header = []
        for i, row in enumerate(rows):
            if i >= MAX_HEADER_ROWS:
                break
            if i == 0:
                header.append(row)
                continue
            if not (row[0] if row else "").strip():
                header.append(row)
                continue
            break
        return header, rows[len(header) :]

    def _merge_headers(self, header_rows):
        """Flatten multi-row headers into one label per column.
        ["2025 performance", "Reported"] will become "2025 performance > Reported".
        """
        if not header_rows:
            return []

        n_cols = max(len(r) for r in header_rows)
        labels = []
        for col in range(n_cols):
            parts = []
            for row in header_rows:
                cell = (row[col] if col < len(row) else "").strip()
                if cell and cell not in parts:
                    parts.append(cell)
            labels.append(" > ".join(parts) if parts else f"Column {col + 1}")
        return labels

    def _chunk_table(self, table, page_num, section, base_metadata):
        header_rows, data_rows = self._split_header(table["rows"])
        headers = self._merge_headers(header_rows)

        chunks = []
        for i, row in enumerate(data_rows):
            metric = (row[0] if row else "").strip()
            if not metric:
                continue

            text = self._format_row(
                metric,
                headers,
                row,
                bank=base_metadata.bank_name,
                section=section,
                page=page_num,
            )

            meta = ChunkMetadata(
                document_id=base_metadata.document_id,
                bank_name=base_metadata.bank_name,
                year=base_metadata.year,
                source_file=base_metadata.source_file,
                chunk_id=f"{base_metadata.document_id}_row_{page_num}_{i}",
                section_title=section,
                page_number=page_num,
                chunk_index=len(chunks),
                chunk_type="row",
            )
            chunks.append(ProcessedChunk(page_content=text, metadata=meta))

        return chunks

    def _format_row(self, metric, headers, row, bank, section, page):
        """Build the embedded text for one row.

        The prefix [Bank | Section | Metric] anchors the vector on the
        specific metric, so BGE-M3 can match specific terms against
        these chunks instead of the whole table.
        """
        prefix = f"[{bank} | {section} | {metric}]"
        lines = [prefix]

        for col in range(1, len(row)):
            value = (row[col] or "").strip()
            if not value:
                continue
            label = headers[col] if col < len(headers) else f"Column {col + 1}"
            lines.append(f"{label}: {value}")

        return "\n".join(lines)
