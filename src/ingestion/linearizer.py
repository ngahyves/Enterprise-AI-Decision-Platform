# src/ingestion/linearizer.py

import re
from markdown_it import MarkdownIt
from src.common.logger import logger_manager

logger = logger_manager.get_logger(__name__)


UNIT_PATTERN = re.compile(
    r"(?:unit|en)\s*:?\s*(millions?|thousands?|billions?|%|EUR|USD|CAD)",
    re.I,
)
NEGATIVE_PATTERN = re.compile(r"^\(([\d\s.,]+)\)$")


class TableLinearizer:
    """Converts Markdown tables into natural language sentences."""

    def __init__(self):
        self.md = MarkdownIt("gfm-like").disable("linkify")

    def linearize(self, content: str) -> str:
        for raw, headers, rows in self._extract_tables(content):
            section = self._find_section(content, raw)
            unit = self._find_unit(content, raw)
            text = self._build_sentences(headers, rows, section, unit)
            if text:
                content = content.replace(raw, text)
        return content

    def _extract_tables(self, content):
        tokens = self.md.parse(content)
        tables = []
        table = None
        row = None
        in_header = False
        start = None
        line_count = 0

        for token in tokens:
            if token.type == "table_open":
                table = {"headers": [], "rows": []}
                start = token.map[0] if token.map else None
                line_count = 0

            elif token.type == "thead_open":
                in_header = True

            elif token.type == "thead_close":
                in_header = False

            elif token.type == "tr_open":
                row = []

            elif token.type == "tr_close":
                if table is not None and row is not None:
                    target = table["headers"] if in_header else table["rows"]
                    target.append(row)
                    line_count += 1
                row = None

            elif token.type == "inline" and row is not None:
                row.append(token.content)

            elif token.type == "table_close":
                if start is not None:
                    end = start + line_count + 1
                    lines = content.split("\n")
                    raw = "\n".join(lines[start:end])
                    headers_flat = table["headers"][0] if table["headers"] else []
                    tables.append((raw, headers_flat, table["rows"]))
                table = None
                start = None
                line_count = 0

        return tables

    def _find_section(self, content, raw):
        pos = content.find(raw)
        if pos == -1:
            return ""
        headers = re.findall(r"^#{1,6}\s+(.+)$", content[:pos], re.M)
        return " > ".join(h.strip() for h in headers[-2:])

    def _find_unit(self, content, raw):
        pos = content.find(raw)
        if pos == -1:
            return ""
        window = content[pos + len(raw) : pos + len(raw) + 400]
        match = UNIT_PATTERN.search(window)
        return match.group(1) if match else ""

    def _clean_cell(self, value):
        value = " ".join(value.strip().strip("|").split())
        negative = NEGATIVE_PATTERN.match(value)
        return f"-{negative.group(1)}" if negative else value

    def _build_sentences(self, headers, rows, section, unit):
        if not headers or not rows:
            return ""

        context = " ".join(filter(None, [section, f"({unit})" if unit else ""]))
        prefix = f"{context} " if context else ""
        columns = headers[1:]
        sentences = []

        for row in rows:
            if not row:
                continue
            metric = self._clean_cell(row[0])
            if not metric:
                continue
            for column, value in zip(columns, row[1:]):
                column = self._clean_cell(column)
                value = self._clean_cell(value) or "n.c."
                if column:
                    sentences.append(f"{prefix}{metric} {column}: {value}.")

        return "\n".join(sentences)
