# src/agents/nodes/extract.py
from src.agents.state import AgentState
from src.agents.schemas import ExtractionOutput
from src.agents.llm import get_fast_agent
from src.common.logger import logger_manager
import time

logger = logger_manager.get_logger(__name__)


def _build_prompt(question: str, bank: str, chunks: list[dict]) -> str:
    MAX_CHARS = 400
    # Build the context block from the retrieved chunks
    parts = []
    for c in chunks:
        header = f"[{c['chunk_id']} | {c['section']}]"
        body = c["content"][:MAX_CHARS]
        parts.append(f"{header}\n{body}")

    context = "\n\n".join(parts)
    return (
        f"Question: {question}\n"
        f"Bank: {bank}\n\n"
        f"Chunks:\n{context}\n\n"
        "Extract the CONSOLIDATED / TOTAL figure for the requested metric. "
        "Prefer the 'financial highlights' or 'consolidated statement of income' section. "
        "Ignore segment-level or quarterly breakdowns unless no total exists. "
        "Return a JSON object with: bank, facts[metric, value, quote, source_chunk_id, bank, page_number]. "
        "Return only the JSON."
    )


def extract_facts(state: AgentState) -> dict:
    """Extracts structured facts (metrics, risks) by bank using an LLM."""
    question = state["question"]
    chunks_by_bank = state.get("chunks_by_bank", {})

    structured_llm = get_fast_agent().with_structured_output(
        ExtractionOutput,
        method="json_schema",
    )

    extracted = {}
    errors = []

    for i, (bank, chunks) in enumerate(chunks_by_bank.items()):
        if not chunks:
            logger.warning(f"No chunks to extract from for {bank}")
            extracted[bank] = []
            continue

        # Respect Groq TPM limits between consecutive LLM calls
        if i > 0:
            logger.info("Sleeping 30s to respect Groq TPM rate limit")
            time.sleep(30)

        prompt = _build_prompt(question, bank, chunks)
        logger.info(f"[{bank}] chunk_ids={[c['chunk_id'] for c in chunks]}")

        try:
            result: ExtractionOutput = structured_llm.invoke(prompt)
            extracted[bank] = [f.model_dump() for f in result.facts]
            logger.info(f"{bank}: {len(result.facts)} facts extracted")
        except Exception as e:
            logger.exception(f"Extraction failed for {bank}")
            extracted[bank] = []
            errors.append(f"Extraction failed for {bank}: {e}")

    return {"extracted_facts": extracted, "errors": errors}
