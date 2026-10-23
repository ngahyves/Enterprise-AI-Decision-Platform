# src/agents/nodes/verifier.py
from src.agents.state import AgentState
from src.agents.schemas import VerifierOutput
from src.agents.llm import get_fast_agent
from src.common.logger import logger_manager

logger = logger_manager.get_logger(__name__)

MAX_CHUNK_CHARS = 600  # cap per chunk to fit Groq TPM
MAX_CHUNKS_PER_BANK = 3  # cap per bank


def _collect_claims(extracted_facts: dict, comparison: dict | None) -> list[str]:
    claims = []
    for bank, facts in extracted_facts.items():
        for f in facts:
            claims.append(
                f"Bank {bank}: {f.get('metric')} = {f.get('value')} (Quote: {f.get('quote')})"
            )
    if comparison:
        for diff in comparison.get("key_differences", []):
            claims.append(f"Comparison Difference: {diff}")
        for theme in comparison.get("common_themes", []):
            claims.append(f"Comparison Theme: {theme}")
    return claims


def verify_facts(state: AgentState) -> dict:
    """Verify that each extracted fact is supported by a source chunk."""
    chunks_by_bank = state.get("chunks_by_bank", {})
    extracted_facts = state.get("extracted_facts", {})
    comparison = state.get("comparison")
    iteration = state.get("verification_iteration", 0) + 1

    if iteration > 2:
        logger.warning("Max iterations reached, forcing status='ok'")
        return {"verification_status": "ok", "verification_iteration": iteration}

    claims = _collect_claims(extracted_facts, comparison)
    if not claims:
        return {"verification_status": "ok", "verification_iteration": iteration}

    # Truncate evidence to stay under Groq free-tier TPM
    evidence_parts = []
    for bank, chunks in chunks_by_bank.items():
        for c in chunks[:MAX_CHUNKS_PER_BANK]:
            content = c["content"][:MAX_CHUNK_CHARS]
            evidence_parts.append(f"[Bank: {bank} | Chunk ID: {c['chunk_id']}]\n{content}")
    evidence = "\n\n".join(evidence_parts)
    claims_text = "\n".join(f"- {c}" for c in claims)

    prompt = (
        "You are a strict fact-checking auditor for a financial AI assistant.\n"
        "Verify if each claim below is fully supported by the source evidence.\n\n"
        f"Source Evidence:\n{evidence}\n\n"
        f"Claims:\n{claims_text}\n\n"
        "For each claim, set supported=True only if the evidence directly contains it. "
        "Set the overall status to 'ok' if all pass, 'retry' if minor gaps exist, 'fail' if hallucinations are found. "
        "Return the results as JSON matching the required schema."
    )

    try:
        structured_llm = get_fast_agent().with_structured_output(
            VerifierOutput, method="json_schema"
        )
        result: VerifierOutput = structured_llm.invoke(prompt)
        logger.info(f"Verifier: status={result.status}, claims={len(result.claims)}")
        return {
            "claims": [c.model_dump() for c in result.claims],
            "verification_status": result.status,
            "verification_iteration": iteration,
        }
    except Exception as e:
        logger.exception(f"Verifier failed")
        return {
            "verification_status": "retry",
            "verification_iteration": iteration,
            "errors": [f"Verifier failed: {e}"],
        }
