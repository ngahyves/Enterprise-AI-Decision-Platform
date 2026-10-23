# src/agents/nodes/synthesizer.py
import json
from src.agents.state import AgentState
from src.agents.llm import get_complex_analyst
from src.common.logger import logger_manager
import time

logger = logger_manager.get_logger(__name__)


def _compute_confidence(extracted_facts: dict, claims: list[dict]) -> str:
    total_facts = sum(len(facts) for facts in extracted_facts.values())
    if total_facts == 0:
        return "low"
    if claims and not all(c.get("supported", True) for c in claims):
        return "medium"
    return "high"


def synthesize_answer(state: AgentState) -> dict:
    """Compile the final answer based on the extracted facts, the comparison, and the verified claims."""
    question = state["question"]
    task_type = state["task_type"]
    extracted_facts = state.get("extracted_facts", {})
    comparison = state.get("comparison")
    claims = state.get("claims", [])

    context = {
        "task_type": task_type,
        "extracted_facts": extracted_facts,
        "comparison": comparison,
        "verified_claims": [c for c in claims if c.get("supported", True)],
    }

    prompt = (
        f"User Original Query: '{question}'\n\n"
        f"Structured Context:\n{json.dumps(context, indent=2, ensure_ascii=False)}\n\n"
        "Task: Write a comprehensive, professional financial response based strictly on this context.\n"
        "Formatting Requirements:\n"
        "1. Structure clearly (Executive Summary, Bank-by-Bank Analysis, Key Takeaways).\n"
        "2. Cite source chunk IDs or bank sections for every financial figure.\n"
        "3. Determine a confidence level ('high', 'medium', 'low') based on evidence clarity.\n"
        "4. Maintain a formal institutional banking tone."
    )

    try:
        time.sleep(3)
        response = get_complex_analyst().invoke(prompt)
        confidence = _compute_confidence(extracted_facts, claims)
        logger.info(f"Synthesis done, confidence={confidence}")
        return {"final_answer": response.content, "confidence": confidence}
    except Exception as e:
        logger.error(f"Synthesizer failed: {e}")
        return {
            "final_answer": "An error occurred while synthesizing the final report.",
            "confidence": "low",
            "errors": [f"Synthesizer failed: {e}"],
        }
