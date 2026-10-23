# src/agents/nodes/refuse.py
from src.agents.state import AgentState
from src.common.logger import logger_manager

logger = logger_manager.get_logger(__name__)

REFUSAL_MESSAGE = (
    "I am an Enterprise Financial Knowledge & Decision Assistant specialized exclusively "
    "in analyzing the 2025 annual and sustainability reports of major Canadian banks "
    "(RBC, TD, BMO, and Scotiabank).\n\n"
    "I cannot answer this query because it falls outside my supported scope or sufficient "
    "source evidence could not be found in the provided corpus. Please ask a question related "
    "to the financial performance, risk disclosures, or strategic priorities of these four banks."
)


def refuse_query(state: AgentState) -> dict:
    """Returns a professional rejection when the request is outside the scope or lacks sufficient evidence."""
    logger.warning(f"Refusal triggered for: '{state.get('question', '')}'")
    return {
        "final_answer": REFUSAL_MESSAGE,
        "confidence": "low",
        "verification_status": "fail",
    }
