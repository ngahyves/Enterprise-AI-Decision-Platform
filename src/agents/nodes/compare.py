# src/agents/nodes/compare.py
from src.agents.state import AgentState
from src.agents.schemas import ComparisonOutput
from src.agents.llm import get_complex_analyst
from src.common.logger import logger_manager
import time

logger = logger_manager.get_logger(__name__)


def _format_facts(extracted_facts: dict) -> str:
    """Formats the facts extracted by bank into a text block that is readable by the LLM."""
    lines = []
    for bank, facts in extracted_facts.items():
        lines.append(f"Bank: {bank}")
        if not facts:
            lines.append("  - No facts extracted.")
            continue
        for f in facts:
            lines.append(
                f'  - Metric: {f.get("metric")} | Value: {f.get("value")} | Quote: "{f.get("quote")}"'
            )
        lines.append("")
    return "\n".join(lines)


def compare_facts(state: AgentState) -> dict:
    """Compare the data extracted from different banks: common themes, differences, and rankings."""
    task_type = state["task_type"]
    extracted_facts = state.get("extracted_facts", {})
    question = state["question"]

    if task_type == "single" or len(extracted_facts) <= 1:
        logger.info("Skipping comparison (single bank or no cross-bank data).")
        return {"comparison": None}

    facts_text = _format_facts(extracted_facts)

    prompt = (
        f"User Original Query: '{question}'\n\n"
        f"Extracted Facts per Bank:\n{facts_text}\n\n"
        "Task: Compare these findings across the banks. Identify common strategic themes or risk priorities, "
        "highlight key numerical or structural differences, provide a comparative ranking if the question implies it, "
        "and write a concise analytical summary paragraph."
    )

    try:
        structured_llm = get_complex_analyst().with_structured_output(
            ComparisonOutput, method="json_schema"
        )
        time.sleep(3)
        result: ComparisonOutput = structured_llm.invoke(prompt)
        logger.info(f"Comparison done across {len(extracted_facts)} banks")
        return {"comparison": result.model_dump()}
    except Exception as e:
        logger.error(f"Comparison failed: {e}")
        return {"comparison": None, "errors": [f"Comparison failed: {e}"]}
