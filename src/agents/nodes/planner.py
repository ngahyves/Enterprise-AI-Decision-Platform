# src/agents/nodes/planner.py
from langchain_groq import ChatGroq
from src.agents.state import AgentState
from src.agents.schemas import PlannerOutput
from src.common.config_manager import env_settings, config
from src.common.logger import logger_manager

logger = logger_manager.get_logger(__name__)

_FAST_AGENT = config["llm"]["fast_agent"]


def _fallback_objective(question: str, bank: str) -> str:
    return f"For {bank}, {question[0].lower() + question[1:]}"


def _align_sub_objectives(objectives: list[str], banks: list[str], question: str) -> list[str]:
    """Guarantee one objective per bank, in the same order as `banks`."""
    aligned = [o.strip() for o in objectives if o.strip()][: len(banks)]
    while len(aligned) < len(banks):
        aligned.append(_fallback_objective(question, banks[len(aligned)]))
    return aligned


def plan_tasks(state: AgentState) -> dict:
    """
    Decomposes a multi-bank or comparison query into one self-contained
    sub-objective per bank. Runs only for task_type in {'multi', 'compare'}.
    """
    question = state["question"]
    banks = state.get("banks", [])
    task_type = state.get("task_type", "single")

    if task_type == "single":
        raise ValueError("planner_node should not be invoked for single-bank tasks")

    if not banks:
        logger.warning("Planner called with no banks — returning empty plan")
        return {"sub_objectives": []}

    logger.info(f"Planner decomposing '{question}' across {len(banks)} banks: {banks}")

    prompt = (
        f"User question: '{question}'\n"
        f"Target banks: {banks}\n\n"
        "For each bank, write ONE short financial query (max 12 words) to find the answer "
        "in that bank's annual report. Use standard financial terminology. "
        "SPELL OUT ALL ACRONYMS (e.g. 'PCL' -> 'provision for credit losses', "
        "'NII' -> 'net interest income', 'ROE' -> 'return on equity'). "
        "No explanations, no synonyms list, no meta-instructions.\n\n"
        f"Return exactly {len(banks)} queries, one per bank, in this order: {banks}."
    )

    try:
        llm = ChatGroq(
            model=_FAST_AGENT["model"],
            temperature=_FAST_AGENT.get("temperature", 0.0),
            api_key=env_settings.groq_api_key,
        )
        structured = llm.with_structured_output(PlannerOutput, method="json_schema")
        result: PlannerOutput = structured.invoke(prompt)
        objectives = result.sub_objectives or []
    except Exception as e:
        logger.error(f"Planner LLM failed: {e}")
        objectives = [_fallback_objective(question, b) for b in banks]

    objectives = _align_sub_objectives(objectives, banks, question)

    logger.info(f"Planner produced {len(objectives)} sub-objectives")
    for bank, obj in zip(banks, objectives):
        logger.debug(f"  [{bank}] {obj}")

    print(f">>> SUB-OBJECTIVES: {objectives}")
    return {"sub_objectives": objectives}
