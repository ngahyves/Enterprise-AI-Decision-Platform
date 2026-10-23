# src/agents/nodes/router.py
import re
from langchain_groq import ChatGroq
from src.agents.state import AgentState
from src.agents.schemas import RouterOutput
from src.common.config_manager import env_settings, config
from src.common.logger import logger_manager

logger = logger_manager.get_logger(__name__)

GROQ_MODEL = config["llm"]["fast_agent"]  # "openai/gpt-oss-20b"

BANK_KEYWORDS = {
    "RBC": ["RBC", "ROYAL BANK", "ROYAL BANK OF CANADA"],
    "TD": ["TD", "TORONTO-DOMINION", "TD BANK"],
    "BMO": ["BMO", "BANK OF MONTREAL"],
    "SCOTIA": ["SCOTIA", "SCOTIABANK", "BANK OF NOVA SCOTIA"],
}

COMPARE_PATTERNS = [
    r"\bcompare\b",
    r"\bversus\b",
    r"\bvs\b",
    r"\bdifference\b",
    r"\bhighest\b",
    r"\blowest\b",
    r"\bbetween\b",
]


def _canonicalize(name: str) -> str | None:
    upper = name.upper().strip()
    for canonical, keywords in BANK_KEYWORDS.items():
        for kw in keywords:
            if re.search(r"\b" + re.escape(kw) + r"\b", upper):
                return canonical
    return None


def route_question(state: AgentState) -> dict:
    question = state["question"]
    q_upper = question.upper()
    q_lower = question.lower()
    logger.info(f"Router processing: '{question}'")

    # 1. Rule-based bank detection
    detected = [
        bank
        for bank, kws in BANK_KEYWORDS.items()
        if any(re.search(r"\b" + re.escape(kw) + r"\b", q_upper) for kw in kws)
    ]
    is_comparison = any(re.search(p, q_lower) for p in COMPARE_PATTERNS)

    if len(detected) == 1 and not is_comparison:
        task_type = "single"
    elif len(detected) > 1:
        task_type = "compare" if is_comparison else "multi"
    elif is_comparison:
        task_type = "compare"
    else:
        task_type = "single"

    method = "rules"

    # 2. LLM fallback — only when no bank was found
    if not detected:
        logger.info("No bank detected by rules — falling back to LLM")
        try:
            llm = ChatGroq(model=GROQ_MODEL, temperature=0.0, api_key=env_settings.groq_api_key)
            structured = llm.with_structured_output(RouterOutput, method="json_schema")
            result: RouterOutput = structured.invoke(
                "Classify this query about Canadian bank annual reports: " + question
            )
            detected = list({b for name in result.banks if (b := _canonicalize(name))})
            task_type = result.task_type
            method = "llm"
        except Exception as e:
            logger.error(f"LLM router failed: {e}")
            detected = list(BANK_KEYWORDS.keys())
            task_type = "compare"
            method = "llm_error_fallback"

    logger.info(f"Router -> method={method} | task={task_type} | banks={detected}")
    return {"task_type": task_type, "banks": detected, "routing_method": method}
