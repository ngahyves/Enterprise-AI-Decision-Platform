# src/agents/graph.py
from langgraph.graph import StateGraph, START, END
from src.agents.state import AgentState
from src.agents.nodes.router import route_question
from src.agents.nodes.planner import plan_tasks
from src.agents.nodes.retriever import retrieve_documents
from src.agents.nodes.extract import extract_facts
from src.agents.nodes.compare import compare_facts
from src.agents.nodes.verifier import verify_facts
from src.agents.nodes.synthesizer import synthesize_answer
from src.agents.nodes.refuse import refuse_query
from src.common.logger import logger_manager

logger = logger_manager.get_logger(__name__)


def _decide_after_router(state: AgentState) -> str:
    """Route to refuse, planner, or retriever based on task type."""
    if not state.get("banks"):
        return "refuse"
    if state.get("task_type") in ("multi", "compare"):
        return "planner"
    return "retriever"


def _decide_after_extract(state: AgentState) -> str:
    """Route to compare for multi-bank tasks, otherwise straight to verifier."""
    return "compare" if state.get("task_type") in ("multi", "compare") else "verifier"


def _decide_after_verifier(state: AgentState) -> str:
    status = state.get("verification_status")
    iteration = state.get("verification_iteration", 0)

    # Don't retry if extraction produced nothing — the issue is upstream, not verification
    total_facts = sum(len(f) for f in state.get("extracted_facts", {}).values())
    if total_facts == 0:
        return "synthesizer"

    if status == "retry" and iteration <= 2:
        logger.warning(f"Verifier requested retry (iteration {iteration})")
        return "extract"
    return "synthesizer"


def build_financial_agent_graph():
    """Build and compile the LangGraph state machine for the financial agent."""
    workflow = StateGraph(AgentState)

    workflow.add_node("router", route_question)
    workflow.add_node("planner", plan_tasks)
    workflow.add_node("retriever", retrieve_documents)
    workflow.add_node("extract", extract_facts)
    workflow.add_node("compare", compare_facts)
    workflow.add_node("verifier", verify_facts)
    workflow.add_node("synthesizer", synthesize_answer)
    workflow.add_node("refuse", refuse_query)

    workflow.add_edge(START, "router")
    workflow.add_conditional_edges(
        "router",
        _decide_after_router,
        {
            "refuse": "refuse",
            "planner": "planner",
            "retriever": "retriever",
        },
    )

    workflow.add_edge("planner", "retriever")
    workflow.add_edge("retriever", "extract")
    workflow.add_conditional_edges(
        "extract",
        _decide_after_extract,
        {
            "compare": "compare",
            "verifier": "verifier",
        },
    )

    workflow.add_edge("compare", "verifier")
    workflow.add_conditional_edges(
        "verifier",
        _decide_after_verifier,
        {
            "extract": "extract",
            "synthesizer": "synthesizer",
        },
    )

    workflow.add_edge("synthesizer", END)
    workflow.add_edge("refuse", END)

    logger.info("Financial agent graph compiled")
    return workflow.compile()


# Module-level compiled graph, ready to invoke
financial_agent = build_financial_agent_graph()
