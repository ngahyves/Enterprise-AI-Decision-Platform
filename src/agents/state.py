# src/agents/state.py
from typing import TypedDict, Literal, Any, Annotated, Optional
from operator import add


# Implementing the reducer
def merge_dict_of_lists(
    left: dict[str, list[dict]] | None,
    right: dict[str, list[dict]] | None,
) -> dict[str, list[dict]]:
    """Merge parallel branches writing to the same dict-of-lists key."""
    merged = dict(left or {})
    for key, values in (right or {}).items():
        merged.setdefault(key, [])
        merged[key].extend(values)
    return merged


class AgentState(TypedDict, total=False):
    """Complete execution state for the Enterprise Financial Decision Agent."""

    # Input
    question: str

    # Router
    task_type: Literal["single", "multi", "compare"]
    banks: list[str]
    routing_method: Literal["rules", "llm"]

    # Planner
    sub_objectives: list[str]

    # Retriever (fan-out with reducer)
    chunks_by_bank: Annotated[dict[str, list[dict[str, Any]]], merge_dict_of_lists]

    # Extract (fan-out capable with reducer)
    extracted_facts: Annotated[dict[str, list[dict[str, Any]]], merge_dict_of_lists]

    # Compare
    comparison: Optional[dict[str, Any]]

    # Verifier
    claims: list[dict[str, Any]]
    verification_iteration: int
    verification_status: Literal["ok", "retry", "fail"]

    # Synthesizer
    final_answer: str
    confidence: Literal["high", "medium", "low"]

    # Debug (accumulate, don't overwrite)
    errors: Annotated[list[str], add]
    trace: Annotated[list[dict[str, Any]], add]
