# src/agents/schemas.py
from typing import Literal, Optional
from pydantic import BaseModel, Field


class RouterOutput(BaseModel):
    """Schema for the router node decision."""

    task_type: Literal["single", "multi", "compare"] = Field(
        description="Classify if the question targets a single bank, multiple independent banks, or requires a direct comparison."
    )
    banks: list[str] = Field(
        description="Target banks identified in the query (uppercase: 'RBC', 'TD', 'BMO', 'SCOTIA')."
    )
    reasoning: str = Field(
        description="Explanation of the routing decision. Mention which keywords or bank names triggered the classification."
    )


class PlannerOutput(BaseModel):
    """Schema for the planner node output."""

    sub_objectives: list[str] = Field(
        description="One self-contained sub-objective per target bank. Must be independently answerable."
    )


class ExtractedFact(BaseModel):
    """A single fact extracted from a document chunk."""

    metric: str = Field(description="Name of the metric, risk factor, or strategic priority.")
    value: str = Field(description="Extracted value, figure, or descriptive statement.")
    quote: str = Field(
        description="Verbatim sentence or table row from the chunk supporting this fact. Must be an exact substring."
    )
    source_chunk_id: str = Field(description="chunk_id of the source.")
    bank: str = Field(description="Bank associated with this fact.")
    page_number: Optional[int] = Field(default=None, description="Page number in the source PDF.")


class ExtractionOutput(BaseModel):
    """Schema for the extraction node output per bank."""

    bank: str
    facts: list[ExtractedFact]


class ComparisonOutput(BaseModel):
    """Schema for the comparison node output."""

    common_themes: list[str] = Field(description="Shared risk priorities or themes across banks.")
    key_differences: list[str] = Field(description="Major divergences or contrasting figures.")
    ranking: list[str] = Field(
        description="Banks ordered from highest to lowest on the comparison metric. Empty if not applicable."
    )
    summary: str = Field(description="One-paragraph analytical summary.")


class ClaimVerification(BaseModel):
    """Verification result for a single claim."""

    claim: str
    bank: str
    supported: bool
    evidence_chunk_id: Optional[str] = None
    reasoning: str


class VerifierOutput(BaseModel):
    """Aggregate verifier output."""

    claims: list[ClaimVerification]
    status: Literal["ok", "retry", "fail"] = Field(
        description="'ok' if all claims pass, 'retry' if minor corrections are needed, 'fail' if hallucinations are critical."
    )
