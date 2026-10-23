# app/schemas/query_schema.py
"""Pydantic schemas for the query API endpoints."""

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------
# Enums — For all strict parameters
# ---------------------------------------------------------------
class ConfidenceLevel(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class TaskType(str, Enum):
    SINGLE = "single"
    MULTI = "multi"
    COMPARE = "compare"
    REFUSE = "refuse"


class BankName(str, Enum):
    RBC = "RBC"
    TD = "TD"
    BMO = "BMO"
    SCOTIA = "SCOTIA"


# ---------------------------------------------------------------
# Source citation — chunk used to build the answer
# ---------------------------------------------------------------
class Source(BaseModel):
    """A single source chunk backing the answer."""

    chunk_id: str = Field(..., description="Unique chunk identifier")
    bank: str = Field(..., description="Bank the chunk belongs to")
    section: Optional[str] = Field(None, description="Report section title")
    page: Optional[int] = Field(None, description="Page number in the source PDF")


# ---------------------------------------------------------------
# Request
# ---------------------------------------------------------------
class QueryRequest(BaseModel):
    """Incoming query from an API client."""

    question: str = Field(
        ...,
        min_length=5,
        max_length=500,
        description="Natural language question about the 2025 annual reports.",
        examples=["What was TD net income in 2025?"],
    )
    use_cache: bool = Field(
        True,
        description="If true, return cached response when available.",
    )


# ---------------------------------------------------------------
# Response
# ---------------------------------------------------------------
class QueryResponse(BaseModel):
    """Answer returned to the API client."""

    question: str = Field(..., description="Echo of the original question")
    answer: str = Field(..., description="Natural language answer")

    confidence: ConfidenceLevel
    task_type: TaskType
    banks: list[BankName] = Field(
        default_factory=list,
        description="Banks referenced in the question.",
    )

    sources: list[Source] = Field(
        default_factory=list,
        description="Source chunks used to build the answer.",
    )
    # Operational metadata
    cached: bool = Field(False, description="True if served from cache")
    latency_ms: float = Field(..., description="End-to-end latency in ms")
    request_id: str = Field(..., description="Unique request identifier for tracing")


# ---------------------------------------------------------------
# Health / ops
# ---------------------------------------------------------------
class HealthResponse(BaseModel):
    """Health check payload."""

    status: str = Field(..., description="'ok' or 'degraded'")
    agent_ready: bool
    cache_ready: bool
    uptime_seconds: float


class ErrorResponse(BaseModel):
    """Standard error envelope."""

    detail: str
    error_code: str
    request_id: str
