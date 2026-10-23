# app/routes.py
"""HTTP endpoints for the RAG agent."""

import uuid
from time import time

from fastapi import APIRouter, Depends, Request, HTTPException

from app.auth import verify_api_key
from app.limiter import limiter
from app.security.guardrails import validate_user_input
from app.schemas.query_schema import (
    ConfidenceLevel,
    QueryRequest,
    QueryResponse,
    Source,
    TaskType,
)
from src.common.logger import logger_manager

logger = logger_manager.get_logger(__name__)

router = APIRouter()


@router.post("/v1/ask", response_model=QueryResponse)
@limiter.limit("10/minute")
def ask_agent(payload: QueryRequest, request: Request, _: str = Depends(verify_api_key)):
    """Run the financial agent on a query and return the structured response."""
    start_time = time()
    request_id = f"req_{uuid.uuid4().hex[:12]}"

    agent = request.app.state.agent
    cache = request.app.state.cache

    # 0. Validate and sanitize input (fail fast on prompt injection)
    try:
        safe_question = validate_user_input(payload.question)
    except HTTPException as exc:
        logger.warning(f"[{request_id}] Input rejected: {exc.detail}")
        raise exc

    # 1. Check cache first if requested
    if payload.use_cache:
        cached_data = cache.get(safe_question)
        if cached_data:
            return QueryResponse(
                **cached_data,
                cached=True,
                latency_ms=round((time() - start_time) * 1000, 2),
                request_id=request_id,
            )

    # 2. Execute the agent graph
    try:
        final_state = agent.invoke({"question": safe_question})
    except Exception as e:
        logger.error(f"Agent execution failed for {request_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Agent error: {str(e)}")

    # 3. Extract sources from extracted facts safely
    sources = []
    seen_chunks = set()
    for bank, facts in final_state.get("extracted_facts", {}).items():
        for fact in facts:
            cid = fact.get("source_chunk_id")
            if cid and cid not in seen_chunks:
                seen_chunks.add(cid)
                sources.append(
                    Source(
                        chunk_id=cid,
                        bank=fact.get("bank", bank),
                        section=None,
                        page=fact.get("page_number"),
                    )
                )

    # 4. Build the response payload
    response_dict = {
        "question": safe_question,
        "answer": final_state.get("final_answer", "No answer generated."),
        "confidence": final_state.get("confidence", ConfidenceLevel.LOW.value),
        "task_type": final_state.get("task_type", TaskType.SINGLE.value),
        "banks": final_state.get("banks", []),
        "sources": [s.model_dump() for s in sources],
    }

    # 5. Save to Redis cache if confidence is acceptable
    if payload.use_cache and response_dict["confidence"] != ConfidenceLevel.LOW.value:
        cache.set(safe_question, response_dict)

    return QueryResponse(
        **response_dict,
        cached=False,
        latency_ms=round((time() - start_time) * 1000, 2),
        request_id=request_id,
    )


@router.get("/v1/cache/stats")
def get_cache_stats(request: Request, _: str = Depends(verify_api_key)):
    """Return Redis cache metrics."""
    return request.app.state.cache.stats()


@router.delete("/v1/cache")
def clear_cache(request: Request, _: str = Depends(verify_api_key)):
    """Flush all RAG cache entries from Redis."""
    deleted_count = request.app.state.cache.clear()
    return {"deleted_entries": deleted_count}
