# app/main.py

from contextlib import asynccontextmanager
from time import time

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, Request
from prometheus_fastapi_instrumentator import Instrumentator
from slowapi.errors import RateLimitExceeded
from slowapi import _rate_limit_exceeded_handler

from src.agents.graph import financial_agent
from src.common.logger import logger_manager
from app.cache import RedisCache
from app.limiter import limiter
from app.routes import router

logger = logger_manager.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Startup: initializing cache and agent")

    from src.agents.nodes.retriever import _retriever, _reranker

    logger.info("Pre-loading HybridRetriever...")
    _retriever()
    logger.info("Pre-loading reranker...")
    _reranker()

    app.state.cache = RedisCache()
    app.state.agent = financial_agent
    app.state.started_at = time()

    logger.info("Startup complete")
    yield


app = FastAPI(
    title="Financial RAG Agent",
    description="REST API for querying 2025 Canadian bank annual reports.",
    version="0.1.0",
    lifespan=lifespan,
)

# Rate limiting
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Insert the routes in our app
app.include_router(router)

# Prometheus instrumentation
Instrumentator().instrument(app).expose(
    app,
    endpoint="/metrics",
    include_in_schema=False,
)


# Health endpoint
@app.get("/health")
def health(request: Request):
    """Liveness and readiness probe."""
    cache = request.app.state.cache
    return {
        "status": "ok",
        "agent_ready": request.app.state.agent is not None,
        "cache_ready": cache.ping(),
        "uptime_seconds": round(time() - request.app.state.started_at, 2),
    }
