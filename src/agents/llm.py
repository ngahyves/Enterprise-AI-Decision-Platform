# src/agents/llm.py
from functools import lru_cache

from langchain_groq import ChatGroq

from src.common.config_manager import env_settings, config
from src.common.logger import logger_manager

logger = logger_manager.get_logger(__name__)


@lru_cache(maxsize=1)
def get_fast_agent():
    """Small Groq model for high-volume nodes (router, planner, extract, verifier)."""
    cfg = config["llm"]["fast_agent"]
    logger.info(f"Building fast_agent: {cfg['model']}")
    return ChatGroq(
        model=cfg["model"],
        temperature=cfg.get("temperature", 0.0),
        api_key=env_settings.groq_api_key,
    )


@lru_cache(maxsize=1)
def get_complex_analyst():
    """Larger Groq model for low-volume, high-reasoning nodes (compare, synthesize)."""
    cfg = config["llm"]["complex_analyst"]
    logger.info(f"Building complex_analyst: {cfg['model']}")
    return ChatGroq(
        model=cfg["model"],
        temperature=cfg.get("temperature", 0.1),
        api_key=env_settings.groq_api_key,
    )
