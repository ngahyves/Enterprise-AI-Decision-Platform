# app/cache.py
"""Redis-backed response cache for the RAG agent."""

import hashlib
import json
import os
from typing import Any, Optional
import redis
from src.common.config_manager import config
from src.common.logger import logger_manager
from dotenv import load_dotenv

load_dotenv()

logger = logger_manager.get_logger(__name__)


class RedisCache:
    """Simple key-value cache for query responses, backed by Redis.

    Keys are SHA256 hashes of the normalized question, so paraphrases
    with different casing or trailing whitespace hit the same entry.
    """

    def __init__(self):
        cfg = config["cache"]
        self.host = os.getenv("REDIS_HOST") or cfg.get("redis_host", "localhost")
        self.port = int(os.getenv("REDIS_PORT") or cfg.get("redis_port", 6379))
        self.ttl = cfg.get("ttl_seconds", 3600)
        self.prefix = cfg.get("key_prefix", "rag:q:")

        self.client = redis.Redis(
            host=self.host,
            port=self.port,
            decode_responses=True,
            socket_connect_timeout=2,
        )
        self._verify_connection()

    # ---------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------
    def _verify_connection(self):
        try:
            self.client.ping()
            logger.info(f"Redis cache connected at {self.host}:{self.port}")
        except redis.ConnectionError as e:
            logger.error(f"Redis connection failed: {e}")
            raise

    def _make_key(self, question: str) -> str:
        normalized = question.lower().strip()
        digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        return f"{self.prefix}{digest}"

    # ---------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------
    def get(self, question: str) -> Optional[dict[str, Any]]:
        """Return the cached response for a question, or None."""
        try:
            raw = self.client.get(self._make_key(question))
        except redis.RedisError as e:
            logger.warning(f"Cache get failed: {e}")
            return None

        if raw is None:
            return None

        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("Corrupted cache entry, ignoring")
            return None

    def set(self, question: str, response: dict[str, Any]) -> None:
        """Store a response, respecting TTL."""
        try:
            self.client.setex(
                self._make_key(question),
                self.ttl,
                json.dumps(response),
            )
        except redis.RedisError as e:
            logger.warning(f"Cache set failed: {e}")

    def clear(self) -> int:
        """Delete all cache entries with the configured prefix."""
        try:
            keys = self.client.keys(f"{self.prefix}*")
            if keys:
                return self.client.delete(*keys)
            return 0
        except redis.RedisError as e:
            logger.warning(f"Cache clear failed: {e}")
            return 0

    def stats(self) -> dict[str, Any]:
        """Return basic cache statistics."""
        try:
            keys = self.client.keys(f"{self.prefix}*")
            return {
                "size": len(keys),
                "ttl_seconds": self.ttl,
                "connected": True,
            }
        except redis.RedisError as e:
            return {
                "size": 0,
                "ttl_seconds": self.ttl,
                "connected": False,
                "error": str(e),
            }

    def ping(self) -> bool:
        """Return True if Redis responds."""
        try:
            return self.client.ping()
        except redis.RedisError:
            return False
