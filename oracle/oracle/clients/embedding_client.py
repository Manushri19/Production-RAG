"""
Embedding Client (Query-Time)

Thin async wrapper around Ollama for generating query embeddings.
At query time we only need embed_text — storage is Meridian's job.
"""

import logging
from typing import List

import httpx

import oracle.config as cfg

logger = logging.getLogger(__name__)


class EmbeddingClient:
    """Async Ollama client for query embedding (read-only, no storage)."""

    def __init__(self):
        self._http = httpx.AsyncClient(
            timeout=httpx.Timeout(cfg.EMBEDDING_TIMEOUT)
        )

    async def embed_text(self, text: str) -> List[float]:
        """
        Embed a single query string.

        Returns:
            Dense embedding vector (list of floats), or empty list on failure.
        """
        try:
            response = await self._http.post(
                f"{cfg.OLLAMA_URL}/api/embed",
                json={
                    "model": cfg.EMBEDDING_MODEL,
                    "input": text,
                    "options": {"num_ctx": 4096},
                },
            )
            response.raise_for_status()
            data = response.json()
            embeddings = data.get("embeddings", [])
            if embeddings:
                emb = embeddings[0]
                # Truncate to configured dimensions if needed
                if len(emb) > cfg.EMBEDDING_DIMENSIONS:
                    emb = emb[: cfg.EMBEDDING_DIMENSIONS]
                return emb
            logger.error("Ollama returned empty embeddings for query")
            return []
        except Exception as e:
            logger.error(f"Embedding request failed: {e}")
            return []

    async def aclose(self):
        await self._http.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.aclose()
