"""
Reranker Client

Async HTTP client for the local reranker sidecar (BAAI/bge-reranker-v2-m3).
The sidecar exposes POST /rerank and returns candidates sorted by cross-encoder score.
"""

import logging
from typing import Any, Dict, List

import httpx

import oracle.config as cfg

logger = logging.getLogger(__name__)


class RerankerClient:
    """Async client for the local BGE reranker sidecar."""

    def __init__(self):
        self._http = httpx.AsyncClient(
            base_url=cfg.RERANKER_URL,
            timeout=httpx.Timeout(cfg.RERANKER_TIMEOUT),
        )

    async def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        """
        Re-rank candidate chunks using the cross-encoder.

        Args:
            query: The user's search query.
            candidates: List of dicts, each must have at least `id` and `text` keys.
            top_k: Number of top results to return.

        Returns:
            Top-k candidates sorted by reranker score (highest first),
            with `rerank_score` added to each dict.

        Raises:
            RuntimeError: If the reranker sidecar is unavailable.
        """
        if not candidates:
            return []

        payload = {
            "query": query,
            "candidates": [
                {"id": str(c.get("id", "")), "text": c.get("text", "")}
                for c in candidates
            ],
            "top_k": top_k,
        }

        try:
            response = await self._http.post("/rerank", json=payload)
            response.raise_for_status()
            data = response.json()
            ranked_ids_scores = {
                item["id"]: item["score"] for item in data.get("ranked", [])
            }

            # Merge scores back into original candidate dicts
            enriched = []
            for c in candidates:
                cid = str(c.get("id", ""))
                if cid in ranked_ids_scores:
                    enriched.append({**c, "rerank_score": ranked_ids_scores[cid]})

            # Sort by rerank score descending, return top_k
            enriched.sort(key=lambda x: x.get("rerank_score", 0.0), reverse=True)
            return enriched[:top_k]

        except httpx.ConnectError:
            raise RuntimeError(
                f"Reranker sidecar unavailable at {cfg.RERANKER_URL}. "
                "Ensure the reranker service is running."
            )
        except Exception as e:
            logger.error(f"Reranker request failed: {e}")
            # Graceful degradation: return candidates sorted by hybrid search score
            logger.warning("Falling back to hybrid search order (no reranking)")
            return candidates[:top_k]

    async def health(self) -> bool:
        """Check if reranker sidecar is reachable."""
        try:
            r = await self._http.get("/health", timeout=3.0)
            return r.status_code == 200
        except Exception:
            return False

    async def aclose(self):
        await self._http.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.aclose()
