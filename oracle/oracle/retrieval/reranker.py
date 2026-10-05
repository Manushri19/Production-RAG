"""
Reranking Layer

Orchestrates the cross-encoder reranking step:
1. Takes raw hybrid search candidates (top-K)
2. Sends them to the reranker sidecar (BAAI/bge-reranker-v2-m3)
3. Returns the top-N most relevant chunks for generation
"""

import logging
from typing import Any, Dict, List

from oracle.clients.reranker_client import RerankerClient

logger = logging.getLogger(__name__)


async def rerank_candidates(
    query: str,
    candidates: List[Dict[str, Any]],
    top_k: int,
    reranker: RerankerClient,
) -> List[Dict[str, Any]]:
    """
    Rerank candidate chunks using the cross-encoder.

    Args:
        query: The user's query string.
        candidates: Hybrid search results (dicts with `id`, `text`, metadata).
        top_k: Number of top chunks to return after reranking.
        reranker: Shared RerankerClient instance.

    Returns:
        Top-k chunks sorted by cross-encoder relevance score (highest first).
    """
    if not candidates:
        logger.warning("Reranker received empty candidate list")
        return []

    logger.info(
        f"Reranking {len(candidates)} candidates → selecting top {top_k}"
    )

    try:
        ranked = await reranker.rerank(
            query=query,
            candidates=candidates,
            top_k=top_k,
        )
        logger.info(
            f"Reranking complete. Top chunk: doc={ranked[0].get('document_id', '?')} "
            f"page={ranked[0].get('page_no', '?')} "
            f"rerank_score={ranked[0].get('rerank_score', 0.0):.4f}"
            if ranked else "Reranking produced no results"
        )
        return ranked

    except RuntimeError as e:
        # Reranker unavailable — fall through with hybrid order
        logger.error(f"Reranker error: {e}")
        return candidates[:top_k]
