"""
Hybrid Search

Executes Qdrant-native hybrid search using:
- Dense HNSW (semantic / conceptual)
- Sparse BM25 (keyword / exact terminology)
- RRF (Reciprocal Rank Fusion) to merge both result sets

This is a single Qdrant round-trip using the `prefetch` + `Query(fusion=Fusion.RRF)` API.
"""

import logging
from typing import Any, Dict, List, Optional

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Filter,
    FieldCondition,
    MatchAny,
    Prefetch,
    FusionQuery,
    Fusion,
    SparseVector,
    SearchParams,
)
from fastembed import SparseTextEmbedding

import oracle.config as cfg

logger = logging.getLogger(__name__)

# BM25 model is CPU-bound and model-init is slow — load once at module level
_bm25_model: Optional[SparseTextEmbedding] = None


def _get_bm25_model() -> SparseTextEmbedding:
    global _bm25_model
    if _bm25_model is None:
        logger.info("Loading BM25 model (Qdrant/bm25) — first call only...")
        _bm25_model = SparseTextEmbedding(model_name="Qdrant/bm25")
        logger.info("BM25 model loaded.")
    return _bm25_model


def _compute_sparse_vector(text: str) -> SparseVector:
    """Convert query text to a BM25 sparse vector."""
    model = _get_bm25_model()
    results = list(model.embed([text]))
    sv = results[0]
    return SparseVector(
        indices=sv.indices.tolist(),
        values=sv.values.tolist(),
    )


async def hybrid_search(
    qdrant: AsyncQdrantClient,
    query_vector: List[float],
    query_text: str,
    top_k: int,
    document_ids: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Run hybrid search with Qdrant-native RRF fusion.

    Args:
        qdrant: Async Qdrant client (caller manages lifecycle).
        query_vector: Dense embedding of the query.
        query_text: Raw query string for BM25 sparse vector computation.
        top_k: Number of fused results to return.
        document_ids: Optional list of document IDs to restrict search scope.

    Returns:
        List of result dicts with text, metadata and hybrid scores.
    """
    # Build optional payload filter for document scoping
    qdrant_filter = None
    if document_ids:
        qdrant_filter = Filter(
            must=[
                FieldCondition(
                    key="document_id",
                    match=MatchAny(any=document_ids),
                )
            ]
        )

    # Compute BM25 sparse vector for the query
    sparse_query = _compute_sparse_vector(query_text)

    try:
        results = await qdrant.query_points(
            collection_name=cfg.QDRANT_COLLECTION,
            prefetch=[
                # Dense branch: HNSW semantic search
                Prefetch(
                    query=query_vector,
                    using=cfg.DENSE_VECTOR_NAME,
                    limit=top_k,
                    filter=qdrant_filter,
                    params=SearchParams(hnsw_ef=cfg.HNSW_EF, exact=False),
                ),
                # Sparse branch: BM25 keyword search
                Prefetch(
                    query=sparse_query,
                    using=cfg.SPARSE_VECTOR_NAME,
                    limit=top_k,
                    filter=qdrant_filter,
                ),
            ],
            # RRF fusion merges both ranked lists mathematically
            query=FusionQuery(fusion=Fusion.RRF),
            limit=top_k,
            with_payload=True,
            with_vectors=False,
        )

        hits = results.points
        logger.info(
            f"Hybrid search returned {len(hits)} fused candidates "
            f"(top_k={top_k}, doc_filter={document_ids})"
        )

        return [
            {
                "id": str(hit.id),
                "score": hit.score,
                "text": hit.payload.get("text", ""),
                "document_id": hit.payload.get("document_id", ""),
                "page_no": hit.payload.get("page_no", 0),
                "order": hit.payload.get("order", 0),
                "chunk_type": hit.payload.get("chunk_type", "text"),
                "headings": hit.payload.get("headings", []),
                "figure_type": hit.payload.get("figure_type"),
                "is_complex_table": hit.payload.get("is_complex_table"),
            }
            for hit in hits
        ]

    except Exception as e:
        logger.error(f"Hybrid search failed: {e}")
        raise RuntimeError(f"Qdrant hybrid search failed: {e}") from e
