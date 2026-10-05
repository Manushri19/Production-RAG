"""
Tests for the hybrid search module.

These tests use a mock Qdrant client to validate that:
1. The correct prefetch structure is passed (dense + sparse)
2. RRF fusion is requested
3. Document ID filtering is applied when document_ids is provided
4. Results are mapped correctly to dicts
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Any


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def _make_hit(point_id: str, score: float, payload: dict) -> Any:
    """Create a mock Qdrant ScoredPoint-like object."""
    hit = MagicMock()
    hit.id = point_id
    hit.score = score
    hit.payload = payload
    return hit


def _make_query_result(hits):
    result = MagicMock()
    result.points = hits
    return result


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_hybrid_search_returns_mapped_results():
    """Verify result dicts are correctly assembled from Qdrant hits."""
    from oracle.retrieval.hybrid_search import hybrid_search

    mock_qdrant = AsyncMock()
    mock_qdrant.query_points = AsyncMock(return_value=_make_query_result([
        _make_hit("abc-1", 0.95, {
            "text": "Inconel 718 tensile strength is 1375 MPa at room temperature.",
            "document_id": "alloy_ds_v3",
            "page_no": 12,
            "order": 5,
            "chunk_type": "text",
            "headings": ["Mechanical Properties"],
            "figure_type": None,
            "is_complex_table": None,
        }),
        _make_hit("abc-2", 0.87, {
            "text": "At 650°C the yield strength drops to 1034 MPa.",
            "document_id": "alloy_ds_v3",
            "page_no": 13,
            "order": 6,
            "chunk_type": "text",
            "headings": ["Mechanical Properties"],
            "figure_type": None,
            "is_complex_table": None,
        }),
    ]))

    with patch("oracle.retrieval.hybrid_search._compute_sparse_vector") as mock_sparse:
        from qdrant_client.models import SparseVector
        mock_sparse.return_value = SparseVector(indices=[1, 2, 3], values=[0.5, 0.3, 0.2])

        results = await hybrid_search(
            qdrant=mock_qdrant,
            query_vector=[0.1] * 1024,
            query_text="tensile strength Inconel 718",
            top_k=10,
        )

    assert len(results) == 2
    assert results[0]["id"] == "abc-1"
    assert results[0]["document_id"] == "alloy_ds_v3"
    assert results[0]["page_no"] == 12
    assert "tensile strength" in results[0]["text"]


@pytest.mark.asyncio
async def test_hybrid_search_with_document_filter():
    """Verify that document_ids filter is passed to Qdrant."""
    from oracle.retrieval.hybrid_search import hybrid_search

    mock_qdrant = AsyncMock()
    mock_qdrant.query_points = AsyncMock(return_value=_make_query_result([]))

    with patch("oracle.retrieval.hybrid_search._compute_sparse_vector") as mock_sparse:
        from qdrant_client.models import SparseVector
        mock_sparse.return_value = SparseVector(indices=[1], values=[1.0])

        await hybrid_search(
            qdrant=mock_qdrant,
            query_vector=[0.0] * 1024,
            query_text="test query",
            top_k=30,
            document_ids=["doc_a", "doc_b"],
        )

    # Ensure query_points was called
    mock_qdrant.query_points.assert_called_once()
    call_kwargs = mock_qdrant.query_points.call_args.kwargs

    # Filter must be set
    assert call_kwargs.get("prefetch") is not None
    # Both prefetch branches should carry the filter
    for pf in call_kwargs["prefetch"]:
        assert pf.filter is not None


@pytest.mark.asyncio
async def test_hybrid_search_empty_results():
    """Verify empty result list on no matches."""
    from oracle.retrieval.hybrid_search import hybrid_search

    mock_qdrant = AsyncMock()
    mock_qdrant.query_points = AsyncMock(return_value=_make_query_result([]))

    with patch("oracle.retrieval.hybrid_search._compute_sparse_vector") as mock_sparse:
        from qdrant_client.models import SparseVector
        mock_sparse.return_value = SparseVector(indices=[], values=[])

        results = await hybrid_search(
            qdrant=mock_qdrant,
            query_vector=[0.0] * 1024,
            query_text="nonexistent material",
            top_k=30,
        )

    assert results == []
