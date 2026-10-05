"""
Tests for the reranker client and reranker sidecar.

Client tests: verify score merging, fallback behavior, top-k slicing.
Sidecar tests: test the /rerank and /health endpoints directly via FastAPI TestClient.
"""

import pytest
from unittest.mock import AsyncMock, patch
from typing import Any


# ---------------------------------------------------------------------------
# Reranker Client Tests (oracle.clients.reranker_client)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_reranker_client_merges_scores():
    """Verify that reranker scores are merged back into candidate dicts."""
    import httpx
    from oracle.clients.reranker_client import RerankerClient

    candidates = [
        {"id": "a", "text": "Tensile strength of steel is 500 MPa.", "document_id": "doc1", "page_no": 1},
        {"id": "b", "text": "Carbon content affects hardness.", "document_id": "doc1", "page_no": 2},
        {"id": "c", "text": "Heat treatment improves toughness.", "document_id": "doc2", "page_no": 5},
    ]

    mock_response = {"ranked": [
        {"id": "b", "score": 0.98},
        {"id": "a", "score": 0.75},
        {"id": "c", "score": 0.41},
    ]}

    client = RerankerClient()
    with patch.object(client._http, "post", new_callable=AsyncMock) as mock_post:
        from unittest.mock import MagicMock
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = mock_response
        mock_post.return_value = mock_resp

        results = await client.rerank(
            query="What affects steel hardness?",
            candidates=candidates,
            top_k=2,
        )

    assert len(results) == 2
    # Top result should be "b" (score 0.98)
    assert results[0]["id"] == "b"
    assert results[0]["rerank_score"] == pytest.approx(0.98)
    # Second should be "a"
    assert results[1]["id"] == "a"
    # Original metadata preserved
    assert results[0]["document_id"] == "doc1"


@pytest.mark.asyncio
async def test_reranker_client_fallback_on_connection_error():
    """Verify graceful fallback to hybrid-search order when reranker is down."""
    import httpx
    from oracle.clients.reranker_client import RerankerClient

    candidates = [
        {"id": "x", "text": "First result", "score": 0.9},
        {"id": "y", "text": "Second result", "score": 0.7},
        {"id": "z", "text": "Third result", "score": 0.5},
    ]

    client = RerankerClient()
    with patch.object(client._http, "post", side_effect=Exception("Connection refused")):
        results = await client.rerank(
            query="any query",
            candidates=candidates,
            top_k=2,
        )

    # Should fall back and return top-2 in original order
    assert len(results) == 2
    assert results[0]["id"] == "x"


# ---------------------------------------------------------------------------
# Reranker Sidecar Tests (reranker/main.py)
# ---------------------------------------------------------------------------

def test_reranker_sidecar_health():
    """Health endpoint returns 200 after model load."""
    from fastapi.testclient import TestClient

    with patch("reranker.main.CrossEncoder") as mock_ce:
        mock_instance = mock_ce.return_value
        import sys
        sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent / "reranker"))
        import importlib
        import main as reranker_main
        importlib.reload(reranker_main)

        client = TestClient(reranker_main.app)
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"


def test_reranker_sidecar_rerank_endpoint():
    """POST /rerank returns candidates sorted by score."""
    import numpy as np
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent / "reranker"))

    with patch("reranker.main.CrossEncoder") as mock_ce_class:
        mock_model = mock_ce_class.return_value
        # Assign descending scores: c2=0.9, c1=0.7, c3=0.3
        mock_model.predict.return_value = np.array([0.7, 0.9, 0.3])

        import importlib
        import main as reranker_main
        importlib.reload(reranker_main)
        reranker_main._model = mock_model

        from fastapi.testclient import TestClient
        client = TestClient(reranker_main.app)

        payload = {
            "query": "tensile strength test",
            "candidates": [
                {"id": "c1", "text": "Chunk about yield strength"},
                {"id": "c2", "text": "Chunk about tensile testing"},
                {"id": "c3", "text": "Unrelated chunk"},
            ],
            "top_k": 2,
        }

        resp = client.post("/rerank", json=payload)
        assert resp.status_code == 200
        ranked = resp.json()["ranked"]
        assert len(ranked) == 2
        # c2 should be first (score 0.9)
        assert ranked[0]["id"] == "c2"
        assert ranked[1]["id"] == "c1"
