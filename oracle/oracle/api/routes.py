"""
Oracle API Routes.

POST /v1/query   — Hybrid search + rerank + stream generation
GET  /v1/health  — Service health check
"""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from oracle.orchestrator.query_orchestrator import QueryOrchestrator

logger = logging.getLogger(__name__)
router = APIRouter()

# Singleton orchestrator (shared across requests)
_orchestrator: Optional[QueryOrchestrator] = None


def _get_orchestrator() -> QueryOrchestrator:
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = QueryOrchestrator()
    return _orchestrator


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    """Incoming query from a client."""

    query: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="The user's question in natural language.",
        examples=["What is the tensile strength of Inconel 718 at 650°C?"],
    )
    document_ids: Optional[list[str]] = Field(
        default=None,
        description=(
            "Optional list of document IDs to scope the search. "
            "If omitted, all documents in the collection are searched."
        ),
        examples=[["alloy_datasheet_v3", "test_report_Q2_2025"]],
    )
    top_k_candidates: int = Field(
        default=30,
        ge=5,
        le=100,
        description="Number of hybrid search candidates before reranking.",
    )
    top_k_final: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of reranked chunks passed to the LLM.",
    )


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.post(
    "/query",
    summary="Query documents with hybrid search and streaming generation",
    response_description=(
        "Server-Sent Events stream. Each event is a JSON object:\n"
        "- `{\"token\": \"...\"}` — progressive LLM output\n"
        "- `{\"sources\": [...]}` — final event with citation metadata\n"
        "- `[DONE]` — stream terminator"
    ),
)
async def query(request: QueryRequest):
    """
    Execute the full RAG pipeline:

    1. Embed query (Ollama)
    2. Hybrid search: dense (HNSW) + sparse (BM25) → RRF fusion → top_k_candidates
    3. Cross-encoder reranking → top_k_final chunks
    4. Strict generation (vLLM, SSE stream) with mandatory inline citations

    Returns a `text/event-stream` response.
    """
    orchestrator = _get_orchestrator()

    try:
        stream = orchestrator.run_stream(
            query=request.query,
            document_ids=request.document_ids,
            top_k_candidates=request.top_k_candidates,
            top_k_final=request.top_k_final,
        )
        return StreamingResponse(
            stream,
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",  # Disable nginx buffering
            },
        )
    except Exception as e:
        logger.exception(f"Query pipeline failed: {e}")
        raise HTTPException(status_code=503, detail=f"Pipeline error: {e}")


@router.get("/health", response_model=HealthResponse)
async def health():
    """Service health check."""
    from oracle import __version__
    return HealthResponse(status="ok", service="oracle-api", version=__version__)
