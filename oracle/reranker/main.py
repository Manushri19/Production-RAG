"""
Reranker Sidecar — FastAPI Service

Serves BAAI/bge-reranker-v2-m3 via sentence-transformers.
Exposes a single endpoint: POST /rerank

The model is loaded once at startup and kept in memory.
CPU-only is sufficient for 30-candidate reranking (~150ms on a modern CPU).
"""

import logging
import os
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import CrossEncoder

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL_NAME = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
HOST = os.getenv("RERANKER_HOST", "0.0.0.0")
PORT = int(os.getenv("RERANKER_PORT", "8011"))
LOG_LEVEL = os.getenv("RERANKER_LOG_LEVEL", "INFO")

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("reranker")

# ---------------------------------------------------------------------------
# Model (loaded once at module import — warm on first request)
# ---------------------------------------------------------------------------

logger.info(f"Loading cross-encoder model: {MODEL_NAME}")
_model: Optional[CrossEncoder] = None


def get_model() -> CrossEncoder:
    global _model
    if _model is None:
        _model = CrossEncoder(MODEL_NAME, max_length=512)
        logger.info(f"Model {MODEL_NAME} loaded and ready.")
    return _model


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Oracle Reranker Sidecar",
    description=f"Cross-encoder reranking using {MODEL_NAME}",
    version="0.1.0",
)


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class CandidateIn(BaseModel):
    id: str
    text: str


class RankedItem(BaseModel):
    id: str
    score: float


class RerankRequest(BaseModel):
    query: str
    candidates: List[CandidateIn]
    top_k: int = 5


class RerankResponse(BaseModel):
    ranked: List[RankedItem]


class HealthResponse(BaseModel):
    status: str
    model: str


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.post("/rerank", response_model=RerankResponse)
def rerank(request: RerankRequest):
    """
    Re-rank candidates using the cross-encoder.

    The cross-encoder jointly encodes (query, candidate) pairs and outputs
    a relevance score for each. Higher score = more relevant.

    Returns candidates sorted by score (descending), limited to top_k.
    """
    if not request.candidates:
        return RerankResponse(ranked=[])

    model = get_model()

    # Build (query, passage) pairs for the cross-encoder
    pairs = [[request.query, c.text] for c in request.candidates]

    try:
        scores: List[float] = model.predict(pairs).tolist()
    except Exception as e:
        logger.error(f"Cross-encoder predict failed: {e}")
        raise HTTPException(status_code=500, detail=f"Reranking failed: {e}")

    # Zip IDs with scores and sort descending
    scored = sorted(
        zip([c.id for c in request.candidates], scores),
        key=lambda x: x[1],
        reverse=True,
    )

    ranked = [
        RankedItem(id=cid, score=score)
        for cid, score in scored[: request.top_k]
    ]

    logger.debug(
        f"Reranked {len(request.candidates)} → top-{request.top_k} | "
        f"best score={ranked[0].score:.4f}" if ranked else "no ranked results"
    )

    return RerankResponse(ranked=ranked)


@app.get("/health", response_model=HealthResponse)
def health():
    """Health check — also triggers model load on first call."""
    get_model()  # Ensure model is loaded
    return HealthResponse(status="ok", model=MODEL_NAME)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT, log_level=LOG_LEVEL.lower())
