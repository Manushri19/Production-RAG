"""
Oracle FastAPI Application Factory.

Usage:
    uvicorn oracle.api.app:app --host 0.0.0.0 --port 8010
"""

import logging
import sys

from fastapi import FastAPI

from oracle import __version__
from oracle.api.routes import router
import oracle.config as cfg

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=getattr(logging, cfg.LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    stream=sys.stdout,
)

logger = logging.getLogger("oracle")


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app() -> FastAPI:
    app = FastAPI(
        title="Oracle — Retrieval & Generation API",
        description=(
            "Hybrid search (dense + BM25 RRF) + cross-encoder reranking + "
            "strict citation-enforced generation for materials engineering R&D."
        ),
        version=__version__,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.include_router(router, prefix="/v1")
    return app


app = create_app()


def main():
    """Entry point for `oracle` CLI script."""
    import uvicorn
    uvicorn.run(
        "oracle.api.app:app",
        host=cfg.SERVICE_HOST,
        port=cfg.SERVICE_PORT,
        log_level=cfg.LOG_LEVEL.lower(),
        reload=False,
    )


if __name__ == "__main__":
    main()
