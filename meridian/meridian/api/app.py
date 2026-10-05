"""
Meridian HTTP API - FastAPI application factory.
"""

from fastapi import FastAPI

from meridian.api.routes import router


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Meridian",
        description="GPU-accelerated document processing API",
        version="0.1.0",
    )

    app.include_router(router, prefix="/v1")

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    return app
