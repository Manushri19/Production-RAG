"""
Celery Task Definitions

Defines the Celery tasks that workers execute.
Tasks are thin wrappers around document_worker functions.

State tracking:
- When batch_id is provided, updates Redis state via StateManager
- State transitions: pending → processing → complete/failed
"""

import logging
from typing import Optional

import redis
from celery import shared_task
from celery.exceptions import SoftTimeLimitExceeded

from meridian.config import REDIS_URL
from meridian.workers.document_worker import process_single_document, process_document_minimal

logger = logging.getLogger(__name__)

# Redis client for state updates (lazy initialized)
_redis_client: Optional[redis.Redis] = None


def _get_redis() -> redis.Redis:
    """Get Redis client for state updates."""
    global _redis_client
    if _redis_client is None:
        _redis_client = redis.from_url(REDIS_URL, decode_responses=False)
    return _redis_client


def _get_state_manager(batch_id: str):
    """Get state manager for batch (import here to avoid circular imports)."""
    from meridian.orchestrator.state_manager import StateManager
    return StateManager(_get_redis(), batch_id)


@shared_task(
    bind=True,
    max_retries=1,  # Retry once for transient errors (network disconnects, etc.)
    default_retry_delay=30,
    autoretry_for=(),  # We handle retries explicitly below
    retry_backoff=False,
    retry_backoff_max=300,
)
def process_document(
    self,
    doc_id: str,
    pdf_path: str,
    batch_id: Optional[str] = None,
    collection: str = "meridian_documents",
    extract_tables: bool = True,
    extract_figures: bool = True,
    extract_formulas: bool = True,
    store_embeddings: bool = True,
):
    """
    Process a single document end-to-end.

    This is the main task for document processing:
    1. Call Docling service (HTTP)
    2. Annotate formula pages (PIL)
    3. Call VLM service (HTTP)
    4. Build chunks (CPU)
    5. Generate embeddings (HTTP)
    6. Store in Qdrant (HTTP)

    Args:
        doc_id: Unique document identifier
        pdf_path: Path to the PDF file
        batch_id: Optional batch ID for state tracking
        collection: Qdrant collection name
        extract_tables: Whether to process tables with VLM
        extract_figures: Whether to process figures with VLM
        extract_formulas: Whether to process formula pages with VLM
        store_embeddings: Whether to generate and store embeddings

    Returns:
        Dict with processing results

    Raises:
        Retries on failure up to max_retries times
    """
    logger.info(f"Task started: process_document({doc_id}, batch={batch_id})")

    # Get state manager if batch tracking enabled
    state = _get_state_manager(batch_id) if batch_id else None

    # Mark as processing
    if state:
        state.mark_processing(doc_id)

    try:
        result = process_single_document(
            doc_id=doc_id,
            pdf_path=pdf_path,
            collection=collection,
            extract_tables=extract_tables,
            extract_figures=extract_figures,
            extract_formulas=extract_formulas,
            store_embeddings=store_embeddings,
        )

        if result.get("success"):
            # Mark complete
            if state:
                state.mark_complete(doc_id, result)
            logger.info(f"Task completed: process_document({doc_id})")
            return result
        else:
            # Processing failed
            error = result.get("error", "Unknown error")
            if state:
                state.mark_failed(doc_id, error)
            logger.warning(f"Document processing failed for {doc_id}: {error}")
            raise RuntimeError(error)

    except SoftTimeLimitExceeded:
        error = "Task exceeded time limit"
        if state:
            state.mark_failed(doc_id, error)
        logger.error(f"Task timeout for {doc_id}")
        return {
            "doc_id": doc_id,
            "success": False,
            "error": error,
        }

    except Exception as e:
        error_str = str(e)

        # Don't retry on errors that won't benefit from retry (corrupt PDFs, etc.)
        no_retry_errors = [
            "Docling conversion failed",  # PDF parsing errors
            "pipeline terminated early",  # Docling internal errors
            "PDF file not found",         # Missing files
            "CUDA out of memory",         # GPU OOM - won't help to retry
        ]

        should_retry = (
            self.request.retries < self.max_retries
            and not any(err in error_str for err in no_retry_errors)
        )

        if should_retry:
            logger.warning(f"Retrying {doc_id} (attempt {self.request.retries + 1}/{self.max_retries}): {e}")
            raise self.retry(exc=e, countdown=30)
        else:
            # Mark as failed permanently
            if state:
                state.mark_failed(doc_id, error_str)
            logger.error(f"Task failed permanently for {doc_id}: {e}")
            return {
                "doc_id": doc_id,
                "success": False,
                "error": error_str,
            }


@shared_task(bind=True, max_retries=3)
def process_document_quick(
    self,
    doc_id: str,
    pdf_path: str,
):
    """
    Quick processing - Docling + chunking only, no VLM or embeddings.

    Useful for:
    - Testing the pipeline
    - When you only need document structure
    - Fast initial processing before full extraction
    """
    logger.info(f"Task started: process_document_quick({doc_id})")

    try:
        result = process_document_minimal(
            doc_id=doc_id,
            pdf_path=pdf_path,
        )

        if not result.get("success"):
            error = result.get("error", "Unknown error")
            raise RuntimeError(error)

        logger.info(f"Task completed: process_document_quick({doc_id})")
        return result

    except Exception as e:
        logger.error(f"Task failed for {doc_id}: {e}")
        self.retry(exc=e, countdown=30)


@shared_task
def health_check():
    """
    Simple health check task to verify worker is running.

    Returns:
        Dict with worker status
    """
    from meridian.clients.docling_client import DoclingClient
    from meridian.clients.vlm_client import VLMClient
    from meridian.clients.embedding_client import EmbeddingClient

    results = {}

    # Check Docling
    try:
        client = DoclingClient()
        health = client.health_all()
        healthy_count = sum(1 for h in health.values() if h.get("models_loaded"))
        results["docling"] = {
            "healthy": healthy_count > 0,
            "instances": f"{healthy_count}/{len(health)}",
        }
        client.close()
    except Exception as e:
        results["docling"] = {"healthy": False, "error": str(e)}

    # Check VLM
    try:
        client = VLMClient()
        health = client.health()
        results["vlm"] = {
            "healthy": health.get("status") == "healthy",
            "model_loaded": health.get("model_loaded", False),
        }
        client.close()
    except Exception as e:
        results["vlm"] = {"healthy": False, "error": str(e)}

    # Check Embedding
    try:
        client = EmbeddingClient()
        health = client.health()
        results["embedding"] = {
            "healthy": health.get("status") == "healthy",
            "ollama": health.get("ollama", {}).get("status"),
            "qdrant": health.get("qdrant", {}).get("status"),
        }
        client.close()
    except Exception as e:
        results["embedding"] = {"healthy": False, "error": str(e)}

    all_healthy = all(r.get("healthy", False) for r in results.values())
    results["overall"] = "healthy" if all_healthy else "degraded"

    return results
