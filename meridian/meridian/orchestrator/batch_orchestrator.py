"""
Batch Orchestrator - Submit and manage batch document processing.

Provides:
- Batch submission (scan directory, queue tasks)
- Progress tracking
- Checkpoint/resume support
- Failed document retry
"""

import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

import redis

from meridian.config import REDIS_URL
from meridian.orchestrator.state_manager import StateManager
from meridian.workers.celery_app import app as celery_app

logger = logging.getLogger(__name__)


class BatchOrchestrator:
    """
    Orchestrates batch document processing.

    Usage:
        orch = BatchOrchestrator()

        # Submit a batch
        batch_id = orch.submit_batch(
            input_dir=Path("/mnt/data/pdfs"),
            collection="my_collection",
        )

        # Check progress
        progress = orch.get_progress(batch_id)
        print(f"Complete: {progress['complete']}/{progress['total']}")

        # Resume after interrupt
        orch.resume(batch_id)

        # Retry failed
        orch.retry_failed(batch_id)
    """

    def __init__(self, redis_url: Optional[str] = None):
        """
        Initialize the orchestrator.

        Args:
            redis_url: Redis connection URL (defaults to REDIS_URL from config)
        """
        self.redis = redis.from_url(redis_url or REDIS_URL, decode_responses=False)
        self.celery = celery_app

    def _generate_batch_id(self) -> str:
        """Generate a unique batch ID."""
        return f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    def _get_state_manager(self, batch_id: str) -> StateManager:
        """Get state manager for a batch."""
        return StateManager(self.redis, batch_id)

    # =========================================================================
    # Batch Submission
    # =========================================================================

    def submit_batch(
        self,
        input_dir: Path,
        collection: str,
        batch_id: Optional[str] = None,
        extract_tables: bool = True,
        extract_figures: bool = True,
        extract_formulas: bool = True,
        store_embeddings: bool = True,
        file_pattern: str = "*.pdf",
    ) -> str:
        """
        Submit all PDFs in directory to processing queue.

        Args:
            input_dir: Directory containing PDF files
            collection: Qdrant collection name
            batch_id: Optional batch ID (generated if not provided)
            extract_tables: Process tables with VLM
            extract_figures: Process figures with VLM
            extract_formulas: Process formula pages with VLM
            store_embeddings: Generate and store embeddings
            file_pattern: Glob pattern for files (default: *.pdf)

        Returns:
            batch_id for tracking
        """
        input_dir = Path(input_dir)
        if not input_dir.exists():
            raise ValueError(f"Input directory not found: {input_dir}")

        # Generate batch ID
        batch_id = batch_id or self._generate_batch_id()
        state = self._get_state_manager(batch_id)

        # Check if batch already exists
        if state.exists():
            raise ValueError(
                f"Batch {batch_id} already exists. Use resume() or clear it first."
            )

        # Scan for PDFs
        pdf_files = sorted(input_dir.glob(file_pattern))
        if not pdf_files:
            raise ValueError(f"No files matching '{file_pattern}' in {input_dir}")

        logger.info(f"Found {len(pdf_files)} files in {input_dir}")

        # Create batch info
        batch_info = {
            "batch_id": batch_id,
            "created_at": datetime.now().isoformat(),
            "input_dir": str(input_dir),
            "collection": collection,
            "total_docs": len(pdf_files),
            "file_pattern": file_pattern,
            "options": {
                "extract_tables": extract_tables,
                "extract_figures": extract_figures,
                "extract_formulas": extract_formulas,
                "store_embeddings": store_embeddings,
            },
            "status": "running",
        }
        state.set_batch_info(batch_info)

        # Build doc_id -> pdf_path mapping and add to pending
        doc_ids = []
        doc_paths = {}

        for pdf_path in pdf_files:
            doc_id = pdf_path.stem  # Use filename without extension
            doc_ids.append(doc_id)
            doc_paths[doc_id] = str(pdf_path)

        # Store doc paths in Redis for task lookup
        if doc_paths:
            self.redis.hset(
                f"meridian:batch:{batch_id}:paths",
                mapping={k: v for k, v in doc_paths.items()},
            )

        # Add all to pending
        state.add_pending(doc_ids)

        # Submit tasks to Celery
        submitted = 0
        for doc_id in doc_ids:
            pdf_path = doc_paths[doc_id]
            self._submit_task(
                batch_id=batch_id,
                doc_id=doc_id,
                pdf_path=pdf_path,
                collection=collection,
                extract_tables=extract_tables,
                extract_figures=extract_figures,
                extract_formulas=extract_formulas,
                store_embeddings=store_embeddings,
            )
            submitted += 1

        logger.info(f"Submitted {submitted} tasks for batch {batch_id}")
        return batch_id

    def _submit_task(
        self,
        batch_id: str,
        doc_id: str,
        pdf_path: str,
        collection: str,
        **options,
    ):
        """Submit a single document processing task."""
        from meridian.workers.tasks import process_document

        process_document.delay(
            doc_id=doc_id,
            pdf_path=pdf_path,
            batch_id=batch_id,
            collection=collection,
            **options,
        )

    # =========================================================================
    # Progress Tracking
    # =========================================================================

    def get_progress(self, batch_id: str) -> Dict[str, Any]:
        """
        Get current progress for a batch.

        Returns:
            {
                "batch_id": "...",
                "status": "running",
                "total": 10000,
                "pending": 5000,
                "processing": 20,
                "complete": 4950,
                "failed": 30,
                "percent_complete": 49.5,
                "elapsed_seconds": 8100,
                "docs_per_minute": 36.7,
                "estimated_remaining_seconds": 8160,
            }
        """
        state = self._get_state_manager(batch_id)

        if not state.exists():
            return {"error": f"Batch {batch_id} not found"}

        batch_info = state.get_batch_info()
        counts = state.get_counts()

        total = counts["total"]
        complete = counts["complete"]
        failed = counts["failed"]
        pending = counts["pending"]
        processing = counts["processing"]

        # Calculate progress
        percent = (complete / total * 100) if total > 0 else 0

        # Calculate timing
        created_at = batch_info.get("created_at", "")
        elapsed_seconds = 0
        docs_per_minute = 0
        estimated_remaining = None

        if created_at:
            try:
                start_time = datetime.fromisoformat(created_at)
                elapsed = datetime.now() - start_time
                elapsed_seconds = elapsed.total_seconds()

                if elapsed_seconds > 0 and complete > 0:
                    docs_per_minute = complete / (elapsed_seconds / 60)
                    remaining_docs = pending + processing
                    if docs_per_minute > 0:
                        estimated_remaining = (remaining_docs / docs_per_minute) * 60
            except:
                pass

        # Get per-document timing summary
        timing_summary = state.get_timing_summary()

        return {
            "batch_id": batch_id,
            "status": batch_info.get("status", "unknown"),
            "collection": batch_info.get("collection", ""),
            "input_dir": batch_info.get("input_dir", ""),
            "total": total,
            "pending": pending,
            "processing": processing,
            "complete": complete,
            "failed": failed,
            "percent_complete": round(percent, 1),
            "elapsed_seconds": int(elapsed_seconds),
            "docs_per_minute": round(docs_per_minute, 1),
            "estimated_remaining_seconds": int(estimated_remaining) if estimated_remaining else None,
            "timing_summary": timing_summary,
        }

    def is_complete(self, batch_id: str) -> bool:
        """Check if batch processing is complete."""
        progress = self.get_progress(batch_id)
        return (
            progress.get("pending", 1) == 0 and
            progress.get("processing", 1) == 0
        )

    # =========================================================================
    # Resume & Retry
    # =========================================================================

    def resume(self, batch_id: str, recover_stale: bool = True, stale_timeout: int = 600) -> int:
        """
        Resume a paused or interrupted batch.

        Re-queues pending documents and optionally recovers stale processing docs.

        Args:
            batch_id: Batch to resume
            recover_stale: Move stale processing docs back to pending
            stale_timeout: Seconds after which a processing doc is considered stale (default 600)

        Returns:
            Number of tasks queued
        """
        state = self._get_state_manager(batch_id)

        if not state.exists():
            raise ValueError(f"Batch {batch_id} not found")

        batch_info = state.get_batch_info()
        collection = batch_info.get("collection", "meridian_documents")
        options = batch_info.get("options", {})

        # Recover stale processing docs
        if recover_stale:
            recovered = state.recover_stale(timeout_seconds=stale_timeout)
            if recovered:
                logger.info(f"Recovered {recovered} stale docs")

        # Get pending docs
        pending = state.get_pending()
        if not pending:
            logger.info("No pending documents to resume")
            return 0

        # Get doc paths
        paths_raw = self.redis.hgetall(f"meridian:batch:{batch_id}:paths")
        doc_paths = {
            (k.decode() if isinstance(k, bytes) else k): (v.decode() if isinstance(v, bytes) else v)
            for k, v in paths_raw.items()
        }

        # Re-submit tasks
        submitted = 0
        for doc_id in pending:
            pdf_path = doc_paths.get(doc_id)
            if not pdf_path:
                logger.warning(f"No path found for {doc_id}, skipping")
                continue

            self._submit_task(
                batch_id=batch_id,
                doc_id=doc_id,
                pdf_path=pdf_path,
                collection=collection,
                **options,
            )
            submitted += 1

        # Update status
        state.update_batch_status("running")

        logger.info(f"Resumed {submitted} tasks for batch {batch_id}")
        return submitted

    def retry_failed(self, batch_id: str) -> int:
        """
        Retry only failed documents.

        Returns:
            Number of tasks queued
        """
        state = self._get_state_manager(batch_id)

        if not state.exists():
            raise ValueError(f"Batch {batch_id} not found")

        # Move failed back to pending
        failed_count = state.retry_failed()

        if failed_count == 0:
            logger.info("No failed documents to retry")
            return 0

        # Resume will pick up the newly pending docs
        return self.resume(batch_id, recover_stale=False)

    # =========================================================================
    # Batch Management
    # =========================================================================

    def pause(self, batch_id: str):
        """
        Pause batch processing.

        Note: This only updates status. Running tasks will complete.
        New tasks won't be submitted until resume().
        """
        state = self._get_state_manager(batch_id)

        if not state.exists():
            raise ValueError(f"Batch {batch_id} not found")

        state.update_batch_status("paused")
        logger.info(f"Paused batch {batch_id}")

    def get_failed_docs(self, batch_id: str) -> List[Dict[str, str]]:
        """Get list of failed documents with error messages."""
        state = self._get_state_manager(batch_id)
        return state.get_failed_with_errors()

    def clear_batch(self, batch_id: str):
        """Remove all state for a batch."""
        state = self._get_state_manager(batch_id)
        state.clear()

        # Also clear paths
        self.redis.delete(f"meridian:batch:{batch_id}:paths")

        logger.info(f"Cleared batch {batch_id}")

    def list_batches(self) -> List[Dict[str, Any]]:
        """List all batches with basic info."""
        # Find all batch info keys
        keys = self.redis.keys("meridian:batch:*:info")
        batches = []

        for key in keys:
            key_str = key.decode() if isinstance(key, bytes) else key
            # Extract batch_id from key
            parts = key_str.split(":")
            if len(parts) >= 3:
                batch_id = parts[2]
                progress = self.get_progress(batch_id)
                if "error" not in progress:
                    batches.append({
                        "batch_id": batch_id,
                        "status": progress.get("status"),
                        "total": progress.get("total"),
                        "complete": progress.get("complete"),
                        "failed": progress.get("failed"),
                        "percent_complete": progress.get("percent_complete"),
                    })

        # Sort by batch_id (newest first)
        batches.sort(key=lambda x: x["batch_id"], reverse=True)
        return batches
