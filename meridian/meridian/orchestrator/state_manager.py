"""
State Manager - Redis state management for batch processing.

Manages document state transitions:
  pending → processing → complete/failed

Thread-safe operations using Redis atomic commands.
"""

import json
import logging
import time
from datetime import datetime
from typing import List, Dict, Any, Optional, Set

import redis

logger = logging.getLogger(__name__)


class StateManager:
    """
    Manages document processing state in Redis.

    Key structure:
        meridian:batch:{batch_id}:info          # Batch metadata (JSON hash)
        meridian:batch:{batch_id}:pending       # Set of pending doc_ids
        meridian:batch:{batch_id}:processing    # Hash: doc_id -> start_timestamp
        meridian:batch:{batch_id}:complete      # Set of complete doc_ids
        meridian:batch:{batch_id}:failed        # Set of failed doc_ids
        meridian:doc:{batch_id}:{doc_id}:result # Doc result (JSON)
        meridian:doc:{batch_id}:{doc_id}:error  # Error message if failed
    """

    def __init__(self, redis_client: redis.Redis, batch_id: str):
        """
        Initialize state manager for a batch.

        Args:
            redis_client: Redis client instance
            batch_id: Unique batch identifier
        """
        self.redis = redis_client
        self.batch_id = batch_id
        self._prefix = f"meridian:batch:{batch_id}"

    # =========================================================================
    # Key Helpers
    # =========================================================================

    def _key(self, suffix: str) -> str:
        """Get full Redis key."""
        return f"{self._prefix}:{suffix}"

    def _doc_key(self, doc_id: str, suffix: str) -> str:
        """Get document-specific key."""
        return f"meridian:doc:{self.batch_id}:{doc_id}:{suffix}"

    # =========================================================================
    # Batch Info
    # =========================================================================

    def set_batch_info(self, info: Dict[str, Any]):
        """Set batch metadata."""
        self.redis.hset(self._key("info"), mapping={
            k: json.dumps(v) if isinstance(v, (dict, list)) else str(v)
            for k, v in info.items()
        })

    def get_batch_info(self) -> Dict[str, Any]:
        """Get batch metadata."""
        raw = self.redis.hgetall(self._key("info"))
        if not raw:
            return {}

        result = {}
        for k, v in raw.items():
            key = k.decode() if isinstance(k, bytes) else k
            val = v.decode() if isinstance(v, bytes) else v
            # Try to parse JSON
            try:
                result[key] = json.loads(val)
            except (json.JSONDecodeError, TypeError):
                result[key] = val
        return result

    def update_batch_status(self, status: str):
        """Update batch status (pending, running, complete, paused)."""
        self.redis.hset(self._key("info"), "status", status)

    # =========================================================================
    # State Transitions
    # =========================================================================

    def add_pending(self, doc_ids: List[str]) -> int:
        """
        Add documents to pending set.

        Args:
            doc_ids: List of document IDs

        Returns:
            Number of documents added
        """
        if not doc_ids:
            return 0

        added = self.redis.sadd(self._key("pending"), *doc_ids)
        logger.debug(f"Added {added} docs to pending")
        return added

    def mark_processing(self, doc_id: str) -> bool:
        """
        Move document from pending to processing.

        Args:
            doc_id: Document ID

        Returns:
            True if moved, False if not in pending
        """
        # Atomic: remove from pending, add to processing with timestamp
        pipe = self.redis.pipeline()
        pipe.srem(self._key("pending"), doc_id)
        pipe.hset(self._key("processing"), doc_id, str(time.time()))
        results = pipe.execute()

        moved = results[0] > 0
        if moved:
            logger.debug(f"Marked {doc_id} as processing")
        return moved

    def mark_complete(self, doc_id: str, result: Optional[Dict] = None) -> bool:
        """
        Move document from processing to complete.

        Args:
            doc_id: Document ID
            result: Optional result data to store

        Returns:
            True if moved, False if not in processing
        """
        # Get start time before removing from processing
        start_time_raw = self.redis.hget(self._key("processing"), doc_id)
        end_time = time.time()

        duration = None
        if start_time_raw:
            start_time = float(start_time_raw.decode() if isinstance(start_time_raw, bytes) else start_time_raw)
            duration = end_time - start_time

        pipe = self.redis.pipeline()
        pipe.hdel(self._key("processing"), doc_id)
        pipe.sadd(self._key("complete"), doc_id)

        # Store timing info
        timing = {
            "start_time": start_time if start_time_raw else None,
            "end_time": end_time,
            "duration_seconds": round(duration, 2) if duration else None,
        }
        pipe.hset(self._key("timings"), doc_id, json.dumps(timing))

        if result:
            # Add timing to result
            result["timing"] = timing
            pipe.set(
                self._doc_key(doc_id, "result"),
                json.dumps(result),
                ex=86400 * 7,  # Expire after 7 days
            )

        results = pipe.execute()
        moved = results[0] > 0

        if moved:
            logger.debug(f"Marked {doc_id} as complete (took {duration:.1f}s)" if duration else f"Marked {doc_id} as complete")
        return moved

    def mark_failed(self, doc_id: str, error: str) -> bool:
        """
        Move document from processing to failed.

        Args:
            doc_id: Document ID
            error: Error message

        Returns:
            True if moved, False if not in processing
        """
        pipe = self.redis.pipeline()
        pipe.hdel(self._key("processing"), doc_id)
        pipe.sadd(self._key("failed"), doc_id)
        pipe.set(
            self._doc_key(doc_id, "error"),
            error,
            ex=86400 * 7,  # Expire after 7 days
        )
        results = pipe.execute()

        moved = results[0] > 0
        if moved:
            logger.debug(f"Marked {doc_id} as failed: {error[:50]}...")
        return moved

    def retry_failed(self, doc_ids: Optional[List[str]] = None) -> int:
        """
        Move failed documents back to pending.

        Args:
            doc_ids: Specific docs to retry, or None for all failed

        Returns:
            Number of documents moved
        """
        if doc_ids is None:
            doc_ids = list(self.get_failed())

        if not doc_ids:
            return 0

        pipe = self.redis.pipeline()
        for doc_id in doc_ids:
            pipe.srem(self._key("failed"), doc_id)
            pipe.sadd(self._key("pending"), doc_id)
            pipe.delete(self._doc_key(doc_id, "error"))

        pipe.execute()
        logger.info(f"Moved {len(doc_ids)} failed docs back to pending")
        return len(doc_ids)

    # =========================================================================
    # Queries
    # =========================================================================

    def get_pending(self) -> Set[str]:
        """Get all pending document IDs."""
        result = self.redis.smembers(self._key("pending"))
        return {d.decode() if isinstance(d, bytes) else d for d in result}

    def get_processing(self) -> Dict[str, float]:
        """Get processing documents with start timestamps."""
        result = self.redis.hgetall(self._key("processing"))
        return {
            (k.decode() if isinstance(k, bytes) else k): float(v.decode() if isinstance(v, bytes) else v)
            for k, v in result.items()
        }

    def get_complete(self) -> Set[str]:
        """Get all complete document IDs."""
        result = self.redis.smembers(self._key("complete"))
        return {d.decode() if isinstance(d, bytes) else d for d in result}

    def get_failed(self) -> Set[str]:
        """Get all failed document IDs."""
        result = self.redis.smembers(self._key("failed"))
        return {d.decode() if isinstance(d, bytes) else d for d in result}

    def get_counts(self) -> Dict[str, int]:
        """Get counts for all states."""
        pipe = self.redis.pipeline()
        pipe.scard(self._key("pending"))
        pipe.hlen(self._key("processing"))
        pipe.scard(self._key("complete"))
        pipe.scard(self._key("failed"))
        results = pipe.execute()

        return {
            "pending": results[0],
            "processing": results[1],
            "complete": results[2],
            "failed": results[3],
            "total": sum(results),
        }

    def get_doc_result(self, doc_id: str) -> Optional[Dict]:
        """Get result for a completed document."""
        raw = self.redis.get(self._doc_key(doc_id, "result"))
        if raw:
            return json.loads(raw.decode() if isinstance(raw, bytes) else raw)
        return None

    def get_doc_error(self, doc_id: str) -> Optional[str]:
        """Get error message for a failed document."""
        raw = self.redis.get(self._doc_key(doc_id, "error"))
        if raw:
            return raw.decode() if isinstance(raw, bytes) else raw
        return None

    def get_failed_with_errors(self) -> List[Dict[str, str]]:
        """Get all failed documents with their error messages."""
        failed = self.get_failed()
        results = []

        for doc_id in failed:
            error = self.get_doc_error(doc_id)
            results.append({
                "doc_id": doc_id,
                "error": error or "Unknown error",
            })

        return results

    def get_timings(self) -> Dict[str, Dict]:
        """Get timing info for all completed documents."""
        raw = self.redis.hgetall(self._key("timings"))
        result = {}
        for k, v in raw.items():
            doc_id = k.decode() if isinstance(k, bytes) else k
            timing = json.loads(v.decode() if isinstance(v, bytes) else v)
            result[doc_id] = timing
        return result

    def get_timing_summary(self) -> Dict[str, Any]:
        """
        Get timing summary for the batch.

        Returns:
            Dict with min, max, avg, total processing time and page stats
        """
        timings = self.get_timings()
        if not timings:
            return {"count": 0}

        durations = [t["duration_seconds"] for t in timings.values() if t.get("duration_seconds")]
        if not durations:
            return {"count": len(timings)}

        # Find actual processing window
        start_times = [t["start_time"] for t in timings.values() if t.get("start_time")]
        end_times = [t["end_time"] for t in timings.values() if t.get("end_time")]

        # Collect page stats from completed doc results
        total_pages = 0
        page_times = []  # (pages, duration) tuples for per-page stats
        for doc_id in timings.keys():
            result = self.get_doc_result(doc_id)
            if result:
                pages = result.get("stats", {}).get("pages", 0)
                duration = timings[doc_id].get("duration_seconds", 0)
                if pages > 0:
                    total_pages += pages
                    if duration and duration > 0:
                        page_times.append((pages, duration))

        summary = {
            "count": len(durations),
            "min_seconds": round(min(durations), 2),
            "max_seconds": round(max(durations), 2),
            "avg_seconds": round(sum(durations) / len(durations), 2),
            "total_seconds": round(sum(durations), 2),
            "first_start": min(start_times) if start_times else None,
            "last_end": max(end_times) if end_times else None,
            "wall_clock_seconds": round(max(end_times) - min(start_times), 2) if start_times and end_times else None,
            # Page stats
            "total_pages": total_pages,
        }

        # Calculate seconds per page
        if page_times:
            secs_per_page = [d / p for p, d in page_times if p > 0]
            if secs_per_page:
                summary["min_sec_per_page"] = round(min(secs_per_page), 2)
                summary["max_sec_per_page"] = round(max(secs_per_page), 2)
                summary["avg_sec_per_page"] = round(sum(secs_per_page) / len(secs_per_page), 2)

        return summary

    # =========================================================================
    # Stale Detection
    # =========================================================================

    def get_stale_processing(self, timeout_seconds: int = 600) -> List[str]:
        """
        Get documents stuck in processing state.

        Args:
            timeout_seconds: Consider stale after this many seconds

        Returns:
            List of stale document IDs
        """
        processing = self.get_processing()
        now = time.time()
        stale = []

        for doc_id, start_time in processing.items():
            if now - start_time > timeout_seconds:
                stale.append(doc_id)

        return stale

    def recover_stale(self, timeout_seconds: int = 600) -> int:
        """
        Move stale processing documents back to pending.

        Args:
            timeout_seconds: Consider stale after this many seconds

        Returns:
            Number of documents recovered
        """
        stale = self.get_stale_processing(timeout_seconds)

        if not stale:
            return 0

        pipe = self.redis.pipeline()
        for doc_id in stale:
            pipe.hdel(self._key("processing"), doc_id)
            pipe.sadd(self._key("pending"), doc_id)

        pipe.execute()
        logger.info(f"Recovered {len(stale)} stale docs back to pending")
        return len(stale)

    # =========================================================================
    # Cleanup
    # =========================================================================

    def clear(self):
        """Clear all state for this batch."""
        keys = self.redis.keys(f"{self._prefix}:*")
        doc_keys = self.redis.keys(f"meridian:doc:{self.batch_id}:*")

        all_keys = list(keys) + list(doc_keys)
        if all_keys:
            self.redis.delete(*all_keys)
            logger.info(f"Cleared {len(all_keys)} keys for batch {self.batch_id}")

    def exists(self) -> bool:
        """Check if batch exists in Redis."""
        return self.redis.exists(self._key("info")) > 0
