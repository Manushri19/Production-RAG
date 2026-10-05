"""
Docling Service Client

HTTP client for workers to call the Docling service.
Includes retry logic, timeout handling, error recovery, and
round-robin load balancing across multiple instances.
"""

import json
import logging
import time
import threading
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field

import httpx
import redis as redis_lib

from meridian.config import DOCLING_INSTANCES as _CONFIG_INSTANCES, REDIS_URL as _CONFIG_REDIS_URL

logger = logging.getLogger(__name__)


# =============================================================================
# Default Instance Configuration
# =============================================================================
# Sourced from meridian.config (env-driven). Falls back to localhost:8001-8008
# only if the env var is unset.

DEFAULT_INSTANCES = list(_CONFIG_INSTANCES)


@dataclass
class DoclingClientConfig:
    """Configuration for the Docling client."""

    # Single instance mode (legacy)
    base_url: str = "http://localhost:8001"

    # Multi-instance mode with round-robin load balancing
    instances: List[str] = field(default_factory=lambda: DEFAULT_INSTANCES.copy())
    use_load_balancing: bool = True  # If True, use round-robin across instances

    # Redis for shared round-robin counter (ensures true distribution across all workers)
    redis_url: str = field(default_factory=lambda: _CONFIG_REDIS_URL)
    redis_counter_key: str = "meridian:docling:instance_counter"

    timeout_seconds: float = 300.0  # 5 minutes default
    max_retries: int = 3
    retry_delay_seconds: float = 5.0
    retry_backoff_multiplier: float = 2.0


class DoclingClientError(Exception):
    """Base exception for Docling client errors."""

    def __init__(self, message: str, status_code: Optional[int] = None, details: Optional[str] = None):
        super().__init__(message)
        self.status_code = status_code
        self.details = details


class DoclingServiceUnavailable(DoclingClientError):
    """Service is not available."""
    pass


class DoclingProcessingError(DoclingClientError):
    """Error during PDF processing."""
    pass


class DoclingClient:
    """
    HTTP client for the Docling service with round-robin load balancing.

    Usage:
        # Multi-instance mode (default) - uses round-robin across 3 instances
        client = DoclingClient()

        # Single instance mode
        config = DoclingClientConfig(use_load_balancing=False, base_url="http://localhost:8001")
        client = DoclingClient(config)

        # Check health
        if client.is_healthy():
            # Process a PDF - automatically routed to next available instance
            result = client.process_pdf("/path/to/document.pdf")

            if result["success"]:
                document = result["document"]
                tables = result["extractions"]["tables"]
                figures = result["extractions"]["figures"]
                formula_pages = result["extractions"]["formula_pages"]
    """

    def __init__(self, config: Optional[DoclingClientConfig] = None):
        """Initialize the client."""
        self.config = config or DoclingClientConfig()

        # Redis client for shared round-robin counter
        self._redis = None
        if self.config.use_load_balancing:
            try:
                self._redis = redis_lib.from_url(self.config.redis_url, decode_responses=True)
            except Exception as e:
                logger.warning(f"Failed to connect to Redis for load balancing: {e}")

        # Create clients for each instance (or single client for legacy mode)
        if self.config.use_load_balancing:
            self._clients = {
                url: httpx.Client(
                    base_url=url,
                    timeout=httpx.Timeout(self.config.timeout_seconds),
                )
                for url in self.config.instances
            }
            self._instances = list(self.config.instances)
            logger.info(f"DoclingClient initialized with {len(self._instances)} instances: {self._instances}")
        else:
            self._client = httpx.Client(
                base_url=self.config.base_url,
                timeout=httpx.Timeout(self.config.timeout_seconds),
            )
            self._clients = None
            self._instances = [self.config.base_url]
            logger.info(f"DoclingClient initialized with single instance: {self.config.base_url}")

    def _get_next_instance(self) -> tuple[str, httpx.Client]:
        """Get the next instance using true round-robin via Redis shared counter."""
        if not self.config.use_load_balancing:
            return self.config.base_url, self._client

        num_instances = len(self._instances)

        # Use Redis atomic counter for true round-robin across all workers
        if self._redis:
            try:
                counter = self._redis.incr(self.config.redis_counter_key)
                instance_index = (counter - 1) % num_instances  # -1 because incr returns after increment
            except Exception as e:
                logger.warning(f"Redis counter failed, falling back to first instance: {e}")
                instance_index = 0
        else:
            # Fallback if Redis not available
            instance_index = 0

        instance_url = self._instances[instance_index]
        client = self._clients[instance_url]
        return instance_url, client

    def _get_client_for_url(self, url: str) -> httpx.Client:
        """Get client for a specific URL."""
        if self.config.use_load_balancing:
            return self._clients.get(url) or list(self._clients.values())[0]
        return self._client

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        """Close all HTTP clients."""
        if self.config.use_load_balancing and self._clients:
            for client in self._clients.values():
                client.close()
        elif hasattr(self, '_client') and self._client:
            self._client.close()
        # Note: Redis client doesn't need explicit close for connection pooling

    def _retry_request(self, method: str, url: str, client: Optional[httpx.Client] = None, **kwargs) -> httpx.Response:
        """Execute request with retry logic."""
        # Use provided client or get default
        if client is None:
            if self.config.use_load_balancing:
                client = list(self._clients.values())[0]
            else:
                client = self._client

        last_error = None
        delay = self.config.retry_delay_seconds

        for attempt in range(self.config.max_retries):
            try:
                response = client.request(method, url, **kwargs)
                return response

            except httpx.TimeoutException as e:
                last_error = e
                logger.warning(f"Request timeout (attempt {attempt + 1}/{self.config.max_retries}): {e}")

            except httpx.ConnectError as e:
                last_error = e
                logger.warning(f"Connection error (attempt {attempt + 1}/{self.config.max_retries}): {e}")

            except httpx.HTTPStatusError as e:
                # Don't retry on client errors (4xx)
                if 400 <= e.response.status_code < 500:
                    raise DoclingClientError(
                        f"Client error: {e.response.status_code}",
                        status_code=e.response.status_code,
                    )
                last_error = e
                logger.warning(f"HTTP error (attempt {attempt + 1}/{self.config.max_retries}): {e}")

            # Wait before retry (exponential backoff)
            if attempt < self.config.max_retries - 1:
                logger.info(f"Retrying in {delay:.1f}s...")
                time.sleep(delay)
                delay *= self.config.retry_backoff_multiplier

        # All retries exhausted
        raise DoclingServiceUnavailable(
            f"Service unavailable after {self.config.max_retries} attempts",
            details=str(last_error),
        )

    # =========================================================================
    # Health & Status
    # =========================================================================

    def health(self, instance_url: Optional[str] = None) -> Dict[str, Any]:
        """
        Check service health for a specific instance or first available.

        Args:
            instance_url: Specific instance to check, or None for first instance

        Returns:
            Dict with status, models_loaded, version, instance
        """
        try:
            if instance_url:
                client = self._get_client_for_url(instance_url)
            elif self.config.use_load_balancing:
                instance_url = self._instances[0]
                client = self._clients[instance_url]
            else:
                instance_url = self.config.base_url
                client = self._client

            response = self._retry_request("GET", "/health", client=client)
            response.raise_for_status()
            result = response.json()
            result["instance"] = instance_url
            return result
        except Exception as e:
            logger.error(f"Health check failed for {instance_url}: {e}")
            return {"status": "unhealthy", "models_loaded": False, "error": str(e), "instance": instance_url}

    def health_all(self) -> Dict[str, Dict[str, Any]]:
        """
        Check health of all instances.

        Returns:
            Dict mapping instance URL to health status
        """
        results = {}
        for instance_url in self._instances:
            results[instance_url] = self.health(instance_url)
        return results

    def is_healthy(self, instance_url: Optional[str] = None) -> bool:
        """Check if service is healthy and models are loaded."""
        health = self.health(instance_url)
        return health.get("status") == "healthy" and health.get("models_loaded", False)

    def all_instances_healthy(self) -> bool:
        """Check if ALL instances are healthy."""
        for instance_url in self._instances:
            if not self.is_healthy(instance_url):
                return False
        return True

    def healthy_instance_count(self) -> int:
        """Count how many instances are healthy."""
        count = 0
        for instance_url in self._instances:
            if self.is_healthy(instance_url):
                count += 1
        return count

    def stats(self, instance_url: Optional[str] = None) -> Dict[str, Any]:
        """Get service statistics for a specific instance."""
        if instance_url:
            client = self._get_client_for_url(instance_url)
        elif self.config.use_load_balancing:
            instance_url = self._instances[0]
            client = self._clients[instance_url]
        else:
            client = self._client

        response = self._retry_request("GET", "/stats", client=client)
        response.raise_for_status()
        result = response.json()
        result["instance"] = instance_url
        return result

    # =========================================================================
    # PDF Processing
    # =========================================================================

    def process_pdf(
        self,
        pdf_path: str | Path,
        extract_tables: bool = True,
        extract_figures: bool = True,
        extract_formula_pages: bool = True,
        page_start: Optional[int] = None,
        page_end: Optional[int] = None,
        image_scale: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Process a PDF file with the Docling service.

        Uses instance failover: if one instance fails with connection error,
        automatically tries the next instance. This handles crashed Docling
        instances gracefully without waiting for watchdog restart.

        Args:
            pdf_path: Path to the PDF file
            extract_tables: Whether to extract table images
            extract_figures: Whether to extract figure images
            extract_formula_pages: Whether to extract full-page images for pages with formulas
            page_start: Start page (1-indexed, optional)
            page_end: End page (1-indexed, optional)
            image_scale: Scale factor for extracted images (default 2.0, use 1.5 for large files)

        Returns:
            Dict with:
                - success: bool
                - document: DoclingDocument as JSON (if successful)
                - extractions: Dict with tables, figures, formula_pages lists
                - metadata: Processing metadata
                - error: Error message (if failed)

            Note: formula_pages contains FULL PAGE images, not cropped formula regions.
            Workers should draw numbered boxes on these pages for VLM processing.

        Raises:
            DoclingClientError: On client-side errors
            DoclingServiceUnavailable: If all instances are unavailable
            DoclingProcessingError: If processing fails
        """
        pdf_path = Path(pdf_path)

        if not pdf_path.exists():
            raise DoclingClientError(f"PDF file not found: {pdf_path}")

        # Build options once (reused across instance attempts)
        options = {
            "extract_tables": extract_tables,
            "extract_figures": extract_figures,
            "extract_formula_pages": extract_formula_pages,
        }
        if page_start is not None:
            options["page_start"] = page_start
        if page_end is not None:
            options["page_end"] = page_end
        if image_scale is not None:
            options["image_scale"] = image_scale

        # Instance failover: try each instance until one succeeds
        tried_instances = []
        last_error = None
        num_instances = len(self._instances) if self.config.use_load_balancing else 1

        for attempt in range(num_instances):
            # Get next instance via round-robin
            instance_url, client = self._get_next_instance()

            # Skip if we've already tried this instance (can happen with few instances)
            if instance_url in tried_instances:
                continue

            tried_instances.append(instance_url)
            logger.info(f"Processing {pdf_path.name} on instance: {instance_url} (attempt {attempt + 1}/{num_instances})")

            try:
                # Read file fresh for each attempt (file handle consumed after first use)
                with open(pdf_path, "rb") as f:
                    files = {"pdf_file": (pdf_path.name, f, "application/pdf")}
                    data = {"options": json.dumps(options)}

                    # Single request attempt (no retry on same instance)
                    response = client.request(
                        "POST",
                        "/process",
                        files=files,
                        data=data,
                    )

                # Parse response
                try:
                    result = response.json()
                except Exception as e:
                    raise DoclingProcessingError(
                        f"Failed to parse response: {e}",
                        status_code=response.status_code,
                    )

                # Check for processing error (this is a valid response, not connection failure)
                if not result.get("success", False):
                    raise DoclingProcessingError(
                        result.get("error", "Unknown processing error"),
                        details=result.get("details"),
                    )

                # Success! Add instance info and return
                result["_instance"] = instance_url
                result["_attempts"] = len(tried_instances)
                return result

            except httpx.ConnectError as e:
                # Instance is down - try next instance
                last_error = e
                logger.warning(f"Instance {instance_url} connection failed: {e}. Trying next instance...")
                continue

            except httpx.TimeoutException as e:
                # Timeout - could be overloaded or crashed mid-request, try next
                last_error = e
                logger.warning(f"Instance {instance_url} timeout: {e}. Trying next instance...")
                continue

            except ConnectionResetError as e:
                # Connection reset - instance crashed mid-processing, try next
                last_error = e
                logger.warning(f"Instance {instance_url} connection reset: {e}. Trying next instance...")
                continue

            except httpx.RemoteProtocolError as e:
                # Server disconnected mid-request (segfault/crash), try next instance
                last_error = e
                logger.warning(f"Instance {instance_url} disconnected mid-request: {e}. Trying next instance...")
                continue

            except httpx.HTTPStatusError as e:
                # HTTP error (4xx/5xx) - don't failover for client errors
                if 400 <= e.response.status_code < 500:
                    raise DoclingClientError(
                        f"Client error: {e.response.status_code}",
                        status_code=e.response.status_code,
                    )
                # Server error (5xx) - try next instance
                last_error = e
                logger.warning(f"Instance {instance_url} server error: {e}. Trying next instance...")
                continue

        # All instances failed
        raise DoclingServiceUnavailable(
            f"All {len(tried_instances)} Docling instances failed. Tried: {tried_instances}",
            details=str(last_error),
        )

    def process_pdf_layout_only(self, pdf_path: str | Path) -> Dict[str, Any]:
        """
        Process a PDF for layout detection only (no image extraction).

        Faster than process_pdf when you only need the document structure.
        Uses instance failover like process_pdf.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            Dict with document and metadata (no extractions)
        """
        pdf_path = Path(pdf_path)

        if not pdf_path.exists():
            raise DoclingClientError(f"PDF file not found: {pdf_path}")

        # Instance failover: try each instance until one succeeds
        tried_instances = []
        last_error = None
        num_instances = len(self._instances) if self.config.use_load_balancing else 1

        for attempt in range(num_instances):
            instance_url, client = self._get_next_instance()

            if instance_url in tried_instances:
                continue

            tried_instances.append(instance_url)
            logger.info(f"Processing (layout-only) {pdf_path.name} on instance: {instance_url} (attempt {attempt + 1}/{num_instances})")

            try:
                with open(pdf_path, "rb") as f:
                    files = {"pdf_file": (pdf_path.name, f, "application/pdf")}

                    response = client.request(
                        "POST",
                        "/process/layout-only",
                        files=files,
                    )

                result = response.json()

                if not result.get("success", False):
                    raise DoclingProcessingError(
                        result.get("error", "Unknown processing error"),
                    )

                result["_instance"] = instance_url
                result["_attempts"] = len(tried_instances)
                return result

            except (httpx.ConnectError, httpx.TimeoutException, ConnectionResetError, httpx.RemoteProtocolError) as e:
                last_error = e
                logger.warning(f"Instance {instance_url} failed: {e}. Trying next instance...")
                continue

            except httpx.HTTPStatusError as e:
                if 400 <= e.response.status_code < 500:
                    raise DoclingClientError(
                        f"Client error: {e.response.status_code}",
                        status_code=e.response.status_code,
                    )
                last_error = e
                logger.warning(f"Instance {instance_url} server error: {e}. Trying next instance...")
                continue

        raise DoclingServiceUnavailable(
            f"All {len(tried_instances)} Docling instances failed. Tried: {tried_instances}",
            details=str(last_error),
        )

    # =========================================================================
    # Convenience Methods
    # =========================================================================

    def wait_for_ready(self, timeout_seconds: float = 120.0, poll_interval: float = 5.0, require_all: bool = True) -> bool:
        """
        Wait for the service(s) to become ready.

        Args:
            timeout_seconds: Maximum time to wait
            poll_interval: Time between health checks
            require_all: If True, wait for ALL instances to be ready (default for load balancing)

        Returns:
            True if service(s) ready, False if timeout
        """
        start_time = time.time()

        while time.time() - start_time < timeout_seconds:
            if require_all and self.config.use_load_balancing:
                healthy_count = self.healthy_instance_count()
                total_count = len(self._instances)
                if healthy_count == total_count:
                    logger.info(f"All {total_count} Docling instances are ready")
                    return True
                logger.info(f"Waiting for Docling instances... {healthy_count}/{total_count} ready ({int(time.time() - start_time)}s)")
            else:
                if self.is_healthy():
                    logger.info("Docling service is ready")
                    return True
                logger.info(f"Waiting for Docling service... ({int(time.time() - start_time)}s)")

            time.sleep(poll_interval)

        logger.error(f"Timeout waiting for Docling service after {timeout_seconds}s")
        return False


# =============================================================================
# Convenience Functions
# =============================================================================

def create_client(
    base_url: str = "http://localhost:8001",
    timeout_seconds: float = 300.0,
) -> DoclingClient:
    """Create a Docling client with custom settings."""
    config = DoclingClientConfig(
        base_url=base_url,
        timeout_seconds=timeout_seconds,
    )
    return DoclingClient(config)


def process_pdf(
    pdf_path: str | Path,
    base_url: str = "http://localhost:8001",
    **kwargs,
) -> Dict[str, Any]:
    """
    Convenience function to process a single PDF.

    Args:
        pdf_path: Path to the PDF file
        base_url: Docling service URL
        **kwargs: Additional arguments for process_pdf

    Returns:
        Processing result dict
    """
    with create_client(base_url) as client:
        return client.process_pdf(pdf_path, **kwargs)
