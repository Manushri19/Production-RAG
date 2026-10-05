"""
Service Management - Start, stop, and monitor Meridian services.

Manages: Docling instances, vLLM, Celery workers, Watchdog.
Reads configuration from .env / environment variables.
"""

import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

STATE_DIR = Path.home() / ".meridian" / "pids"
LOG_DIR = Path.home() / ".meridian" / "logs"


def _ensure_dirs():
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)


def _load_env():
    """Load .env file if present."""
    env_path = Path(".env")
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                os.environ.setdefault(key.strip(), value.strip())


def _get_docling_instances():
    """Get list of Docling instance URLs from config."""
    from meridian.config import DOCLING_INSTANCES
    return DOCLING_INSTANCES


def _write_pid(name: str, pid: int):
    _ensure_dirs()
    (STATE_DIR / f"{name}.pid").write_text(str(pid))


def _read_pid(name: str) -> Optional[int]:
    pid_file = STATE_DIR / f"{name}.pid"
    if pid_file.exists():
        try:
            pid = int(pid_file.read_text().strip())
            # Check if process is alive
            os.kill(pid, 0)
            return pid
        except (ValueError, ProcessLookupError, PermissionError):
            pid_file.unlink(missing_ok=True)
    return None


def _kill_pid(name: str) -> bool:
    pid = _read_pid(name)
    if pid:
        try:
            os.kill(pid, signal.SIGTERM)
            # Wait for graceful shutdown
            for _ in range(10):
                try:
                    os.kill(pid, 0)
                    time.sleep(0.5)
                except ProcessLookupError:
                    break
            else:
                # Force kill
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        except ProcessLookupError:
            pass
        (STATE_DIR / f"{name}.pid").unlink(missing_ok=True)
        return True
    return False


def _check_health(url: str, timeout: float = 3.0) -> bool:
    try:
        resp = httpx.get(url, timeout=timeout)
        return resp.status_code == 200
    except Exception:
        return False


def _start_docling_instances():
    """Start Docling service instances."""
    instances = _get_docling_instances()
    started = 0

    for i, url in enumerate(instances):
        name = f"docling_{i}"
        if _read_pid(name):
            print(f"  Docling instance {i} already running")
            started += 1
            continue

        # Extract port from URL
        port = url.rstrip("/").split(":")[-1]

        log_file = LOG_DIR / f"docling_{port}.log"
        proc = subprocess.Popen(
            [
                sys.executable, "-m", "uvicorn",
                "meridian.services.docling_service.main:app",
                "--host", "0.0.0.0",
                "--port", str(port),
            ],
            stdout=open(log_file, "w"),
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        _write_pid(name, proc.pid)
        started += 1
        print(f"  Docling instance {i} started on port {port} (PID: {proc.pid})")

    return started


def _start_vllm():
    """Start vLLM as a Docker container."""
    # Check if already running
    result = subprocess.run(
        ["docker", "ps", "-q", "-f", "name=vllm"],
        capture_output=True, text=True,
    )
    if result.stdout.strip():
        print("  vLLM container already running")
        return

    # Remove stopped container if exists
    subprocess.run(
        ["docker", "rm", "-f", "vllm"],
        capture_output=True, text=True,
    )

    model = os.environ.get("VLLM_MODEL", "Qwen/Qwen3-VL-8B-Instruct")
    gpu_util = os.environ.get("VLLM_GPU_UTILIZATION", "0.55")

    print(f"  Starting vLLM ({model}, gpu_util={gpu_util})...")
    proc = subprocess.run(
        [
            "docker", "run", "-d", "--name", "vllm",
            "--gpus", "all",
            "-p", "8000:8000",
            "-v", f"{Path.home()}/.cache/huggingface:/root/.cache/huggingface",
            "vllm/vllm-openai:latest",
            "--model", model,
            "--quantization", "fp8",
            "--gpu-memory-utilization", gpu_util,
            "--max-num-seqs", "128",
            "--max-model-len", "8192",
            "--enable-chunked-prefill",
            "--enable-prefix-caching",
        ],
        capture_output=True, text=True,
    )
    if proc.returncode == 0:
        print(f"  vLLM container started (loading model, may take 1-2 min)")
    else:
        print(f"  vLLM start failed: {proc.stderr.strip()}")


def _stop_vllm():
    """Stop vLLM Docker container."""
    result = subprocess.run(
        ["docker", "stop", "vllm"],
        capture_output=True, text=True,
    )
    if result.returncode == 0:
        subprocess.run(["docker", "rm", "vllm"], capture_output=True, text=True)
        return True
    return False


def _wait_for_vllm(timeout: int = 180):
    """Wait for vLLM to become healthy."""
    vllm_url = os.environ.get("VLLM_API_URL", "http://localhost:8000/v1/chat/completions")
    vllm_base = vllm_url.rsplit("/v1", 1)[0]
    health_url = f"{vllm_base}/v1/models"

    print(f"\nWaiting for vLLM to load model...")
    start = time.time()
    while time.time() - start < timeout:
        if _check_health(health_url):
            print(f"  vLLM ready!")
            return True
        elapsed = int(time.time() - start)
        print(f"  Loading... ({elapsed}s)", end="\r")
        time.sleep(5)

    print(f"\n  vLLM timeout after {timeout}s. Check: docker logs vllm")
    return False


def _start_worker():
    """Start Celery worker."""
    if _read_pid("celery_worker"):
        print("  Celery worker already running")
        return

    concurrency = os.environ.get("CELERY_CONCURRENCY", "3")
    log_file = LOG_DIR / "celery_worker.log"

    proc = subprocess.Popen(
        [
            sys.executable, "-m", "celery",
            "-A", "meridian.workers.celery_app",
            "worker",
            "--loglevel=info",
            f"--concurrency={concurrency}",
        ],
        stdout=open(log_file, "w"),
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    _write_pid("celery_worker", proc.pid)
    print(f"  Celery worker started (PID: {proc.pid}, concurrency: {concurrency})")


def _wait_for_docling(timeout: int = 120):
    """Wait for Docling instances to become healthy."""
    instances = _get_docling_instances()
    print(f"\nWaiting for {len(instances)} Docling instance(s) to load models...")

    start = time.time()
    while time.time() - start < timeout:
        healthy = sum(1 for url in instances if _check_health(f"{url}/health"))
        if healthy == len(instances):
            print(f"  All {healthy} instance(s) ready!")
            return True
        elapsed = int(time.time() - start)
        print(f"  {healthy}/{len(instances)} ready ({elapsed}s)...", end="\r")
        time.sleep(5)

    healthy = sum(1 for url in instances if _check_health(f"{url}/health"))
    print(f"\n  Timeout: {healthy}/{len(instances)} instances ready")
    return healthy > 0


def cmd_start(args):
    """Start services."""
    _load_env()
    _ensure_dirs()

    service = getattr(args, "service", None)

    print("Starting Meridian services...")
    print()

    # Check Redis
    try:
        import redis
        r = redis.from_url(os.environ.get("REDIS_URL", "redis://localhost:6379/0"))
        r.ping()
        print("  Redis: connected")
    except Exception:
        print("  Redis: NOT AVAILABLE - please start Redis first")
        if not service:
            sys.exit(1)

    if not service or service == "vllm":
        _start_vllm()

    if not service or service == "docling":
        count = _start_docling_instances()
        _wait_for_docling()

    if not service or service == "vllm":
        _wait_for_vllm()

    if not service or service == "worker":
        _start_worker()

    print()
    print("Services started. Run 'meridian status' to check health.")


def cmd_stop(args):
    """Stop all services."""
    _load_env()
    print("Stopping Meridian services...")

    # Stop worker
    if _kill_pid("celery_worker"):
        print("  Celery worker stopped")

    # Stop Docling instances
    instances = _get_docling_instances()
    for i in range(len(instances)):
        name = f"docling_{i}"
        if _kill_pid(name):
            print(f"  Docling instance {i} stopped")

    # Stop vLLM
    if _stop_vllm():
        print("  vLLM container stopped")

    # Stop watchdog
    if _kill_pid("watchdog"):
        print("  Watchdog stopped")

    print()
    print("All services stopped.")


def cmd_status(args):
    """Check service status."""
    _load_env()
    instances = _get_docling_instances()

    print()
    print("Meridian Service Status")
    print("=" * 50)

    # Redis
    try:
        import redis
        r = redis.from_url(os.environ.get("REDIS_URL", "redis://localhost:6379/0"))
        r.ping()
        print(f"  Redis:          OK")
    except Exception:
        print(f"  Redis:          DOWN")

    # Docling instances
    healthy = 0
    for i, url in enumerate(instances):
        is_healthy = _check_health(f"{url}/health")
        port = url.rstrip("/").split(":")[-1]
        status = "OK" if is_healthy else "DOWN"
        pid = _read_pid(f"docling_{i}")
        pid_str = f" (PID: {pid})" if pid else ""
        print(f"  Docling [{port}]:   {status}{pid_str}")
        if is_healthy:
            healthy += 1

    print(f"  Docling total:    {healthy}/{len(instances)} healthy")

    # vLLM
    vllm_url = os.environ.get("VLLM_API_URL", "http://localhost:8000/v1/chat/completions")
    vllm_base = vllm_url.rsplit("/v1", 1)[0]
    vllm_ok = _check_health(f"{vllm_base}/v1/models")
    print(f"  vLLM:            {'OK' if vllm_ok else 'DOWN'}")

    # Ollama
    ollama_url = os.environ.get("OLLAMA_URL", "http://localhost:11434")
    ollama_ok = _check_health(ollama_url)
    print(f"  Ollama:          {'OK' if ollama_ok else 'DOWN'}")

    # Qdrant
    qdrant_host = os.environ.get("QDRANT_HOST", "localhost")
    qdrant_port = os.environ.get("QDRANT_PORT", "6333")
    qdrant_ok = _check_health(f"http://{qdrant_host}:{qdrant_port}/collections")
    print(f"  Qdrant:          {'OK' if qdrant_ok else 'DOWN'}")

    # Celery worker
    worker_pid = _read_pid("celery_worker")
    print(f"  Celery worker:   {'OK (PID: ' + str(worker_pid) + ')' if worker_pid else 'DOWN'}")

    print()
    print("=" * 50)

    # Overall
    all_ok = healthy > 0 and vllm_ok and ollama_ok
    if all_ok:
        print("Ready to process documents.")
    else:
        print("Some services are down. Run 'meridian start' to start them.")
    print()
