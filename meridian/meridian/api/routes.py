"""
Meridian API Routes.

POST /v1/parse              Upload PDF -> returns JSON (sync)
POST /v1/parse/async        Upload PDF -> returns job_id
GET  /v1/jobs/{job_id}      Get status + results
GET  /v1/health             Service health
"""

import asyncio
import tempfile
import uuid
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from typing import Dict, Any, Optional
from urllib.parse import urlparse

import httpx
from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel

from meridian.output import format_result_json

_executor = ThreadPoolExecutor(max_workers=2)

router = APIRouter()

# In-memory job store (replace with Redis for production)
_jobs: Dict[str, Dict[str, Any]] = {}


@router.post("/parse")
async def parse_sync(file: UploadFile = File(...)):
    """
    Upload a PDF and get structured JSON back (synchronous).

    Blocks until processing is complete.
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "File must be a PDF")

    # Save uploaded file to temp location
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        from meridian.workers.document_worker import process_single_document

        doc_id = Path(file.filename).stem
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            _executor,
            partial(
                process_single_document,
                doc_id=doc_id,
                pdf_path=tmp_path,
                store_embeddings=False,
            ),
        )

        if not result.get("success"):
            raise HTTPException(500, f"Processing failed: {result.get('error')}")

        return format_result_json(result)

    finally:
        Path(tmp_path).unlink(missing_ok=True)


@router.post("/parse/async")
async def parse_async(file: UploadFile = File(...)):
    """
    Upload a PDF and get a job_id back. Poll /v1/jobs/{job_id} for results.
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "File must be a PDF")

    # Save uploaded file
    job_id = str(uuid.uuid4())[:8]
    tmp_dir = Path(tempfile.gettempdir()) / "meridian_jobs"
    tmp_dir.mkdir(exist_ok=True)
    tmp_path = tmp_dir / f"{job_id}.pdf"

    content = await file.read()
    tmp_path.write_bytes(content)

    _jobs[job_id] = {
        "status": "queued",
        "filename": file.filename,
        "pdf_path": str(tmp_path),
        "result": None,
    }

    # Submit to Celery
    try:
        import meridian.workers.celery_app  # noqa: F401 - ensures app is configured
        from meridian.workers.tasks import process_document

        doc_id = Path(file.filename).stem
        task = process_document.delay(
            doc_id=doc_id,
            pdf_path=str(tmp_path),
            store_embeddings=False,
        )
        _jobs[job_id]["celery_task_id"] = task.id
        _jobs[job_id]["status"] = "processing"
    except Exception as e:
        _jobs[job_id]["status"] = "error"
        _jobs[job_id]["error"] = str(e)

    return {"job_id": job_id, "status": _jobs[job_id]["status"]}


@router.get("/jobs/{job_id}")
async def get_job(job_id: str):
    """Get the status and results of an async job."""
    if job_id not in _jobs:
        raise HTTPException(404, f"Job {job_id} not found")

    job = _jobs[job_id]

    # Check Celery task status if processing
    if job["status"] == "processing" and job.get("celery_task_id"):
        try:
            from meridian.workers.celery_app import app as celery_app

            task_result = celery_app.AsyncResult(job["celery_task_id"])
            if task_result.ready():
                result = task_result.result
                if isinstance(result, dict) and result.get("success"):
                    job["status"] = "completed"
                    job["result"] = format_result_json(result)
                else:
                    job["status"] = "failed"
                    job["error"] = result.get("error", "Unknown error") if isinstance(result, dict) else str(result)
        except Exception:
            pass

    response = {"job_id": job_id, "status": job["status"]}
    if job.get("result"):
        response["result"] = job["result"]
    if job.get("error"):
        response["error"] = job["error"]

    return response


@router.get("/health")
async def health():
    """Service health check."""
    return {"status": "ok", "service": "meridian-api"}
