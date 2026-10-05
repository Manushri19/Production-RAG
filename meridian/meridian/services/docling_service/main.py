"""
Docling Service - FastAPI Application

HTTP service that wraps Docling for PDF processing.
GPU models are loaded once at startup and shared across all requests.
"""

import faulthandler
import json
import logging
import shutil
import sys
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Enable faulthandler to dump tracebacks on crashes (SIGSEGV, SIGABRT, etc.)
# This helps debug C-level crashes in dependencies like PyMuPDF, EasyOCR
faulthandler.enable(file=sys.stderr, all_threads=True)

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import JSONResponse

from meridian.config import SERVICE_HOST, SERVICE_PORT, LOG_LEVEL, ensure_temp_dir
from .processor import init_processor, get_processor, shutdown_processor
from .extractors import extract_all
from .models import (
    ProcessOptions,
    ProcessResponse,
    LayoutOnlyResponse,
    HealthResponse,
    StatsResponse,
    ErrorResponse,
    Extractions,
    ProcessingMetadata,
    TableExtraction,
    FigureExtraction,
    FormulaPageExtraction,
    BoundingBox,
    BatchProcessResponse,
    BatchDocumentResult,
)

# Configure logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Service statistics
_stats = {
    "requests_processed": 0,
    "total_processing_time_ms": 0,
}

# Thread pool for CPU-bound PDF processing (to avoid blocking async)
_executor = ThreadPoolExecutor(max_workers=4)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    Load GPU models on startup, cleanup on shutdown.
    """
    logger.info("Starting Docling service...")
    ensure_temp_dir()

    # Initialize processor (loads GPU models)
    try:
        init_processor()
        logger.info("Docling service ready")
    except Exception as e:
        logger.error(f"Failed to initialize processor: {e}")
        raise

    yield

    # Shutdown
    logger.info("Shutting down Docling service...")
    shutdown_processor()
    _executor.shutdown(wait=True)
    logger.info("Docling service stopped")


# Create FastAPI app
app = FastAPI(
    title="Docling Service",
    description="PDF processing service with GPU-accelerated layout detection and content extraction",
    version="1.0.0",
    lifespan=lifespan,
)


# =============================================================================
# Health & Status Endpoints
# =============================================================================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    try:
        processor = get_processor()
        return HealthResponse(
            status="healthy" if processor.is_ready else "unhealthy",
            models_loaded=processor.is_ready,
            version="1.0.0",
        )
    except Exception as e:
        return HealthResponse(
            status="unhealthy",
            models_loaded=False,
            version="1.0.0",
        )


@app.get("/stats", response_model=StatsResponse)
async def get_stats():
    """Get service statistics."""
    try:
        processor = get_processor()
        proc_stats = processor.get_stats()

        avg_time = 0.0
        if _stats["requests_processed"] > 0:
            avg_time = _stats["total_processing_time_ms"] / _stats["requests_processed"]

        return StatsResponse(
            models_loaded=proc_stats["models_loaded"],
            ocr_enabled=proc_stats["ocr_enabled"],
            table_structure_enabled=proc_stats["table_structure_enabled"],
            formula_enrichment_enabled=proc_stats["formula_enrichment_enabled"],
            picture_extraction_enabled=proc_stats["picture_extraction_enabled"],
            accelerator_device=proc_stats["accelerator_device"],
            requests_processed=_stats["requests_processed"],
            total_processing_time_ms=_stats["total_processing_time_ms"],
            average_processing_time_ms=avg_time,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Processing Endpoints
# =============================================================================

def _process_pdf_sync(
    pdf_path: Path,
    options: ProcessOptions,
    extract_images: bool = True,
) -> dict:
    """
    Synchronous PDF processing (runs in thread pool).

    Returns dict with document, extractions, metadata, or error.
    """
    start_time = time.time()
    processor = get_processor()

    # Process PDF with Docling
    doc, backend, error = processor.process_pdf(pdf_path)

    if error:
        return {"error": error}

    # Get page count
    try:
        page_count = len(list(doc.pages))
    except:
        page_count = 0

    # Convert document to JSON
    try:
        doc_json = json.loads(doc.model_dump_json())
    except Exception as e:
        logger.warning(f"Failed to serialize document: {e}")
        doc_json = None

    result = {
        "document": doc_json,
        "extractions": None,
        "metadata": {
            "filename": pdf_path.name,
            "pages": page_count,
            "processing_time_ms": 0,
            "tables_count": 0,
            "figures_count": 0,
            "formula_pages_count": 0,
        }
    }

    # Extract images if requested
    if extract_images and backend is not None:
        extractions = extract_all(
            doc,
            backend,
            extract_tables_flag=options.extract_tables,
            extract_figures_flag=options.extract_figures,
            extract_formula_pages_flag=options.extract_formula_pages,
            scale=options.image_scale,
        )

        result["extractions"] = extractions
        result["metadata"]["tables_count"] = len(extractions["tables"])
        result["metadata"]["figures_count"] = len(extractions["figures"])
        result["metadata"]["formula_pages_count"] = len(extractions["formula_pages"])

    # Record processing time
    processing_time_ms = int((time.time() - start_time) * 1000)
    result["metadata"]["processing_time_ms"] = processing_time_ms

    return result


@app.post("/process", response_model=ProcessResponse)
async def process_pdf(
    pdf_file: UploadFile = File(..., description="PDF file to process"),
    options: str = Form(default="{}", description="Processing options as JSON"),
):
    """
    Process a PDF file with layout detection and content extraction.

    Returns the full DoclingDocument plus extracted images (tables, figures, formulas)
    as base64-encoded PNGs.
    """
    global _stats

    # Parse options
    try:
        options_dict = json.loads(options)
        proc_options = ProcessOptions(**options_dict)
    except Exception as e:
        return ProcessResponse(
            success=False,
            error=f"Invalid options: {str(e)}",
        )

    # Save uploaded file to temp location
    temp_dir = ensure_temp_dir()
    temp_path = temp_dir / f"upload_{int(time.time() * 1000)}_{pdf_file.filename}"

    try:
        with open(temp_path, "wb") as f:
            content = await pdf_file.read()
            f.write(content)

        # Process in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            _executor,
            _process_pdf_sync,
            temp_path,
            proc_options,
            True,  # extract_images
        )

        # Check for error
        if "error" in result and result.get("document") is None:
            return ProcessResponse(
                success=False,
                error=result["error"],
            )

        # Update stats
        _stats["requests_processed"] += 1
        _stats["total_processing_time_ms"] += result["metadata"]["processing_time_ms"]

        # Build response
        extractions = None
        if result.get("extractions"):
            ext = result["extractions"]
            extractions = Extractions(
                tables=[
                    TableExtraction(
                        id=t["id"],
                        page=t["page"],
                        bbox=BoundingBox(**t["bbox"]),
                        image_base64=t["image_base64"],
                        source=t["source"],
                        title=t.get("title"),
                        reason=t.get("reason"),
                        structure=t.get("structure"),
                    )
                    for t in ext["tables"]
                ],
                figures=[
                    FigureExtraction(
                        id=f["id"],
                        page=f["page"],
                        bbox=BoundingBox(**f["bbox"]),
                        image_base64=f["image_base64"],
                        context_above=f.get("context_above", ""),
                        context_below=f.get("context_below", ""),
                    )
                    for f in ext["figures"]
                ],
                # Formula pages are FULL PAGE images, not cropped regions
                # Workers will draw numbered boxes for VLM processing
                formula_pages=[
                    FormulaPageExtraction(
                        page=fp["page"],
                        image_base64=fp["image_base64"],
                    )
                    for fp in ext["formula_pages"]
                ],
            )

        return ProcessResponse(
            success=True,
            document=result["document"],
            extractions=extractions,
            metadata=ProcessingMetadata(**result["metadata"]),
        )

    except Exception as e:
        logger.exception(f"Error processing PDF: {e}")
        return ProcessResponse(
            success=False,
            error=str(e),
        )

    finally:
        # Cleanup temp file
        if temp_path.exists():
            try:
                temp_path.unlink()
            except:
                pass


@app.post("/process/layout-only", response_model=LayoutOnlyResponse)
async def process_layout_only(
    pdf_file: UploadFile = File(..., description="PDF file to process"),
):
    """
    Process a PDF file for layout detection only (no image extraction).

    Faster than /process when you only need the document structure.
    """
    global _stats

    # Save uploaded file to temp location
    temp_dir = ensure_temp_dir()
    temp_path = temp_dir / f"upload_{int(time.time() * 1000)}_{pdf_file.filename}"

    try:
        with open(temp_path, "wb") as f:
            content = await pdf_file.read()
            f.write(content)

        # Process in thread pool
        loop = asyncio.get_event_loop()
        proc_options = ProcessOptions()  # Default options
        result = await loop.run_in_executor(
            _executor,
            _process_pdf_sync,
            temp_path,
            proc_options,
            False,  # extract_images = False
        )

        # Check for error
        if "error" in result and result.get("document") is None:
            return LayoutOnlyResponse(
                success=False,
                error=result["error"],
            )

        # Update stats
        _stats["requests_processed"] += 1
        _stats["total_processing_time_ms"] += result["metadata"]["processing_time_ms"]

        return LayoutOnlyResponse(
            success=True,
            document=result["document"],
            metadata=ProcessingMetadata(**result["metadata"]),
        )

    except Exception as e:
        logger.exception(f"Error processing PDF: {e}")
        return LayoutOnlyResponse(
            success=False,
            error=str(e),
        )

    finally:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except:
                pass


# =============================================================================
# Batch Processing Endpoint
# =============================================================================

def _build_extractions_response(ext: dict) -> Extractions:
    """Build Extractions response from extraction dict."""
    return Extractions(
        tables=[
            TableExtraction(
                id=t["id"],
                page=t["page"],
                bbox=BoundingBox(**t["bbox"]),
                image_base64=t["image_base64"],
                source=t["source"],
                title=t.get("title"),
                reason=t.get("reason"),
                structure=t.get("structure"),
            )
            for t in ext["tables"]
        ],
        figures=[
            FigureExtraction(
                id=f["id"],
                page=f["page"],
                bbox=BoundingBox(**f["bbox"]),
                image_base64=f["image_base64"],
                context_above=f.get("context_above", ""),
                context_below=f.get("context_below", ""),
            )
            for f in ext["figures"]
        ],
        formula_pages=[
            FormulaPageExtraction(
                page=fp["page"],
                image_base64=fp["image_base64"],
            )
            for fp in ext["formula_pages"]
        ],
    )


def _process_batch_sync(
    pdf_paths: list[Path],
    filenames: list[str],
    options: ProcessOptions,
) -> dict:
    """
    Synchronous batch PDF processing using Docling's convert_all().

    This leverages Docling's internal parallelism (when env vars are set):
    - DOCLING_PERF_DOC_BATCH_SIZE > 1
    - DOCLING_PERF_DOC_BATCH_CONCURRENCY > 1
    """
    start_time = time.time()
    processor = get_processor()

    # Process all PDFs in batch
    batch_results = processor.process_pdfs_batch(pdf_paths)

    results = []
    successful_count = 0

    for idx, (doc, backend, error) in enumerate(batch_results):
        pdf_path = pdf_paths[idx]
        filename = filenames[idx]

        if error:
            results.append({
                "filename": filename,
                "success": False,
                "error": error,
            })
            continue

        # Get page count
        try:
            page_count = len(list(doc.pages))
        except:
            page_count = 0

        # Convert document to JSON
        try:
            doc_json = json.loads(doc.model_dump_json())
        except Exception as e:
            logger.warning(f"Failed to serialize document {filename}: {e}")
            doc_json = None

        result_data = {
            "filename": filename,
            "success": True,
            "document": doc_json,
            "extractions": None,
            "metadata": {
                "filename": filename,
                "pages": page_count,
                "processing_time_ms": 0,  # Will set at end
                "tables_count": 0,
                "figures_count": 0,
                "formula_pages_count": 0,
            }
        }

        # Extract images
        if backend is not None:
            extractions = extract_all(
                doc,
                backend,
                extract_tables_flag=options.extract_tables,
                extract_figures_flag=options.extract_figures,
                extract_formula_pages_flag=options.extract_formula_pages,
                scale=options.image_scale,
            )
            result_data["extractions"] = extractions
            result_data["metadata"]["tables_count"] = len(extractions["tables"])
            result_data["metadata"]["figures_count"] = len(extractions["figures"])
            result_data["metadata"]["formula_pages_count"] = len(extractions["formula_pages"])

        results.append(result_data)
        successful_count += 1

    total_time_ms = int((time.time() - start_time) * 1000)

    # Set per-document processing time (averaged for batch)
    if successful_count > 0:
        avg_time = total_time_ms // successful_count
        for r in results:
            if r.get("success"):
                r["metadata"]["processing_time_ms"] = avg_time

    return {
        "results": results,
        "total_processing_time_ms": total_time_ms,
        "successful_count": successful_count,
        "failed_count": len(pdf_paths) - successful_count,
    }


@app.post("/process-batch", response_model=BatchProcessResponse)
async def process_batch(
    pdf_files: list[UploadFile] = File(..., description="PDF files to process"),
    options: str = Form(default="{}", description="Processing options as JSON"),
):
    """
    Process multiple PDF files in a single batch.

    IMPORTANT: This endpoint uses Docling's internal parallelism when:
    - DOCLING_PERF_DOC_BATCH_SIZE > 1
    - DOCLING_PERF_DOC_BATCH_CONCURRENCY > 1

    Set these environment variables before starting the service to enable
    true parallel document processing.

    Example:
        DOCLING_PERF_DOC_BATCH_SIZE=4 DOCLING_PERF_DOC_BATCH_CONCURRENCY=4 \
        python -m uvicorn meridian.services.docling_service.main:app --port 8001
    """
    global _stats

    if not pdf_files:
        return BatchProcessResponse(
            success=False,
            total_documents=0,
            successful_count=0,
            failed_count=0,
            total_processing_time_ms=0,
            results=[],
        )

    # Parse options
    try:
        options_dict = json.loads(options)
        proc_options = ProcessOptions(**options_dict)
    except Exception as e:
        return BatchProcessResponse(
            success=False,
            total_documents=len(pdf_files),
            successful_count=0,
            failed_count=len(pdf_files),
            total_processing_time_ms=0,
            results=[
                BatchDocumentResult(
                    filename=f.filename or "unknown",
                    success=False,
                    error=f"Invalid options: {str(e)}",
                )
                for f in pdf_files
            ],
        )

    # Save all uploaded files to temp location
    temp_dir = ensure_temp_dir()
    temp_paths = []
    filenames = []

    try:
        for pdf_file in pdf_files:
            temp_path = temp_dir / f"batch_{int(time.time() * 1000)}_{pdf_file.filename}"
            content = await pdf_file.read()
            with open(temp_path, "wb") as f:
                f.write(content)
            temp_paths.append(temp_path)
            filenames.append(pdf_file.filename or "unknown")

        # Process batch in thread pool
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            _executor,
            _process_batch_sync,
            temp_paths,
            filenames,
            proc_options,
        )

        # Update stats
        _stats["requests_processed"] += result["successful_count"]
        _stats["total_processing_time_ms"] += result["total_processing_time_ms"]

        # Build response
        batch_results = []
        for r in result["results"]:
            extractions = None
            if r.get("extractions"):
                extractions = _build_extractions_response(r["extractions"])

            batch_results.append(BatchDocumentResult(
                filename=r["filename"],
                success=r.get("success", False),
                document=r.get("document"),
                extractions=extractions,
                metadata=ProcessingMetadata(**r["metadata"]) if r.get("metadata") else None,
                error=r.get("error"),
            ))

        return BatchProcessResponse(
            success=result["failed_count"] == 0,
            total_documents=len(pdf_files),
            successful_count=result["successful_count"],
            failed_count=result["failed_count"],
            total_processing_time_ms=result["total_processing_time_ms"],
            results=batch_results,
        )

    except Exception as e:
        logger.exception(f"Error in batch processing: {e}")
        return BatchProcessResponse(
            success=False,
            total_documents=len(pdf_files),
            successful_count=0,
            failed_count=len(pdf_files),
            total_processing_time_ms=0,
            results=[
                BatchDocumentResult(
                    filename=fn,
                    success=False,
                    error=str(e),
                )
                for fn in filenames
            ],
        )

    finally:
        # Cleanup temp files
        for temp_path in temp_paths:
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except:
                    pass


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Run the service with uvicorn."""
    import uvicorn

    uvicorn.run(
        "meridian.services.docling_service.main:app",
        host=SERVICE_HOST,
        port=SERVICE_PORT,
        reload=False,  # Disable reload to avoid reloading GPU models
        workers=1,  # Single worker to share GPU models
    )


if __name__ == "__main__":
    main()
