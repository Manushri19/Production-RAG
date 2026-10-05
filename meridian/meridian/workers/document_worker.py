"""
Document Worker - Main Processing Logic

Orchestrates the full document processing pipeline:
1. Call Docling service (HTTP)
2. Annotate formula pages (PIL)
3. Call VLM service (HTTP)
4. Build chunks (CPU)
5. Generate embeddings (HTTP)
6. Store in Qdrant (HTTP)

CRITICAL: NO DOCLING IMPORTS (except docling_core.types for data types)
"""

import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional

import asyncio


def _run_async(coro):
    """Run an async coroutine, safe to call from threads with a running event loop."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        # We're in a thread but there's an event loop in the main thread.
        # Create a fresh loop for this thread.
        new_loop = asyncio.new_event_loop()
        try:
            return new_loop.run_until_complete(coro)
        finally:
            new_loop.close()
    else:
        return asyncio.run(coro)

from meridian.clients.docling_client import DoclingClient
from meridian.clients.vlm_backend import get_process_vlm_tasks_async
from meridian.clients.embedding_client_async import AsyncEmbeddingClient
from meridian.workers.box_annotator import annotate_formula_pages, get_formula_pages_from_doc
from meridian.workers.chunker import UnifiedChunker

# Resolved once at import: 'vllm' (default) or 'bedrock', selected by VLM_BACKEND.
process_vlm_tasks_async = get_process_vlm_tasks_async()

logger = logging.getLogger(__name__)


def get_image_scale_for_file(file_path: str) -> float:
    """
    Determine image scale based on file size.

    Larger files use lower scale for faster processing.

    Returns:
        Scale factor (0.75 to 2.0)
    """
    try:
        file_size_mb = Path(file_path).stat().st_size / (1024 * 1024)

        if file_size_mb < 30:
            return 2.0      # High quality (default)
        elif file_size_mb < 100:
            return 1.5      # Medium-high
        elif file_size_mb < 200:
            return 1.0      # Medium
        else:
            return 0.75     # Lower (for very large scanned docs)
    except Exception:
        return 2.0  # Default on error


def process_single_document(
    doc_id: str,
    pdf_path: str,
    collection: Optional[str] = None,
    extract_tables: bool = True,
    extract_figures: bool = True,
    extract_formulas: bool = True,
    store_embeddings: bool = True,
    image_scale: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Process a single document end-to-end.

    This is the main worker function - NO DOCLING IMPORTS allowed.
    All PDF processing happens via HTTP calls to the Docling service.

    Args:
        doc_id: Unique document identifier
        pdf_path: Path to the PDF file
        collection: Qdrant collection name (default: meridian_documents)
        extract_tables: Whether to process tables with VLM
        extract_figures: Whether to process figures with VLM
        extract_formulas: Whether to process formula pages with VLM
        store_embeddings: Whether to generate and store embeddings
        image_scale: Image scale factor (None = auto-detect based on file size)

    Returns:
        Dict with processing results and statistics
    """
    start_time = time.time()
    collection = collection or "meridian_documents"

    # Auto-detect image scale based on file size if not provided
    if image_scale is None:
        image_scale = get_image_scale_for_file(pdf_path)

    logger.info(f"Processing document: {doc_id} ({pdf_path}) [scale={image_scale}]")

    result = {
        "doc_id": doc_id,
        "pdf_path": pdf_path,
        "success": False,
        "error": None,
        "stats": {},
        "timing": {},
    }

    try:
        # =====================================================================
        # Step 1: Call Docling Service
        # =====================================================================
        step_start = time.time()
        logger.info(f"[{doc_id}] Step 1: Calling Docling service...")

        docling_client = DoclingClient()
        docling_result = docling_client.process_pdf(
            pdf_path,
            extract_tables=extract_tables,
            extract_figures=extract_figures,
            extract_formula_pages=extract_formulas,
            image_scale=image_scale,
        )

        if not docling_result.get("success"):
            raise RuntimeError(f"Docling processing failed: {docling_result.get('error')}")

        docling_doc = docling_result["document"]
        tables = docling_result["extractions"]["tables"]
        figures = docling_result["extractions"]["figures"]
        formula_pages = docling_result["extractions"]["formula_pages"]

        result["timing"]["docling"] = time.time() - step_start
        result["stats"]["pages"] = docling_result.get("metadata", {}).get("pages", 0)
        result["stats"]["tables_detected"] = len(tables)
        result["stats"]["figures_detected"] = len(figures)
        result["stats"]["formula_pages"] = len(formula_pages)

        logger.info(
            f"[{doc_id}] Docling complete: {result['stats']['pages']} pages, "
            f"{len(tables)} tables, {len(figures)} figures, {len(formula_pages)} formula pages"
        )

        docling_client.close()

        # =====================================================================
        # Step 2: Annotate Formula Pages (PIL only)
        # =====================================================================
        step_start = time.time()
        annotated_pages = []
        blocks_by_page = {}

        if formula_pages and extract_formulas:
            logger.info(f"[{doc_id}] Step 2: Annotating formula pages...")

            annotated_pages, blocks_by_page = annotate_formula_pages(
                docling_doc,
                formula_pages,
            )

            result["stats"]["formula_pages_annotated"] = len(annotated_pages)
            logger.info(f"[{doc_id}] Annotated {len(annotated_pages)} formula pages")
        else:
            logger.info(f"[{doc_id}] Step 2: No formula pages to annotate")

        result["timing"]["annotate"] = time.time() - step_start

        # =====================================================================
        # Step 3: Call VLM Service (ASYNC - Phase 4B optimization)
        # =====================================================================
        step_start = time.time()
        logger.info(f"[{doc_id}] Step 3: Processing with VLM (async/concurrent)...")

        vlm_results = {"tables": [], "pictures": [], "formulas": []}
        formula_detection_results = {"results": []}

        # Prepare VLM tasks
        vlm_tables = tables if (tables and extract_tables) else []
        vlm_figures = figures if (figures and extract_figures) else []
        vlm_formulas = annotated_pages if (annotated_pages and extract_formulas) else []

        # Normalize page key: extractors output "page", VLM client reads "page_no"
        for t in vlm_tables:
            if "page_no" not in t and "page" in t:
                t["page_no"] = t["page"]
        for f in vlm_figures:
            if "page_no" not in f and "page" in f:
                f["page_no"] = f["page"]

        # Build text context for each formula page from blocks_by_page
        # Format: [box_num] text — helps VLM with formula descriptions
        for fp in vlm_formulas:
            fp_page = fp.get("page_no", fp.get("page", 0))
            page_blocks = blocks_by_page.get(fp_page, [])
            text_parts = []
            for i, block in enumerate(page_blocks):
                box_num = i + 1
                block_text = block.get("text", "").strip()
                if block_text and block_text not in ("[Picture]", "[Table]"):
                    text_parts.append(f"[{box_num}] {block_text}")
            fp["text_context"] = "\n".join(text_parts) if text_parts else ""

        total_vlm_tasks = len(vlm_tables) + len(vlm_figures) + len(vlm_formulas)

        if total_vlm_tasks > 0:
            logger.info(f"[{doc_id}] Sending {total_vlm_tasks} VLM requests concurrently...")

            # Process ALL VLM tasks concurrently (Phase 4B key optimization)
            vlm_results = _run_async(process_vlm_tasks_async(
                tables=vlm_tables,
                figures=vlm_figures,
                formulas=vlm_formulas,
                context="",
            ))

            # Enrich VLM table results with source from extraction pipeline
            for i, vlm_table in enumerate(vlm_results.get("tables", [])):
                if i < len(vlm_tables):
                    vlm_table["source"] = vlm_tables[i].get("source", "detected")

            # Build formula detection results from VLM output
            # VLM now returns {standalone_formulas: [...], inline_math_regions: [...]}
            if vlm_formulas:
                for i, formula_result in enumerate(vlm_results.get("formulas", [])):
                    page_data = vlm_formulas[i] if i < len(vlm_formulas) else {}
                    page_no = formula_result.get("page_no", page_data.get("page_no", 0))

                    # Get standalone formulas — VLM returns them directly with after_box
                    standalone = formula_result.get("standalone_formulas", [])

                    # Fallback: old format had "formulas" key
                    if not standalone:
                        standalone = formula_result.get("formulas", [])

                    # Assign formula_ids if missing
                    for j, formula in enumerate(standalone):
                        if "formula_id" not in formula:
                            formula["formula_id"] = f"p{page_no}_f{j}"

                    formula_detection_results["results"].append({
                        "page_no": page_no,
                        "num_blocks": page_data.get("num_blocks", 0),
                        "standalone_formulas": standalone,
                        "inline_math_regions": formula_result.get("inline_math_regions", []),
                    })
        else:
            logger.info(f"[{doc_id}] No VLM tasks to process")

        result["timing"]["vlm"] = time.time() - step_start
        result["stats"]["tables_processed"] = len(vlm_results.get("tables", []))
        result["stats"]["figures_processed"] = len(vlm_results.get("pictures", []))

        logger.info(
            f"[{doc_id}] VLM complete: {len(vlm_results.get('tables', []))} tables, "
            f"{len(vlm_results.get('pictures', []))} figures "
            f"in {result['timing']['vlm']:.1f}s"
        )

        # =====================================================================
        # Step 4: Build Chunks
        # =====================================================================
        step_start = time.time()
        logger.info(f"[{doc_id}] Step 4: Building chunks...")

        chunker = UnifiedChunker()
        chunks = chunker.build_chunks(
            docling_doc,
            vlm_results,
            formula_results=formula_detection_results if formula_detection_results["results"] else None,
        )

        # Merge small chunks
        chunks = chunker.merge_small_chunks(chunks)

        # Convert to dict for storage
        chunks_dict = chunker.chunks_to_dict(chunks)

        result["timing"]["chunking"] = time.time() - step_start
        result["stats"]["chunks"] = len(chunks_dict)
        result["chunks"] = chunks_dict
        logger.info(f"[{doc_id}] Built {len(chunks_dict)} chunks")

        # =====================================================================
        # Step 5 & 6: Generate Embeddings and Store (ASYNC - Phase 4 optimization)
        # =====================================================================
        if store_embeddings and chunks_dict:
            step_start = time.time()
            logger.info(f"[{doc_id}] Step 5-6: Generating embeddings and storing (async)...")

            # Use async embedding client for non-blocking operations
            async def store_embeddings_async():
                async with AsyncEmbeddingClient(collection_name=collection) as embedding_client:
                    return await embedding_client.store_document_chunks_async(
                        document_id=doc_id,
                        chunks=chunks_dict,
                    )

            stored_count = _run_async(store_embeddings_async())

            result["timing"]["embedding"] = time.time() - step_start
            result["stats"]["chunks_stored"] = stored_count

            logger.info(f"[{doc_id}] Stored {stored_count} chunks in Qdrant (async)")
        else:
            logger.info(f"[{doc_id}] Skipping embedding storage")
            result["stats"]["chunks_stored"] = 0

        # =====================================================================
        # Complete
        # =====================================================================
        result["success"] = True
        result["timing"]["total"] = time.time() - start_time

        logger.info(
            f"[{doc_id}] Processing complete in {result['timing']['total']:.1f}s: "
            f"{result['stats']['chunks']} chunks"
        )

    except Exception as e:
        logger.error(f"[{doc_id}] Processing failed: {e}", exc_info=True)
        result["success"] = False
        result["error"] = str(e)
        result["timing"]["total"] = time.time() - start_time

    return result


def process_document_minimal(
    doc_id: str,
    pdf_path: str,
) -> Dict[str, Any]:
    """
    Minimal processing - just Docling + chunking, no VLM or embeddings.

    Useful for testing the pipeline quickly.
    """
    return process_single_document(
        doc_id=doc_id,
        pdf_path=pdf_path,
        extract_tables=False,
        extract_figures=False,
        extract_formulas=False,
        store_embeddings=False,
    )
