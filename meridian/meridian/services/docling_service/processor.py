"""
Docling Processor

Wrapper around Docling's DocumentConverter that loads GPU models once
and reuses them across multiple PDF processing requests.
"""

import gc
import logging
import time
from pathlib import Path
from typing import Optional, Tuple

import torch

# IMPORTANT: Import config FIRST to set Docling env vars before Docling is imported
from meridian.config import (
    OCR_ENABLED,
    OCR_ENGINE,
    OCR_BITMAP_THRESHOLD,
    TABLE_STRUCTURE_ENABLED,
    TABLE_STRUCTURE_MODE,
    TABLE_CELL_MATCHING,
    FORMULA_ENRICHMENT_ENABLED,
    PICTURE_EXTRACTION_ENABLED,
    ACCELERATOR_DEVICE,
    ACCELERATOR_THREADS,
    IMAGE_SCALE,
    DOC_BATCH_SIZE,
    DOC_BATCH_CONCURRENCY,
    PROFILE_PIPELINE_TIMINGS,
    PDF_BACKEND,
    LAYOUT_MODEL,
)

# Now import Docling (after env vars are set)
from docling.document_converter import DocumentConverter, PdfFormatOption, ConversionResult
from docling.datamodel.base_models import InputFormat, ConversionStatus
from docling.datamodel.settings import settings as docling_settings
from docling.datamodel.pipeline_options import (
    ThreadedPdfPipelineOptions,  # Use threaded version for parallelism!
    TableStructureOptions,
    TableFormerMode,
    EasyOcrOptions,
)
from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling.datamodel.document import InputDocument
from docling_core.types.doc.document import DoclingDocument

logger = logging.getLogger(__name__)


class DoclingProcessor:
    """
    Singleton-style Docling processor that loads models once at initialization.

    This class is designed to be instantiated once at service startup and reused
    for all incoming PDF processing requests.
    """

    def __init__(self):
        """Initialize the Docling processor and load GPU models."""
        logger.info("Initializing Docling processor...")
        start_time = time.time()

        # Enable pipeline profiling if configured
        if PROFILE_PIPELINE_TIMINGS:
            docling_settings.debug.profile_pipeline_timings = True
            logger.info("Pipeline profiling ENABLED - timing data will be logged")

        self.converter = self._create_converter()
        self._models_loaded = True

        load_time = time.time() - start_time
        logger.info(f"Docling models loaded in {load_time:.2f}s")

    def _get_accelerator_device(self) -> AcceleratorDevice:
        """Map config string to AcceleratorDevice enum."""
        device_map = {
            "auto": AcceleratorDevice.AUTO,
            "cuda": AcceleratorDevice.CUDA,
            "mps": AcceleratorDevice.MPS,
            "cpu": AcceleratorDevice.CPU,
        }
        return device_map.get(ACCELERATOR_DEVICE.lower(), AcceleratorDevice.AUTO)

    def _create_converter(self) -> DocumentConverter:
        """Create and configure the Docling document converter."""
        # Use ThreadedPdfPipelineOptions for parallelism support!
        pipeline_options = ThreadedPdfPipelineOptions()

        # Configure OCR
        if not OCR_ENABLED:
            pipeline_options.do_ocr = False
            logger.info("OCR disabled - using PDF text layer + VLMs for visual content")
        else:
            if OCR_ENGINE == "easyocr":
                pipeline_options.ocr_options = EasyOcrOptions(
                    bitmap_area_threshold=OCR_BITMAP_THRESHOLD
                )
            logger.info(f"OCR enabled: engine={OCR_ENGINE}, threshold={OCR_BITMAP_THRESHOLD}")

        # Configure accelerator (GPU)
        pipeline_options.accelerator_options = AcceleratorOptions(
            num_threads=ACCELERATOR_THREADS,
            device=self._get_accelerator_device(),
        )
        logger.info(f"Accelerator: device={ACCELERATOR_DEVICE}, threads={ACCELERATOR_THREADS}")

        # Table structure extraction
        pipeline_options.do_table_structure = TABLE_STRUCTURE_ENABLED
        if TABLE_STRUCTURE_ENABLED:
            mode = TableFormerMode.ACCURATE if TABLE_STRUCTURE_MODE == "accurate" else TableFormerMode.FAST
            pipeline_options.table_structure_options = TableStructureOptions(
                do_cell_matching=TABLE_CELL_MATCHING,
                mode=mode,
            )
            logger.info(f"Table structure: enabled, mode={TABLE_STRUCTURE_MODE}, cell_matching={TABLE_CELL_MATCHING}")

        # Layout model selection
        layout_model_map = {
            "heron": "DOCLING_LAYOUT_HERON",
            "heron-101": "DOCLING_LAYOUT_HERON_101",
            "egret-medium": "DOCLING_LAYOUT_EGRET_MEDIUM",
            "egret-large": "DOCLING_LAYOUT_EGRET_LARGE",
            "egret-xlarge": "DOCLING_LAYOUT_EGRET_XLARGE",
            "v2": "DOCLING_LAYOUT_V2",
        }
        layout_spec_name = layout_model_map.get(LAYOUT_MODEL.lower())
        if layout_spec_name:
            from docling.datamodel import layout_model_specs
            from docling.datamodel.pipeline_options import LayoutOptions
            layout_spec = getattr(layout_model_specs, layout_spec_name, None)
            if layout_spec:
                pipeline_options.layout_options = LayoutOptions(model_spec=layout_spec)
                logger.info(f"Layout model: {LAYOUT_MODEL} ({layout_spec.repo_id})")
            else:
                logger.warning(f"Layout model spec '{layout_spec_name}' not found, using default")
        else:
            logger.info(f"Layout model: {LAYOUT_MODEL} (default)")

        # Formula enrichment
        pipeline_options.do_formula_enrichment = FORMULA_ENRICHMENT_ENABLED
        logger.info(f"Formula enrichment: {FORMULA_ENRICHMENT_ENABLED}")

        # Picture extraction
        pipeline_options.generate_picture_images = PICTURE_EXTRACTION_ENABLED
        if PICTURE_EXTRACTION_ENABLED:
            pipeline_options.images_scale = IMAGE_SCALE
        logger.info(f"Picture extraction: {PICTURE_EXTRACTION_ENABLED}, scale={IMAGE_SCALE}")

        # Configure batch sizes for within-document parallelism
        # These control how many pages/items are batched at each pipeline stage
        pipeline_options.ocr_batch_size = 8       # Default: 4
        pipeline_options.layout_batch_size = 8    # Default: 4, Layout model batches
        pipeline_options.table_batch_size = 8     # Default: 4, TableFormer batches
        pipeline_options.queue_max_size = 200     # Default: 100, allow more buffering
        pipeline_options.batch_polling_interval_seconds = 0.2  # Default: 0.5, tighter polling
        logger.info(
            f"Pipeline batching: ocr={pipeline_options.ocr_batch_size}, "
            f"layout={pipeline_options.layout_batch_size}, "
            f"table={pipeline_options.table_batch_size}, "
            f"queue_max={pipeline_options.queue_max_size}"
        )

        # Log Docling's document-level parallelism settings
        logger.info(
            f"Docling parallelism: doc_batch_size={docling_settings.perf.doc_batch_size}, "
            f"doc_batch_concurrency={docling_settings.perf.doc_batch_concurrency}"
        )

        # Configure PDF backend
        logger.info(f"PDF backend: {PDF_BACKEND}")

        # Build format options with backend selection
        pdf_format_option = PdfFormatOption(pipeline_options=pipeline_options)

        # Set backend based on config
        # pypdfium2 = fastest, v2 = default (moderate), v4 = best semantic
        if PDF_BACKEND == "pypdfium2":
            pdf_format_option.backend = PyPdfiumDocumentBackend
            logger.info("Using PyPdfium2 backend (fastest, basic structure)")
        # For v2 and v4, we use the default pipeline behavior
        # v2 is the default when no backend is specified

        return DocumentConverter(
            format_options={
                InputFormat.PDF: pdf_format_option
            }
        )

    @property
    def is_ready(self) -> bool:
        """Check if the processor is ready to handle requests."""
        return self._models_loaded and self.converter is not None

    def process_pdf(
        self,
        pdf_path: Path,
    ) -> Tuple[Optional[DoclingDocument], Optional[PyPdfiumDocumentBackend], Optional[str]]:
        """
        Process a PDF file and return the Docling document + PDF backend.

        The backend is returned so that image extraction can be performed
        using the same loaded PDF.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            Tuple of (DoclingDocument, PyPdfiumDocumentBackend, error_message)
            - On success: (document, backend, None)
            - On failure: (None, None, error_message)
        """
        if not self.is_ready:
            return None, None, "Processor not initialized"

        if not pdf_path.exists():
            return None, None, f"PDF file not found: {pdf_path}"

        logger.info(f"Processing PDF: {pdf_path.name}")
        start_time = time.time()

        try:
            # Convert document using Docling
            result = self.converter.convert(str(pdf_path))

            if result.status != ConversionStatus.SUCCESS:
                errors = [str(e.error_message) for e in result.errors] if result.errors else ["Unknown error"]
                error_msg = f"Docling conversion failed: {'; '.join(errors)}"
                logger.error(error_msg)
                return None, None, error_msg

            doc = result.document

            # Log profiling data if enabled
            if PROFILE_PIPELINE_TIMINGS and hasattr(result, 'timings') and result.timings:
                logger.info(f"=== PIPELINE TIMINGS for {pdf_path.name} ===")
                # Sort by total time (descending) to show biggest bottlenecks first
                sorted_timings = sorted(
                    result.timings.items(),
                    key=lambda x: x[1].total() if hasattr(x[1], 'total') else 0,
                    reverse=True
                )
                for stage_name, timing_item in sorted_timings:
                    if hasattr(timing_item, 'total'):
                        total = timing_item.total()
                        count = timing_item.count
                        avg = timing_item.avg() if count > 0 else 0
                        scope = timing_item.scope.value if hasattr(timing_item.scope, 'value') else str(timing_item.scope)
                        logger.info(f"  {stage_name}: total={total:.2f}s, count={count}, avg={avg:.3f}s ({scope})")
                    else:
                        logger.info(f"  {stage_name}: {timing_item}")
                logger.info("=== END TIMINGS ===")

            # Load PDF backend for image extraction
            input_doc = InputDocument(
                path_or_stream=pdf_path,
                format=InputFormat.PDF,
                backend=PyPdfiumDocumentBackend,
            )
            backend = input_doc._backend

            process_time = time.time() - start_time
            page_count = len(list(doc.pages)) if hasattr(doc, 'pages') else 0
            logger.info(f"Processed {pdf_path.name}: {page_count} pages in {process_time:.2f}s")

            return doc, backend, None

        except Exception as e:
            error_msg = f"Exception processing PDF: {str(e)}"
            logger.exception(error_msg)
            return None, None, error_msg

        finally:
            # CRITICAL: Clean up GPU memory after each document
            # This prevents memory accumulation over thousands of documents
            self._cleanup_gpu_memory()

    def process_pdfs_batch(
        self,
        pdf_paths: list[Path],
    ) -> list[Tuple[Optional[DoclingDocument], Optional[PyPdfiumDocumentBackend], Optional[str]]]:
        """
        Process multiple PDF files in parallel using Docling's convert_all().

        This leverages Docling's internal parallelism when:
        - DOCLING_PERF_DOC_BATCH_SIZE > 1
        - DOCLING_PERF_DOC_BATCH_CONCURRENCY > 1

        Args:
            pdf_paths: List of paths to PDF files

        Returns:
            List of tuples (DoclingDocument, PyPdfiumDocumentBackend, error_message)
            In same order as input pdf_paths.
        """
        if not self.is_ready:
            return [(None, None, "Processor not initialized") for _ in pdf_paths]

        # Validate all paths exist and build mapping
        valid_paths = []
        valid_indices = []  # Maps valid_paths index -> original pdf_paths index
        results = [None] * len(pdf_paths)

        for idx, pdf_path in enumerate(pdf_paths):
            if not pdf_path.exists():
                results[idx] = (None, None, f"PDF file not found: {pdf_path}")
            else:
                valid_paths.append(pdf_path)
                valid_indices.append(idx)

        if not valid_paths:
            return results

        logger.info(f"Batch processing {len(valid_paths)} PDFs (concurrency={docling_settings.perf.doc_batch_concurrency}, batch_size={docling_settings.perf.doc_batch_size})")
        start_time = time.time()

        try:
            # Convert all documents using Docling's parallel processing
            # convert_all() uses ThreadPoolExecutor when doc_batch_concurrency > 1
            # Results are yielded in the same order as input
            conversion_results: list[ConversionResult] = list(
                self.converter.convert_all([str(p) for p in valid_paths])
            )

            # Process each result - results are in same order as valid_paths
            for result_idx, conv_result in enumerate(conversion_results):
                # Map back to original index
                original_idx = valid_indices[result_idx]
                pdf_path = pdf_paths[original_idx]

                if conv_result.status != ConversionStatus.SUCCESS:
                    errors = [str(e.error_message) for e in conv_result.errors] if conv_result.errors else ["Unknown error"]
                    error_msg = f"Docling conversion failed: {'; '.join(errors)}"
                    logger.error(f"Failed {pdf_path.name}: {error_msg}")
                    results[original_idx] = (None, None, error_msg)
                    continue

                doc = conv_result.document

                # Load PDF backend for image extraction
                input_doc = InputDocument(
                    path_or_stream=pdf_path,
                    format=InputFormat.PDF,
                    backend=PyPdfiumDocumentBackend,
                )
                backend = input_doc._backend

                page_count = len(list(doc.pages)) if hasattr(doc, 'pages') else 0
                logger.info(f"Converted {pdf_path.name}: {page_count} pages")

                results[original_idx] = (doc, backend, None)

            total_time = time.time() - start_time
            logger.info(f"Batch processing complete: {len(valid_paths)} docs in {total_time:.2f}s ({total_time/len(valid_paths):.2f}s/doc avg)")

            return results

        except Exception as e:
            error_msg = f"Exception in batch processing: {str(e)}"
            logger.exception(error_msg)
            # Fill remaining None results with error
            for idx, result in enumerate(results):
                if result is None:
                    results[idx] = (None, None, error_msg)
            return results

        finally:
            # Clean up GPU memory after batch processing
            self._cleanup_gpu_memory()

    def _cleanup_gpu_memory(self):
        """
        Release cached GPU memory without unloading models.

        This is CRITICAL for long-running batch processing (10K+ documents).
        PyTorch caches freed GPU memory blocks for reuse, which accumulates
        over time. This releases those cached blocks back to CUDA.

        NOTE: This does NOT affect:
        - Model weights (stay in GPU memory)
        - vLLM's KV cache (separate process)
        - Processing performance (models stay loaded)
        """
        gc.collect()  # Python garbage collection first
        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # Release cached CUDA memory blocks

    def get_stats(self) -> dict:
        """Get processor statistics."""
        return {
            "models_loaded": self._models_loaded,
            "ocr_enabled": OCR_ENABLED,
            "table_structure_enabled": TABLE_STRUCTURE_ENABLED,
            "formula_enrichment_enabled": FORMULA_ENRICHMENT_ENABLED,
            "picture_extraction_enabled": PICTURE_EXTRACTION_ENABLED,
            "accelerator_device": ACCELERATOR_DEVICE,
        }


# Global processor instance (initialized by FastAPI lifespan)
_processor: Optional[DoclingProcessor] = None


def get_processor() -> DoclingProcessor:
    """Get the global processor instance."""
    global _processor
    if _processor is None:
        raise RuntimeError("Processor not initialized. Call init_processor() first.")
    return _processor


def init_processor() -> DoclingProcessor:
    """Initialize the global processor instance."""
    global _processor
    if _processor is None:
        _processor = DoclingProcessor()
    return _processor


def shutdown_processor():
    """Shutdown the global processor instance."""
    global _processor
    _processor = None
    logger.info("Processor shutdown complete")
