"""
Meridian Configuration

Single source of truth for all configuration.
Consolidates settings from:
- Docling service config (env-var driven)
- Worker chunker constants
- Worker box_annotator constants

All settings are configurable via environment variables with sensible defaults.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # Load .env file from project root


# =============================================================================
# Redis
# =============================================================================

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")


# =============================================================================
# Docling Service Settings
# =============================================================================

SERVICE_HOST = os.getenv("DOCLING_SERVICE_HOST", "0.0.0.0")
SERVICE_PORT = int(os.getenv("DOCLING_SERVICE_PORT", "8001"))

# Request handling
MAX_CONCURRENT_REQUESTS = int(os.getenv("DOCLING_MAX_CONCURRENT", "10"))
REQUEST_TIMEOUT_SECONDS = int(os.getenv("DOCLING_REQUEST_TIMEOUT", "300"))

# Temporary file storage
TEMP_DIR = Path(os.getenv("DOCLING_TEMP_DIR", "/tmp/docling_service"))


# =============================================================================
# Docling Pipeline Settings
# =============================================================================

# OCR Settings
# Disable OCR to save ~11GB GPU memory - VLMs handle visual content
OCR_ENABLED = os.getenv("DOCLING_OCR_ENABLED", "false").lower() == "true"
OCR_ENGINE = os.getenv("DOCLING_OCR_ENGINE", "easyocr")
OCR_BITMAP_THRESHOLD = float(os.getenv("DOCLING_OCR_BITMAP_THRESHOLD", "0.74"))

# Table Structure
TABLE_STRUCTURE_ENABLED = os.getenv("DOCLING_TABLE_STRUCTURE", "true").lower() == "true"
TABLE_STRUCTURE_MODE = os.getenv("DOCLING_TABLE_MODE", "accurate")  # "accurate" or "fast"
TABLE_CELL_MATCHING = os.getenv("DOCLING_TABLE_CELL_MATCHING", "true").lower() == "true"

# Formula Enrichment
# IMPORTANT: Disabled by default - formula DETECTION still works via layout model.
# The CodeFormulaModel (enrichment) converts formulas to LaTeX via VLM, which is:
# 1. Extremely slow on scanned documents (46s/formula vs 0.7s on digital PDFs)
# 2. Not needed if you send entire pages to your own VLM for formula understanding
FORMULA_ENRICHMENT_ENABLED = os.getenv("DOCLING_FORMULA_ENRICHMENT", "false").lower() == "true"

# Picture Extraction
PICTURE_EXTRACTION_ENABLED = os.getenv("DOCLING_PICTURE_EXTRACTION", "true").lower() == "true"


# =============================================================================
# GPU / Accelerator Settings
# =============================================================================

# Device selection: "auto", "cuda", "mps", "cpu"
ACCELERATOR_DEVICE = os.getenv("DOCLING_ACCELERATOR_DEVICE", "auto")
ACCELERATOR_THREADS = int(os.getenv("DOCLING_ACCELERATOR_THREADS", "4"))


# =============================================================================
# Batch Processing & Parallelism Settings
# =============================================================================

# Number of documents per batch for convert_all()
DOC_BATCH_SIZE = int(os.getenv("DOCLING_DOC_BATCH_SIZE", "4"))

# Number of parallel threads for document processing
DOC_BATCH_CONCURRENCY = int(os.getenv("DOCLING_DOC_BATCH_CONCURRENCY", "4"))

# Page batch size for within-document parallelism
PAGE_BATCH_SIZE = int(os.getenv("DOCLING_PAGE_BATCH_SIZE", "4"))

# Set Docling environment variables so they're picked up by Docling's settings
# These MUST be set before Docling's settings module is imported
os.environ.setdefault("DOCLING_PERF_DOC_BATCH_SIZE", str(DOC_BATCH_SIZE))
os.environ.setdefault("DOCLING_PERF_DOC_BATCH_CONCURRENCY", str(DOC_BATCH_CONCURRENCY))
os.environ.setdefault("DOCLING_PERF_PAGE_BATCH_SIZE", str(PAGE_BATCH_SIZE))


# =============================================================================
# Image Extraction Settings
# =============================================================================

# Scale factor for extracted images (2.0 = 144 DPI)
IMAGE_SCALE = float(os.getenv("DOCLING_IMAGE_SCALE", "2.0"))

# Padding around bounding boxes (in points)
TABLE_PADDING_PX = int(os.getenv("DOCLING_TABLE_PADDING", "70"))
FORMULA_PADDING_PX = int(os.getenv("DOCLING_FORMULA_PADDING", "20"))
PICTURE_PADDING_PX = int(os.getenv("DOCLING_PICTURE_PADDING", "10"))

# Picture context extraction
PICTURE_CONTEXT_CHARS = int(os.getenv("DOCLING_PICTURE_CONTEXT_CHARS", "300"))

# Formula extraction mode
# "bbox" = crop formula bounding boxes
# "page" = extract full page for pages with formulas (better for complex layouts)
FORMULA_EXTRACTION_MODE = os.getenv("DOCLING_FORMULA_MODE", "bbox")

# Vertical expansion for formula bbox (fraction of height to add above/below)
FORMULA_VERTICAL_EXPANSION = float(os.getenv("DOCLING_FORMULA_VERTICAL_EXPANSION", "0.75"))


# =============================================================================
# Content Block Filtering
# =============================================================================
# These thresholds are tuned and used by both chunker and box_annotator.
# They MUST be identical in both to ensure correct formula placement.

CONTENT_BLOCK_MIN_WIDTH_PT = float(os.getenv("DOCLING_MIN_WIDTH_PT", "20"))
CONTENT_BLOCK_MIN_HEIGHT_PT = float(os.getenv("DOCLING_MIN_HEIGHT_PT", "10"))
CONTENT_BLOCK_MIN_AREA_PT2 = float(os.getenv("DOCLING_MIN_AREA_PT2", "500"))

# Nested box containment threshold
CONTENT_BLOCK_CONTAINMENT_THRESHOLD = float(os.getenv("DOCLING_CONTAINMENT_THRESHOLD", "0.8"))


# =============================================================================
# Table Detection Settings (multi-method)
# =============================================================================

DETECT_MISSED_TABLES = os.getenv("DOCLING_DETECT_MISSED_TABLES", "true").lower() == "true"
DETECT_CONTINUATION_PAGES = os.getenv("DOCLING_DETECT_CONTINUATION", "false").lower() == "true"

# Pictures above this area threshold (in pt²) are sent through VLM table pipeline
# as candidates — VLM decides if they contain tabular data
PICTURE_TABLE_CANDIDATE_MIN_AREA = float(os.getenv("DOCLING_PICTURE_TABLE_MIN_AREA", "50000"))


# =============================================================================
# Layout Model Selection
# =============================================================================

# Layout model for document element detection (tables, figures, text, etc.)
# Options: "heron" (default, ResNet50), "heron-101" (ResNet101),
#          "egret-medium", "egret-large", "egret-xlarge", "v2" (legacy)
LAYOUT_MODEL = os.getenv("DOCLING_LAYOUT_MODEL", "heron")


# =============================================================================
# Performance Profiling & Backend Selection
# =============================================================================

PROFILE_PIPELINE_TIMINGS = os.getenv("DOCLING_PROFILE_TIMINGS", "false").lower() == "true"

# PDF backend: "pypdfium2" (fastest), "v2" (default), "v4" (best semantic)
PDF_BACKEND = os.getenv("DOCLING_PDF_BACKEND", "v2")


# =============================================================================
# Logging
# =============================================================================

LOG_LEVEL = os.getenv("DOCLING_LOG_LEVEL", "INFO")


# =============================================================================
# VLM Service
# =============================================================================

VLLM_API_URL = os.getenv("VLLM_API_URL", "http://localhost:8000/v1/chat/completions")
VLLM_MODEL = os.getenv("VLLM_MODEL", "Qwen/Qwen3-VL-8B-Instruct")
VLLM_API_KEY = os.getenv("VLLM_API_KEY", "EMPTY")


# =============================================================================
# Embedding / Vector Store
# =============================================================================

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "qwen3-embedding:4b-q8_0")
EMBEDDING_DIMENSIONS = int(os.getenv("EMBEDDING_DIMENSIONS", "1024"))

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "meridian_documents")


# =============================================================================
# Docling Client (load balancing)
# =============================================================================

_default_instances = "http://localhost:8001,http://localhost:8002,http://localhost:8003,http://localhost:8004,http://localhost:8005,http://localhost:8006,http://localhost:8007,http://localhost:8008"
DOCLING_INSTANCES = os.getenv("DOCLING_INSTANCES", _default_instances).split(",")


# =============================================================================
# Image Service
# =============================================================================

IMAGE_SERVICE_PORT = int(os.getenv("IMAGE_SERVICE_PORT", "9847"))
PDF_DIR = Path(os.getenv("PDF_DIR", "/mnt/pdfs"))


# =============================================================================
# Derived Settings
# =============================================================================

def ensure_temp_dir():
    """Ensure temp directory exists."""
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    return TEMP_DIR
