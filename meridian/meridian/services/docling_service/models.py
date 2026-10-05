"""
Pydantic Models for Docling Service API

Request and response schemas for the HTTP API.
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


# =============================================================================
# Request Models
# =============================================================================

class ProcessOptions(BaseModel):
    """Options for PDF processing."""

    extract_tables: bool = Field(default=True, description="Extract table images")
    extract_figures: bool = Field(default=True, description="Extract figure/picture images")
    extract_formula_pages: bool = Field(default=True, description="Extract full-page images for pages with formulas")

    # Optional page range
    page_start: Optional[int] = Field(default=None, description="Start page (1-indexed, inclusive)")
    page_end: Optional[int] = Field(default=None, description="End page (1-indexed, inclusive)")

    # Image settings
    image_scale: float = Field(default=2.0, description="Scale factor for extracted images")

    # Table detection options
    detect_missed_tables: bool = Field(default=True, description="Detect tables from section headers")
    detect_continuation_pages: bool = Field(default=True, description="Detect table continuation pages")


# =============================================================================
# Response Models - Bounding Box
# =============================================================================

class BoundingBox(BaseModel):
    """Bounding box coordinates."""

    l: float = Field(description="Left coordinate")
    t: float = Field(description="Top coordinate")
    r: float = Field(description="Right coordinate")
    b: float = Field(description="Bottom coordinate")


# =============================================================================
# Response Models - Extractions
# =============================================================================

class TableExtraction(BaseModel):
    """Extracted table data."""

    id: str = Field(description="Unique identifier (e.g., 'table_0')")
    page: int = Field(description="Page number (1-indexed)")
    bbox: BoundingBox = Field(description="Bounding box on the page")
    image_base64: str = Field(description="Base64 encoded PNG image")
    source: str = Field(description="Detection source: 'detected', 'missed', or 'continuation'")
    title: Optional[str] = Field(default=None, description="Table title if found")
    reason: Optional[str] = Field(default=None, description="Reason for continuation detection")
    structure: Optional[Dict[str, Any]] = Field(default=None, description="Table structure data")


class FigureExtraction(BaseModel):
    """Extracted figure/picture data."""

    id: str = Field(description="Unique identifier (e.g., 'figure_0')")
    page: int = Field(description="Page number (1-indexed)")
    bbox: BoundingBox = Field(description="Bounding box on the page")
    image_base64: str = Field(description="Base64 encoded PNG image")
    context_above: str = Field(default="", description="Text context above the figure")
    context_below: str = Field(default="", description="Text context below the figure")


class FormulaPageExtraction(BaseModel):
    """
    Full page image for a page containing formulas.

    IMPORTANT: This is a FULL PAGE image, not a cropped formula region.
    Workers will draw numbered boxes on these pages for VLM processing.
    This design keeps box numbering logic tightly coupled with unified_chunker.
    """

    page: int = Field(description="Page number (1-indexed)")
    image_base64: str = Field(description="Base64 encoded PNG of the FULL PAGE")


class Extractions(BaseModel):
    """All extractions from a document."""

    tables: List[TableExtraction] = Field(default_factory=list)
    figures: List[FigureExtraction] = Field(default_factory=list)
    formula_pages: List[FormulaPageExtraction] = Field(default_factory=list)


# =============================================================================
# Response Models - Metadata
# =============================================================================

class ProcessingMetadata(BaseModel):
    """Metadata about the processing result."""

    filename: str = Field(description="Original filename")
    pages: int = Field(description="Total number of pages")
    processing_time_ms: int = Field(description="Processing time in milliseconds")
    tables_count: int = Field(default=0, description="Number of tables extracted")
    figures_count: int = Field(default=0, description="Number of figures extracted")
    formula_pages_count: int = Field(default=0, description="Number of pages with formulas")


# =============================================================================
# Response Models - Full Response
# =============================================================================

class ProcessResponse(BaseModel):
    """Response from /process endpoint."""

    success: bool = Field(description="Whether processing was successful")
    document: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Full DoclingDocument as JSON (if successful)"
    )
    extractions: Optional[Extractions] = Field(
        default=None,
        description="Extracted tables, figures, and formulas"
    )
    metadata: Optional[ProcessingMetadata] = Field(
        default=None,
        description="Processing metadata"
    )
    error: Optional[str] = Field(
        default=None,
        description="Error message (if failed)"
    )


class LayoutOnlyResponse(BaseModel):
    """Response from /process/layout-only endpoint (no images)."""

    success: bool = Field(description="Whether processing was successful")
    document: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Full DoclingDocument as JSON"
    )
    metadata: Optional[ProcessingMetadata] = Field(
        default=None,
        description="Processing metadata"
    )
    error: Optional[str] = Field(
        default=None,
        description="Error message (if failed)"
    )


# =============================================================================
# Health & Status Models
# =============================================================================

class HealthResponse(BaseModel):
    """Response from /health endpoint."""

    status: str = Field(description="Service status: 'healthy' or 'unhealthy'")
    models_loaded: bool = Field(description="Whether GPU models are loaded")
    version: str = Field(default="1.0.0", description="Service version")


class StatsResponse(BaseModel):
    """Response from /stats endpoint."""

    models_loaded: bool
    ocr_enabled: bool
    table_structure_enabled: bool
    formula_enrichment_enabled: bool
    picture_extraction_enabled: bool
    accelerator_device: str
    requests_processed: int = Field(default=0)
    total_processing_time_ms: int = Field(default=0)
    average_processing_time_ms: float = Field(default=0.0)


# =============================================================================
# Error Models
# =============================================================================

class ErrorResponse(BaseModel):
    """Error response."""

    success: bool = Field(default=False)
    error: str = Field(description="Error message")
    details: Optional[str] = Field(default=None, description="Additional error details")


# =============================================================================
# Batch Processing Models
# =============================================================================

class BatchDocumentResult(BaseModel):
    """Result for a single document in a batch."""

    filename: str = Field(description="Original filename")
    success: bool = Field(description="Whether processing was successful")
    document: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Full DoclingDocument as JSON (if successful)"
    )
    extractions: Optional[Extractions] = Field(
        default=None,
        description="Extracted tables, figures, and formulas"
    )
    metadata: Optional[ProcessingMetadata] = Field(
        default=None,
        description="Processing metadata"
    )
    error: Optional[str] = Field(
        default=None,
        description="Error message (if failed)"
    )


class BatchProcessResponse(BaseModel):
    """Response from /process-batch endpoint."""

    success: bool = Field(description="Whether all documents processed successfully")
    total_documents: int = Field(description="Total number of documents submitted")
    successful_count: int = Field(description="Number of successfully processed documents")
    failed_count: int = Field(description="Number of failed documents")
    total_processing_time_ms: int = Field(description="Total processing time in milliseconds")
    results: List[BatchDocumentResult] = Field(description="Results for each document")
