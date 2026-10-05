"""
Docling Service

GPU-accelerated PDF processing service using Docling.
Models are loaded once at startup and shared across all requests.
"""

from .processor import DoclingProcessor, init_processor, get_processor
from .extractors import extract_all, extract_tables, extract_figures, extract_formula_pages
from .models import (
    ProcessOptions,
    ProcessResponse,
    HealthResponse,
    Extractions,
    TableExtraction,
    FigureExtraction,
    FormulaPageExtraction,
    BatchProcessResponse,
    BatchDocumentResult,
)

__version__ = "1.0.0"
__all__ = [
    "DoclingProcessor",
    "init_processor",
    "get_processor",
    "extract_all",
    "extract_tables",
    "extract_figures",
    "extract_formula_pages",
    "ProcessOptions",
    "ProcessResponse",
    "HealthResponse",
    "Extractions",
    "TableExtraction",
    "FigureExtraction",
    "FormulaPageExtraction",
    "BatchProcessResponse",
    "BatchDocumentResult",
]
