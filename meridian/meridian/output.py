"""
Output Formatters

Converts processing results to structured output files (JSON).
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


def format_result_json(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Format a process_single_document() result into the output JSON schema.

    Args:
        result: Raw result from process_single_document()

    Returns:
        Structured JSON dict ready for serialization
    """
    chunks = result.get("chunks", [])

    # Count chunk types
    type_counts = {}
    for chunk in chunks:
        ctype = chunk.get("type", "text")
        type_counts[ctype] = type_counts.get(ctype, 0) + 1

    return {
        "document_id": result.get("doc_id", ""),
        "source": result.get("pdf_path", ""),
        "pages": result.get("stats", {}).get("pages", 0),
        "processing_time_seconds": round(result.get("timing", {}).get("total", 0), 2),
        "chunks": chunks,
        "stats": {
            "tables": type_counts.get("table", 0),
            "figures": type_counts.get("picture", 0),
            "formulas": type_counts.get("formula", 0),
            "text_chunks": type_counts.get("text", 0),
            "total_chunks": len(chunks),
        },
        "timing": result.get("timing", {}),
    }


def write_result_json(
    result: Dict[str, Any],
    output_dir: Path,
    doc_id: Optional[str] = None,
) -> Path:
    """
    Write a processing result as a JSON file.

    Args:
        result: Raw result from process_single_document()
        output_dir: Directory to write output to
        doc_id: Override document ID for filename

    Returns:
        Path to the written JSON file
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    formatted = format_result_json(result)
    doc_id = doc_id or formatted["document_id"] or "document"

    output_path = output_dir / f"{doc_id}.json"

    with open(output_path, "w") as f:
        json.dump(formatted, f, indent=2, default=str)

    logger.info(f"Wrote {output_path} ({formatted['stats']['total_chunks']} chunks)")
    return output_path
