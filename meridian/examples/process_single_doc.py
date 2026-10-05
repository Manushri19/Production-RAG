#!/usr/bin/env python3
"""
Example: Process a single PDF document.

Prerequisites:
- Docling service running (./scripts/start_docling_instances.sh)
- VLM service running (./scripts/start_vllm.sh)
- Redis running
- Ollama running with embedding model

Usage:
    python examples/process_single_doc.py /path/to/document.pdf
"""

import sys
from meridian.workers.document_worker import process_single_document


def main():
    if len(sys.argv) < 2:
        print("Usage: python examples/process_single_doc.py <pdf_path> [collection]")
        sys.exit(1)

    pdf_path = sys.argv[1]
    collection = sys.argv[2] if len(sys.argv) > 2 else "meridian_documents"

    # Use filename as document ID
    from pathlib import Path
    doc_id = Path(pdf_path).stem

    print(f"Processing: {pdf_path}")
    print(f"Document ID: {doc_id}")
    print(f"Collection: {collection}")
    print()

    result = process_single_document(
        doc_id=doc_id,
        pdf_path=pdf_path,
        collection=collection,
    )

    if result["success"]:
        print(f"Success!")
        print(f"  Pages: {result['stats'].get('pages', 0)}")
        print(f"  Tables: {result['stats'].get('tables_detected', 0)}")
        print(f"  Figures: {result['stats'].get('figures_detected', 0)}")
        print(f"  Chunks: {result['stats'].get('chunks', 0)}")
        print(f"  Total time: {result['timing'].get('total', 0):.1f}s")
    else:
        print(f"Failed: {result.get('error')}")
        sys.exit(1)


if __name__ == "__main__":
    main()
