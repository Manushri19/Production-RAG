#!/usr/bin/env python3
"""
Example: Submit a batch of PDFs for processing.

Prerequisites:
- All services running (./scripts/start_all.sh)

Usage:
    python examples/batch_process.py /path/to/pdf/directory [collection_name]
"""

import sys
import time
from pathlib import Path
from meridian.orchestrator.batch_orchestrator import BatchOrchestrator
from meridian.orchestrator.progress import print_progress


def main():
    if len(sys.argv) < 2:
        print("Usage: python examples/batch_process.py <pdf_dir> [collection]")
        sys.exit(1)

    pdf_dir = Path(sys.argv[1])
    collection = sys.argv[2] if len(sys.argv) > 2 else "meridian_documents"

    if not pdf_dir.exists():
        print(f"Directory not found: {pdf_dir}")
        sys.exit(1)

    orch = BatchOrchestrator()

    # Submit batch
    print(f"Submitting batch from: {pdf_dir}")
    print(f"Collection: {collection}")
    print()

    batch_id = orch.submit_batch(
        input_dir=pdf_dir,
        collection=collection,
    )

    print(f"Batch submitted: {batch_id}")
    print()

    # Poll progress until complete
    while True:
        progress = orch.get_progress(batch_id)
        print_progress(progress)

        if orch.is_complete(batch_id):
            print("Batch processing complete!")
            break

        time.sleep(10)


if __name__ == "__main__":
    main()
