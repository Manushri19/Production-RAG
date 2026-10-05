"""
Parse Command - Process PDF documents and output structured JSON.

For small jobs (1-10 files): processes directly using thread pool.
For large jobs: uses Celery batch system.
"""

import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List

from meridian.output import write_result_json

logger = logging.getLogger(__name__)

BATCH_THRESHOLD = 10  # Use Celery for jobs larger than this


def find_pdfs(path: Path) -> List[Path]:
    """Find all PDF files at the given path."""
    path = Path(path)

    if path.is_file():
        if path.suffix.lower() == ".pdf":
            return [path]
        else:
            print(f"Error: {path} is not a PDF file", file=sys.stderr)
            sys.exit(1)

    if path.is_dir():
        pdfs = sorted(path.glob("**/*.pdf"))
        # Also catch uppercase .PDF
        pdfs += sorted(p for p in path.glob("**/*.PDF") if p not in pdfs)
        return pdfs

    print(f"Error: {path} not found", file=sys.stderr)
    sys.exit(1)


def process_one(pdf_path: Path, args) -> dict:
    """Process a single PDF file. Returns result dict."""
    from meridian.workers.document_worker import process_single_document

    doc_id = pdf_path.stem
    result = process_single_document(
        doc_id=doc_id,
        pdf_path=str(pdf_path),
        collection=args.collection if args.store else None,
        extract_tables=not args.no_tables,
        extract_figures=not args.no_figures,
        extract_formulas=not args.no_formulas,
        store_embeddings=args.store,
    )
    return result


def cmd_parse(args):
    """Handle the parse command."""
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    path = Path(args.path)
    output_dir = Path(args.output)
    pdfs = find_pdfs(path)

    if not pdfs:
        print(f"No PDF files found at {path}")
        sys.exit(1)

    print(f"Found {len(pdfs)} PDF file(s)")
    print(f"Output: {output_dir.resolve()}")

    options = []
    if args.no_tables:
        options.append("no-tables")
    if args.no_figures:
        options.append("no-figures")
    if args.no_formulas:
        options.append("no-formulas")
    if args.store:
        options.append(f"store:{args.collection}")
    if options:
        print(f"Options: {', '.join(options)}")

    print()

    # For large jobs, suggest using batch mode
    if len(pdfs) > BATCH_THRESHOLD:
        print(f"Processing {len(pdfs)} files (this may take a while)...")
        print(f"Tip: For 10+ files, consider 'meridian batch submit' for queue management.")
        print()

    start_time = time.time()
    succeeded = 0
    failed = 0
    total_chunks = 0

    # Determine parallelism - match Celery concurrency from config
    max_workers = args.workers
    if max_workers is None:
        concurrency = int(os.environ.get("CELERY_CONCURRENCY", "3"))
        max_workers = min(concurrency, len(pdfs))

    if len(pdfs) == 1:
        # Single file - process directly
        pdf = pdfs[0]
        print(f"Processing: {pdf.name}...", end="", flush=True)
        result = process_one(pdf, args)

        if result.get("success"):
            out_path = write_result_json(result, output_dir)
            chunks = result.get("stats", {}).get("chunks", 0)
            total_time = result.get("timing", {}).get("total", 0)
            print(f" done ({chunks} chunks, {total_time:.1f}s)")
            print(f"  -> {out_path}")
            succeeded = 1
            total_chunks = chunks
        else:
            print(f" FAILED: {result.get('error', 'Unknown error')}")
            failed = 1

    else:
        # Multiple files - use thread pool
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            for pdf in pdfs:
                future = executor.submit(process_one, pdf, args)
                futures[future] = pdf

            for i, future in enumerate(as_completed(futures), 1):
                pdf = futures[future]
                try:
                    result = future.result()

                    if result.get("success"):
                        out_path = write_result_json(result, output_dir)
                        chunks = result.get("stats", {}).get("chunks", 0)
                        total_time = result.get("timing", {}).get("total", 0)
                        total_chunks += chunks
                        succeeded += 1
                        print(f"  [{i}/{len(pdfs)}] {pdf.name}: {chunks} chunks ({total_time:.1f}s)")
                    else:
                        failed += 1
                        print(f"  [{i}/{len(pdfs)}] {pdf.name}: FAILED - {result.get('error', 'Unknown')}")

                except Exception as e:
                    failed += 1
                    print(f"  [{i}/{len(pdfs)}] {pdf.name}: ERROR - {e}")

    # Summary
    elapsed = time.time() - start_time
    print()
    print(f"Complete: {succeeded} succeeded, {failed} failed ({elapsed:.1f}s)")
    print(f"Output: {total_chunks} total chunks in {output_dir.resolve()}/")

    if failed > 0:
        sys.exit(1)
