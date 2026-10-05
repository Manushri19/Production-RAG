#!/usr/bin/env python3
"""
Measure Bedrock VLM cost and throughput over a directory of PDFs.

WHY THIS EXISTS
---------------
Meridian's VLM step normally runs on a locally hosted vLLM server, which needs
a GPU large enough to hold a vision model alongside Docling. The Bedrock
backend (``VLM_BACKEND=bedrock``) removes that requirement by sending the VLM
work to an API instead. That trade is only worth making if you know what it
costs, so this script was written to answer one question: what does a document
actually cost, end to end, on Bedrock?

It runs the real pipeline over a folder of PDFs and reports per-document
timings, token counts and dollar cost, then extrapolates to a corpus size of
your choosing. It is a measurement tool, not a test -- the test suite lives in
``tests/``. Nothing here asserts anything or fails a build.

DISCLAIMERS
-----------
* **Prices go stale.** ``--price-in`` / ``--price-out`` default to Amazon Nova 2
  Lite on-demand pricing in ``us-east-1`` as of early 2026. Check the current
  AWS pricing page and pass your own numbers rather than trusting these.
* **Costs are estimates.** They are computed from the token counts the Bedrock
  Converse API reports back, so they exclude retried requests and any Bedrock
  features billed separately.
* **Your numbers will differ.** Throughput depends on your account's TPS quota,
  ``BEDROCK_MAX_CONCURRENCY``, document complexity, and how many tables and
  figures per page your corpus actually contains.

PREREQUISITES
-------------
* Docling service reachable at ``http://localhost:8001``
* ``pip install 'meridian[bedrock]'``
* AWS credentials available to boto3, and model access enabled in your region
* ``VLM_BACKEND=bedrock`` in the environment (this script sets it if unset)

USAGE
-----
    python examples/bedrock_cost_benchmark.py ./pdfs
    python examples/bedrock_cost_benchmark.py ./pdfs --out results.csv --project-to 108000
    python examples/bedrock_cost_benchmark.py ./pdfs --price-in 0.06 --price-out 0.24
"""

import argparse
import csv
import logging
import os
import sys
import time
from pathlib import Path

# The backend is resolved at import time inside meridian.workers.document_worker,
# so this must be set before that module is imported below.
os.environ.setdefault("VLM_BACKEND", "bedrock")

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
for noisy in ("httpx", "botocore", "urllib3", "meridian.clients.docling_client"):
    logging.getLogger(noisy).setLevel(logging.ERROR)

from meridian.clients import vlm_client_bedrock  # noqa: E402
from meridian.workers.document_worker import process_single_document  # noqa: E402


# Amazon Nova 2 Lite, us-east-1, on-demand, USD per 1M tokens. See disclaimer above.
DEFAULT_PRICE_IN_PER_M = 0.06
DEFAULT_PRICE_OUT_PER_M = 0.24

CSV_FIELDS = [
    "name", "size_mb", "success", "pages", "tables", "figures", "formula_pages",
    "chunks", "t_docling", "t_vlm", "t_total", "wall_s", "in_tok", "out_tok",
    "cost_usd", "error",
]


def collect_usage(doc_result):
    """Sum input/output tokens the Bedrock client recorded on each VLM result.

    The Bedrock client attaches ``_usage`` to every VLM result; the chunker
    carries it through into chunk metadata but never aggregates it.
    """
    in_tok = out_tok = 0
    for chunk in doc_result.get("chunks", []):
        usage = chunk.get("metadata", {}).get("_usage") or {}
        in_tok += int(usage.get("inputTokens", 0) or 0)
        out_tok += int(usage.get("outputTokens", 0) or 0)
    return in_tok, out_tok


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description="Measure Bedrock VLM cost and throughput over a directory of PDFs.",
    )
    ap.add_argument("pdf_dir", type=Path, help="Directory of PDFs to process")
    ap.add_argument("--out", type=Path, default=Path("bedrock_benchmark.csv"),
                    help="CSV output path (default: ./bedrock_benchmark.csv)")
    ap.add_argument("--price-in", type=float, default=DEFAULT_PRICE_IN_PER_M,
                    help=f"USD per 1M input tokens (default: {DEFAULT_PRICE_IN_PER_M})")
    ap.add_argument("--price-out", type=float, default=DEFAULT_PRICE_OUT_PER_M,
                    help=f"USD per 1M output tokens (default: {DEFAULT_PRICE_OUT_PER_M})")
    ap.add_argument("--project-to", type=int, default=0, metavar="N",
                    help="Also project cost and time to a corpus of N documents")
    return ap.parse_args(argv)


def blank_row(pdf, size_mb, wall, error):
    row = {field: 0 for field in CSV_FIELDS}
    row.update(name=pdf.name, size_mb=round(size_mb, 2), success=False,
               t_total=round(wall, 1), wall_s=round(wall, 1), error=error[:200])
    return row


def main(argv=None):
    args = parse_args(argv)

    pdfs = sorted(args.pdf_dir.glob("*.pdf"))
    if not pdfs:
        sys.exit(f"No PDFs found in {args.pdf_dir}")

    print(f"=== Bedrock benchmark: {len(pdfs)} PDFs ===")
    print(f"Model:            {vlm_client_bedrock.BEDROCK_MODEL_ID}")
    print(f"Region:           {vlm_client_bedrock.BEDROCK_REGION}")
    print(f"Concurrency cap:  {vlm_client_bedrock.BEDROCK_MAX_CONCURRENCY}")
    print(f"Pricing (per 1M): ${args.price_in} in / ${args.price_out} out")
    print(f"Output CSV:       {args.out}\n")

    rows = []
    t_start = time.time()

    for i, pdf in enumerate(pdfs, 1):
        size_mb = pdf.stat().st_size / 1024 / 1024
        print(f"[{i}/{len(pdfs)}] {pdf.name}  ({size_mb:.1f} MB) ... ", end="", flush=True)
        t0 = time.time()
        try:
            result = process_single_document(
                doc_id=pdf.stem,
                pdf_path=str(pdf),
                store_embeddings=False,
            )
        except Exception as exc:  # noqa: BLE001 -- a crash is a data point, keep going
            wall = time.time() - t0
            print(f"CRASH  {wall:.1f}s  {type(exc).__name__}: {str(exc)[:120]}")
            rows.append(blank_row(pdf, size_mb, wall, f"{type(exc).__name__}: {exc}"))
            continue

        wall = time.time() - t0
        stats = result.get("stats", {})
        timing = result.get("timing", {})
        in_tok, out_tok = collect_usage(result)
        cost = in_tok / 1_000_000 * args.price_in + out_tok / 1_000_000 * args.price_out
        success = result.get("success", False)

        row = {
            "name": pdf.name,
            "size_mb": round(size_mb, 2),
            "success": success,
            "pages": stats.get("pages", 0),
            "tables": stats.get("tables_detected", 0),
            "figures": stats.get("figures_detected", 0),
            "formula_pages": stats.get("formula_pages", 0),
            "chunks": stats.get("chunks", 0),
            "t_docling": round(timing.get("docling", 0), 1),
            "t_vlm": round(timing.get("vlm", 0), 1),
            "t_total": round(timing.get("total", 0), 1),
            "wall_s": round(wall, 1),
            "in_tok": in_tok,
            "out_tok": out_tok,
            "cost_usd": round(cost, 4),
            "error": (result.get("error") or "")[:200],
        }
        rows.append(row)
        print(f"{'OK ' if success else 'FAIL'}  {row['pages']}p  "
              f"vlm={row['t_vlm']}s  total={row['wall_s']}s  ${row['cost_usd']:.4f}")

    with args.out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    total_wall = time.time() - t_start
    ok = sum(1 for r in rows if r["success"])
    cost = sum(r["cost_usd"] for r in rows)
    per_doc_cost = cost / max(ok, 1)
    per_doc_time = total_wall / max(ok, 1)

    print("\n=== Summary ===")
    print(f"Docs:       {ok}/{len(rows)} succeeded")
    print(f"Pages:      {sum(r['pages'] for r in rows)}")
    print(f"Chunks:     {sum(r['chunks'] for r in rows)}")
    print(f"Tables:     {sum(r['tables'] for r in rows)} | "
          f"Figures: {sum(r['figures'] for r in rows)}")
    print(f"Tokens:     {sum(r['in_tok'] for r in rows):,} in / "
          f"{sum(r['out_tok'] for r in rows):,} out")
    print(f"Cost:       ${cost:.3f} for this batch  ->  ${per_doc_cost:.4f}/doc avg")
    print(f"Wall time:  {total_wall:.1f}s ({per_doc_time:.1f}s/doc avg)")

    if args.project_to:
        n = args.project_to
        print(f"\nProjection to {n:,} docs at this rate (estimate, retail pricing):")
        print(f"  Cost:      ${per_doc_cost * n:,.0f}")
        print(f"  Sequential time: {per_doc_time * n / 3600:,.0f} hrs "
              f"(divide by your worker count for wall time)")

    print(f"\nCSV written to {args.out}")


if __name__ == "__main__":
    main()
