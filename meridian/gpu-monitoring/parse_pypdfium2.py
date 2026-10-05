#!/usr/bin/env python3
"""
Production text-only parser using pypdfium2.

Bulk-parse a directory of PDFs into per-doc JSON files at ~800 docs/min on
this Blackwell pod (CPU-bound; GPU not used). Output shape is the minimum
needed to embed later — keeps page_texts as the unit of work.

Quality-flagging built in:
    - Soft-hyphen cleanup (joins line-break hyphens common in Apollo PDFs)
    - Two-part low-quality detection (per page → per doc):
        a. Empty/sparse: page has <100 alphanumeric characters
        b. Garbled: alphanum / non-whitespace ratio <0.7 (catches OCR garbage
           where the page has letters but most chars are punctuation noise)
      A doc is is_low_quality=true if >40% of pages match either condition.
    Both empty scans (no text layer) and OCR-garbage scans (corrupt text
    layer like Apollo doc 19690026608) get caught.

Output JSON shape:
    {
        "document_id": str,
        "source": str,
        "pages": int,
        "extractor": "pypdfium2",
        "elapsed_s": float,
        "is_low_quality": bool,
        "quality_stats": {
            "low_alphanum_pages": int,
            "total_pages": int,
            "alphanum_chars": int
        },
        "page_texts": list[str]
    }

Usage:
    /workspace/meridian/venv/bin/python /workspace/meridian/parse_pypdfium2.py \\
        --pdf-dir ./pdfs \\
        --out-dir /workspace/meridian/output_pypdfium2 \\
        --workers 32
"""
from __future__ import annotations

import argparse
import json
import re
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pypdfium2 as pdfium

ALPHANUM_RE = re.compile(r"[A-Za-z0-9]")
LOW_PAGE_ALPHANUM = 100         # page has <this many alphanumerics → empty/sparse
LOW_PAGE_ALNUM_RATIO = 0.70     # alphanum / non-whitespace below this → garbled
GARBLED_NONEMPTY_FRAC = 0.30    # doc with >this fraction of garbled non-empty pages → needs OCR


def page_class(text: str) -> str:
    """Classify a page as 'ok', 'empty', or 'garbled'.

    - 'empty'   : <100 alphanumeric characters. Almost always a figure page
                  with just a caption — text content is fine, OCR won't help.
    - 'garbled' : has plenty of letters but mostly punctuation noise
                  (alphanum / non-whitespace < 0.7). Indicates a corrupted
                  embedded text layer. OCR can recover real text by re-reading
                  the page image.
    - 'ok'      : real text.
    """
    alphanum = sum(1 for c in text if c.isalnum())
    if alphanum < LOW_PAGE_ALPHANUM:
        return "empty"
    non_ws = sum(1 for c in text if not c.isspace())
    if non_ws == 0:
        return "empty"
    if (alphanum / non_ws) < LOW_PAGE_ALNUM_RATIO:
        return "garbled"
    return "ok"


def clean_soft_hyphens(text: str) -> str:
    """Drop soft hyphens (U+00AD) and the corrupted-BOM (U+FFFE) seen in
    Apollo PDFs. These appear at line-break hyphenation points and should
    be removed so words rejoin cleanly."""
    return text.replace("­", "").replace("￾", "")


def parse_one(pdf_path_str: str, out_dir_str: str) -> dict:
    pdf_path = Path(pdf_path_str)
    out_dir = Path(out_dir_str)
    doc_id = pdf_path.stem
    out_path = out_dir / f"{doc_id}.json"

    t0 = time.time()
    try:
        pdf = pdfium.PdfDocument(str(pdf_path))
        page_texts: list[str] = []
        for i in range(len(pdf)):
            page = pdf[i]
            tp = page.get_textpage()
            page_texts.append(clean_soft_hyphens(tp.get_text_range()))
            tp.close()
            page.close()
        pdf.close()
        elapsed = time.time() - t0

        page_classes = [page_class(p) for p in page_texts]
        empty_pages   = sum(1 for c in page_classes if c == "empty")
        garbled_pages = sum(1 for c in page_classes if c == "garbled")
        total_alphanum = sum(len(ALPHANUM_RE.findall(p)) for p in page_texts)
        # A doc "needs_ocr" if many of its non-empty pages are GARBLED.
        # Excluding empty pages from the denominator avoids false-flagging
        # figure-heavy docs (Apollo reports often have 50%+ figure pages
        # whose text content is just a brief caption).
        n = len(page_texts)
        non_empty = n - empty_pages
        needs_ocr = (non_empty > 0 and garbled_pages / non_empty > GARBLED_NONEMPTY_FRAC)

        result = {
            "document_id": doc_id,
            "source": str(pdf_path),
            "pages": n,
            "extractor": "pypdfium2",
            "elapsed_s": round(elapsed, 3),
            "needs_ocr": needs_ocr,
            "quality_stats": {
                "ok_pages": n - empty_pages - garbled_pages,
                "empty_pages": empty_pages,
                "garbled_pages": garbled_pages,
                "total_pages": n,
                "alphanum_chars": total_alphanum,
            },
            "page_classes": page_classes,
            "page_texts": page_texts,
        }
        out_path.write_text(json.dumps(result, ensure_ascii=False))
        return {
            "doc_id": doc_id, "ok": True, "pages": n,
            "chars": sum(len(t) for t in page_texts),
            "elapsed_s": elapsed, "needs_ocr": needs_ocr,
            "garbled_pages": garbled_pages, "empty_pages": empty_pages,
        }
    except Exception as e:
        return {
            "doc_id": doc_id, "ok": False, "error": str(e),
            "elapsed_s": time.time() - t0,
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf-dir", default="./pdfs")
    ap.add_argument("--out-dir", default="/workspace/meridian/output_pypdfium2")
    ap.add_argument("--workers", type=int, default=32)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pdfs = sorted(Path(args.pdf_dir).glob("*.pdf"))
    print(f"PDFs: {len(pdfs)}  workers: {args.workers}  out: {out_dir}")

    t0 = time.time()
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(parse_one, str(p), str(out_dir)): p for p in pdfs}
        done = 0
        for f in as_completed(futs):
            r = f.result()
            results.append(r)
            done += 1
            if done % 100 == 0 or done == len(pdfs):
                tag = ("ok" if r["ok"] else "FAIL")
                if r.get("is_low_quality"):
                    tag += " low-quality"
                print(f"  [{done}/{len(pdfs)}] last: {r['doc_id']} ({tag}, {r.get('elapsed_s', 0):.2f}s)")

    wall = time.time() - t0
    ok = sum(1 for r in results if r["ok"])
    fail = len(results) - ok
    needs_ocr = sum(1 for r in results if r.get("needs_ocr"))
    pages = sum(r.get("pages", 0) for r in results if r["ok"])
    chars = sum(r.get("chars", 0) for r in results if r["ok"])
    cpu_s = sum(r.get("elapsed_s", 0) for r in results if r["ok"])
    sum_garbled = sum(r.get("garbled_pages", 0) for r in results if r["ok"])
    sum_empty   = sum(r.get("empty_pages",   0) for r in results if r["ok"])

    print()
    print(f"Wall clock      : {wall:.1f}s ({wall/60:.1f} min)")
    print(f"Docs            : {ok} ok ({needs_ocr} need OCR), {fail} fail")
    print(f"Pages           : {pages:,}  (garbled={sum_garbled}, empty={sum_empty})")
    print(f"Total pages     : {pages:,}")
    print(f"Total chars     : {chars:,}")
    print(f"Sum CPU-s       : {cpu_s:.1f}s   (= effective parallel: {cpu_s/wall:.1f}x of {args.workers})")
    print(f"Throughput      : {ok/wall*60:.1f} docs/min  =  {pages/wall*60:.0f} pages/min")
    print(f"108k extrapolation @ this rate: {108_000/(ok/wall)/3600:.2f} hours")

    # Persist run metadata + needs-OCR list
    (out_dir / "_run_meta.json").write_text(json.dumps({
        "wall_s": wall, "workers": args.workers,
        "ok": ok, "fail": fail, "needs_ocr": needs_ocr,
        "pages": pages, "chars": chars, "cpu_s": cpu_s,
        "sum_garbled_pages": sum_garbled, "sum_empty_pages": sum_empty,
    }, indent=2))
    needs_ocr_ids = sorted(r["doc_id"] for r in results if r.get("needs_ocr"))
    (out_dir / "_needs_ocr_list.txt").write_text("\n".join(needs_ocr_ids) + "\n")
    if fail:
        failed_ids = sorted(r["doc_id"] for r in results if not r["ok"])
        (out_dir / "_failed_list.txt").write_text("\n".join(failed_ids) + "\n")

    print()
    print(f"Needs-OCR doc IDs   : {out_dir}/_needs_ocr_list.txt ({needs_ocr} docs)")
    if fail:
        print(f"Failed doc IDs      : {out_dir}/_failed_list.txt ({fail} docs)")


if __name__ == "__main__":
    main()
