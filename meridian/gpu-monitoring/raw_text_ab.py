#!/usr/bin/env python3
"""
Three-way A/B of raw-text PDF extractors vs the existing Docling output.

For each PDF in --sample, runs:
  - pdftotext --layout (poppler)
  - pypdfium2 (Google PDFium via Python)
  - PyMuPDF (fitz)

Reports per-tool wall time and lets you spot-check text fidelity by writing
each tool's output to a side-by-side directory tree.

Usage:
    /workspace/meridian/venv/bin/python /workspace/meridian/raw_text_ab.py \
        --sample-from /workspace/meridian/output \
        --pdf-dir ./pdfs \
        --out-dir /workspace/meridian/output_ab \
        --n 30
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pypdfium2 as pdfium  # already installed via meridian deps
import fitz  # PyMuPDF, already installed


def extract_pdftotext(pdf_path: Path) -> tuple[str, float]:
    t0 = time.time()
    r = subprocess.run(
        ["pdftotext", "-layout", "-enc", "UTF-8", str(pdf_path), "-"],
        capture_output=True, text=True, timeout=300,
    )
    return r.stdout, time.time() - t0


def extract_pypdfium2(pdf_path: Path) -> tuple[str, float]:
    t0 = time.time()
    pdf = pdfium.PdfDocument(str(pdf_path))
    parts = []
    for i in range(len(pdf)):
        page = pdf[i]
        textpage = page.get_textpage()
        parts.append(textpage.get_text_range())
        textpage.close()
        page.close()
    pdf.close()
    return "\n\f\n".join(parts), time.time() - t0


def extract_pymupdf(pdf_path: Path) -> tuple[str, float]:
    t0 = time.time()
    doc = fitz.open(str(pdf_path))
    parts = []
    for page in doc:
        # "text" preserves reading order best for multi-column
        parts.append(page.get_text("text"))
    doc.close()
    return "\n\f\n".join(parts), time.time() - t0


EXTRACTORS = {
    "pdftotext": extract_pdftotext,
    "pypdfium2": extract_pypdfium2,
    "pymupdf":   extract_pymupdf,
}


def process_one(args_tuple):
    pdf_path, out_dir = args_tuple
    pdf_path = Path(pdf_path)
    out_dir = Path(out_dir)
    doc_id = pdf_path.stem
    results = {"doc_id": doc_id, "pages_via_pdftotext": None, "tools": {}}

    for name, fn in EXTRACTORS.items():
        try:
            text, elapsed = fn(pdf_path)
        except Exception as e:
            results["tools"][name] = {"ok": False, "error": str(e), "elapsed_s": None}
            continue
        # Persist for spot-check
        per_tool_dir = out_dir / name
        per_tool_dir.mkdir(parents=True, exist_ok=True)
        (per_tool_dir / f"{doc_id}.txt").write_text(text, encoding="utf-8")
        results["tools"][name] = {
            "ok": True,
            "elapsed_s": round(elapsed, 3),
            "chars": len(text),
            "pages_split": text.count("\f") + 1,
        }
    return results


def pick_sample(sample_from: Path, pdf_dir: Path, n: int, seed: int = 7) -> list[Path]:
    """Pick N PDFs we already have Docling output for, weighted to span small/med/large."""
    json_files = sorted(sample_from.glob("*.json"))
    docs = []
    for jf in json_files:
        try:
            d = json.loads(jf.read_text())
            pages = d.get("pages", 0)
            pdf = pdf_dir / f"{jf.stem}.pdf"
            if pdf.exists() and pages > 0:
                docs.append((pdf, pages))
        except Exception:
            continue
    docs.sort(key=lambda x: x[1])
    if not docs:
        sys.exit("No Docling-parsed docs found in --sample-from to pick from.")
    # Stratified: small (<=20p), medium (21-100p), large (>100p)
    small = [d for d in docs if d[1] <= 20]
    med   = [d for d in docs if 20 < d[1] <= 100]
    large = [d for d in docs if d[1] > 100]
    rng = random.Random(seed)
    pick = []
    n_small = max(1, n // 3)
    n_med   = max(1, n // 3)
    n_large = n - n_small - n_med
    for bucket, k in ((small, n_small), (med, n_med), (large, n_large)):
        rng.shuffle(bucket)
        pick.extend(b[0] for b in bucket[:k])
    return pick


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-from", default="/workspace/meridian/output",
                    help="Dir of existing Docling JSONs (used to know page counts)")
    ap.add_argument("--pdf-dir", default="./pdfs")
    ap.add_argument("--out-dir", default="/workspace/meridian/output_ab")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--workers", type=int, default=8,
                    help="Parallel processes for the 3-way extraction (per process: 3 tools serially)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    sample = pick_sample(Path(args.sample_from), Path(args.pdf_dir), args.n)
    print(f"Sample: {len(sample)} PDFs")
    for p in sample:
        print(f"  {p.name}")

    overall_t0 = time.time()
    all_results = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(process_one, (str(p), str(out_dir))) for p in sample]
        for f in as_completed(futs):
            all_results.append(f.result())

    overall = time.time() - overall_t0

    # Aggregate
    print()
    print(f"Wall clock: {overall:.1f}s ({args.workers} parallel processes, 3 tools each)")
    print()
    summary = {}
    for r in all_results:
        for tool, m in r["tools"].items():
            s = summary.setdefault(tool, {"sum_t": 0.0, "sum_chars": 0, "n_ok": 0, "n_fail": 0})
            if m["ok"]:
                s["sum_t"] += m["elapsed_s"]
                s["sum_chars"] += m["chars"]
                s["n_ok"] += 1
            else:
                s["n_fail"] += 1
    print(f"{'tool':<12} {'ok':>4} {'fail':>4} {'tot CPU-s':>10} {'avg s/doc':>10} {'tot chars':>12}")
    for tool, s in summary.items():
        avg = s["sum_t"] / s["n_ok"] if s["n_ok"] else 0
        print(f"{tool:<12} {s['n_ok']:>4} {s['n_fail']:>4} {s['sum_t']:>10.1f} {avg:>10.3f} {s['sum_chars']:>12,}")

    # Save raw results for further analysis
    (out_dir / "_results.json").write_text(json.dumps(all_results, indent=2))
    print()
    print(f"Per-doc text dumps written under {out_dir}/<tool>/")
    print(f"Spot-check a multi-column doc:")
    print(f"  diff <(head -100 {out_dir}/pdftotext/<doc>.txt) <(head -100 {out_dir}/pymupdf/<doc>.txt)")


if __name__ == "__main__":
    main()
