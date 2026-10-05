#!/usr/bin/env python3
"""
Fetch N PDFs from NASA NTRS into a local directory.

Pages through the search API and downloads any result whose `downloads`
array contains a PDF link. Stops after N successful downloads.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import httpx

API = "https://ntrs.nasa.gov/api/citations"


def search_page(client: httpx.Client, q: str, page_from: int, page_size: int) -> dict:
    body = {
        "page": {"size": page_size, "from": page_from},
    }
    if q:
        body["q"] = q
    r = client.post(f"{API}/search", json=body, timeout=30)
    r.raise_for_status()
    return r.json()


def pick_pdf_url(record: dict) -> str | None:
    for dl in record.get("downloads", []):
        if dl.get("mimetype") == "application/pdf":
            link = dl.get("links", {}).get("pdf") or dl.get("links", {}).get("original")
            if link:
                return link
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="./pdfs/ntrs")
    ap.add_argument("--count", type=int, default=100)
    ap.add_argument("--query", default="apollo")
    ap.add_argument("--page-size", type=int, default=100)
    ap.add_argument("--max-mb", type=float, default=80.0,
                    help="skip PDFs larger than this (avoid huge scans)")
    ap.add_argument("--min-mb", type=float, default=0.05,
                    help="skip PDFs smaller than this (placeholder/cover-only)")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    client = httpx.Client(
        base_url="https://ntrs.nasa.gov",
        timeout=120,
        headers={
            "Accept": "application/json",
            "User-Agent": "Mozilla/5.0 (research; meridian-pipeline)",
        },
    )

    downloaded = 0
    skipped = 0
    page_from = 0
    seen_ids = {p.stem for p in out_dir.glob("*.pdf")}
    if seen_ids:
        print(f"already have {len(seen_ids)} files in {out_dir}, will skip those")
        downloaded = len(seen_ids)

    while downloaded < args.count:
        try:
            data = search_page(client, args.query, page_from, args.page_size)
        except Exception as e:
            print(f"search page from={page_from} failed: {e}", file=sys.stderr)
            break

        results = data.get("results", [])
        if not results:
            print(f"no more results at from={page_from}")
            break

        for rec in results:
            if downloaded >= args.count:
                break
            rid = str(rec.get("id"))
            if rid in seen_ids:
                continue
            url = pick_pdf_url(rec)
            if not url:
                skipped += 1
                continue

            target = out_dir / f"{rid}.pdf"
            try:
                with client.stream("GET", url, timeout=120) as r:
                    if r.status_code != 200:
                        print(f"  {rid}: HTTP {r.status_code}, skipped")
                        skipped += 1
                        continue
                    cl = r.headers.get("content-length")
                    if cl and int(cl) > args.max_mb * 1024 * 1024:
                        print(f"  {rid}: too big ({int(cl)/1024/1024:.1f} MB), skipped")
                        skipped += 1
                        continue
                    with open(target, "wb") as f:
                        for chunk in r.iter_bytes(chunk_size=64 * 1024):
                            f.write(chunk)
                size = target.stat().st_size
                if size < args.min_mb * 1024 * 1024:
                    target.unlink()
                    skipped += 1
                    continue
                downloaded += 1
                print(f"[{downloaded}/{args.count}] {rid} ({size/1024/1024:.1f} MB)")
                seen_ids.add(rid)
            except Exception as e:
                print(f"  {rid}: download failed: {e}", file=sys.stderr)
                skipped += 1

        page_from += args.page_size
        # be nice to NTRS
        time.sleep(0.5)

    client.close()
    print(f"\nDone: {downloaded} downloaded, {skipped} skipped")


if __name__ == "__main__":
    main()
