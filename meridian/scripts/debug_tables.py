#!/usr/bin/env python3
"""
Debug script: Save table images extracted by the pipeline.

Shows exactly what the VLM receives — the cropped table image with padding.
Saves as PNGs so you can visually verify bounding boxes are correct.

Usage:
    python scripts/debug_tables.py <pdf_path> [--limit 10] [--output debug_tables]
"""

import argparse
import base64
import json
import sys
from io import BytesIO
from pathlib import Path

from PIL import Image


def save_table_images(pdf_path: str, limit: int = 10, output_dir: str = "debug_tables"):
    # Import here so .env is loaded
    from meridian.clients.docling_client import DoclingClient

    pdf_path = str(Path(pdf_path).resolve())
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Processing: {Path(pdf_path).name}")
    print(f"Output dir: {out.resolve()}")
    print()

    # Step 1: Call Docling (same as the real pipeline)
    client = DoclingClient()
    result = client.process_pdf(
        pdf_path,
        extract_tables=True,
        extract_figures=False,
        extract_formula_pages=False,
    )
    client.close()

    if not result.get("success"):
        print(f"Docling failed: {result.get('error')}")
        sys.exit(1)

    tables = result["extractions"]["tables"]
    pages = result.get("metadata", {}).get("pages", "?")
    print(f"Document: {pages} pages, {len(tables)} tables detected")
    print()

    # Step 2: Save images
    count = min(limit, len(tables))
    manifest = []

    for i, table in enumerate(tables[:count]):
        table_id = table.get("id", f"table_{i}")
        page = table.get("page", "?")
        source = table.get("source", "detected")
        bbox = table.get("bbox", {})
        title = table.get("title") or ""

        # Decode base64 to PNG
        img_data = base64.b64decode(table["image_base64"])
        img = Image.open(BytesIO(img_data))

        filename = f"{table_id}_p{page}_{source}.png"
        filepath = out / filename
        img.save(filepath, "PNG")

        info = {
            "file": filename,
            "table_id": table_id,
            "page": page,
            "source": source,
            "bbox": bbox,
            "title": title,
            "image_size": f"{img.width}x{img.height}",
        }
        manifest.append(info)
        img.close()

        print(f"  [{i+1}/{count}] {filename}  (page {page}, {source}, {img.width}x{img.height})")
        if title:
            print(f"           title: {title}")
        print(f"           bbox:  l={bbox.get('l',0):.0f} t={bbox.get('t',0):.0f} r={bbox.get('r',0):.0f} b={bbox.get('b',0):.0f}")

    # Save manifest
    manifest_path = out / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print()
    print(f"Saved {count} table images to {out.resolve()}/")
    print(f"Manifest: {manifest_path}")

    if len(tables) > limit:
        print(f"\n({len(tables) - limit} more tables not saved — use --limit to increase)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Debug table extraction — save table images as PNGs")
    parser.add_argument("pdf", help="Path to PDF file")
    parser.add_argument("--limit", type=int, default=10, help="Max tables to save (default: 10)")
    parser.add_argument("--output", default="debug_tables", help="Output directory (default: debug_tables)")
    args = parser.parse_args()

    save_table_images(args.pdf, limit=args.limit, output_dir=args.output)
