"""
Image Extractors

Extract tables, figures, and formulas as base64 images from processed PDFs.
Combines logic from table_extractor.py, formula_extractor.py, and picture_extractor.py.
"""

import base64
import io
import logging
import re
from typing import List, Optional, Set

from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling_core.types.doc.document import DoclingDocument
from docling_core.types.doc.base import BoundingBox, CoordOrigin
from PIL import Image

from meridian.config import (
    IMAGE_SCALE,
    TABLE_PADDING_PX,
    FORMULA_PADDING_PX,
    PICTURE_PADDING_PX,
    PICTURE_CONTEXT_CHARS,
    FORMULA_VERTICAL_EXPANSION,
    DETECT_MISSED_TABLES,
    DETECT_CONTINUATION_PAGES,
    PICTURE_TABLE_CANDIDATE_MIN_AREA,
)

logger = logging.getLogger(__name__)

# Pattern to match table titles (TABLE I, TABLE 1, TABLE II, etc.)
TABLE_TITLE_PATTERN = re.compile(r'^TABLE\s+([IVXLCDM]+|\d+)', re.IGNORECASE)


def image_to_base64(image: Image.Image, format: str = "PNG") -> str:
    """Convert PIL Image to base64 string and clean up."""
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    result = base64.b64encode(buffer.getvalue()).decode("utf-8")
    # Explicitly close image to release memory
    image.close()
    buffer.close()
    return result


def extract_image_from_bbox(
    backend: PyPdfiumDocumentBackend,
    page_no: int,
    bbox: dict,
    padding: int = 0,
    scale: float = IMAGE_SCALE,
) -> Optional[str]:
    """
    Extract image from PDF page using bounding box.

    Args:
        backend: PDF backend
        page_no: Page number (1-indexed)
        bbox: Bounding box dict with l, t, r, b keys
        padding: Padding around bbox in points
        scale: Image scale factor

    Returns:
        Base64 encoded image string, or None on error
    """
    try:
        page_backend = backend.load_page(page_no - 1)  # 0-indexed

        # Get coordinate origin (default to BOTTOMLEFT for PDF)
        coord_origin = bbox.get("coord_origin", CoordOrigin.BOTTOMLEFT)
        if isinstance(coord_origin, str):
            coord_origin = CoordOrigin.BOTTOMLEFT

        crop_bbox = BoundingBox(
            l=max(0, bbox["l"] - padding),
            t=bbox["t"] + padding,
            r=bbox["r"] + padding,
            b=max(0, bbox["b"] - padding),
            coord_origin=coord_origin,
        )

        image = page_backend.get_page_image(scale=scale, cropbox=crop_bbox)
        result = image_to_base64(image)
        # image_to_base64 now closes the image, but ensure page_backend cleanup
        del page_backend
        return result

    except Exception as e:
        logger.error(f"Error extracting image from page {page_no}: {e}")
        return None


# =============================================================================
# Table Extraction
# =============================================================================

def find_missed_tables(doc: DoclingDocument) -> List[dict]:
    """
    Find tables that were mentioned as section headers but not detected by TableFormer.

    Args:
        doc: The processed DoclingDocument

    Returns:
        List of missed table info dicts with title, page, and bbox
    """
    # Get pages where tables were already detected
    detected_table_pages = set()
    for table in doc.tables:
        if table.prov:
            detected_table_pages.add(table.prov[0].page_no)

    missed_tables = []
    items_list = list(doc.texts)

    for i, item in enumerate(items_list):
        if item.label != "section_header":
            continue

        text = item.text.strip()
        if not TABLE_TITLE_PATTERN.match(text):
            continue

        if not item.prov:
            continue

        page_no = item.prov[0].page_no
        title_bbox = item.prov[0].bbox

        # Check if already detected
        already_detected = False
        for table in doc.tables:
            if table.prov and table.prov[0].page_no == page_no:
                table_bbox = table.prov[0].bbox
                if abs(title_bbox.t - table_bbox.t) < 50:
                    already_detected = True
                    break

        if already_detected:
            continue

        # Collect content items that follow
        content_items = [item]
        for j in range(i + 1, len(items_list)):
            next_item = items_list[j]
            if not next_item.prov:
                continue

            next_page = next_item.prov[0].page_no
            if next_page != page_no:
                break

            if next_item.label == "section_header" and TABLE_TITLE_PATTERN.match(next_item.text.strip()):
                break

            if next_item.label == "section_header":
                upper_text = next_item.text.upper()
                if any(kw in upper_text for kw in ["REFERENCE", "CONCLUSION", "APPENDIX", "FIGURE"]):
                    break

            content_items.append(next_item)

        if content_items:
            all_bboxes = [item.prov[0].bbox for item in content_items if item.prov]
            if all_bboxes:
                missed_tables.append({
                    "title": text,
                    "page_no": page_no,
                    "bbox": {
                        "l": min(b.l for b in all_bboxes),
                        "t": max(b.t for b in all_bboxes),
                        "r": max(b.r for b in all_bboxes),
                        "b": min(b.b for b in all_bboxes),
                        "coord_origin": all_bboxes[0].coord_origin,
                    },
                })
                logger.debug(f"Found missed table: '{text}' on page {page_no}")

    return missed_tables


def find_continuation_pages(doc: DoclingDocument) -> List[dict]:
    """
    Find pages that likely contain table continuations.

    Args:
        doc: The processed DoclingDocument

    Returns:
        List of continuation page info dicts
    """
    detected_pages = sorted(set(
        table.prov[0].page_no for table in doc.tables if table.prov
    ))

    if len(detected_pages) < 2:
        return []

    continuation_pages = []

    # Find gap pages between consecutive detected tables
    for i in range(len(detected_pages) - 1):
        start_page = detected_pages[i]
        end_page = detected_pages[i + 1]

        for gap_page in range(start_page + 1, end_page):
            if gap_page not in detected_pages:
                page_items = [
                    item for item in doc.texts
                    if item.prov and item.prov[0].page_no == gap_page
                ]

                if page_items:
                    all_bboxes = [item.prov[0].bbox for item in page_items]
                    continuation_pages.append({
                        "title": f"Table continuation (page {gap_page})",
                        "page_no": gap_page,
                        "bbox": {
                            "l": min(b.l for b in all_bboxes),
                            "t": max(b.t for b in all_bboxes),
                            "r": max(b.r for b in all_bboxes),
                            "b": min(b.b for b in all_bboxes),
                            "coord_origin": all_bboxes[0].coord_origin,
                        },
                        "reason": "gap_page",
                    })

    # Look for "Continued" section headers
    for item in doc.texts:
        if item.label != "section_header":
            continue

        text = item.text.strip().lower()
        if "continued" not in text:
            continue

        if not item.prov:
            continue

        page_no = item.prov[0].page_no

        if page_no in detected_pages:
            continue
        if any(cp["page_no"] == page_no for cp in continuation_pages):
            continue

        page_items = [
            item for item in doc.texts
            if item.prov and item.prov[0].page_no == page_no
        ]

        if page_items:
            all_bboxes = [item.prov[0].bbox for item in page_items]
            continuation_pages.append({
                "title": f"Table continuation: {item.text.strip()[:50]}",
                "page_no": page_no,
                "bbox": {
                    "l": min(b.l for b in all_bboxes),
                    "t": max(b.t for b in all_bboxes),
                    "r": max(b.r for b in all_bboxes),
                    "b": min(b.b for b in all_bboxes),
                    "coord_origin": all_bboxes[0].coord_origin,
                },
                "reason": "continued_header",
            })

    return continuation_pages


def extract_tables(
    doc: DoclingDocument,
    backend: PyPdfiumDocumentBackend,
    detect_missed: bool = DETECT_MISSED_TABLES,
    detect_continuation: bool = DETECT_CONTINUATION_PAGES,
    scale: float = IMAGE_SCALE,
) -> List[dict]:
    """
    Extract all tables from the document.

    Args:
        doc: Processed DoclingDocument
        scale: Image scale factor (default 2.0)
        backend: PDF backend for image extraction
        detect_missed: Whether to detect missed tables
        detect_continuation: Whether to detect continuation pages

    Returns:
        List of table dicts with id, page, bbox, image_base64, source, etc.
    """
    tables = []
    table_idx = 0

    # 1. Extract detected tables
    for i, table in enumerate(doc.tables):
        if not table.prov or len(table.prov) == 0:
            continue

        prov = table.prov[0]
        page_no = prov.page_no
        bbox = prov.bbox

        bbox_dict = {"l": bbox.l, "t": bbox.t, "r": bbox.r, "b": bbox.b, "coord_origin": bbox.coord_origin}
        image_b64 = extract_image_from_bbox(backend, page_no, bbox_dict, TABLE_PADDING_PX, scale=scale)

        if image_b64:
            tables.append({
                "id": f"table_{table_idx}",
                "page": page_no,
                "bbox": {"l": bbox.l, "t": bbox.t, "r": bbox.r, "b": bbox.b},
                "image_base64": image_b64,
                "source": "detected",
                "title": None,
                "structure": None,  # Could add table structure data here
            })
            table_idx += 1
            logger.debug(f"Extracted detected table {i} from page {page_no}")

    # 2. Extract missed tables
    if detect_missed:
        missed = find_missed_tables(doc)
        for mt in missed:
            image_b64 = extract_image_from_bbox(backend, mt["page_no"], mt["bbox"], TABLE_PADDING_PX, scale=scale)
            if image_b64:
                tables.append({
                    "id": f"table_{table_idx}",
                    "page": mt["page_no"],
                    "bbox": {k: v for k, v in mt["bbox"].items() if k != "coord_origin"},
                    "image_base64": image_b64,
                    "source": "missed",
                    "title": mt["title"],
                })
                table_idx += 1

    # 3. Extract continuation pages
    if detect_continuation:
        continuations = find_continuation_pages(doc)
        for cp in continuations:
            image_b64 = extract_image_from_bbox(backend, cp["page_no"], cp["bbox"], TABLE_PADDING_PX, scale=scale)
            if image_b64:
                tables.append({
                    "id": f"table_{table_idx}",
                    "page": cp["page_no"],
                    "bbox": {k: v for k, v in cp["bbox"].items() if k != "coord_origin"},
                    "image_base64": image_b64,
                    "source": "continuation",
                    "title": cp["title"],
                    "reason": cp.get("reason"),
                })
                table_idx += 1

    # 4. Large pictures as table candidates — VLM decides if they contain tables
    if PICTURE_TABLE_CANDIDATE_MIN_AREA > 0:
        candidate_count = 0
        for item in doc.pictures:
            if not item.prov or len(item.prov) == 0:
                continue

            prov = item.prov[0]
            page_no = prov.page_no
            bbox = prov.bbox

            # Check area threshold
            width = abs(bbox.r - bbox.l)
            height = abs(bbox.t - bbox.b)
            area = width * height
            if area < PICTURE_TABLE_CANDIDATE_MIN_AREA:
                continue

            bbox_dict = {"l": bbox.l, "t": bbox.t, "r": bbox.r, "b": bbox.b, "coord_origin": bbox.coord_origin}
            image_b64 = extract_image_from_bbox(backend, page_no, bbox_dict, TABLE_PADDING_PX, scale=scale)

            if image_b64:
                tables.append({
                    "id": f"table_{table_idx}",
                    "page": page_no,
                    "bbox": {"l": bbox.l, "t": bbox.t, "r": bbox.r, "b": bbox.b},
                    "image_base64": image_b64,
                    "source": "picture_table",
                    "title": None,
                })
                table_idx += 1
                candidate_count += 1
                logger.debug(f"Large picture on page {page_no} ({area:.0f} pt²) added as table candidate")

        if candidate_count:
            logger.info(f"Added {candidate_count} large pictures as table candidates")

    # Sort by document order
    tables.sort(key=lambda t: (t["page"], -t["bbox"]["t"]))

    pic_table_count = sum(1 for t in tables if t.get('source') == 'picture_table')
    logger.info(f"Extracted {len(tables)} tables: "
                f"{sum(1 for t in tables if t['source'] == 'detected')} detected, "
                f"{sum(1 for t in tables if t['source'] == 'missed')} missed, "
                f"{sum(1 for t in tables if t['source'] == 'continuation')} continuation"
                f"{f', {pic_table_count} picture candidates' if pic_table_count else ''}")

    return tables


# =============================================================================
# Figure/Picture Extraction
# =============================================================================

def extract_figures(
    doc: DoclingDocument,
    backend: PyPdfiumDocumentBackend,
    context_chars: int = PICTURE_CONTEXT_CHARS,
    scale: float = IMAGE_SCALE,
) -> List[dict]:
    """
    Extract all figures/pictures from the document with surrounding context.

    Args:
        doc: Processed DoclingDocument
        backend: PDF backend for image extraction
        scale: Image scale factor (default 2.0)
        context_chars: Characters of context to extract above/below

    Returns:
        List of figure dicts with id, page, bbox, image_base64, context, etc.
    """
    # Collect text items by page for context extraction
    text_items_by_page = {}
    for item in doc.texts:
        if item.prov and len(item.prov) > 0:
            page_no = item.prov[0].page_no
            if page_no not in text_items_by_page:
                text_items_by_page[page_no] = []
            text_items_by_page[page_no].append({
                "text": item.text or "",
                "bbox": item.prov[0].bbox,
            })

    # Sort by vertical position
    for page_no in text_items_by_page:
        text_items_by_page[page_no].sort(key=lambda x: -x["bbox"].t)

    figures = []
    figure_idx = 0

    for item in doc.pictures:
        if not item.prov or len(item.prov) == 0:
            continue

        prov = item.prov[0]
        page_no = prov.page_no
        bbox = prov.bbox

        # Extract image
        bbox_dict = {"l": bbox.l, "t": bbox.t, "r": bbox.r, "b": bbox.b, "coord_origin": bbox.coord_origin}
        image_b64 = extract_image_from_bbox(backend, page_no, bbox_dict, PICTURE_PADDING_PX, scale=scale)

        if not image_b64:
            continue

        # Extract context
        context_above = ""
        context_below = ""

        if page_no in text_items_by_page:
            page_texts = text_items_by_page[page_no]

            texts_above = [t for t in page_texts if t["bbox"].b > bbox.t]
            texts_below = [t for t in page_texts if t["bbox"].t < bbox.b]

            if texts_above:
                texts_above_sorted = sorted(texts_above, key=lambda x: x["bbox"].b)
                above_text = " ".join([t["text"] for t in texts_above_sorted])
                context_above = above_text[-context_chars:] if len(above_text) > context_chars else above_text

            if texts_below:
                texts_below_sorted = sorted(texts_below, key=lambda x: -x["bbox"].t)
                below_text = " ".join([t["text"] for t in texts_below_sorted])
                context_below = below_text[:context_chars] if len(below_text) > context_chars else below_text

        figures.append({
            "id": f"figure_{figure_idx}",
            "page": page_no,
            "bbox": {"l": bbox.l, "t": bbox.t, "r": bbox.r, "b": bbox.b},
            "image_base64": image_b64,
            "context_above": context_above.strip(),
            "context_below": context_below.strip(),
        })
        figure_idx += 1
        logger.debug(f"Extracted figure {figure_idx - 1} from page {page_no}")

    logger.info(f"Extracted {len(figures)} figures")
    return figures


# =============================================================================
# Formula Page Extraction
# =============================================================================


def get_formula_pages(doc: DoclingDocument) -> Set[int]:
    """
    Get pages that contain formulas.

    COPIED FROM BATTLE-TESTED CODE: src/visualize_boxes.py:77-84
    DO NOT MODIFY without checking the original.
    """
    formula_pages = set()
    for item in doc.texts:
        if "formula" in str(item.label).lower():
            if item.prov and len(item.prov) > 0:
                formula_pages.add(item.prov[0].page_no)
    return formula_pages


def extract_formula_pages(
    doc: DoclingDocument,
    backend: PyPdfiumDocumentBackend,
    scale: float = IMAGE_SCALE,
) -> List[dict]:
    """
    Extract FULL PAGE images for pages that contain formulas.

    IMPORTANT: This returns full-page images, NOT cropped formula regions.
    The worker will draw numbered boxes on these pages for VLM processing.
    This design keeps the box numbering logic (visualize_boxes) in the worker,
    which is tightly coupled with unified_chunker for correct formula placement.

    PAGE EXTRACTION LOGIC COPIED FROM BATTLE-TESTED CODE:
    - get_formula_pages(): src/visualize_boxes.py:77-84
    - Page image extraction: src/visualize_boxes.py:311-312, 533-534

    Args:
        doc: Processed DoclingDocument
        backend: PDF backend for image extraction
        scale: Scale factor for page images (default 2.0 = 144 DPI)

    Returns:
        List of formula page dicts with page number and full-page image.
        Format:
        [
            {
                "page": 7,
                "image_base64": "...",  # Full page image
            }
        ]
    """
    # Step 1: Find all pages that contain formulas
    # Uses battle-tested get_formula_pages() helper
    formula_page_set = get_formula_pages(doc)

    if not formula_page_set:
        logger.info("No formula pages detected")
        return []

    logger.info(f"Detected formulas on {len(formula_page_set)} pages: {sorted(formula_page_set)}")

    # Step 2: Extract full-page images for each formula page
    # Page extraction logic from src/visualize_boxes.py:311-312, 533-534
    results = []

    for page_no in sorted(formula_page_set):
        try:
            # Load page (0-indexed) - matches visualize_boxes.py:311, 533
            page_backend = backend.load_page(page_no - 1)

            # Extract FULL PAGE image - matches visualize_boxes.py:312, 534
            # No cropbox = entire page
            page_image = page_backend.get_page_image(scale=scale)

            # Convert to base64 (also closes the image)
            image_b64 = image_to_base64(page_image)

            results.append({
                "page": page_no,
                "image_base64": image_b64,
            })

            # Cleanup page backend
            del page_backend

            logger.debug(f"Extracted full page {page_no}")

        except Exception as e:
            logger.error(f"Error extracting formula page {page_no}: {e}")
            continue

    logger.info(f"Extracted {len(results)} formula page images")
    return results


# =============================================================================
# Combined Extraction
# =============================================================================

def extract_all(
    doc: DoclingDocument,
    backend: PyPdfiumDocumentBackend,
    extract_tables_flag: bool = True,
    extract_figures_flag: bool = True,
    extract_formula_pages_flag: bool = True,
    scale: float = IMAGE_SCALE,
) -> dict:
    """
    Extract all content (tables, figures, formula pages) from the document.

    Args:
        doc: Processed DoclingDocument
        backend: PDF backend for image extraction
        extract_tables_flag: Whether to extract tables
        extract_figures_flag: Whether to extract figures
        extract_formula_pages_flag: Whether to extract formula page images
        scale: Image scale factor (default 2.0, use lower for large files)

    Returns:
        Dict with tables, figures, formula_pages lists.
        Note: formula_pages contains FULL PAGE images, not cropped formula regions.
        Workers will draw numbered boxes on these pages for VLM processing.
    """
    result = {
        "tables": [],
        "figures": [],
        "formula_pages": [],
    }

    if extract_tables_flag:
        result["tables"] = extract_tables(doc, backend, scale=scale)

    if extract_figures_flag:
        result["figures"] = extract_figures(doc, backend, scale=scale)

    if extract_formula_pages_flag:
        result["formula_pages"] = extract_formula_pages(doc, backend, scale=scale)

    return result
