"""
Box Annotator - Draw numbered boxes on formula pages.

Adapted from src/visualize_boxes.py but uses PIL ONLY (no pypdfium2).
Receives page images as base64 from Docling service.

CRITICAL: NO DOCLING IMPORTS (except docling_core.types for data types)
"""

import base64
import logging
from io import BytesIO
from typing import List, Dict, Any, Tuple, Optional

from PIL import Image, ImageDraw, ImageFont

# ALLOWED: Data types only from docling_core
from docling_core.types.doc import TextItem, PictureItem, TableItem

from meridian.config import (
    CONTENT_BLOCK_MIN_WIDTH_PT,
    CONTENT_BLOCK_MIN_HEIGHT_PT,
    CONTENT_BLOCK_MIN_AREA_PT2,
    CONTENT_BLOCK_CONTAINMENT_THRESHOLD,
)
from meridian.utils.bbox import filter_contained_boxes, deduplicate_overlapping_types

logger = logging.getLogger(__name__)


# =============================================================================
# Content Block Extraction
# =============================================================================

def get_content_blocks_for_page(
    docling_doc: dict,
    page_no: int,
    scale: float = 2.0,
) -> List[dict]:
    """
    Get all content blocks for a page from DoclingDocument dict.

    This matches the logic in visualize_boxes.py exactly:
    - Includes TextItem, PictureItem, TableItem
    - Excludes formulas
    - Filters out very small items
    - Sorts top to bottom
    - Removes contained boxes

    Args:
        docling_doc: DoclingDocument as dict (from Docling service)
        page_no: Page number to extract blocks for
        scale: Image scale factor

    Returns:
        List of block dicts with text, label, item_type, bbox
    """
    blocks = []

    # Helper to process items from a list
    def process_items(items: List[dict], item_type: str):
        for item in items:
            # Get provenance
            prov_list = item.get("prov", [])
            if not prov_list:
                continue

            prov = prov_list[0]
            if prov.get("page_no") != page_no:
                continue

            label = item.get("label", "unknown")

            # Skip formulas (detected by VLM)
            if "formula" in str(label).lower():
                continue

            # Get bbox
            bbox = prov.get("bbox")
            if not bbox:
                continue

            # Handle bbox as dict or BoundingBox object
            if isinstance(bbox, dict):
                l, r, t, b = bbox.get("l", 0), bbox.get("r", 0), bbox.get("t", 0), bbox.get("b", 0)
            else:
                # BoundingBox object
                l, r, t, b = bbox.l, bbox.r, bbox.t, bbox.b

            # Filter out very small items
            bbox_width = r - l
            bbox_height = t - b
            bbox_area = bbox_width * bbox_height

            if (bbox_width < CONTENT_BLOCK_MIN_WIDTH_PT and bbox_height < CONTENT_BLOCK_MIN_HEIGHT_PT) \
                    or bbox_area < CONTENT_BLOCK_MIN_AREA_PT2:
                continue

            # Get display text
            if item_type == "text":
                text = item.get("text", "")
                display_text = text[:100] if text else ""
            elif item_type == "picture":
                display_text = "[Picture]"
            elif item_type == "table":
                display_text = "[Table]"
            else:
                display_text = ""

            blocks.append({
                "text": display_text,
                "label": str(label),
                "item_type": item_type,
                "bbox": {
                    "l": l * scale,
                    "t": t * scale,
                    "r": r * scale,
                    "b": b * scale,
                },
            })

    # Process texts
    texts = docling_doc.get("texts", [])
    process_items(texts, "text")

    # Process pictures
    pictures = docling_doc.get("pictures", [])
    process_items(pictures, "picture")

    # Process tables
    tables = docling_doc.get("tables", [])
    process_items(tables, "table")

    # Sort top to bottom (by -bbox.t in Docling coord system)
    blocks.sort(key=lambda x: -x["bbox"]["t"])

    # Filter out contained boxes
    blocks = filter_contained_boxes(blocks, CONTENT_BLOCK_CONTAINMENT_THRESHOLD)

    # Remove duplicate picture/table for same bbox (e.g., pages 41/42)
    blocks = deduplicate_overlapping_types(blocks)

    return blocks


# =============================================================================
# Box Drawing (Pure PIL)
# =============================================================================

def draw_boxes_on_image(
    image: Image.Image,
    blocks: List[dict],
    line_width: int = 3,
) -> Image.Image:
    """
    Draw numbered boxes on image with alternating colors.

    Exactly matches visualize_boxes.py:
    - Odd boxes (1, 3, 5...): RED
    - Even boxes (2, 4, 6...): BLUE
    - Labels positioned to avoid overlapping

    Args:
        image: PIL Image to annotate
        blocks: List of block dicts with bbox
        line_width: Box border width

    Returns:
        Annotated PIL Image
    """
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)

    # Try to load a font
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 28)
    except:
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 28)
        except:
            font = ImageFont.load_default()

    img_height = image.height

    # Colors for odd/even boxes
    color_odd = (220, 20, 20)    # Red
    color_even = (20, 20, 220)   # Blue

    # Convert all boxes to image coords
    box_coords = []
    for block in blocks:
        bbox = block["bbox"]
        x1 = bbox["l"]
        y1 = img_height - bbox["t"]  # Convert from Docling to image coords
        x2 = bbox["r"]
        y2 = img_height - bbox["b"]
        box_coords.append((x1, y1, x2, y2))

    # Track placed label positions
    placed_labels = []

    # Draw boxes and labels
    for i, block in enumerate(blocks):
        box_num = i + 1
        x1, y1, x2, y2 = box_coords[i]

        # Select color
        box_color = color_odd if box_num % 2 == 1 else color_even

        # Draw rectangle
        draw.rectangle([x1, y1, x2, y2], outline=box_color, width=line_width)

        # Calculate label dimensions
        label = f"[{box_num}]"
        label_bbox = draw.textbbox((0, 0), label, font=font)
        label_width = label_bbox[2] - label_bbox[0]
        label_height = label_bbox[3] - label_bbox[1]

        bg_padding = 2
        label_margin = 8

        def check_overlap(lx, ly):
            """Check if label overlaps with boxes or placed labels."""
            lx1, ly1 = lx - bg_padding, ly - bg_padding
            lx2, ly2 = lx + label_width + bg_padding, ly + label_height + bg_padding

            # Check against all boxes
            for j, (ox1, oy1, ox2, oy2) in enumerate(box_coords):
                if i == j:
                    continue
                if not (lx2 < ox1 or lx1 > ox2 or ly2 < oy1 or ly1 > oy2):
                    return True

            # Check against placed labels
            for (plx1, ply1, plx2, ply2) in placed_labels:
                if not (lx2 + label_margin < plx1 or lx1 > plx2 + label_margin or
                        ly2 + label_margin < ply1 or ly1 > ply2 + label_margin):
                    return True

            return False

        # Try placing label on left first
        label_x_left = x1 - label_width - 5
        label_y = y1

        left_overlaps = label_x_left < 0 or check_overlap(label_x_left, label_y)

        # Choose position
        if left_overlaps:
            label_x = x2 + 5
        else:
            label_x = label_x_left

        # Record label position
        placed_labels.append((
            label_x - bg_padding,
            label_y - bg_padding,
            label_x + label_width + bg_padding,
            label_y + label_height + bg_padding
        ))

        # Draw label background
        draw.rectangle(
            [label_x - bg_padding, label_y - bg_padding,
             label_x + label_width + bg_padding, label_y + label_height + bg_padding],
            fill=(255, 255, 255),
            outline=box_color,
            width=1
        )

        # Draw label text
        draw.text((label_x, label_y), label, fill=box_color, font=font)

    return annotated


# =============================================================================
# Main Entry Point
# =============================================================================

def annotate_formula_pages(
    docling_doc: dict,
    formula_pages: List[dict],
    scale: float = 2.0,
) -> Tuple[List[dict], Dict[int, List[dict]]]:
    """
    Draw numbered boxes on formula pages.

    Args:
        docling_doc: DoclingDocument as dict (from Docling service)
        formula_pages: List of {"page_no": int, "image_base64": str}

    Returns:
        annotated_pages: List of {"page_no": int, "image_base64": str, "num_blocks": int}
        blocks_by_page: Dict mapping page_no -> list of blocks (for chunker reference)
    """
    annotated_pages = []
    blocks_by_page = {}

    for page_data in formula_pages:
        page_no = page_data.get("page_no", page_data.get("page", 0))
        image_base64 = page_data.get("image_base64", "")

        if not image_base64:
            logger.warning(f"No image data for formula page {page_no}")
            continue

        # Decode image
        try:
            img_bytes = base64.b64decode(image_base64)
            image = Image.open(BytesIO(img_bytes))
        except Exception as e:
            logger.error(f"Failed to decode image for page {page_no}: {e}")
            continue

        # Get content blocks for this page
        blocks = get_content_blocks_for_page(docling_doc, page_no, scale)
        blocks_by_page[page_no] = blocks

        logger.info(f"Page {page_no}: {len(blocks)} content blocks")

        if not blocks:
            # No blocks to annotate - return original image
            annotated_pages.append({
                "page_no": page_no,
                "image_base64": image_base64,
                "num_blocks": 0,
            })
            continue

        # Draw boxes
        annotated = draw_boxes_on_image(image, blocks)

        # Encode back to base64
        buffered = BytesIO()
        annotated.save(buffered, format="PNG")
        annotated_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        annotated_pages.append({
            "page_no": page_no,
            "image_base64": annotated_base64,
            "num_blocks": len(blocks),
        })

    return annotated_pages, blocks_by_page


def get_formula_pages_from_doc(docling_doc: dict) -> List[int]:
    """
    Get list of page numbers that contain formulas.

    Args:
        docling_doc: DoclingDocument as dict

    Returns:
        List of page numbers with formulas
    """
    formula_pages = set()

    for item in docling_doc.get("texts", []):
        label = item.get("label", "")
        if "formula" in str(label).lower():
            prov_list = item.get("prov", [])
            if prov_list:
                page_no = prov_list[0].get("page_no")
                if page_no:
                    formula_pages.add(page_no)

    return sorted(formula_pages)
