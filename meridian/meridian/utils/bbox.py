"""
Bounding Box Utilities

Deduplicated from workers/chunker.py and workers/box_annotator.py.
These functions are byte-identical in both files - consolidated here as
the single source of truth.

CRITICAL: Both chunker and box_annotator MUST use these functions to
ensure identical box counting for correct formula placement.
"""

import logging
from typing import List, Tuple

logger = logging.getLogger(__name__)


def calculate_intersection_area(
    box1: Tuple[float, float, float, float],
    box2: Tuple[float, float, float, float]
) -> float:
    """Calculate intersection area of two boxes (l, b, r, t)."""
    l1, b1, r1, t1 = box1
    l2, b2, r2, t2 = box2

    inter_left = max(l1, l2)
    inter_bottom = max(b1, b2)
    inter_right = min(r1, r2)
    inter_top = min(t1, t2)

    if inter_left >= inter_right or inter_bottom >= inter_top:
        return 0.0

    return (inter_right - inter_left) * (inter_top - inter_bottom)


def calculate_box_area(box: Tuple[float, float, float, float]) -> float:
    """Calculate area of a box (l, b, r, t)."""
    l, b, r, t = box
    return (r - l) * (t - b)


def filter_contained_boxes(
    items: List[dict],
    containment_threshold: float = 0.8
) -> List[dict]:
    """
    Remove boxes that are mostly contained within other larger boxes.

    If box A's intersection with box B is >= threshold of A's area,
    and B is larger than A, then A is considered "contained" and removed.
    """
    if not items:
        return items

    boxes_with_area = []
    for i, item in enumerate(items):
        bbox = item.get("bbox", {})
        l = bbox.get("l", 0)
        r = bbox.get("r", 0)
        t = bbox.get("t", 0)
        b = bbox.get("b", 0)
        box_tuple = (l, b, r, t)
        area = calculate_box_area(box_tuple)
        boxes_with_area.append((i, box_tuple, area))

    indices_to_remove = set()

    for i, (idx_a, box_a, area_a) in enumerate(boxes_with_area):
        if area_a <= 0:
            continue

        for j, (idx_b, box_b, area_b) in enumerate(boxes_with_area):
            if i == j:
                continue
            if area_b <= area_a:
                continue

            intersection = calculate_intersection_area(box_a, box_b)
            containment_ratio = intersection / area_a

            if containment_ratio >= containment_threshold:
                indices_to_remove.add(idx_a)
                break

    return [item for i, item in enumerate(items) if i not in indices_to_remove]


def deduplicate_overlapping_types(
    items: List[dict],
    overlap_threshold: float = 0.9,
) -> List[dict]:
    """
    Remove PictureItem when a TableItem covers the same region.

    filter_contained_boxes can't handle identical bboxes because areas are equal
    (area_b <= area_a skips when equal). This function resolves that by preferring
    TableItem over PictureItem for overlapping regions (>90% mutual overlap).

    Works with both:
    - box_annotator dicts: {"item_type": "picture"/"table", "bbox": {...}}
    - chunker dicts: {"item": DocItem instance, "bbox": {...}}
    """
    if len(items) < 2:
        return items

    indices_to_remove = set()

    for i, item_a in enumerate(items):
        if i in indices_to_remove:
            continue

        # Determine type of item_a
        type_a = item_a.get("item_type", "")
        if not type_a and "item" in item_a:
            type_a = type(item_a["item"]).__name__.lower().replace("item", "")

        bbox_a = item_a.get("bbox", {})
        box_a = (bbox_a.get("l", 0), bbox_a.get("b", 0), bbox_a.get("r", 0), bbox_a.get("t", 0))
        area_a = calculate_box_area(box_a)
        if area_a <= 0:
            continue

        for j, item_b in enumerate(items):
            if i == j or j in indices_to_remove:
                continue

            type_b = item_b.get("item_type", "")
            if not type_b and "item" in item_b:
                type_b = type(item_b["item"]).__name__.lower().replace("item", "")

            # Only resolve picture vs table conflicts
            if not (
                (type_a == "picture" and type_b == "table") or
                (type_a == "table" and type_b == "picture")
            ):
                continue

            bbox_b = item_b.get("bbox", {})
            box_b = (bbox_b.get("l", 0), bbox_b.get("b", 0), bbox_b.get("r", 0), bbox_b.get("t", 0))
            area_b = calculate_box_area(box_b)
            if area_b <= 0:
                continue

            intersection = calculate_intersection_area(box_a, box_b)
            overlap_a = intersection / area_a
            overlap_b = intersection / area_b

            # Both boxes must have high mutual overlap
            if overlap_a >= overlap_threshold and overlap_b >= overlap_threshold:
                # Remove the picture, keep the table
                if type_a == "picture":
                    indices_to_remove.add(i)
                    break
                else:
                    indices_to_remove.add(j)

    if indices_to_remove:
        logger.info(f"Dedup: removed {len(indices_to_remove)} overlapping picture(s) in favor of table(s)")

    return [item for i, item in enumerate(items) if i not in indices_to_remove]
