"""Tests for meridian.utils.bbox

CRITICAL: These functions must produce identical results to ensure
box_annotator and chunker count boxes the same way.
"""

import pytest
from meridian.utils.bbox import (
    calculate_intersection_area,
    calculate_box_area,
    filter_contained_boxes,
)


class TestCalculateIntersectionArea:
    """Tests for calculate_intersection_area."""

    def test_no_overlap(self):
        """Non-overlapping boxes have zero intersection."""
        box1 = (0, 0, 10, 10)
        box2 = (20, 20, 30, 30)
        assert calculate_intersection_area(box1, box2) == 0.0

    def test_full_overlap(self):
        """Identical boxes have full intersection."""
        box = (0, 0, 10, 10)
        assert calculate_intersection_area(box, box) == 100.0

    def test_partial_overlap(self):
        """Partially overlapping boxes."""
        box1 = (0, 0, 10, 10)
        box2 = (5, 5, 15, 15)
        assert calculate_intersection_area(box1, box2) == 25.0

    def test_contained_box(self):
        """Small box contained in larger box."""
        outer = (0, 0, 100, 100)
        inner = (10, 10, 20, 20)
        assert calculate_intersection_area(inner, outer) == 100.0

    def test_touching_edges(self):
        """Boxes touching at edge have zero intersection."""
        box1 = (0, 0, 10, 10)
        box2 = (10, 0, 20, 10)
        assert calculate_intersection_area(box1, box2) == 0.0

    def test_negative_coords(self):
        """Boxes with negative coordinates."""
        box1 = (-10, -10, 0, 0)
        box2 = (-5, -5, 5, 5)
        assert calculate_intersection_area(box1, box2) == 25.0


class TestCalculateBoxArea:
    """Tests for calculate_box_area."""

    def test_simple_area(self):
        """Simple box area calculation."""
        assert calculate_box_area((0, 0, 10, 10)) == 100.0

    def test_rectangular_area(self):
        """Rectangular box area."""
        assert calculate_box_area((0, 0, 5, 20)) == 100.0

    def test_zero_area(self):
        """Zero-area box (line)."""
        assert calculate_box_area((0, 0, 10, 0)) == 0.0

    def test_unit_area(self):
        """Unit area box."""
        assert calculate_box_area((0, 0, 1, 1)) == 1.0


class TestFilterContainedBoxes:
    """Tests for filter_contained_boxes."""

    def test_empty_list(self):
        """Empty list returns empty list."""
        assert filter_contained_boxes([]) == []

    def test_single_item(self):
        """Single item list is returned as-is."""
        items = [{"bbox": {"l": 0, "r": 10, "t": 10, "b": 0}}]
        result = filter_contained_boxes(items)
        assert len(result) == 1

    def test_non_overlapping(self):
        """Non-overlapping boxes are all kept."""
        items = [
            {"bbox": {"l": 0, "r": 10, "t": 10, "b": 0}},
            {"bbox": {"l": 20, "r": 30, "t": 30, "b": 20}},
        ]
        result = filter_contained_boxes(items)
        assert len(result) == 2

    def test_contained_removed(self):
        """Small box contained in larger box is removed."""
        items = [
            {"bbox": {"l": 0, "r": 100, "t": 100, "b": 0}},   # Large
            {"bbox": {"l": 10, "r": 20, "t": 20, "b": 10}},    # Small (contained)
        ]
        result = filter_contained_boxes(items, containment_threshold=0.8)
        assert len(result) == 1
        # The large box should remain
        assert result[0]["bbox"]["r"] == 100

    def test_threshold_respected(self):
        """Containment threshold is respected."""
        # Create two overlapping boxes where overlap is ~50% of smaller
        items = [
            {"bbox": {"l": 0, "r": 100, "t": 100, "b": 0}},   # Large
            {"bbox": {"l": 50, "r": 150, "t": 50, "b": 0}},    # 50% overlap
        ]
        # With 0.8 threshold, should keep both
        result = filter_contained_boxes(items, containment_threshold=0.8)
        assert len(result) == 2

        # With 0.4 threshold, smaller should be removed
        result = filter_contained_boxes(items, containment_threshold=0.4)
        assert len(result) == 1

    def test_preserves_order(self):
        """Filtered items preserve original order."""
        items = [
            {"bbox": {"l": 0, "r": 10, "t": 10, "b": 0}, "id": "a"},
            {"bbox": {"l": 20, "r": 30, "t": 30, "b": 20}, "id": "b"},
            {"bbox": {"l": 40, "r": 50, "t": 50, "b": 40}, "id": "c"},
        ]
        result = filter_contained_boxes(items)
        assert [r["id"] for r in result] == ["a", "b", "c"]
