"""
Unified Chunker for Workers

Adapted from src/unified_chunker.py to work with DoclingDocument as dict.
Uses docling_core.types for data type definitions only (no processing code).

CRITICAL: NO DOCLING IMPORTS (except docling_core.types for data types)
"""

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set, Tuple

# ALLOWED: Data types only from docling_core
from docling_core.types.doc.document import DoclingDocument

from meridian.config import (
    CONTENT_BLOCK_MIN_WIDTH_PT,
    CONTENT_BLOCK_MIN_HEIGHT_PT,
    CONTENT_BLOCK_MIN_AREA_PT2,
    CONTENT_BLOCK_CONTAINMENT_THRESHOLD,
)
from meridian.utils.bbox import filter_contained_boxes, deduplicate_overlapping_types

logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class UnifiedChunk:
    """A chunk with text and all relevant metadata."""
    text: str
    chunk_type: str  # "text", "table", "picture", "formula"
    page_no: int
    order: int
    headings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Chunker
# =============================================================================

class UnifiedChunker:
    """
    Builds chunks in reading order, matching VLM results by bbox location.

    This version works with DoclingDocument deserialized from JSON.
    Uses DoclingDocument.model_validate() to reconstruct the object,
    which allows using iterate_items() method.
    """

    def __init__(self, max_text_chunk_chars: int = 2000, min_text_chunk_chars: int = 150):
        self.max_text_chunk_chars = max_text_chunk_chars
        self.min_text_chunk_chars = min_text_chunk_chars

    def _build_formula_lookup(
        self,
        formula_results: Dict[str, List[dict]],
    ) -> Tuple[Dict[int, List[dict]], Dict[int, Dict[int, dict]], Dict[int, Set[int]]]:
        """
        Build lookup structures for formula detection results.

        Returns:
            - formulas_by_page: {page_no: [standalone_formulas sorted by after_box]}
            - inline_regions_by_page: {page_no: {box_num: region_data}}
            - boxes_in_regions: {page_no: set of box numbers that are in inline regions}
        """
        formulas_by_page: Dict[int, List[dict]] = {}
        inline_regions_by_page: Dict[int, Dict[int, dict]] = {}
        boxes_in_regions: Dict[int, Set[int]] = {}

        for page_result in formula_results.get("results", []):
            page_no = page_result.get("page_no", 0)

            # Standalone formulas sorted by after_box.
            # `or 0` handles VLMs that emit "after_box": null (dict.get default
            # only fires when the key is absent, not when its value is None).
            standalone = sorted(
                page_result.get("standalone_formulas", []),
                key=lambda x: x.get("after_box") or 0
            )
            formulas_by_page[page_no] = standalone

            # Inline math regions
            inline_regions_by_page[page_no] = {}
            boxes_in_regions[page_no] = set()

            for region in page_result.get("inline_math_regions", []):
                start_box = region.get("start_box") or 0
                last_box = region.get("last_box") or region.get("end_box") or start_box

                for box_num in range(start_box, last_box + 1):
                    inline_regions_by_page[page_no][box_num] = region
                    boxes_in_regions[page_no].add(box_num)

        return formulas_by_page, inline_regions_by_page, boxes_in_regions

    def _build_vlm_lookup(
        self,
        vlm_items: List[dict],
    ) -> Tuple[Dict[Tuple[int, float], dict], List[dict]]:
        """
        Build lookup for VLM results keyed on (page_no, rounded_bbox_t).

        Detected items go in the lookup dict for bbox matching.
        Continuation/missed items go in extras for standalone insertion.

        Returns:
            - lookup: {(page_no, rounded_bbox_t): vlm_result}
            - extras: [vlm_results for missed/continuation tables]
        """
        lookup: Dict[Tuple[int, float], dict] = {}
        extras: List[dict] = []

        for item in vlm_items:
            source = item.get("source", "detected")
            page_no = item.get("page_no", 0)
            bbox_t = item.get("bbox", {}).get("t", 0)

            if source in ("missed", "continuation", "picture_table"):
                extras.append(item)
            else:
                key = (page_no, round(bbox_t))
                lookup[key] = item

        return lookup, extras

    def _find_vlm_match(
        self,
        lookup: Dict[Tuple[int, float], dict],
        page_no: int,
        bbox_t: float,
        tolerance: float = 5.0,
    ) -> Optional[dict]:
        """
        Find closest VLM match by page_no and bbox.t, then pop from lookup.

        Returns:
            Matched VLM result dict, or None if no match found
        """
        rounded_t = round(bbox_t)

        # Try exact match first
        key = (page_no, rounded_t)
        if key in lookup:
            return lookup.pop(key)

        # Try nearby values within tolerance
        best_key = None
        best_dist = tolerance + 1

        for candidate_key in list(lookup.keys()):
            if candidate_key[0] != page_no:
                continue
            dist = abs(candidate_key[1] - rounded_t)
            if dist <= tolerance and dist < best_dist:
                best_dist = dist
                best_key = candidate_key

        if best_key is not None:
            return lookup.pop(best_key)

        return None

    # =========================================================================
    # Multi-Page Table Linking
    # =========================================================================

    @staticmethod
    def _is_separator_row(line: str) -> bool:
        """Check if a markdown table line is a separator row (e.g. |---|---|)."""
        stripped = line.strip()
        if not stripped.startswith('|'):
            return False
        content = stripped.replace('|', '').replace('-', '').replace(':', '').replace(' ', '')
        return content == '' and '-' in stripped

    @staticmethod
    def _count_table_columns(line: str) -> int:
        """Count columns in a markdown table line by counting internal pipe separators."""
        stripped = line.strip()
        if not stripped.startswith('|'):
            return 0
        inner = stripped.strip('|')
        return inner.count('|') + 1

    def _parse_table_markdown(self, markdown: str) -> dict:
        """
        Parse markdown table into structural components.

        Returns dict with:
            header_row: The header line (or None)
            separator_row: The |---|---| line (or None)
            data_rows: Lines after separator
            col_count: Number of columns (from separator or first row)
            pre_table: Text before the table (title etc.)
            all_table_lines: All lines starting with |
        """
        if not markdown:
            return {
                "header_row": None, "separator_row": None, "data_rows": [],
                "col_count": 0, "pre_table": "", "all_table_lines": [],
            }

        lines = markdown.split('\n')
        table_lines = []
        pre_table = []
        in_table = False

        for line in lines:
            stripped = line.strip()
            if stripped.startswith('|'):
                in_table = True
                table_lines.append(stripped)
            elif not in_table:
                pre_table.append(line)

        if not table_lines:
            return {
                "header_row": None, "separator_row": None, "data_rows": [],
                "col_count": 0, "pre_table": "\n".join(pre_table).strip(),
                "all_table_lines": [],
            }

        # Find first separator row
        sep_idx = None
        for i, line in enumerate(table_lines):
            if self._is_separator_row(line):
                sep_idx = i
                break

        if sep_idx is None:
            col_count = self._count_table_columns(table_lines[0])
            return {
                "header_row": None, "separator_row": None,
                "data_rows": table_lines, "col_count": col_count,
                "pre_table": "\n".join(pre_table).strip(),
                "all_table_lines": table_lines,
            }

        separator = table_lines[sep_idx]
        col_count = self._count_table_columns(separator)
        header_rows = table_lines[:sep_idx]
        header_row = header_rows[-1] if header_rows else None
        data_rows = table_lines[sep_idx + 1:]

        return {
            "header_row": header_row,
            "separator_row": separator,
            "data_rows": data_rows,
            "col_count": col_count,
            "pre_table": "\n".join(pre_table).strip(),
            "all_table_lines": table_lines,
        }

    def _link_multipage_tables(self, vlm_tables: List[dict]) -> None:
        """
        Detect multi-page table groups and carry headers from first table to continuations.

        Modifies vlm_tables in-place. For each multi-page group:
        1. The first table's header row becomes the canonical header
        2. Continuation tables get their fabricated/missing headers replaced
        3. All non-separator data rows in continuations are preserved

        Detection heuristics (all must be satisfied):
        - Last table on page N + first table on page N+1
        - Same column count (exact match), minimum 3 columns
        - Continuation starts in upper portion of page (bbox_t > 650)
        - Both successfully extracted with markdown

        The col_count >= 3 filter avoids linking separate 2-column key-value tables
        (e.g., case study profiles) that share the same template structure.
        """
        if len(vlm_tables) < 2:
            return

        # Filter to successful tables with markdown
        valid = [(i, t) for i, t in enumerate(vlm_tables)
                 if t.get("success") and t.get("markdown")]

        if len(valid) < 2:
            return

        # Parse markdown for each valid table
        parsed = {}
        for idx, table in valid:
            parsed[idx] = self._parse_table_markdown(table.get("markdown", ""))

        # Sort by (page_no, -bbox.t) — reading order
        valid.sort(key=lambda x: (
            x[1].get("page_no", 0),
            -x[1].get("bbox", {}).get("t", 0),
        ))

        # Group by page
        by_page: Dict[int, List[Tuple[int, dict]]] = {}
        for idx, table in valid:
            page = table.get("page_no", 0)
            if page not in by_page:
                by_page[page] = []
            by_page[page].append((idx, table))

        pages = sorted(by_page.keys())

        # Track header carriers: page -> (header_row, separator_row, origin_page)
        header_for_page: Dict[int, Tuple[str, str, int]] = {}
        linked_count = 0

        for pi in range(1, len(pages)):
            curr_page = pages[pi]
            prev_page = pages[pi - 1]

            if curr_page != prev_page + 1:
                continue

            # Last table on previous page, first table on current page
            prev_last_idx, prev_last = by_page[prev_page][-1]
            curr_first_idx, curr_first = by_page[curr_page][0]

            prev_parsed = parsed[prev_last_idx]
            curr_parsed = parsed[curr_first_idx]

            if prev_parsed["col_count"] == 0 or curr_parsed["col_count"] == 0:
                continue

            if prev_parsed["col_count"] != curr_parsed["col_count"]:
                continue

            # Skip 2-column tables: these are typically key-value tables
            # (e.g., case study profiles) where each page is a separate instance
            # sharing the same template, not a single table spanning pages
            if prev_parsed["col_count"] < 3:
                continue

            # Continuation must start near the top of the page (upper ~15%)
            # Tables starting mid-page or lower are likely separate tables
            cont_bbox_t = curr_first.get("bbox", {}).get("t", 0)
            if cont_bbox_t < 650:
                continue

            # Get canonical header — either from chain or from previous table
            if prev_page in header_for_page:
                header, separator, origin_page = header_for_page[prev_page]
            elif prev_parsed["header_row"] and prev_parsed["separator_row"]:
                header = prev_parsed["header_row"]
                separator = prev_parsed["separator_row"]
                origin_page = prev_page
            else:
                continue

            # Store for further chaining (3+ page tables)
            header_for_page[curr_page] = (header, separator, origin_page)

            # Rebuild continuation markdown:
            # Keep all non-separator table rows as data (preserves everything,
            # even if VLM treated data as header or fabricated headers)
            cont_data_rows = [
                line for line in curr_parsed["all_table_lines"]
                if not self._is_separator_row(line)
            ]

            # Strip fabricated header row: if the first data row looks like
            # column labels (all cells short) rather than real data (at least
            # one cell with substantial text), remove it to avoid a duplicate
            # header. Real data rows almost always have a cell > 35 chars.
            if cont_data_rows:
                first_row = cont_data_rows[0]
                cells = [c.strip() for c in first_row.strip().strip('|').split('|')]
                max_cell_len = max((len(c) for c in cells), default=0)
                if max_cell_len <= 35:
                    logger.debug(
                        f"Stripping fabricated header from p{curr_page} continuation: "
                        f"{first_row[:80]}"
                    )
                    cont_data_rows = cont_data_rows[1:]

            parts = []
            if curr_parsed.get("pre_table"):
                parts.append(curr_parsed["pre_table"])
            parts.append(header)
            parts.append(separator)
            parts.extend(cont_data_rows)

            curr_first["markdown"] = "\n".join(parts)
            curr_first["_continuation_of_page"] = origin_page
            linked_count += 1

            logger.info(
                f"Multi-page table: page {curr_page} linked to page {origin_page} "
                f"({prev_parsed['col_count']} cols)"
            )

        if linked_count > 0:
            logger.info(f"Linked {linked_count} continuation tables with correct headers")

    def build_chunks(
        self,
        docling_doc_dict: dict,
        vlm_results: Dict[str, List[dict]],
        formula_results: Optional[Dict] = None,
    ) -> List[UnifiedChunk]:
        """
        Build chunks in reading order with VLM results and formulas merged.

        Args:
            docling_doc_dict: DoclingDocument as dict (from Docling service JSON)
            vlm_results: Dict with "tables", "pictures" lists from VLM
            formula_results: Dict with "results" list containing per-page
                           standalone_formulas and inline_math_regions

        Returns:
            List of UnifiedChunk in reading order
        """
        # Deserialize to DoclingDocument object to use iterate_items()
        doc = DoclingDocument.model_validate(docling_doc_dict)

        # Link multi-page tables: carry headers from first table to continuations
        if vlm_results.get("tables"):
            self._link_multipage_tables(vlm_results["tables"])

        # Build VLM lookups keyed on (page_no, bbox.t) instead of sequential index
        table_lookup, extra_tables = self._build_vlm_lookup(
            vlm_results.get("tables", [])
        )
        picture_lookup, extra_pictures = self._build_vlm_lookup(
            vlm_results.get("pictures", [])
        )

        logger.info(
            f"VLM lookup: {len(table_lookup)} detected tables, {len(extra_tables)} extra tables, "
            f"{len(picture_lookup)} detected pictures, {len(extra_pictures)} extra pictures"
        )

        # Build formula lookup structures
        if formula_results:
            formulas_by_page, inline_regions_by_page, boxes_in_regions = \
                self._build_formula_lookup(formula_results)
        else:
            formulas_by_page = {}
            inline_regions_by_page = {}
            boxes_in_regions = {}

        # Track box numbers per page (for formula matching)
        current_page = -1
        box_num_on_page = 0

        # Track processed regions and inserted formulas
        processed_regions: Set[Tuple[int, int, int]] = set()
        inserted_formulas: Set[Tuple[int, str]] = set()

        chunks = []
        order = 0
        current_headings = []

        # Collect all content items with position info
        from docling_core.types.doc import TextItem, PictureItem, TableItem

        items_with_pos = []
        for item, level in doc.iterate_items():
            if not isinstance(item, (TextItem, PictureItem, TableItem)):
                continue

            if not item.prov or len(item.prov) == 0:
                continue

            prov = item.prov[0]
            bbox = prov.bbox
            if not bbox:
                continue

            # Filter out very small items
            bbox_width = bbox.r - bbox.l
            bbox_height = bbox.t - bbox.b
            bbox_area = bbox_width * bbox_height
            if (bbox_width < CONTENT_BLOCK_MIN_WIDTH_PT and bbox_height < CONTENT_BLOCK_MIN_HEIGHT_PT) \
                    or bbox_area < CONTENT_BLOCK_MIN_AREA_PT2:
                continue

            page_no = prov.page_no

            items_with_pos.append({
                "item": item,
                "level": level,
                "page_no": page_no,
                "bbox": {"l": bbox.l, "r": bbox.r, "t": bbox.t, "b": bbox.b},
            })

        # Sort by (page_no, -bbox_t) to match visualize_boxes order
        items_with_pos.sort(key=lambda x: (x["page_no"], -x["bbox"]["t"]))

        # Filter contained boxes per page
        items_by_page = defaultdict(list)
        for item_dict in items_with_pos:
            items_by_page[item_dict["page_no"]].append(item_dict)

        filtered_items = []
        for page_no in sorted(items_by_page.keys()):
            page_items = items_by_page[page_no]
            page_items = filter_contained_boxes(page_items, CONTENT_BLOCK_CONTAINMENT_THRESHOLD)
            page_items = deduplicate_overlapping_types(page_items)
            filtered_items.extend(page_items)

        items_with_pos = filtered_items

        # Process items — bbox matching instead of sequential indexing
        for idx, item_dict in enumerate(items_with_pos):
            item = item_dict["item"]
            level = item_dict["level"]
            page_no = item_dict["page_no"]
            bbox = item_dict["bbox"]

            # Reset box counter when page changes
            if page_no != current_page:
                current_page = page_no
                box_num_on_page = 0

            # Get item label
            label = str(item.label).lower() if hasattr(item, 'label') else ""

            # Track headings
            if label == 'section_header':
                if hasattr(item, 'text') and item.text:
                    current_headings = current_headings[:level] + [item.text]

            # Skip formula items (handled via VLM results)
            if "formula" in label:
                continue

            # Increment box number
            box_num_on_page += 1
            current_box = box_num_on_page

            if isinstance(item, TextItem):
                text = item.text if hasattr(item, 'text') else ""

                if not text.strip():
                    pass  # Empty text - counted but not added
                elif page_no in boxes_in_regions and current_box in boxes_in_regions[page_no]:
                    # Part of inline math region
                    region = inline_regions_by_page[page_no][current_box]
                    start_box = region.get("start_box", 0)
                    last_box = region.get("last_box", region.get("end_box", start_box))
                    region_key = (page_no, start_box, last_box)

                    if region_key not in processed_regions and current_box == start_box:
                        processed_regions.add(region_key)
                        formatted_text = region.get("formatted_text", text)

                        chunks.append(UnifiedChunk(
                            text=formatted_text,
                            chunk_type="text",
                            page_no=page_no,
                            order=order,
                            headings=current_headings.copy(),
                            metadata={
                                "label": "inline_math_region",
                                "formula_refs": region.get("formula_refs", []),
                                "boxes": f"{start_box}-{last_box}",
                                "bbox": bbox,
                            },
                        ))
                        order += 1
                else:
                    # Regular text box
                    chunks.append(UnifiedChunk(
                        text=text,
                        chunk_type="text",
                        page_no=page_no,
                        order=order,
                        headings=current_headings.copy(),
                        metadata={
                            "label": str(item.label) if hasattr(item, 'label') else None,
                            "bbox": bbox,
                        },
                    ))
                    order += 1

            elif isinstance(item, TableItem):
                # Match VLM result by bbox location instead of sequential index
                vlm = self._find_vlm_match(table_lookup, page_no, bbox["t"])

                if vlm and vlm.get("success"):
                    text = vlm.get("markdown", "[Table]")
                else:
                    text = "[Table]"

                table_metadata = {
                    "vlm_extracted": vlm is not None and vlm.get("success", False),
                    "bbox": bbox,
                }
                if vlm and vlm.get("notes"):
                    table_metadata["notes"] = vlm.get("notes")
                if vlm and vlm.get("_continuation_of_page"):
                    table_metadata["continuation_of_page"] = vlm["_continuation_of_page"]

                chunks.append(UnifiedChunk(
                    text=text,
                    chunk_type="table",
                    page_no=page_no,
                    order=order,
                    headings=current_headings.copy(),
                    metadata=table_metadata,
                ))
                order += 1

            elif isinstance(item, PictureItem):
                # Match VLM result by bbox location instead of sequential index
                vlm = self._find_vlm_match(picture_lookup, page_no, bbox["t"])

                if vlm and not vlm.get("relevant", True):
                    continue

                if vlm and vlm.get("success"):
                    desc = vlm.get("description", "")
                    fig_type = vlm.get("figure_type", "figure")
                    caption = vlm.get("caption", "")
                    key_elements = vlm.get("key_elements", [])
                    text = f"[{fig_type.title()}: {caption + ' - ' if caption else ''}{desc}]"
                else:
                    desc = ""
                    fig_type = "figure"
                    caption = ""
                    key_elements = []
                    text = "[Figure]"

                picture_metadata = {
                    "vlm_extracted": vlm is not None and vlm.get("success", False),
                    "figure_type": fig_type,
                    "key_elements": key_elements,
                    "bbox": bbox,
                }
                if caption:
                    picture_metadata["caption"] = caption

                chunks.append(UnifiedChunk(
                    text=text,
                    chunk_type="picture",
                    page_no=page_no,
                    order=order,
                    headings=current_headings.copy(),
                    metadata=picture_metadata,
                ))
                order += 1

            # Check if any standalone formulas should be inserted after this box
            # (battle-tested: formula after_box matching from visualize_boxes pipeline)
            if page_no in formulas_by_page:
                for formula in formulas_by_page[page_no]:
                    formula_id = formula.get("formula_id", "")
                    after_box = formula.get("after_box", 0)

                    formula_key = (page_no, formula_id)
                    if after_box == current_box and formula_key not in inserted_formulas:
                        inserted_formulas.add(formula_key)

                        latex = formula.get("latex", "")
                        eq_num = formula.get("equation_number", "")
                        description = formula.get("description")
                        eq_display = f" {eq_num}" if eq_num else ""

                        if description:
                            text = f"{description}\n$${latex}$${eq_display}"
                        else:
                            text = f"$${latex}$${eq_display}"

                        # Estimate formula bbox from gap between current item and next
                        formula_bbox = None
                        if bbox:
                            formula_top = bbox["b"]
                            formula_left = bbox["l"]
                            formula_right = bbox["r"]

                            formula_bottom = None
                            for next_idx in range(idx + 1, len(items_with_pos)):
                                next_item = items_with_pos[next_idx]
                                if next_item["page_no"] != page_no:
                                    break
                                next_label = str(next_item["item"].label).lower() if hasattr(next_item["item"], 'label') else ""
                                if "formula" in next_label:
                                    continue
                                formula_bottom = next_item["bbox"]["t"]
                                break

                            if formula_bottom is None:
                                formula_bottom = formula_top - 100

                            formula_bbox = {
                                "l": formula_left,
                                "r": formula_right,
                                "t": formula_top,
                                "b": formula_bottom,
                            }

                        chunks.append(UnifiedChunk(
                            text=text,
                            chunk_type="formula",
                            page_no=page_no,
                            order=order,
                            headings=current_headings.copy(),
                            metadata={
                                "formula_id": formula_id,
                                "equation_number": eq_num,
                                "latex": latex,
                                "description": description,
                                "bbox": formula_bbox,
                            },
                        ))
                        order += 1

        # =================================================================
        # Insert extra tables (missed/continuation/picture_table) as standalone chunks
        # =================================================================
        for extra in extra_tables:
            if not extra.get("success"):
                continue

            extra_page = extra.get("page_no", 0)
            extra_bbox = extra.get("bbox", {})
            source = extra.get("source", "unknown")

            text = extra.get("markdown", "[Table]")

            # For picture_table candidates: only keep if VLM returned real table markdown
            # (must have | characters indicating columns, and at least 2 columns)
            if source == "picture_table":
                # Check if VLM explicitly said this is not a table (graph, chart, diagram, etc.)
                if extra.get("not_table"):
                    reason = extra.get("reason", "non-tabular content")
                    logger.info(f"Dropping picture_table on page {extra_page}: VLM says not a table ({reason})")
                    continue

                lines_with_pipes = [l for l in text.split('\n') if l.strip().startswith('|')]
                if len(lines_with_pipes) < 2:
                    logger.debug(f"Dropping picture_table on page {extra_page}: no table markdown")
                    continue
                # Check for at least 2 columns
                first_table_line = lines_with_pipes[0].strip().strip('|')
                col_count = first_table_line.count('|') + 1
                if col_count < 2:
                    logger.debug(f"Dropping picture_table on page {extra_page}: only {col_count} column")
                    continue
                logger.info(f"Picture on page {extra_page} confirmed as table ({col_count} columns)")

            extra_metadata = {
                "vlm_extracted": True,
                "source": source,
                "bbox": extra_bbox,
            }
            if extra.get("notes"):
                extra_metadata["notes"] = extra.get("notes")

            chunks.append(UnifiedChunk(
                text=text,
                chunk_type="table",
                page_no=extra_page,
                order=order,
                headings=[],
                metadata=extra_metadata,
            ))
            order += 1

        # =================================================================
        # Insert formulas that were not placed via after_box matching
        # (fallback: append at page end, re-sort will position them)
        # =================================================================
        for page_no in sorted(formulas_by_page.keys()):
            for formula in formulas_by_page[page_no]:
                formula_id = formula.get("formula_id", "")
                if (page_no, formula_id) in inserted_formulas:
                    continue  # Already inserted via after_box

                latex = formula.get("latex", "")
                eq_num = formula.get("equation_number", "")
                description = formula.get("description")
                eq_display = f" {eq_num}" if eq_num else ""

                if not latex:
                    continue

                if description:
                    text = f"{description}\n$${latex}$${eq_display}"
                else:
                    text = f"$${latex}$${eq_display}"

                chunks.append(UnifiedChunk(
                    text=text,
                    chunk_type="formula",
                    page_no=page_no,
                    order=order,
                    headings=[],
                    metadata={
                        "formula_id": formula_id,
                        "equation_number": eq_num,
                        "latex": latex,
                        "description": description,
                    },
                ))
                order += 1

        # =================================================================
        # Re-sort all chunks by reading order and reassign sequential order
        # =================================================================
        def chunk_sort_key(c: UnifiedChunk) -> Tuple[int, float]:
            bbox = c.metadata.get("bbox")
            if bbox and isinstance(bbox, dict) and "t" in bbox:
                return (c.page_no, -bbox["t"])
            else:
                # No bbox (formulas/extras) → sort to end of page
                return (c.page_no, float('inf'))

        chunks.sort(key=chunk_sort_key)

        for i, chunk in enumerate(chunks):
            chunk.order = i

        logger.info(f"Built {len(chunks)} chunks in reading order")
        return chunks

    def chunks_to_dict(self, chunks: List[UnifiedChunk]) -> List[dict]:
        """Convert chunks to dict for JSON/storage."""
        return [
            {
                "text": c.text,
                "type": c.chunk_type,
                "page_no": c.page_no,
                "order": c.order,
                "headings": c.headings,
                "metadata": c.metadata,
            }
            for c in chunks
        ]

    def merge_text_chunks(
        self,
        chunks: List[UnifiedChunk],
        max_chars: Optional[int] = None,
    ) -> List[UnifiedChunk]:
        """Merge consecutive text chunks. Tables/pictures stay separate."""
        if not chunks:
            return []

        max_chars = max_chars or self.max_text_chunk_chars
        merged = []
        current_text = []
        current_start = None

        def flush():
            nonlocal current_text, current_start
            if current_text and current_start:
                merged.append(UnifiedChunk(
                    text="\n\n".join(current_text),
                    chunk_type="text",
                    page_no=current_start.page_no,
                    order=current_start.order,
                    headings=current_start.headings,
                    metadata={"merged": len(current_text) > 1},
                ))
            current_text = []
            current_start = None

        for chunk in chunks:
            if chunk.chunk_type != "text":
                flush()
                merged.append(chunk)
            else:
                if sum(len(t) for t in current_text) + len(chunk.text) > max_chars and current_text:
                    flush()
                if not current_text:
                    current_start = chunk
                current_text.append(chunk.text)

        flush()
        return merged

    def merge_small_chunks(
        self,
        chunks: List[UnifiedChunk],
        min_chars: Optional[int] = None,
    ) -> List[UnifiedChunk]:
        """Merge small text chunks with the previous text chunk."""
        if not chunks:
            return []

        min_chars = min_chars or self.min_text_chunk_chars
        result = []

        for chunk in chunks:
            if (chunk.chunk_type == "text" and
                len(chunk.text) < min_chars and
                result and
                result[-1].chunk_type == "text"):

                prev = result[-1]
                result[-1] = UnifiedChunk(
                    text=prev.text + "\n" + chunk.text,
                    chunk_type="text",
                    page_no=prev.page_no,
                    order=prev.order,
                    headings=prev.headings,
                    metadata={**prev.metadata, "merged_small": True},
                )
            else:
                result.append(chunk)

        return result
