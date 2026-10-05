"""
Async VLM Service Client

Phase 4B: Async HTTP client for vision-language model processing via vLLM.
Sends ALL VLM requests concurrently to maximize GPU batching efficiency.

Key difference from vlm_client.py:
- Uses aiohttp instead of httpx
- process_all_concurrent() sends all requests at once
- vLLM's continuous batching automatically batches concurrent requests
"""

import asyncio
import base64
import json
import logging
from io import BytesIO
from pathlib import Path
from typing import Optional, Dict, Any, List, Union
from dataclasses import dataclass

import aiohttp
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class VLMAsyncConfig:
    """Configuration for async VLM client."""
    api_url: str = "http://localhost:8000/v1/chat/completions"
    model: str = "Qwen/Qwen3-VL-8B-Instruct"
    api_key: str = "EMPTY"

    # Token limits per task type
    max_tokens_table: int = 4000
    max_tokens_formula: int = 4000
    max_tokens_picture: int = 1500

    # Retry configuration
    max_retries: int = 3
    retry_delay_seconds: float = 2.0

    # Request settings
    timeout_seconds: float = 300.0
    temperature: float = 0.2

    # Image processing
    max_image_dimension: int = 1536

    # Concurrency control (optional - vLLM handles this, but can limit if needed)
    max_concurrent_requests: Optional[int] = None  # None = unlimited


class VLMClientAsync:
    """
    Async HTTP client for vLLM vision-language model processing.

    Usage:
        async with VLMClientAsync() as client:
            # Process all VLM tasks concurrently
            results = await client.process_all_concurrent(
                tables=[{"image": img1, "page_no": 1}, ...],
                figures=[{"image": img2, "page_no": 2}, ...],
                formulas=[{"image": img3, "page_no": 3}, ...],
            )
    """

    def __init__(self, config: Optional[VLMAsyncConfig] = None):
        """Initialize the async VLM client."""
        if config is None:
            from meridian.config import VLLM_API_URL, VLLM_MODEL, VLLM_API_KEY
            config = VLMAsyncConfig(
                api_url=VLLM_API_URL,
                model=VLLM_MODEL,
                api_key=VLLM_API_KEY,
            )
        self.config = config
        self._session: Optional[aiohttp.ClientSession] = None
        self._semaphore: Optional[asyncio.Semaphore] = None

    async def __aenter__(self):
        """Async context manager entry."""
        timeout = aiohttp.ClientTimeout(total=self.config.timeout_seconds)
        self._session = aiohttp.ClientSession(timeout=timeout)
        if self.config.max_concurrent_requests:
            self._semaphore = asyncio.Semaphore(self.config.max_concurrent_requests)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self._session:
            await self._session.close()

    # =========================================================================
    # Image Handling (same as sync client)
    # =========================================================================

    def _prepare_image(self, image: Union[str, Path, bytes]) -> str:
        """Prepare image for API request."""
        # Raw bytes
        if isinstance(image, bytes):
            return self._resize_and_encode(image)

        # Path object
        if isinstance(image, Path):
            if not image.exists():
                raise ValueError(f"Image file not found: {image}")
            with open(image, "rb") as f:
                return self._resize_and_encode(f.read())

        # String - could be base64 or file path
        if isinstance(image, str):
            # If string is long (>200 chars), it's almost certainly base64, not a path
            # This avoids "File name too long" errors from Path().exists()
            if len(image) > 200:
                img_bytes = base64.b64decode(image)
                return self._resize_and_encode(img_bytes)

            # Short string - check if it's a file path
            try:
                path = Path(image)
                if path.exists():
                    with open(path, "rb") as f:
                        return self._resize_and_encode(f.read())
            except OSError:
                # Path check failed (e.g., invalid characters), treat as base64
                pass

            # Not a valid path, try as base64
            img_bytes = base64.b64decode(image)
            return self._resize_and_encode(img_bytes)

        raise ValueError(f"Unsupported image type: {type(image)}")

    def _resize_and_encode(self, img_bytes: bytes) -> str:
        """Resize image if too large and encode to base64."""
        img = Image.open(BytesIO(img_bytes))

        max_dim = self.config.max_image_dimension
        if max(img.size) > max_dim:
            ratio = max_dim / max(img.size)
            new_size = tuple(int(d * ratio) for d in img.size)
            resized = img.resize(new_size, Image.Resampling.LANCZOS)
            img.close()  # Close original
            img = resized

        if img.mode in ('RGBA', 'P'):
            converted = img.convert('RGB')
            img.close()  # Close before conversion
            img = converted

        buffered = BytesIO()
        img.save(buffered, format="PNG")
        result = base64.b64encode(buffered.getvalue()).decode("utf-8")

        # Cleanup
        img.close()
        buffered.close()

        return result

    # =========================================================================
    # Prompts (same as sync client)
    # =========================================================================

    def _get_table_prompt(self, context: str = "") -> str:
        """Get prompt for table extraction."""
        context_section = f"\n\nDOCUMENT CONTEXT:\n{context}\n" if context else ""
        return f"""Extract this table to markdown format.
{context_section}
NOTE: The image includes extra padding around the table to provide context.
Focus ONLY on extracting the main table. Use surrounding context for title if visible.

If this image does not contain a table, respond with:
{{
  "not_table": true,
  "reason": "Brief description of what the image actually shows"
}}

ALWAYS extract the full table as markdown, regardless of complexity. Do your best even if the table is dense, has merged cells, or unusual formatting.

CONTINUATION TABLES: If this appears to be a continuation of a table from a previous page (no header row visible at the top, data rows start immediately), do NOT fabricate or guess column headers. Start your markdown directly with the data rows and separator — omit any header row. The correct headers will be added automatically in post-processing.

INCLUDE IN YOUR EXTRACTION:
1. Table title/heading (e.g., "TABLE I. - SUMMARY OF..." or "Table 1: Results...")
2. Any subtitle or description below the title
3. The full table with all rows and columns
4. Any footnotes or notes that belong to THIS table

ALWAYS include "notes" for any nuances that cannot be fully captured in markdown:
- Merged cells, spanning headers, or unusual layouts
- Units, abbreviations, or symbols that need clarification
- Footnote references and their meanings
- Any context important for understanding the data

Output JSON:
{{
  "markdown": "## TABLE I. - TITLE\\n\\n| Header1 | Header2 |...\\n|---|---|...\\n| val1 | val2 |...",
  "notes": "Optional notes about merged cells, units, abbreviations, or other nuances not captured in markdown"
}}

Output ONLY the JSON object."""

    def _get_formula_prompt(self, text_context: str = "") -> str:
        """
        Get prompt for formula extraction from annotated page images.

        Battle-tested prompt from visualize_boxes.py - explains numbered box
        system and asks for after_box positioning + inline_math_regions.
        """
        context_section = ""
        if text_context:
            context_section = f"""
---

## Reference Text Context (for description field ONLY)

The following is OCR-extracted text from the page. This text may be NOISY (OCR errors, spacing issues, old document artifacts).

USE THIS TEXT ONLY to generate the "description" field for each formula. Do NOT let this text affect your extraction of latex, equation_number, or after_box - those must come purely from looking at the IMAGE.

```
{text_context}
```

---

"""

        return f"""{context_section}## Background: How Your Output Will Be Used

We are building a document processing pipeline that reconstructs documents in correct reading order. The numbered boxes [1], [2], [3], etc. represent text blocks detected by OCR, ordered from top to bottom on the page.

**How we use your output:**
1. **standalone_formulas**: We INSERT each formula into the document AFTER the box you specify in "after_box". If you say after_box=5, the formula appears right after box 5's content in the final document.

2. **inline_math_regions**: We REPLACE the original OCR text for the specified boxes with your "formatted_text". If you specify start_box=3, last_box=4, we replace boxes 3 and 4's text entirely with your formatted version.

**Why accuracy matters:**
- Wrong "after_box" = formula appears in wrong position, breaking reading flow
- Wrong box range in inline regions = content gets lost or duplicated
- Grouping unrelated paragraphs = one paragraph's content overwrites another

Your goal: Identify the correct box numbers so formulas and math appear in their natural reading position.

---

## Task

Look at this page image with numbered boxes. Return a JSON object with:

1. "standalone_formulas": Main equations displayed on their own line (not inside a text paragraph).
   - "formula_id": unique identifier like "FORMULA_1"
   - "latex": the equation in LaTeX (extract from IMAGE only)
   - "equation_number": e.g. "(1)" if labeled (extract from IMAGE only)
   - "after_box": the box number IMMEDIATELY BEFORE this formula in reading order (extract from IMAGE only)
   - "description": Brief description of what this formula represents, based on surrounding text context. If no context provided or unclear, use null.

2. "inline_math_regions": Text blocks containing inline math expressions (variables, symbols within text).
   - "start_box": first box in this continuous region
   - "last_box": last box in this region (INCLUSIVE - this box IS part of the region)
   - "formatted_text": the COMPLETE text of ALL boxes from start_box through last_box, with math symbols wrapped in $...$
   - "formula_refs": list of formula IDs this text references

**Rules for inline_math_regions:**
- Only group boxes from the SAME paragraph or continuous explanation
- STOP the region when a new section heading or unrelated topic begins
- last_box is inclusive: start_box=3, last_box=5 means boxes 3, 4, AND 5
- formatted_text must contain ONLY text visible inside the specified boxes - do NOT complete sentences beyond box boundaries

## Example Output

{{{{
  "standalone_formulas": [
    {{{{"formula_id": "FORMULA_1", "latex": "F = ma", "equation_number": "(1)", "after_box": 3, "description": "Newton's second law relating force, mass, and acceleration"}}}}
  ],
  "inline_math_regions": [
    {{{{"start_box": 4, "last_box": 4, "formatted_text": "where $F$ is force and $m$ is mass.", "formula_refs": ["FORMULA_1"]}}}}
  ]
}}}}

Use $ for inline math. Use \\\\ for LaTeX escaping in JSON.
Output ONLY the JSON object."""

    def _get_picture_prompt(self, context: str = "") -> str:
        """Get prompt for picture description."""
        context_section = f"\n\nDOCUMENT CONTEXT:\n{context}\n" if context else ""
        return f"""Describe this figure/diagram from a technical document.
{context_section}
Provide:
1. A concise description of what the figure shows
2. Key elements, labels, and annotations
3. The type of figure (schematic, graph, photo, flowchart, etc.)
4. Whether this figure is relevant for technical understanding

Output JSON:
{{
  "description": "Detailed description of the figure",
  "figure_type": "schematic|graph|photograph|flowchart|diagram|other",
  "key_elements": ["element1", "element2"],
  "relevant": true/false,
  "caption": "Figure caption if visible"
}}

Output ONLY the JSON object."""

    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """Parse JSON from VLM response (handles markdown code blocks)."""
        text = response.strip()

        if text.startswith("```"):
            lines = text.split("\n")
            lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines)

        start = text.find("{")
        end = text.rfind("}") + 1

        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end])
            except json.JSONDecodeError:
                pass

        return {"error": "Failed to parse JSON", "raw_response": response[:500]}

    # =========================================================================
    # Core Async API Call
    # =========================================================================

    async def _call_vlm_async(
        self,
        image_base64: str,
        prompt: str,
        max_tokens: int,
        task_id: str = "",
    ) -> str:
        """Call VLM API asynchronously with retry logic."""
        if not self._session:
            raise RuntimeError("Client not initialized. Use 'async with' context manager.")

        last_error = None
        delay = self.config.retry_delay_seconds

        for attempt in range(self.config.max_retries):
            try:
                # Use semaphore if concurrency limiting is enabled
                if self._semaphore:
                    async with self._semaphore:
                        return await self._make_request(image_base64, prompt, max_tokens)
                else:
                    return await self._make_request(image_base64, prompt, max_tokens)

            except asyncio.TimeoutError as e:
                last_error = e
                logger.warning(f"VLM timeout for {task_id} (attempt {attempt + 1}/{self.config.max_retries})")

            except aiohttp.ClientError as e:
                last_error = e
                logger.warning(f"VLM connection error for {task_id} (attempt {attempt + 1}/{self.config.max_retries}): {e}")

            except Exception as e:
                last_error = e
                logger.warning(f"VLM call failed for {task_id} (attempt {attempt + 1}/{self.config.max_retries}): {e}")

            if attempt < self.config.max_retries - 1:
                await asyncio.sleep(delay)
                delay *= 2  # Exponential backoff

        raise RuntimeError(f"VLM service unavailable after {self.config.max_retries} attempts: {last_error}")

    async def _make_request(self, image_base64: str, prompt: str, max_tokens: int) -> str:
        """Make single HTTP request to vLLM."""
        async with self._session.post(
            self.config.api_url,
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.config.model,
                "messages": [{
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{image_base64}"},
                        },
                        {"type": "text", "text": prompt},
                    ],
                }],
                "max_tokens": max_tokens,
                "temperature": self.config.temperature,
            },
        ) as response:
            response.raise_for_status()
            data = await response.json()

            if "choices" in data and len(data["choices"]) > 0:
                return data["choices"][0]["message"]["content"]

            raise ValueError("No 'choices' in response")

    # =========================================================================
    # Individual Processing Methods (async)
    # =========================================================================

    async def process_table_async(
        self,
        image: Union[str, bytes],
        context: str = "",
        page_no: int = 0,
        bbox: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """Process a table image asynchronously."""
        try:
            img_base64 = self._prepare_image(image)
            prompt = self._get_table_prompt(context)
            response = await self._call_vlm_async(
                img_base64, prompt, self.config.max_tokens_table,
                task_id=f"table_p{page_no}"
            )
            result = self._parse_json_response(response)
            result["success"] = "error" not in result
            result["page_no"] = page_no
            result["type"] = "table"
            result["bbox"] = bbox or {}
            return result
        except Exception as e:
            logger.error(f"Table processing failed (page {page_no}): {e}")
            return {
                "success": False,
                "error": str(e),
                "page_no": page_no,
                "type": "table",
                "bbox": bbox or {},
            }

    async def process_formula_async(
        self,
        image: Union[str, bytes],
        context: str = "",
        page_no: int = 0,
        text_context: str = "",
    ) -> Dict[str, Any]:
        """Process a formula page asynchronously with numbered-box prompt."""
        try:
            img_base64 = self._prepare_image(image)
            prompt = self._get_formula_prompt(text_context=text_context or context)
            response = await self._call_vlm_async(
                img_base64, prompt, self.config.max_tokens_formula,
                task_id=f"formula_p{page_no}"
            )
            result = self._parse_json_response(response)
            result["success"] = "error" not in result
            result["page_no"] = page_no
            result["type"] = "formula"
            return result
        except Exception as e:
            logger.error(f"Formula processing failed (page {page_no}): {e}")
            return {
                "success": False,
                "error": str(e),
                "page_no": page_no,
                "type": "formula",
            }

    async def process_picture_async(
        self,
        image: Union[str, bytes],
        context: str = "",
        page_no: int = 0,
        bbox: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """Process a figure/picture asynchronously."""
        try:
            img_base64 = self._prepare_image(image)
            prompt = self._get_picture_prompt(context)
            response = await self._call_vlm_async(
                img_base64, prompt, self.config.max_tokens_picture,
                task_id=f"picture_p{page_no}"
            )
            result = self._parse_json_response(response)
            result["success"] = "error" not in result
            result["page_no"] = page_no
            result["type"] = "picture"
            result["bbox"] = bbox or {}
            return result
        except Exception as e:
            logger.error(f"Picture processing failed (page {page_no}): {e}")
            return {
                "success": False,
                "error": str(e),
                "page_no": page_no,
                "type": "picture",
                "bbox": bbox or {},
            }

    # =========================================================================
    # Main Concurrent Processing Method
    # =========================================================================

    async def process_all_concurrent(
        self,
        tables: List[Dict[str, Any]],
        figures: List[Dict[str, Any]],
        formulas: List[Dict[str, Any]],
        context: str = "",
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Process ALL VLM tasks concurrently.

        This is the key method for Phase 4B optimization.
        All requests are sent at once, and vLLM's continuous batching
        automatically groups them for efficient GPU processing.

        Args:
            tables: List of {"image": base64, "page_no": int, "bbox": dict}
            figures: List of {"image": base64, "page_no": int, "bbox": dict}
            formulas: List of {"image": base64, "page_no": int, "num_blocks": int}
            context: Optional document context string

        Returns:
            {"tables": [...], "pictures": [...], "formulas": [...]}
        """
        tasks = []
        task_types = []

        # Create all coroutines
        for table in tables:
            tasks.append(self.process_table_async(
                image=table.get("image") or table.get("image_base64"),
                context=context,
                page_no=table.get("page_no", 0),
                bbox=table.get("bbox"),
            ))
            task_types.append("table")

        for figure in figures:
            tasks.append(self.process_picture_async(
                image=figure.get("image") or figure.get("image_base64"),
                context=context,
                page_no=figure.get("page_no", 0),
                bbox=figure.get("bbox"),
            ))
            task_types.append("picture")

        for formula in formulas:
            tasks.append(self.process_formula_async(
                image=formula.get("image") or formula.get("image_base64"),
                context=context,
                page_no=formula.get("page_no", 0),
                text_context=formula.get("text_context", ""),
            ))
            task_types.append("formula")

        total_tasks = len(tasks)
        logger.info(f"Sending {total_tasks} VLM requests concurrently "
                    f"({len(tables)} tables, {len(figures)} figures, {len(formulas)} formulas)")

        # Execute ALL tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Organize results by type
        organized = {"tables": [], "pictures": [], "formulas": []}

        for i, result in enumerate(results):
            task_type = task_types[i]

            if isinstance(result, Exception):
                # Handle exceptions from gather
                error_result = {
                    "success": False,
                    "error": str(result),
                    "type": task_type,
                }
                if task_type == "table":
                    organized["tables"].append(error_result)
                elif task_type == "picture":
                    organized["pictures"].append(error_result)
                else:
                    organized["formulas"].append(error_result)
            else:
                if task_type == "table":
                    organized["tables"].append(result)
                elif task_type == "picture":
                    organized["pictures"].append(result)
                else:
                    organized["formulas"].append(result)

        # Sort by page number
        for key in organized:
            organized[key].sort(key=lambda r: r.get("page_no", 0))

        successful = sum(1 for r in results if not isinstance(r, Exception) and r.get("success", False))
        logger.info(f"VLM processing complete: {successful}/{total_tasks} successful")

        return organized


# =============================================================================
# Convenience function for use in document_worker.py
# =============================================================================

async def process_vlm_tasks_async(
    tables: List[Dict],
    figures: List[Dict],
    formulas: List[Dict],
    context: str = "",
) -> Dict[str, List[Dict]]:
    """
    Process all VLM tasks concurrently.

    Convenience wrapper for use in document_worker.py.

    Example:
        vlm_results = asyncio.run(process_vlm_tasks_async(
            tables=docling_tables,
            figures=docling_figures,
            formulas=annotated_pages,
        ))
    """
    async with VLMClientAsync() as client:
        return await client.process_all_concurrent(
            tables=tables,
            figures=figures,
            formulas=formulas,
            context=context,
        )
