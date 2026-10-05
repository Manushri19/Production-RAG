"""
VLM Service Client

HTTP client for vision-language model processing via vLLM.
Supports tables, formulas, and figures with type-specific prompts.
Accepts both base64 images (from Docling service) and file paths.
"""

import base64
import json
import logging
import time
from io import BytesIO
from pathlib import Path
from typing import Optional, Dict, Any, List, Union
from dataclasses import dataclass

import httpx
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class VLMClientConfig:
    """Configuration for VLM client."""
    api_url: str = "http://localhost:8000/v1/chat/completions"
    model: str = "Qwen/Qwen3-VL-8B-Instruct"  # Current deployed model
    api_key: str = "EMPTY"  # vLLM doesn't require a key

    # Token limits per task type
    max_tokens_table: int = 4000
    max_tokens_formula: int = 2000
    max_tokens_picture: int = 1500

    # Retry configuration
    max_retries: int = 5
    retry_delay_seconds: float = 2.0
    retry_backoff_multiplier: float = 2.0

    # Request settings
    timeout_seconds: float = 300.0
    temperature: float = 0.2

    # Image processing
    max_image_dimension: int = 1536


class VLMClientError(Exception):
    """Base exception for VLM client errors."""
    pass


class VLMServiceUnavailable(VLMClientError):
    """VLM service is not available."""
    pass


class VLMProcessingError(VLMClientError):
    """Error during VLM processing."""
    pass


class VLMClient:
    """
    HTTP client for vLLM vision-language model processing.

    Supports:
    - Table extraction to markdown
    - Formula extraction to LaTeX
    - Figure description generation
    - Both base64 images and file paths

    Usage:
        client = VLMClient()

        # Process table (base64 from Docling)
        result = client.process_table(base64_image, context="Document about physics")

        # Process formula page (file path)
        result = client.process_formula(Path("page_5.png"))

        # Batch processing
        results = client.process_batch([
            {"type": "table", "image": base64_img1},
            {"type": "formula", "image": base64_img2},
        ])
    """

    def __init__(self, config: Optional[VLMClientConfig] = None):
        """Initialize the VLM client."""
        self.config = config or VLMClientConfig()
        self._client = httpx.Client(timeout=httpx.Timeout(self.config.timeout_seconds))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        """Close the HTTP client."""
        self._client.close()

    # =========================================================================
    # Health & Status
    # =========================================================================

    def health(self) -> Dict[str, Any]:
        """Check vLLM service health."""
        try:
            # vLLM doesn't have a dedicated health endpoint, check models
            response = self._client.get(
                self.config.api_url.replace("/chat/completions", "/models"),
                timeout=10.0,
            )
            if response.status_code == 200:
                data = response.json()
                models = data.get("data", [])
                model_ids = [m.get("id", "") for m in models]
                return {
                    "status": "healthy",
                    "models": model_ids,
                    "model_loaded": self.config.model in model_ids or any(
                        self.config.model in m for m in model_ids
                    ),
                }
            return {"status": "unhealthy", "error": f"Status {response.status_code}"}
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}

    def is_healthy(self) -> bool:
        """Check if service is healthy and model is loaded."""
        health = self.health()
        return health.get("status") == "healthy" and health.get("model_loaded", False)

    # =========================================================================
    # Image Handling
    # =========================================================================

    def _prepare_image(self, image: Union[str, Path, bytes]) -> str:
        """
        Prepare image for API request.

        Args:
            image: Base64 string, file path, or raw bytes

        Returns:
            Base64-encoded string (resized if needed)
        """
        # Already base64 string
        if isinstance(image, str) and not Path(image).exists():
            # Decode, resize if needed, re-encode
            img_bytes = base64.b64decode(image)
            return self._resize_and_encode(img_bytes)

        # File path
        if isinstance(image, (str, Path)):
            path = Path(image)
            if not path.exists():
                raise VLMClientError(f"Image file not found: {path}")
            with open(path, "rb") as f:
                return self._resize_and_encode(f.read())

        # Raw bytes
        if isinstance(image, bytes):
            return self._resize_and_encode(image)

        raise VLMClientError(f"Unsupported image type: {type(image)}")

    def _resize_and_encode(self, img_bytes: bytes) -> str:
        """Resize image if too large and encode to base64."""
        img = Image.open(BytesIO(img_bytes))

        max_dim = self.config.max_image_dimension
        if max(img.size) > max_dim:
            ratio = max_dim / max(img.size)
            new_size = tuple(int(d * ratio) for d in img.size)
            img = img.resize(new_size, Image.Resampling.LANCZOS)

        # Convert to RGB if needed (for RGBA/P modes)
        if img.mode in ('RGBA', 'P'):
            img = img.convert('RGB')

        buffered = BytesIO()
        img.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    # =========================================================================
    # Core API Call
    # =========================================================================

    def _call_vlm(
        self,
        image_base64: str,
        prompt: str,
        max_tokens: int,
    ) -> str:
        """
        Call VLM API with retry logic.

        Returns:
            Raw response content string
        """
        last_error = None
        delay = self.config.retry_delay_seconds

        for attempt in range(self.config.max_retries):
            try:
                response = self._client.post(
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
                )
                response.raise_for_status()
                data = response.json()

                if "choices" in data and len(data["choices"]) > 0:
                    return data["choices"][0]["message"]["content"]

                logger.warning(f"No 'choices' in response, attempt {attempt + 1}")
                last_error = "No choices in response"

            except httpx.TimeoutException as e:
                last_error = e
                logger.warning(f"VLM timeout (attempt {attempt + 1}/{self.config.max_retries})")

            except httpx.ConnectError as e:
                last_error = e
                logger.warning(f"VLM connection error (attempt {attempt + 1}/{self.config.max_retries})")

            except Exception as e:
                last_error = e
                logger.warning(f"VLM call failed (attempt {attempt + 1}/{self.config.max_retries}): {e}")

            # Exponential backoff
            if attempt < self.config.max_retries - 1:
                time.sleep(delay)
                delay *= self.config.retry_backoff_multiplier

        raise VLMServiceUnavailable(
            f"VLM service unavailable after {self.config.max_retries} attempts",
        )

    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """Parse JSON from VLM response (handles markdown code blocks)."""
        # Try to find JSON in response
        text = response.strip()

        # Remove markdown code blocks if present
        if text.startswith("```"):
            lines = text.split("\n")
            # Skip first line (```json or ```)
            lines = lines[1:]
            # Remove trailing ```
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines)

        # Find JSON object
        start = text.find("{")
        end = text.rfind("}") + 1

        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end])
            except json.JSONDecodeError:
                pass

        return {"error": "Failed to parse JSON", "raw_response": response[:500]}

    # =========================================================================
    # Prompts (from battle-tested batch_processor.py)
    # =========================================================================

    def _get_table_prompt(self, context: str = "") -> str:
        """Get prompt for table extraction."""
        context_section = f"\n\nDOCUMENT CONTEXT:\n{context}\n" if context else ""

        return f"""Extract this table to markdown format.
{context_section}
IMPORTANT: You must be 100% accurate. Every cell, every value, every detail must be correct.

NOTE: The image includes extra padding around the table to provide context.
Focus ONLY on extracting the main table. Use surrounding context for title if visible.

INCLUDE IN YOUR EXTRACTION:
1. Table title/heading (e.g., "TABLE I. - SUMMARY OF..." or "Table 1: Results...")
2. Any subtitle or description below the title
3. The full table with all rows and columns
4. Any footnotes or notes that belong to THIS table

If you are confident you can extract this table with 100% accuracy:
- Set "complex": false
- Provide the COMPLETE extraction in "markdown" including title, table, and footnotes

If you CANNOT guarantee 100% accuracy (unclear text, unusual formatting, too dense):
- Set "complex": true
- Provide a BRIEF description (max 250 words) in "description"

ALWAYS include "notes" for any nuances that cannot be captured in markdown:
- Merged cells, spanning headers, or unusual layouts
- Units, abbreviations, or symbols that need clarification
- Footnote references and their meanings
- Any context important for understanding the data

Output JSON:
{{
  "complex": false,
  "markdown": "## TABLE I. - TITLE\\n\\n| Header1 | Header2 |...\\n|---|---|...\\n| val1 | val2 |...",
  "notes": "Optional notes about merged cells, units, abbreviations, or other nuances not captured in markdown"
}}

OR

{{
  "complex": true,
  "description": "TABLE X - [title]. X rows x Y columns. [brief description]",
  "notes": "Notes about table structure, data relationships, or context needed to understand the table"
}}

Output ONLY the JSON object."""

    def _get_formula_prompt(self, context: str = "") -> str:
        """Get prompt for formula extraction."""
        context_section = f"\n\nDOCUMENT CONTEXT:\n{context}\n" if context else ""

        return f"""Extract all mathematical formulas from this page.
{context_section}
For each formula:
1. Extract the equation in LaTeX format
2. Include the equation number if present
3. Note any variable definitions nearby

Output JSON:
{{
  "formulas": [
    {{
      "latex": "E = mc^2",
      "equation_number": "(1)",
      "description": "Energy-mass equivalence"
    }}
  ],
  "page_has_formulas": true/false
}}

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

    # =========================================================================
    # Processing Methods
    # =========================================================================

    def process_table(
        self,
        image: Union[str, Path, bytes],
        context: str = "",
        page_no: int = 0,
    ) -> Dict[str, Any]:
        """
        Process a table image and extract to markdown.

        Args:
            image: Base64 string, file path, or raw bytes
            context: Optional document context
            page_no: Page number for metadata

        Returns:
            Dict with 'success', 'complex', 'markdown' or 'description', 'notes'
        """
        try:
            img_base64 = self._prepare_image(image)
            prompt = self._get_table_prompt(context)
            response = self._call_vlm(img_base64, prompt, self.config.max_tokens_table)
            result = self._parse_json_response(response)
            result["success"] = "error" not in result
            result["page_no"] = page_no
            result["type"] = "table"
            return result
        except Exception as e:
            logger.error(f"Table processing failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "page_no": page_no,
                "type": "table",
            }

    def process_formula(
        self,
        image: Union[str, Path, bytes],
        context: str = "",
        page_no: int = 0,
    ) -> Dict[str, Any]:
        """
        Process a formula page and extract LaTeX.

        Args:
            image: Base64 string, file path, or raw bytes
            context: Optional document context
            page_no: Page number for metadata

        Returns:
            Dict with 'success', 'formulas' list, 'page_has_formulas'
        """
        try:
            img_base64 = self._prepare_image(image)
            prompt = self._get_formula_prompt(context)
            response = self._call_vlm(img_base64, prompt, self.config.max_tokens_formula)
            result = self._parse_json_response(response)
            result["success"] = "error" not in result
            result["page_no"] = page_no
            result["type"] = "formula"
            return result
        except Exception as e:
            logger.error(f"Formula processing failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "page_no": page_no,
                "type": "formula",
            }

    def process_picture(
        self,
        image: Union[str, Path, bytes],
        context: str = "",
        page_no: int = 0,
    ) -> Dict[str, Any]:
        """
        Process a figure/picture and generate description.

        Args:
            image: Base64 string, file path, or raw bytes
            context: Optional document context
            page_no: Page number for metadata

        Returns:
            Dict with 'success', 'description', 'figure_type', 'key_elements'
        """
        try:
            img_base64 = self._prepare_image(image)
            prompt = self._get_picture_prompt(context)
            response = self._call_vlm(img_base64, prompt, self.config.max_tokens_picture)
            result = self._parse_json_response(response)
            result["success"] = "error" not in result
            result["page_no"] = page_no
            result["type"] = "picture"
            return result
        except Exception as e:
            logger.error(f"Picture processing failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "page_no": page_no,
                "type": "picture",
            }

    def process_batch(
        self,
        tasks: List[Dict[str, Any]],
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Process multiple VLM tasks.

        Note: Unlike batch_processor.py which uses ThreadPoolExecutor,
        this processes sequentially to avoid overwhelming the VLM service.
        For true parallelism, use multiple Celery workers.

        Args:
            tasks: List of task dicts with keys:
                - type: "table" | "formula" | "picture"
                - image: Base64 string, file path, or bytes
                - context: Optional context string
                - page_no: Optional page number

        Returns:
            Dict with results grouped by type: {"tables": [...], "formulas": [...], "pictures": [...]}
        """
        results = {"tables": [], "formulas": [], "pictures": []}

        for task in tasks:
            task_type = task.get("type", "")
            image = task.get("image")
            context = task.get("context", "")
            page_no = task.get("page_no", 0)

            if task_type == "table":
                result = self.process_table(image, context, page_no)
                results["tables"].append(result)
            elif task_type == "formula":
                result = self.process_formula(image, context, page_no)
                results["formulas"].append(result)
            elif task_type == "picture":
                result = self.process_picture(image, context, page_no)
                results["pictures"].append(result)
            else:
                logger.warning(f"Unknown task type: {task_type}")

        # Sort by page number
        for key in results:
            results[key].sort(key=lambda r: r.get("page_no", 0))

        return results


# =============================================================================
# Convenience Functions
# =============================================================================

def create_client(
    api_url: str = "http://localhost:8000/v1/chat/completions",
    model: str = "Qwen/Qwen2.5-VL-72B-Instruct-AWQ",
) -> VLMClient:
    """Create a VLM client with custom settings."""
    config = VLMClientConfig(api_url=api_url, model=model)
    return VLMClient(config)
