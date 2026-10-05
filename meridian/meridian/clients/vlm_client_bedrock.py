"""
Bedrock VLM client — mirrors vlm_client_async.process_vlm_tasks_async signature
so it can be swapped in via monkey-patch in the document worker.

Uses the Bedrock Converse API. Images are sent as raw bytes (PNG), max
dimension is downscaled to keep tokens bounded. Prompts and JSON parsing are
imported unchanged from vlm_client_async so model outputs follow the same
schema the chunker already understands.

Concurrency: boto3 is sync. We get true concurrency by wrapping each call in
asyncio.to_thread, then asyncio.gather across all VLM tasks for a doc — same
shape as the existing aiohttp client.
"""

import asyncio
import base64
import logging
import os
import time
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import boto3
from botocore.config import Config
from PIL import Image

from meridian.clients.vlm_client_async import VLMClientAsync  # for prompts + parser reuse

logger = logging.getLogger(__name__)


BEDROCK_REGION = os.getenv("BEDROCK_REGION", "us-east-1")
BEDROCK_MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "us.amazon.nova-2-lite-v1:0")
BEDROCK_MAX_IMAGE_DIM = int(os.getenv("BEDROCK_MAX_IMAGE_DIM", "1536"))
BEDROCK_MAX_RETRIES = int(os.getenv("BEDROCK_MAX_RETRIES", "5"))
BEDROCK_RETRY_DELAY = float(os.getenv("BEDROCK_RETRY_DELAY", "2.0"))
# Cap concurrent in-flight Converse calls per worker to avoid TPS throttling.
# Tune up if account TPS allows; 5 is conservative.
BEDROCK_MAX_CONCURRENCY = int(os.getenv("BEDROCK_MAX_CONCURRENCY", "5"))

MAX_TOKENS_TABLE = int(os.getenv("BEDROCK_MAX_TOKENS_TABLE", "4000"))
MAX_TOKENS_FORMULA = int(os.getenv("BEDROCK_MAX_TOKENS_FORMULA", "4000"))
MAX_TOKENS_PICTURE = int(os.getenv("BEDROCK_MAX_TOKENS_PICTURE", "1500"))
TEMPERATURE = float(os.getenv("BEDROCK_TEMPERATURE", "0.2"))


def _build_client():
    cfg = Config(
        retries={"max_attempts": BEDROCK_MAX_RETRIES, "mode": "adaptive"},
        read_timeout=300,
        connect_timeout=30,
    )
    return boto3.client("bedrock-runtime", region_name=BEDROCK_REGION, config=cfg)


# Module-level prompts/parser borrowed from the async client (same schema)
_helper = VLMClientAsync.__new__(VLMClientAsync)
_get_table_prompt = _helper._get_table_prompt
_get_formula_prompt = _helper._get_formula_prompt
_get_picture_prompt = _helper._get_picture_prompt
_parse_json_response = _helper._parse_json_response


def _to_png_bytes(image: Union[str, bytes, Path]) -> bytes:
    """Normalize input (path / base64 / raw bytes) to resized PNG bytes."""
    if isinstance(image, bytes):
        raw = image
    elif isinstance(image, Path):
        raw = image.read_bytes()
    elif isinstance(image, str):
        if len(image) > 200:
            raw = base64.b64decode(image)
        else:
            p = Path(image)
            if p.exists():
                raw = p.read_bytes()
            else:
                raw = base64.b64decode(image)
    else:
        raise TypeError(f"Unsupported image type: {type(image)}")

    img = Image.open(BytesIO(raw))
    if max(img.size) > BEDROCK_MAX_IMAGE_DIM:
        ratio = BEDROCK_MAX_IMAGE_DIM / max(img.size)
        img = img.resize(tuple(int(d * ratio) for d in img.size), Image.Resampling.LANCZOS)
    if img.mode in ("RGBA", "P"):
        img = img.convert("RGB")
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _converse_sync(client, image_bytes: bytes, prompt: str, max_tokens: int) -> Dict[str, Any]:
    """One Converse call. Returns {'text': str, 'usage': {...}}."""
    resp = client.converse(
        modelId=BEDROCK_MODEL_ID,
        messages=[{
            "role": "user",
            "content": [
                {"image": {"format": "png", "source": {"bytes": image_bytes}}},
                {"text": prompt},
            ],
        }],
        inferenceConfig={"maxTokens": max_tokens, "temperature": TEMPERATURE},
    )
    text = resp["output"]["message"]["content"][0]["text"]
    return {"text": text, "usage": resp.get("usage", {})}


async def _call_with_retry(
    client, image_bytes: bytes, prompt: str, max_tokens: int, task_id: str,
    semaphore: asyncio.Semaphore,
) -> Dict[str, Any]:
    delay = BEDROCK_RETRY_DELAY
    last_err = None
    for attempt in range(BEDROCK_MAX_RETRIES):
        try:
            async with semaphore:
                return await asyncio.to_thread(_converse_sync, client, image_bytes, prompt, max_tokens)
        except Exception as e:
            last_err = e
            err_name = type(e).__name__
            # Throttling-aware backoff: longer initial wait for ThrottlingException
            is_throttle = "Throttl" in err_name or "Throttl" in str(e)
            logger.warning(f"Bedrock call failed for {task_id} (attempt {attempt + 1}/{BEDROCK_MAX_RETRIES}, {err_name}): {e}")
            if attempt < BEDROCK_MAX_RETRIES - 1:
                wait = delay * (2 if is_throttle else 1)
                await asyncio.sleep(wait)
                delay *= 2
    raise RuntimeError(f"Bedrock unavailable after {BEDROCK_MAX_RETRIES} attempts: {last_err}")


async def _process_table(client, item: Dict, context: str, sem: asyncio.Semaphore) -> Dict[str, Any]:
    page_no = item.get("page_no", item.get("page", 0))
    try:
        img_bytes = _to_png_bytes(item.get("image") or item.get("image_base64"))
        prompt = _get_table_prompt(context)
        out = await _call_with_retry(client, img_bytes, prompt, MAX_TOKENS_TABLE, f"table_p{page_no}", sem)
        parsed = _parse_json_response(out["text"])
        parsed["success"] = "error" not in parsed
        parsed["page_no"] = page_no
        parsed["type"] = "table"
        parsed["bbox"] = item.get("bbox", {})
        parsed["_usage"] = out.get("usage", {})
        return parsed
    except Exception as e:
        logger.error(f"Table failed (page {page_no}): {e}")
        return {"success": False, "error": str(e), "page_no": page_no, "type": "table", "bbox": item.get("bbox", {})}


async def _process_picture(client, item: Dict, context: str, sem: asyncio.Semaphore) -> Dict[str, Any]:
    page_no = item.get("page_no", item.get("page", 0))
    try:
        img_bytes = _to_png_bytes(item.get("image") or item.get("image_base64"))
        prompt = _get_picture_prompt(context)
        out = await _call_with_retry(client, img_bytes, prompt, MAX_TOKENS_PICTURE, f"picture_p{page_no}", sem)
        parsed = _parse_json_response(out["text"])
        parsed["success"] = "error" not in parsed
        parsed["page_no"] = page_no
        parsed["type"] = "picture"
        parsed["bbox"] = item.get("bbox", {})
        parsed["_usage"] = out.get("usage", {})
        return parsed
    except Exception as e:
        logger.error(f"Picture failed (page {page_no}): {e}")
        return {"success": False, "error": str(e), "page_no": page_no, "type": "picture", "bbox": item.get("bbox", {})}


async def _process_formula(client, item: Dict, context: str, sem: asyncio.Semaphore) -> Dict[str, Any]:
    page_no = item.get("page_no", item.get("page", 0))
    try:
        img_bytes = _to_png_bytes(item.get("image") or item.get("image_base64"))
        prompt = _get_formula_prompt(text_context=item.get("text_context", "") or context)
        out = await _call_with_retry(client, img_bytes, prompt, MAX_TOKENS_FORMULA, f"formula_p{page_no}", sem)
        parsed = _parse_json_response(out["text"])
        parsed["success"] = "error" not in parsed
        parsed["page_no"] = page_no
        parsed["type"] = "formula"
        parsed["_usage"] = out.get("usage", {})
        return parsed
    except Exception as e:
        logger.error(f"Formula failed (page {page_no}): {e}")
        return {"success": False, "error": str(e), "page_no": page_no, "type": "formula"}


async def process_vlm_tasks_async(
    tables: List[Dict],
    figures: List[Dict],
    formulas: List[Dict],
    context: str = "",
) -> Dict[str, List[Dict]]:
    """Drop-in replacement for vlm_client_async.process_vlm_tasks_async."""
    client = _build_client()
    sem = asyncio.Semaphore(BEDROCK_MAX_CONCURRENCY)
    coros: List = []
    types: List[str] = []
    for t in tables:
        coros.append(_process_table(client, t, context, sem)); types.append("table")
    for f in figures:
        coros.append(_process_picture(client, f, context, sem)); types.append("picture")
    for fo in formulas:
        coros.append(_process_formula(client, fo, context, sem)); types.append("formula")

    total = len(coros)
    logger.info(f"Bedrock VLM: dispatching {total} tasks ({len(tables)} tables, {len(figures)} figures, {len(formulas)} formulas) "
                f"on {BEDROCK_MODEL_ID} [concurrency={BEDROCK_MAX_CONCURRENCY}]")
    t0 = time.time()
    results = await asyncio.gather(*coros, return_exceptions=True)
    elapsed = time.time() - t0

    organized: Dict[str, List[Dict]] = {"tables": [], "pictures": [], "formulas": []}
    total_in_tok = total_out_tok = 0
    success = 0
    for i, r in enumerate(results):
        ttype = types[i]
        bucket = "tables" if ttype == "table" else ("pictures" if ttype == "picture" else "formulas")
        if isinstance(r, Exception):
            organized[bucket].append({"success": False, "error": str(r), "type": ttype})
        else:
            organized[bucket].append(r)
            if r.get("success"):
                success += 1
            u = r.get("_usage") or {}
            total_in_tok += int(u.get("inputTokens", 0) or 0)
            total_out_tok += int(u.get("outputTokens", 0) or 0)

    for k in organized:
        organized[k].sort(key=lambda x: x.get("page_no", 0))

    logger.info(
        f"Bedrock VLM: {success}/{total} succeeded in {elapsed:.1f}s "
        f"(input={total_in_tok} tok, output={total_out_tok} tok)"
    )
    return organized
