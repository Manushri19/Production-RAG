"""
Generation Streamer

Handles SSE streaming of LLM output and citation extraction.

SSE event format (each event is a JSON line prefixed with "data: "):
  data: {"token": "..."}\n\n       — progressive token
  data: {"sources": [...]}\n\n     — final event with parsed citations
  data: [DONE]\n\n                 — stream terminator

Citation parsing:
  The LLM outputs a ```sources [...] ``` fenced JSON block at the end.
  We buffer the full response, parse the block, and emit it as the final event.
"""

import json
import logging
import re
from typing import Any, AsyncIterator, Dict, List

from oracle.clients.llm_client import LLMClient
from oracle.generation.prompt_builder import SYSTEM_PROMPT, build_context_block, build_user_prompt

logger = logging.getLogger(__name__)

# Regex to extract the ```sources [...] ``` fenced block from LLM output
_SOURCES_PATTERN = re.compile(
    r"```sources\s*(\[.*?\])\s*```",
    re.DOTALL | re.IGNORECASE,
)


def _parse_sources(full_text: str) -> List[Dict[str, Any]]:
    """
    Extract and parse the SOURCES JSON block from LLM output.

    Returns empty list if the block is absent or malformed.
    """
    match = _SOURCES_PATTERN.search(full_text)
    if not match:
        logger.warning("LLM output did not contain a valid ```sources``` block")
        return []
    try:
        return json.loads(match.group(1))
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse sources JSON: {e}")
        return []


def _sse(payload: Any) -> str:
    """Format a payload as an SSE data line."""
    return f"data: {json.dumps(payload)}\n\n"


async def stream_generation(
    query: str,
    chunks: List[Dict[str, Any]],
    llm: LLMClient,
) -> AsyncIterator[str]:
    """
    Stream generation output as SSE events.

    Args:
        query: The user's question.
        chunks: Reranked context chunks (top-k after cross-encoder).
        llm: Shared LLMClient instance.

    Yields:
        SSE-formatted strings:
            - Token events: `data: {"token": "..."}\n\n`
            - Sources event: `data: {"sources": [...]}\n\n`
            - Done terminator: `data: [DONE]\n\n`
    """
    context_block = build_context_block(chunks)
    user_prompt = build_user_prompt(query=query, context_block=context_block)

    full_response = []

    try:
        async for token in llm.stream_completion(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
        ):
            full_response.append(token)
            yield _sse({"token": token})

    except RuntimeError as e:
        # LLM unreachable — emit error event and stop
        logger.error(f"LLM stream error: {e}")
        yield _sse({"error": str(e)})
        yield "data: [DONE]\n\n"
        return

    # Parse citations from the complete response
    complete_text = "".join(full_response)
    sources = _parse_sources(complete_text)

    # Enrich sources with chunk metadata from our context (ground truth)
    chunk_map = {str(i + 1): c for i, c in enumerate(chunks)}
    enriched_sources = []
    for src in sources:
        ref = str(src.get("ref", ""))
        chunk = chunk_map.get(ref, {})
        enriched_sources.append({
            "ref": src.get("ref"),
            "document_id": src.get("document_id") or chunk.get("document_id", ""),
            "page_no": src.get("page_no") if src.get("page_no") is not None else chunk.get("page_no"),
            "chunk_type": src.get("chunk_type") or chunk.get("chunk_type", "text"),
            "excerpt": (src.get("excerpt") or chunk.get("text", ""))[:120],
        })

    yield _sse({"sources": enriched_sources})
    yield "data: [DONE]\n\n"
