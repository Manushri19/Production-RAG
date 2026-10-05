"""
LLM Streaming Client

Async SSE streaming client for vLLM (OpenAI-compatible API).
Handles token-by-token streaming and yields each token as it arrives.
"""

import json
import logging
from typing import AsyncIterator

import httpx

import oracle.config as cfg

logger = logging.getLogger(__name__)


class LLMClient:
    """Async streaming client for vLLM / OpenAI-compatible LLM endpoint."""

    def __init__(self):
        self._http = httpx.AsyncClient(
            base_url=cfg.VLLM_BASE_URL,
            headers={
                "Authorization": f"Bearer {cfg.VLLM_API_KEY}",
                "Content-Type": "application/json",
            },
            timeout=httpx.Timeout(cfg.VLLM_TIMEOUT),
        )

    async def stream_completion(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> AsyncIterator[str]:
        """
        Stream a chat completion from vLLM.

        Yields:
            Token strings as they arrive from the LLM.
            Raises RuntimeError if the LLM endpoint is unreachable.
        """
        payload = {
            "model": cfg.VLLM_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": cfg.VLLM_MAX_TOKENS,
            "temperature": cfg.VLLM_TEMPERATURE,
            "stream": True,
        }

        try:
            async with self._http.stream("POST", "/chat/completions", json=payload) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    # OpenAI SSE format: "data: {...}" or "data: [DONE]"
                    if line.startswith("data: "):
                        line = line[len("data: "):]
                    if line == "[DONE]":
                        return
                    try:
                        chunk = json.loads(line)
                        delta = chunk["choices"][0].get("delta", {})
                        token = delta.get("content", "")
                        if token:
                            yield token
                    except (json.JSONDecodeError, KeyError, IndexError):
                        continue  # Skip malformed chunks

        except httpx.ConnectError:
            raise RuntimeError(
                f"LLM endpoint unreachable at {cfg.VLLM_BASE_URL}. "
                "Ensure vLLM is running and VLLM_BASE_URL is set correctly."
            )
        except httpx.HTTPStatusError as e:
            raise RuntimeError(f"LLM HTTP error {e.response.status_code}: {e.response.text}")

    async def aclose(self):
        await self._http.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.aclose()
