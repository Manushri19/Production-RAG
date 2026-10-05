"""
Query Orchestrator

Top-level pipeline wiring:
  embed → hybrid_search → rerank → stream_generation

All shared clients are instantiated once and reused across requests.
"""

import logging
from typing import Any, AsyncIterator, Dict, List, Optional

from qdrant_client import AsyncQdrantClient

import oracle.config as cfg
from oracle.clients.embedding_client import EmbeddingClient
from oracle.clients.llm_client import LLMClient
from oracle.clients.reranker_client import RerankerClient
from oracle.generation.streamer import stream_generation
from oracle.retrieval.hybrid_search import hybrid_search
from oracle.retrieval.reranker import rerank_candidates

logger = logging.getLogger(__name__)


class QueryOrchestrator:
    """
    Stateless orchestrator that wires all pipeline stages together.

    Shared clients are created once at startup and reused across requests.
    The Qdrant client uses HTTP (not gRPC) for simplicity; upgrade to gRPC
    by setting prefer_grpc=True if p99 latency becomes critical.
    """

    def __init__(self):
        self._embedding = EmbeddingClient()
        self._reranker = RerankerClient()
        self._llm = LLMClient()
        self._qdrant = AsyncQdrantClient(
            host=cfg.QDRANT_HOST,
            port=cfg.QDRANT_PORT,
            timeout=30,
        )
        logger.info(
            f"QueryOrchestrator initialized | Qdrant={cfg.QDRANT_HOST}:{cfg.QDRANT_PORT} "
            f"| Collection={cfg.QDRANT_COLLECTION} "
            f"| Reranker={cfg.RERANKER_URL} "
            f"| LLM={cfg.VLLM_BASE_URL}"
        )

    async def run_stream(
        self,
        query: str,
        document_ids: Optional[List[str]],
        top_k_candidates: int,
        top_k_final: int,
    ) -> AsyncIterator[str]:
        """
        Execute the full RAG pipeline and yield SSE events.

        Stages:
            1. Embed query (Ollama)
            2. Hybrid search → top_k_candidates fused chunks (Qdrant RRF)
            3. Cross-encoder reranking → top_k_final chunks
            4. Streaming generation with citation guardrails (vLLM SSE)

        Args:
            query: User question string.
            document_ids: Optional scope filter.
            top_k_candidates: Hybrid search pool size.
            top_k_final: Final context window size after reranking.

        Yields:
            SSE-formatted strings.
        """
        # ── Stage 1: Embed query ──────────────────────────────────────────
        logger.info(f"[Query] '{query[:80]}{'...' if len(query) > 80 else ''}'")
        query_vector = await self._embedding.embed_text(query)
        if not query_vector:
            yield 'data: {"error": "Failed to embed query — Ollama unreachable"}\n\n'
            yield "data: [DONE]\n\n"
            return

        # ── Stage 2: Hybrid Search (dense + BM25 → RRF) ──────────────────
        try:
            candidates = await hybrid_search(
                qdrant=self._qdrant,
                query_vector=query_vector,
                query_text=query,
                top_k=top_k_candidates,
                document_ids=document_ids,
            )
        except RuntimeError as e:
            yield f'data: {{"error": "{str(e)}"}}\n\n'
            yield "data: [DONE]\n\n"
            return

        if not candidates:
            yield 'data: {"token": "I cannot answer this question based on the provided documents."}\n\n'
            yield 'data: {"sources": []}\n\n'
            yield "data: [DONE]\n\n"
            return

        # ── Stage 3: Cross-Encoder Reranking ─────────────────────────────
        top_chunks = await rerank_candidates(
            query=query,
            candidates=candidates,
            top_k=top_k_final,
            reranker=self._reranker,
        )

        logger.info(
            f"Pipeline: {len(candidates)} candidates → reranked → {len(top_chunks)} chunks "
            f"sent to LLM"
        )

        # ── Stage 4: Streaming Generation ────────────────────────────────
        async for sse_event in stream_generation(
            query=query,
            chunks=top_chunks,
            llm=self._llm,
        ):
            yield sse_event
