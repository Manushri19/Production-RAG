"""
Async Embedding and Vector Store Client

Phase 4 Optimization: Non-blocking embedding generation and storage.
Uses AsyncQdrantClient and async HTTP for Ollama to prevent pipeline stalls.

Key difference from embedding_client.py:
- Uses httpx.AsyncClient instead of httpx.Client
- Uses AsyncQdrantClient instead of QdrantClient
- All methods are async and non-blocking
- Can be used with asyncio.gather() for concurrent operations

Hybrid search support:
- Dense vectors: Ollama qwen3-embedding (semantic)
- Sparse vectors: FastEmbed BM25 (keyword / exact match)
"""

import asyncio
import logging
import uuid
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field

import httpx
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    SparseVectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    SearchParams,
    SparseVector,
)

# FastEmbed BM25 for sparse keyword vectors (lazy import)
try:
    from fastembed import SparseTextEmbedding
    _FASTEMBED_AVAILABLE = True
except ImportError:
    _FASTEMBED_AVAILABLE = False

# Name constants for named vector spaces in Qdrant
DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"


logger = logging.getLogger(__name__)


# =============================================================================
# Configuration (same as sync version)
# =============================================================================

@dataclass
class EmbeddingConfig:
    """Configuration for Ollama embedding service."""
    model: str = "qwen3-embedding:4b-q8_0"
    ollama_url: str = "http://localhost:11434"
    dimensions: int = 1024
    batch_size: int = 512  # Increased from 256 - Ollama handles large batches efficiently
    timeout_seconds: float = 120.0
    keep_alive: str = "240h"  # Keep model loaded for 10 days
    # Concurrency control for embedding batches
    max_concurrent_batches: int = 4


@dataclass
class VectorStoreConfig:
    """Configuration for Qdrant vector store."""
    host: str = "localhost"
    port: int = 6333
    collection_name: str = "meridian_documents"
    embedding_dimension: int = 1024
    distance_metric: Distance = Distance.COSINE
    indexed_fields: List[str] = field(default_factory=lambda: [
        "document_id",
        "page_no",
        "chunk_type",
        "figure_type",
    ])


@dataclass
class AsyncEmbeddingClientConfig:
    """Combined configuration for async embedding client."""
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    vector_store: VectorStoreConfig = field(default_factory=VectorStoreConfig)


# =============================================================================
# Exceptions
# =============================================================================

class AsyncEmbeddingClientError(Exception):
    """Base exception for async embedding client errors."""
    pass


# =============================================================================
# Async Embedding Client
# =============================================================================

class AsyncEmbeddingClient:
    """
    Async client for embeddings and vector storage.

    Non-blocking version that prevents pipeline stalls during document processing.
    Uses the same patterns as VLMClientAsync for consistency.

    Usage:
        async with AsyncEmbeddingClient() as client:
            # Embed and store chunks (non-blocking)
            stored_count = await client.store_document_chunks_async(
                document_id="doc_123",
                chunks=chunks_list,
            )

            # Semantic search
            results = await client.search_async("aerodynamic equations", limit=5)
    """

    def __init__(self, config: Optional[AsyncEmbeddingClientConfig] = None, collection_name: Optional[str] = None):
        """Initialize async embedding client.

        Args:
            config: Full configuration object (optional)
            collection_name: Override collection name (convenience parameter)
        """
        self.config = config or AsyncEmbeddingClientConfig()
        # Allow overriding collection name via convenience parameter
        if collection_name:
            self.config.vector_store.collection_name = collection_name
        self._http_client: Optional[httpx.AsyncClient] = None
        self._qdrant: Optional[AsyncQdrantClient] = None
        self._semaphore: Optional[asyncio.Semaphore] = None

    async def __aenter__(self):
        """Async context manager entry."""
        # Create async HTTP client for Ollama
        self._http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(self.config.embedding.timeout_seconds)
        )

        # Create async Qdrant client. Default httpx timeout is 5s, which is
        # too low under concurrent write load (large PDFs, many chunks).
        self._qdrant = AsyncQdrantClient(
            host=self.config.vector_store.host,
            port=self.config.vector_store.port,
            timeout=120,
        )

        # Semaphore for concurrent batch control
        if self.config.embedding.max_concurrent_batches:
            self._semaphore = asyncio.Semaphore(
                self.config.embedding.max_concurrent_batches
            )

        # Ensure collection exists
        await self._ensure_collection_async()

        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self._http_client:
            await self._http_client.aclose()
        if self._qdrant:
            await self._qdrant.close()

    # =========================================================================
    # Collection Management
    # =========================================================================

    async def _ensure_collection_async(self):
        """Create collection if it doesn't exist."""
        try:
            collections = await self._qdrant.get_collections()
            collection_names = [c.name for c in collections.collections]

            if self.config.vector_store.collection_name not in collection_names:
                # Named vectors: 'dense' for semantic, 'sparse' for BM25 keyword search
                await self._qdrant.create_collection(
                    collection_name=self.config.vector_store.collection_name,
                    vectors_config={
                        DENSE_VECTOR_NAME: VectorParams(
                            size=self.config.vector_store.embedding_dimension,
                            distance=self.config.vector_store.distance_metric,
                        ),
                    },
                    sparse_vectors_config={
                        SPARSE_VECTOR_NAME: SparseVectorParams(),
                    },
                )
                logger.info(f"Created collection with dense+sparse vectors: {self.config.vector_store.collection_name}")

                # Create payload indexes
                for field_name in self.config.vector_store.indexed_fields:
                    try:
                        await self._qdrant.create_payload_index(
                            collection_name=self.config.vector_store.collection_name,
                            field_name=field_name,
                            field_schema="keyword",
                        )
                    except Exception:
                        pass  # Index may already exist
            else:
                logger.debug(f"Collection '{self.config.vector_store.collection_name}' exists")
        except Exception as e:
            logger.warning(f"Could not ensure collection: {e}")

    # =========================================================================
    # Async Embedding
    # =========================================================================

    async def _embed_single_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a single batch of texts asynchronously.

        Args:
            texts: List of texts to embed (up to batch_size)

        Returns:
            List of embedding vectors
        """
        try:
            # Use semaphore if concurrency limiting is enabled
            if self._semaphore:
                async with self._semaphore:
                    return await self._make_embed_request(texts)
            else:
                return await self._make_embed_request(texts)

        except Exception as e:
            logger.error(f"Error embedding batch of {len(texts)} texts: {e}")
            # Return empty embeddings for failed batch
            return [[] for _ in texts]

    async def _make_embed_request(self, texts: List[str]) -> List[List[float]]:
        """Make the actual HTTP request to Ollama."""
        response = await self._http_client.post(
            f"{self.config.embedding.ollama_url}/api/embed",
            json={
                "model": self.config.embedding.model,
                "input": texts,
                "keep_alive": self.config.embedding.keep_alive,
                # Cap context — embeddings only do one forward pass on small
                # chunks; the model's max ctx (40960) is wasted KV/activation buffer.
                "options": {"num_ctx": 4096},
            },
        )
        response.raise_for_status()

        data = response.json()
        embeddings = data.get("embeddings", [])

        # Truncate to configured dimensions if needed
        result = []
        for emb in embeddings:
            if len(emb) > self.config.embedding.dimensions:
                emb = emb[:self.config.embedding.dimensions]
            result.append(emb)

        return result

    async def embed_batch_async(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts concurrently.

        Splits texts into batches and processes them in parallel.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors (same order as input)
        """
        if not texts:
            return []

        batch_size = self.config.embedding.batch_size
        all_embeddings = []

        # Split into batches
        batches = []
        for i in range(0, len(texts), batch_size):
            batches.append(texts[i:i + batch_size])

        if len(batches) == 1:
            # Single batch - no need for gather
            return await self._embed_single_batch(batches[0])

        # Create tasks for all batches
        tasks = [self._embed_single_batch(batch) for batch in batches]

        # Execute all batches concurrently
        logger.debug(f"Embedding {len(texts)} texts in {len(batches)} concurrent batches")
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Flatten results in order, handling exceptions
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Batch {i} embedding failed: {result}")
                # Add empty embeddings for failed batch
                all_embeddings.extend([[] for _ in batches[i]])
            else:
                all_embeddings.extend(result)

        return all_embeddings

    async def embed_text_async(self, text: str) -> List[float]:
        """
        Generate embedding for a single text asynchronously.

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        embeddings = await self.embed_batch_async([text])
        return embeddings[0] if embeddings else []

    # =========================================================================
    # Async Document Storage
    # =========================================================================

    async def store_document_chunks_async(
        self,
        document_id: str,
        chunks: List[Dict[str, Any]],
        embeddings: Optional[List[List[float]]] = None,
    ) -> int:
        """
        Store document chunks with embeddings asynchronously.

        If embeddings are not provided, they will be generated concurrently.

        Args:
            document_id: Unique document identifier
            chunks: List of chunk dicts with keys:
                - text: Chunk text (required)
                - type: "text", "table", "formula", "picture" (optional)
                - page_no: Page number (optional)
                - order: Chunk order (optional)
                - headings: List of headings (optional)
                - metadata: Additional metadata dict (optional)
            embeddings: Optional pre-computed embeddings

        Returns:
            Number of chunks stored
        """
        if not chunks:
            return 0

        # Generate embeddings concurrently if not provided
        if embeddings is None:
            texts = [c.get("text", "") for c in chunks]
            embeddings = await self.embed_batch_async(texts)

        if len(chunks) != len(embeddings):
            raise AsyncEmbeddingClientError(
                f"Chunks ({len(chunks)}) and embeddings ({len(embeddings)}) count mismatch"
            )

        # Compute BM25 sparse vectors in a thread (fastembed is CPU-bound)
        sparse_vectors_list: List[Optional[SparseVector]] = [None] * len(chunks)
        if _FASTEMBED_AVAILABLE:
            try:
                def _compute_sparse(texts: List[str]) -> List[SparseVector]:
                    bm25_model = SparseTextEmbedding(model_name="Qdrant/bm25")
                    results = list(bm25_model.embed(texts))
                    return [
                        SparseVector(indices=sv.indices.tolist(), values=sv.values.tolist())
                        for sv in results
                    ]

                texts = [c.get("text", "") for c in chunks]
                loop = asyncio.get_event_loop()
                sparse_vectors_list = await loop.run_in_executor(None, _compute_sparse, texts)
                logger.debug(f"Computed async BM25 sparse vectors for {len(texts)} chunks")
            except Exception as e:
                logger.warning(f"Async BM25 sparse vector computation failed, dense-only: {e}")
        else:
            logger.warning("fastembed not installed — dense-only storage. Run: pip install fastembed")

        # Build points
        points = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            if not embedding:
                logger.warning(f"Skipping chunk with empty embedding: {chunk.get('text', '')[:50]}...")
                continue

            # Build payload
            payload = {
                "document_id": document_id,
                "text": chunk.get("text", ""),
                "chunk_type": chunk.get("type", "text"),
                "page_no": chunk.get("page_no", 0),
                "order": chunk.get("order", 0),
                "headings": chunk.get("headings", []),
            }

            # Add metadata
            metadata = chunk.get("metadata", {})
            if metadata:
                for key, value in metadata.items():
                    payload[f"meta_{key}"] = value
                if "figure_type" in metadata:
                    payload["figure_type"] = metadata["figure_type"]
                if "complex" in metadata:
                    payload["is_complex_table"] = metadata["complex"]

            point_id = str(uuid.uuid4())
            sparse_vec = sparse_vectors_list[i] if i < len(sparse_vectors_list) else None

            # Build named vector dict: always include dense; add sparse when available
            vector_dict: Dict[str, Any] = {DENSE_VECTOR_NAME: embedding}
            if sparse_vec is not None:
                vector_dict[SPARSE_VECTOR_NAME] = sparse_vec

            points.append(PointStruct(
                id=point_id,
                vector=vector_dict,
                payload=payload,
            ))

        if points:
            # Upsert asynchronously with wait=True for consistency
            await self._qdrant.upsert(
                collection_name=self.config.vector_store.collection_name,
                points=points,
                wait=True,  # Ensures data is written before returning
            )
            logger.info(f"Stored {len(points)} chunks async (dense+sparse) for document '{document_id}'")

        return len(points)

    # =========================================================================
    # Async Search
    # =========================================================================

    async def search_async(
        self,
        query: str,
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        score_threshold: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """
        Async semantic search for similar chunks.

        Args:
            query: Search query text
            limit: Max results to return
            filters: Optional metadata filters (e.g., {"document_id": "doc_123"})
            score_threshold: Minimum similarity score

        Returns:
            List of results with text, metadata, and score
        """
        # Generate query embedding
        query_embedding = await self.embed_text_async(query)
        if not query_embedding:
            logger.error("Failed to generate query embedding")
            return []

        return await self.search_by_vector_async(
            query_embedding, limit, filters, score_threshold
        )

    async def search_by_vector_async(
        self,
        query_embedding: List[float],
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        score_threshold: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """
        Async search by pre-computed embedding vector.

        Args:
            query_embedding: Query vector
            limit: Max results
            filters: Optional metadata filters
            score_threshold: Minimum similarity score

        Returns:
            List of results
        """
        # Build Qdrant filter
        qdrant_filter = None
        if filters:
            conditions = [
                FieldCondition(key=k, match=MatchValue(value=v))
                for k, v in filters.items()
            ]
            qdrant_filter = Filter(must=conditions)

        try:
            results = await self._qdrant.query_points(
                collection_name=self.config.vector_store.collection_name,
                query=query_embedding,
                using=DENSE_VECTOR_NAME,
                limit=limit,
                query_filter=qdrant_filter,
                score_threshold=score_threshold,
                search_params=SearchParams(exact=False, hnsw_ef=128),
            )

            return [
                {
                    "id": hit.id,
                    "score": hit.score,
                    "text": hit.payload.get("text", ""),
                    "document_id": hit.payload.get("document_id", ""),
                    "page_no": hit.payload.get("page_no", 0),
                    "order": hit.payload.get("order", 0),
                    "chunk_type": hit.payload.get("chunk_type", ""),
                    "headings": hit.payload.get("headings", []),
                    "figure_type": hit.payload.get("figure_type"),
                    "is_complex_table": hit.payload.get("is_complex_table"),
                    "metadata": {
                        k.replace("meta_", ""): v
                        for k, v in hit.payload.items()
                        if k.startswith("meta_")
                    },
                }
                for hit in results.points
            ]
        except Exception as e:
            logger.error(f"Async search failed: {e}")
            return []

    # =========================================================================
    # Async Document Operations
    # =========================================================================

    async def delete_document_async(self, document_id: str) -> int:
        """
        Delete all chunks for a document asynchronously.

        Returns:
            Number of chunks deleted (approximate)
        """
        try:
            await self._qdrant.delete(
                collection_name=self.config.vector_store.collection_name,
                points_selector=Filter(
                    must=[
                        FieldCondition(
                            key="document_id",
                            match=MatchValue(value=document_id),
                        )
                    ]
                ),
                wait=True,
            )
            logger.info(f"Deleted chunks for document '{document_id}'")
            return 1  # Approximate - actual count requires separate query
        except Exception as e:
            logger.error(f"Error deleting document: {e}")
            return 0

    async def get_collection_stats_async(self) -> Dict[str, Any]:
        """Get collection statistics asynchronously."""
        try:
            info = await self._qdrant.get_collection(
                self.config.vector_store.collection_name
            )
            vectors_count = getattr(info, 'vectors_count', None)
            if vectors_count is None:
                vectors_count = info.points_count if hasattr(info, 'points_count') else 0
            return {
                "collection_name": self.config.vector_store.collection_name,
                "vectors_count": vectors_count,
                "points_count": getattr(info, 'points_count', vectors_count),
                "status": info.status.name if hasattr(info.status, 'name') else str(info.status),
            }
        except Exception as e:
            return {"error": str(e)}


# =============================================================================
# Convenience Function for Document Worker
# =============================================================================

async def store_chunks_async(
    document_id: str,
    chunks: List[Dict[str, Any]],
) -> int:
    """
    Convenience wrapper for storing document chunks asynchronously.

    Usage in document_worker.py:
        stored_count = asyncio.run(store_chunks_async(doc_id, chunks_dict))

    Or combined with VLM processing:
        async def process_vlm_and_store():
            vlm_results = await process_vlm_tasks_async(...)
            # ... build chunks ...
            stored_count = await store_chunks_async(doc_id, chunks)
            return vlm_results, stored_count
    """
    async with AsyncEmbeddingClient() as client:
        return await client.store_document_chunks_async(
            document_id=document_id,
            chunks=chunks,
        )
