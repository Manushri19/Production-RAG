"""
Embedding and Vector Store Client

Combined client for:
1. Ollama embedding service (text → vectors)
2. Qdrant vector store (storage and retrieval)

Provides a unified interface for document chunk storage and semantic search.

Hybrid search support:
- Dense vectors: Ollama qwen3-embedding (semantic)
- Sparse vectors: FastEmbed BM25 (keyword / exact match)
Both are stored per-point so Oracle's retrieval pipeline can run RRF fusion.
"""

import logging
import uuid
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field

import httpx
from qdrant_client import QdrantClient
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
    NamedVector,
    NamedSparseVector,
)

# FastEmbed BM25 for sparse keyword vectors (lazy import to keep startup fast)
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
# Configuration
# =============================================================================

@dataclass
class EmbeddingConfig:
    """Configuration for Ollama embedding service."""
    model: str = "qwen3-embedding:4b-q8_0"
    ollama_url: str = "http://localhost:11434"
    dimensions: int = 1024  # qwen3-embedding supports 32-4096
    batch_size: int = 32
    timeout_seconds: float = 120.0
    keep_alive: str = "240h"  # Keep model loaded for 10 days


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
class EmbeddingClientConfig:
    """Combined configuration for embedding client."""
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    vector_store: VectorStoreConfig = field(default_factory=VectorStoreConfig)


# =============================================================================
# Exceptions
# =============================================================================

class EmbeddingClientError(Exception):
    """Base exception for embedding client errors."""
    pass


class OllamaServiceUnavailable(EmbeddingClientError):
    """Ollama service is not available."""
    pass


class QdrantServiceUnavailable(EmbeddingClientError):
    """Qdrant service is not available."""
    pass


# =============================================================================
# Embedding Client
# =============================================================================

class EmbeddingClient:
    """
    Combined client for embeddings and vector storage.

    Handles:
    - Text embedding via Ollama
    - Batch embedding with chunking
    - Qdrant collection management
    - Document storage and retrieval
    - Semantic search with filters

    Usage:
        client = EmbeddingClient()

        # Check services
        if client.is_healthy():
            # Embed and store chunks
            chunks = [
                {"text": "Chapter 1...", "type": "text", "page_no": 1},
                {"text": "| Table |", "type": "table", "page_no": 2},
            ]
            client.store_document_chunks("doc_123", chunks)

            # Semantic search
            results = client.search("aerodynamic equations", limit=5)
    """

    def __init__(self, config: Optional[EmbeddingClientConfig] = None):
        """Initialize embedding and vector store clients."""
        self.config = config or EmbeddingClientConfig()

        # HTTP client for Ollama
        self._http_client = httpx.Client(
            timeout=httpx.Timeout(self.config.embedding.timeout_seconds)
        )

        # Qdrant client
        self._qdrant = QdrantClient(
            host=self.config.vector_store.host,
            port=self.config.vector_store.port,
        )

        # Ensure collection exists
        self._ensure_collection()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        """Close all clients."""
        self._http_client.close()

    # =========================================================================
    # Health & Status
    # =========================================================================

    def health_ollama(self) -> Dict[str, Any]:
        """Check Ollama service health."""
        try:
            response = self._http_client.get(
                f"{self.config.embedding.ollama_url}/api/tags",
                timeout=5.0,
            )
            if response.status_code == 200:
                data = response.json()
                models = data.get("models", [])
                model_names = [m.get("name", "") for m in models]
                model_available = any(
                    self.config.embedding.model in name for name in model_names
                )
                return {
                    "status": "healthy",
                    "models": model_names,
                    "model_available": model_available,
                    "model": self.config.embedding.model,
                }
            return {"status": "unhealthy", "error": f"Status {response.status_code}"}
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}

    def health_qdrant(self) -> Dict[str, Any]:
        """Check Qdrant service health."""
        try:
            collections = self._qdrant.get_collections().collections
            collection_names = [c.name for c in collections]
            return {
                "status": "healthy",
                "collections": collection_names,
                "target_collection": self.config.vector_store.collection_name,
                "collection_exists": self.config.vector_store.collection_name in collection_names,
            }
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}

    def health(self) -> Dict[str, Any]:
        """Check all services health."""
        ollama = self.health_ollama()
        qdrant = self.health_qdrant()
        all_healthy = (
            ollama.get("status") == "healthy"
            and ollama.get("model_available", False)
            and qdrant.get("status") == "healthy"
        )
        return {
            "status": "healthy" if all_healthy else "unhealthy",
            "ollama": ollama,
            "qdrant": qdrant,
        }

    def is_healthy(self) -> bool:
        """Check if all services are healthy."""
        return self.health().get("status") == "healthy"

    # =========================================================================
    # Collection Management
    # =========================================================================

    def _ensure_collection(self):
        """Create collection if it doesn't exist."""
        try:
            collections = self._qdrant.get_collections().collections
            collection_names = [c.name for c in collections]

            if self.config.vector_store.collection_name not in collection_names:
                # Named vectors: 'dense' for semantic, 'sparse' for BM25 keyword search
                self._qdrant.create_collection(
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
                        self._qdrant.create_payload_index(
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

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics."""
        try:
            info = self._qdrant.get_collection(self.config.vector_store.collection_name)
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

    def delete_collection(self) -> bool:
        """Delete the collection. Returns True if deleted."""
        try:
            collections = self._qdrant.get_collections().collections
            if self.config.vector_store.collection_name in [c.name for c in collections]:
                self._qdrant.delete_collection(self.config.vector_store.collection_name)
                logger.info(f"Deleted collection: {self.config.vector_store.collection_name}")
                return True
            return False
        except Exception as e:
            logger.error(f"Error deleting collection: {e}")
            return False

    # =========================================================================
    # Embedding
    # =========================================================================

    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.

        Args:
            text: Text to embed

        Returns:
            List of floats (embedding vector)
        """
        try:
            response = self._http_client.post(
                f"{self.config.embedding.ollama_url}/api/embed",
                json={
                    "model": self.config.embedding.model,
                    "input": text,
                    "keep_alive": self.config.embedding.keep_alive,
                },
            )
            response.raise_for_status()

            data = response.json()
            embeddings = data.get("embeddings", [])

            if embeddings and len(embeddings) > 0:
                embedding = embeddings[0]
                # Truncate to configured dimensions if needed
                if len(embedding) > self.config.embedding.dimensions:
                    embedding = embedding[:self.config.embedding.dimensions]
                return embedding
            else:
                logger.error(f"No embeddings returned for text: {text[:50]}...")
                return []
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return []

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        all_embeddings = []
        batch_size = self.config.embedding.batch_size

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]

            try:
                response = self._http_client.post(
                    f"{self.config.embedding.ollama_url}/api/embed",
                    json={
                        "model": self.config.embedding.model,
                        "input": batch,
                        "keep_alive": self.config.embedding.keep_alive,
                    },
                )
                response.raise_for_status()

                data = response.json()
                embeddings = data.get("embeddings", [])

                for emb in embeddings:
                    if len(emb) > self.config.embedding.dimensions:
                        emb = emb[:self.config.embedding.dimensions]
                    all_embeddings.append(emb)

                logger.debug(f"Embedded batch {i//batch_size + 1}: {len(batch)} texts")

            except Exception as e:
                logger.error(f"Error embedding batch: {e}")
                all_embeddings.extend([[] for _ in batch])

        return all_embeddings

    # =========================================================================
    # Document Storage
    # =========================================================================

    def store_document_chunks(
        self,
        document_id: str,
        chunks: List[Dict[str, Any]],
        embeddings: Optional[List[List[float]]] = None,
    ) -> int:
        """
        Store document chunks with embeddings.

        If embeddings are not provided, they will be generated.

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

        # Generate embeddings if not provided
        if embeddings is None:
            texts = [c.get("text", "") for c in chunks]
            embeddings = self.embed_batch(texts)

        if len(chunks) != len(embeddings):
            raise EmbeddingClientError(
                f"Chunks ({len(chunks)}) and embeddings ({len(embeddings)}) count mismatch"
            )

        # Compute BM25 sparse vectors for keyword search (if fastembed is available)
        sparse_vectors_list: List[Optional[SparseVector]] = [None] * len(chunks)
        if _FASTEMBED_AVAILABLE:
            try:
                bm25_model = SparseTextEmbedding(model_name="Qdrant/bm25")
                texts = [c.get("text", "") for c in chunks]
                sparse_results = list(bm25_model.embed(texts))
                sparse_vectors_list = [
                    SparseVector(indices=sv.indices.tolist(), values=sv.values.tolist())
                    for sv in sparse_results
                ]
                logger.debug(f"Computed BM25 sparse vectors for {len(texts)} chunks")
            except Exception as e:
                logger.warning(f"BM25 sparse vector computation failed, storing dense-only: {e}")
        else:
            logger.warning("fastembed not installed — storing dense vectors only. Run: pip install fastembed")

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
            self._qdrant.upsert(
                collection_name=self.config.vector_store.collection_name,
                points=points,
            )
            logger.info(f"Stored {len(points)} chunks (dense+sparse) for document '{document_id}'")

        return len(points)

    # =========================================================================
    # Search
    # =========================================================================

    def search(
        self,
        query: str,
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        score_threshold: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """
        Semantic search for similar chunks.

        Args:
            query: Search query text
            limit: Max results to return
            filters: Optional metadata filters (e.g., {"document_id": "doc_123"})
            score_threshold: Minimum similarity score

        Returns:
            List of results with text, metadata, and score
        """
        # Generate query embedding
        query_embedding = self.embed_text(query)
        if not query_embedding:
            logger.error("Failed to generate query embedding")
            return []

        return self.search_by_vector(query_embedding, limit, filters, score_threshold)

    def search_by_vector(
        self,
        query_embedding: List[float],
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        score_threshold: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """
        Search by pre-computed embedding vector.

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
            results = self._qdrant.query_points(
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
            logger.error(f"Search failed: {e}")
            return []

    # =========================================================================
    # Document Operations
    # =========================================================================

    def get_document_chunks(
        self,
        document_id: str,
        chunk_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get all chunks for a document.

        Args:
            document_id: Document identifier
            chunk_type: Optional filter by type ("text", "table", etc.)

        Returns:
            List of chunks sorted by order
        """
        conditions = [
            FieldCondition(key="document_id", match=MatchValue(value=document_id))
        ]
        if chunk_type:
            conditions.append(
                FieldCondition(key="chunk_type", match=MatchValue(value=chunk_type))
            )

        try:
            results, _ = self._qdrant.scroll(
                collection_name=self.config.vector_store.collection_name,
                scroll_filter=Filter(must=conditions),
                limit=10000,
                with_payload=True,
                with_vectors=False,
            )

            chunks = [
                {
                    "id": point.id,
                    "text": point.payload.get("text", ""),
                    "page_no": point.payload.get("page_no", 0),
                    "chunk_type": point.payload.get("chunk_type", ""),
                    "order": point.payload.get("order", 0),
                    "headings": point.payload.get("headings", []),
                }
                for point in results
            ]

            # Sort by order
            chunks.sort(key=lambda c: c.get("order", 0))
            return chunks
        except Exception as e:
            logger.error(f"Error getting document chunks: {e}")
            return []

    def delete_document(self, document_id: str) -> int:
        """
        Delete all chunks for a document.

        Returns:
            Number of chunks deleted
        """
        try:
            # Count before deletion
            chunks = self.get_document_chunks(document_id)
            count = len(chunks)

            self._qdrant.delete(
                collection_name=self.config.vector_store.collection_name,
                points_selector=Filter(
                    must=[
                        FieldCondition(
                            key="document_id",
                            match=MatchValue(value=document_id),
                        )
                    ]
                ),
            )

            logger.info(f"Deleted {count} chunks for document '{document_id}'")
            return count
        except Exception as e:
            logger.error(f"Error deleting document: {e}")
            return 0

    def list_documents(self) -> List[str]:
        """
        Get list of all document IDs in the collection.

        Returns:
            List of unique document IDs
        """
        try:
            results, _ = self._qdrant.scroll(
                collection_name=self.config.vector_store.collection_name,
                limit=10000,
                with_payload=["document_id"],
                with_vectors=False,
            )

            document_ids = set()
            for point in results:
                doc_id = point.payload.get("document_id")
                if doc_id:
                    document_ids.add(doc_id)

            return sorted(document_ids)
        except Exception as e:
            logger.error(f"Error listing documents: {e}")
            return []


# =============================================================================
# Convenience Functions
# =============================================================================

def create_client(
    ollama_url: str = "http://localhost:11434",
    qdrant_host: str = "localhost",
    qdrant_port: int = 6333,
    collection_name: str = "meridian_documents",
) -> EmbeddingClient:
    """Create an embedding client with custom settings."""
    config = EmbeddingClientConfig(
        embedding=EmbeddingConfig(ollama_url=ollama_url),
        vector_store=VectorStoreConfig(
            host=qdrant_host,
            port=qdrant_port,
            collection_name=collection_name,
        ),
    )
    return EmbeddingClient(config)
