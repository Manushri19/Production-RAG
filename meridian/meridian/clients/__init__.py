"""
Clients package for Meridian.

HTTP clients for calling GPU services:
- DoclingClient: PDF processing with round-robin load balancing
- VLMClient: Vision-language model processing (tables, formulas, figures)
- EmbeddingClient: Text embeddings + Qdrant vector store
"""

from meridian.clients.docling_client import (
    DoclingClient,
    DoclingClientConfig,
    DoclingClientError,
    DoclingServiceUnavailable,
    DoclingProcessingError,
)

from meridian.clients.vlm_client import (
    VLMClient,
    VLMClientConfig,
    VLMClientError,
    VLMServiceUnavailable,
    VLMProcessingError,
)

from meridian.clients.embedding_client import (
    EmbeddingClient,
    EmbeddingClientConfig,
    EmbeddingConfig,
    VectorStoreConfig,
    EmbeddingClientError,
    OllamaServiceUnavailable,
    QdrantServiceUnavailable,
)

__all__ = [
    "DoclingClient", "DoclingClientConfig", "DoclingClientError",
    "DoclingServiceUnavailable", "DoclingProcessingError",
    "VLMClient", "VLMClientConfig", "VLMClientError",
    "VLMServiceUnavailable", "VLMProcessingError",
    "EmbeddingClient", "EmbeddingClientConfig", "EmbeddingConfig",
    "VectorStoreConfig", "EmbeddingClientError",
    "OllamaServiceUnavailable", "QdrantServiceUnavailable",
]
