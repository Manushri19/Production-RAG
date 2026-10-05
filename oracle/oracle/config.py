"""
Oracle Configuration

Single source of truth for all configuration.
All settings are configurable via environment variables with sensible defaults.
"""

import os
from dotenv import load_dotenv

load_dotenv()  # Load .env file from project root


# =============================================================================
# Oracle API Service
# =============================================================================

SERVICE_HOST = os.getenv("ORACLE_HOST", "0.0.0.0")
SERVICE_PORT = int(os.getenv("ORACLE_PORT", "8010"))


# =============================================================================
# Qdrant Vector Store
# =============================================================================

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "meridian_documents")

# Named vector spaces (must match Meridian's embedding_client constants)
DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"


# =============================================================================
# Ollama Embedding Service
# =============================================================================

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "qwen3-embedding:4b-q8_0")
EMBEDDING_DIMENSIONS = int(os.getenv("EMBEDDING_DIMENSIONS", "1024"))
EMBEDDING_TIMEOUT = float(os.getenv("EMBEDDING_TIMEOUT_SECONDS", "30.0"))


# =============================================================================
# Retrieval Pipeline
# =============================================================================

# Number of candidates from hybrid search (dense + sparse RRF fusion)
HYBRID_TOP_K = int(os.getenv("HYBRID_TOP_K", "30"))

# Final number of chunks sent to LLM after reranking
FINAL_TOP_K = int(os.getenv("FINAL_TOP_K", "5"))

# HNSW search parameter for dense vector recall quality
HNSW_EF = int(os.getenv("HNSW_EF", "128"))


# =============================================================================
# Reranker Sidecar
# =============================================================================

RERANKER_URL = os.getenv("RERANKER_URL", "http://localhost:8011")
RERANKER_TIMEOUT = float(os.getenv("RERANKER_TIMEOUT_SECONDS", "30.0"))


# =============================================================================
# LLM (vLLM / OpenAI-compatible)
# =============================================================================

VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
VLLM_API_KEY = os.getenv("VLLM_API_KEY", "EMPTY")
VLLM_MODEL = os.getenv("VLLM_MODEL", "Qwen/Qwen3-VL-8B-Instruct")
VLLM_MAX_TOKENS = int(os.getenv("VLLM_MAX_TOKENS", "1024"))
VLLM_TEMPERATURE = float(os.getenv("VLLM_TEMPERATURE", "0.1"))  # Low temp for factual precision
VLLM_TIMEOUT = float(os.getenv("VLLM_TIMEOUT_SECONDS", "120.0"))


# =============================================================================
# Logging
# =============================================================================

LOG_LEVEL = os.getenv("ORACLE_LOG_LEVEL", "INFO")
