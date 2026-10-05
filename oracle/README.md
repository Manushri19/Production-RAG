# Oracle — Retrieval & Generation Pipeline

> **Companion to [Meridian](../meridian/)** (offline ingestion).
> Oracle is the online, synchronous query layer for materials engineering R&D.

---

## Architecture

```
User Query
    │
    ▼
1. Embed (Ollama qwen3-embedding:4b-q8_0)
    │
    ▼
2. Hybrid Search — Qdrant RRF Fusion
   ├── Dense HNSW (semantic meaning)       ─┐
   └── Sparse BM25 (exact terms/acronyms)  ─┴─► top-30 fused candidates
    │
    ▼
3. Cross-Encoder Reranking (BAAI/bge-reranker-v2-m3)
   └── top-5 final chunks
    │
    ▼
4. Strict Generation (vLLM, streaming SSE)
   ├── Hard guardrails: refuse if context insufficient
   ├── Mandatory inline citations [N]
   └── Structured SOURCES block {doc_id, page_no, excerpt}
```

---

## Quick Start

### Prerequisites

Ensure Meridian's infrastructure is running (Qdrant, Ollama, vLLM):
```bash
cd ../meridian
docker compose up -d qdrant ollama vllm
```

### Start Oracle

```bash
cd retrieval-and-generation

# Copy env config
cp .env.example .env

# Option A — Docker (recommended)
docker compose up -d

# Option B — Local dev
pip install -e ".[dev]"
uvicorn oracle.api.app:app --host 0.0.0.0 --port 8010 --reload

# Start reranker sidecar separately:
pip install -r reranker/requirements.txt
python reranker/main.py
```

---

## API

### `POST /v1/query`

Stream a query through the full RAG pipeline.

**Request body:**
```json
{
  "query": "What is the tensile strength of Inconel 718 at 650°C?",
  "document_ids": ["alloy_datasheet_v3"],
  "top_k_candidates": 30,
  "top_k_final": 5
}
```

**Response:** `text/event-stream`
```
data: {"token": "The"}

data: {"token": " tensile"}

data: {"token": " strength"}

data: {"token": " of"}

...

data: {"sources": [{"ref": 1, "document_id": "alloy_datasheet_v3", "page_no": 12, "chunk_type": "text", "excerpt": "Inconel 718 tensile strength is 1375 MPa..."}]}

data: [DONE]
```

### `GET /v1/health`
```json
{"status": "ok", "service": "oracle-api", "version": "0.1.0"}
```

---

## Service Ports

| Service | Port | Purpose |
|---|---|---|
| Oracle API | 8010 | Main RAG query endpoint |
| Reranker sidecar | 8011 | BGE cross-encoder reranking |
| Qdrant | 6333 | Vector store (shared with Meridian) |
| Ollama | 11434 | Embedding model (shared with Meridian) |
| vLLM | 8000 | LLM generation (shared with Meridian) |

---

## Key Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| BM25 backend | Qdrant native sparse vectors (FastEmbed) | Zero extra services; single Qdrant round-trip for RRF |
| Fusion | Qdrant RRF `prefetch` API | Mathematically balanced; no manual score normalization |
| Reranker | BAAI/bge-reranker-v2-m3 | Best multilingual cross-encoder for technical/scientific text |
| LLM backend | Existing vLLM (OpenAI-compat) | Reuses Meridian infra; no extra GPU required |
| Streaming | SSE | Low perceived latency; progressive token delivery |
| Citations | Inline [N] + SOURCES JSON block | Every claim auditable against source doc + page |
| Hallucination guard | Hard system prompt refusal | LLM instructed to output exact refusal phrase if context is insufficient |

---

## Running Tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

---

## Meridian Integration Note

Oracle reads from the Qdrant collection created by Meridian.
To enable hybrid (BM25) search, **new documents must be ingested after**
adding `fastembed` to Meridian:

```bash
cd ../meridian
pip install fastembed
# Re-ingest documents to populate sparse vectors
meridian ingest --pdf-dir /your/pdfs
```

Existing points (dense-only) will gracefully degrade to dense-only retrieval
until re-indexed.
