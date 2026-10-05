# Meridian — Local Mac Setup

This documents the local development setup for running Meridian on macOS (Apple Silicon).

## Architecture

```
Local (Mac)                          Remote (API)
───────────────                      ────────────
Docling service  (MPS, port 8001)    OpenRouter VLM API
Ollama embeddings (port 11434)         └─ qwen/qwen3-vl-235b-a22b-instruct
Redis            (Docker, port 6399)
Qdrant           (Docker, port 6363)
Celery worker    (CPU)
```

## Prerequisites

- Python 3.12 (`/opt/homebrew/bin/python3.12`)
- Docker Desktop
- Ollama (native install via `brew install ollama`)

## Virtual Environment

Created with Python 3.12 (3.14 is too new for PyTorch/docling):

```bash
/opt/homebrew/bin/python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[gpu]"
```

## Docker Containers (Isolated)

Meridian runs its own Redis and Qdrant on offset ports (6399, 6363) with named volumes, so it will not collide with any other Redis or Qdrant you already have running.

```bash
# Start
docker compose -f docker-compose.local.yml up -d

# Stop (keeps data)
docker compose -f docker-compose.local.yml down

# Delete everything (wipes data)
docker compose -f docker-compose.local.yml down -v
```

| Container | Image | Host Port | Purpose |
|-----------|-------|-----------|---------|
| `meridian-redis` | `redis:7-alpine` | **6399** | Task broker + state |
| `meridian-qdrant` | `qdrant/qdrant:latest` | **6363** (HTTP), 6364 (gRPC) | Vector store |

Data volumes: `meridian_redis_data`, `meridian_qdrant_data`

## Ollama (Embeddings)

Ollama runs natively (not in Docker). The embedding model is already pulled:

```bash
ollama serve              # if not already running
ollama pull qwen3-embedding:4b-q8_0
```

- URL: `http://localhost:11434`
- Model: `qwen3-embedding:4b-q8_0` (1024 dimensions)

## VLM (Vision-Language Model)

Runs via OpenRouter API — no local GPU needed.

- URL: `https://openrouter.ai/api/v1/chat/completions`
- Model: `qwen/qwen3-vl-235b-a22b-instruct`
- API key is in `.env` (`VLLM_API_KEY`)

## Configuration

All config lives in `.env` (gitignored). Created from `.env.example` with these Mac-specific changes:

| Setting | Default (H100) | Mac Local |
|---------|----------------|-----------|
| `DOCLING_ACCELERATOR_DEVICE` | `auto` (CUDA) | `mps` |
| `DOCLING_INSTANCES` | 8 instances (8001-8008) | 1 instance (8001) |
| `VLLM_API_URL` | `localhost:8000` | OpenRouter |
| `REDIS_URL` | `localhost:6379` | `localhost:6399` |
| `QDRANT_PORT` | `6333` | `6363` |
| Batch sizes | 4 | 2 |
| Worker concurrency | 3 | 2 |

## Code Changes Made

1. **Added `python-dotenv`** — `pyproject.toml` dependency + `load_dotenv()` call in `meridian/config.py` so `.env` is loaded automatically.

## Running the Pipeline

```bash
# 1. Activate venv
source .venv/bin/activate

# 2. Start Docker infrastructure
docker compose -f docker-compose.local.yml up -d

# 3. Start Docling service (terminal 1)
python -m uvicorn meridian.services.docling_service.main:app --host 0.0.0.0 --port 8001

# 4. Start Celery worker (terminal 2)
celery -A meridian.workers.celery_app worker --loglevel=info --concurrency=2

# 5. Process a document
meridian parse /path/to/document.pdf
```

## Verify Services

```bash
# Redis
docker exec meridian-redis redis-cli ping

# Qdrant
curl -s http://localhost:6363/collections

# Ollama
curl -s http://localhost:11434/api/tags

# Docling (after starting)
curl -s http://localhost:8001/health
```
