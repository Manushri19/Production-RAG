# Meridian

Large-scale GPU-accelerated document processing pipeline. Extracts tables, figures and formulas from PDFs, reads them with a vision-language model, and stores searchable embeddings in a vector database. Measured at **up to 118 pages/minute on a single H200**; see [Performance](#performance) for the full numbers and how to reproduce them.

## Architecture Overview

```
                          +------------------+
                          |   CLI / Submit   |
                          +--------+---------+
                                   |
                          +--------v---------+
                          |    Redis Queue    |
                          |  (Celery Broker)  |
                          +--------+---------+
                                   |
                    +--------------+--------------+
                    |                              |
           +-------v--------+            +--------v-------+
           | Celery Workers  |            | Celery Workers  |
           | (Stateless CPU) |            | (Stateless CPU) |
           +---+----+---+---+            +---+----+---+---+
               |    |   |                    |    |   |
      +--------+    |   +--------+  +--------+    |   +--------+
      |             |            |  |              |            |
+-----v-----+ +----v----+ +-----v--v--+ +----v----+ +----------v+
|  Docling   | |  vLLM   | |  Ollama   | |  Qdrant | |  Image    |
| (8 inst.)  | | (VLM)   | | (Embed)   | | (Vector)| | Service   |
| GPU:~10GB  | | GPU:55% | | GPU/CPU   | |  CPU    | |   CPU     |
+------------+ +---------+ +-----------+ +---------+ +-----------+
```

**GPU Services** run once, loaded with models, shared via HTTP:
- **Docling** (ports 8001-8008) -- PDF parsing with layout detection, table structure, formula/figure extraction
- **vLLM** (port 8000) -- Vision-language model for understanding tables, figures, and formulas
- **Ollama** (port 11434) -- Embedding generation for vector search

**Infrastructure:**
- **Redis** (port 6379) -- Celery broker/backend + batch state management
- **Qdrant** (port 6333) -- Vector storage for document embeddings

**Stateless Workers** call services via HTTP and run the 6-step pipeline:
1. **Docling** -- Parse PDF, detect layout, extract table/figure/formula regions
2. **Annotate** -- Draw numbered boxes on formula pages for VLM identification
3. **VLM** -- Send all tables, figures, and annotated formula pages concurrently to vLLM
4. **Chunk** -- Build ordered document chunks with injected VLM descriptions
5. **Embed** -- Generate vector embeddings via Ollama
6. **Store** -- Persist embeddings and metadata in Qdrant

## Key Features

- **8-instance Docling load balancing** with round-robin distribution via Redis atomic counter
- **Instance failover** -- automatically routes to next instance on crash (handles RemoteProtocolError, ConnectionResetError)
- **Watchdog auto-restart** -- monitors all instances every 30 seconds, restarts crashed ones
- **Async VLM processing** -- all table/figure/formula VLM requests sent concurrently
- **Async embedding pipeline** -- non-blocking embedding generation and storage
- **Batch orchestration** -- submit/pause/resume/retry with full Redis state tracking
- **GPU memory management** -- cleanup after each document to prevent accumulation
- **Adaptive image scaling** -- adjusts resolution based on file size (2.0x for small, 0.75x for large)
- **Content block filtering** -- tuned thresholds eliminate noise while preserving content
- **Up to 118 pages/minute** on a single H200, sustained across multi-document batches
- **Pluggable VLM backend** -- local vLLM by default, or AWS Bedrock (`VLM_BACKEND=bedrock`) when you have no GPU to spare for a vision model

## Quick Start

### 1. Install

```bash
# Clone the repository
git clone https://github.com/Rajsuthan/meridian.git
cd meridian

# Install base package (for CLI and workers)
pip install -e .

# Or install with GPU support (for running Docling service locally)
pip install -e ".[gpu]"
```

### 2. Configure

```bash
# Copy the example environment file
cp .env.example .env

# Edit with your settings (Redis URL, model paths, etc.)
```

See [Configuration Reference](docs/configuration.md) for all available settings.

### 3. Start Services

```bash
# Start everything (Redis, Docling instances, vLLM, workers, watchdog)
make start

# Or start individually:
./scripts/start_docling_instances.sh start
./scripts/start_vllm.sh
./scripts/start_worker.sh start
./scripts/docling_watchdog.sh
```

### 4. Submit a Batch

```bash
# Submit a directory of PDFs for processing
meridian submit /path/to/pdfs --collection my_documents

# Monitor progress
meridian status

# Check specific batch
meridian status batch_20250101_120000
```

## Performance

All figures below are measured, not projected. Each run is a full pipeline pass --
Docling layout detection, VLM reading of every table/figure/formula, chunking,
embedding and Qdrant storage -- on a single **NVIDIA H200 (143 GB)** RunPod instance.

| Batch | Docs | Layout model | Docling instances | vLLM util / max_seqs | Wall clock | Throughput |
|---|---|---|---|---|---|---|
| Academic papers (~30 pg avg) | 20 | heron | 12 | 0.40 / 256 | 4m 53s | **118.7 pg/min** |
| NASA NTRS Apollo reports, v1 | 25 | heron | 12 | 0.40 / 256 | 25m 45s | **82.9 pg/min** |
| NASA NTRS Apollo reports, v2 | 25 | egret-large | 5 | 0.65 / 1024 | 27m 40s | **77.2 pg/min** |

Reading these honestly:

- **Corpus matters more than tuning.** The Apollo reports are scan-heavy with far more
  tables and figures per page than the academic set, so every page costs more VLM work.
  Compare configurations within a corpus, not across.
- **v2 is the better production config despite the lower headline.** It completed with
  zero retries where v1 needed 5. A single pathological document -- 355 pages, 677 VLM
  requests -- took 20 minutes on its own and dominated the wall clock. The other 24 docs
  sustained ~95-100 pg/min.
- **Throughput scales with documents in flight,** not with any single document. One
  15-page PDF runs at ~10.6 pg/min because the embedding step is serial per request.
  Meridian's design assumes many documents in different pipeline stages at once via
  Celery, which is where the numbers above come from.

Resource profile at these settings: Docling ~1.2 GB VRAM per instance, vLLM configured
to 40-65% of the remainder, and cleanup after every document to stop GPU memory
accumulating across a long batch.

### Reproducing these numbers

```bash
python gpu-monitoring/fetch_ntrs.py --count 25 --out ./pdfs   # public NASA corpus
meridian parse ./pdfs --workers 12
```

[SETUP_NOTES.md](SETUP_NOTES.md) is the full runbook these runs came from, including
per-VRAM-tier tuning tables, the bugs found along the way, and what did not work.

### A note on a different pipeline

Meridian also ships a text-only path (`gpu-monitoring/parse_pypdfium2.py`) that skips
Docling and the VLM entirely and pulls raw text with pypdfium2. It processed 1,000 NASA
documents (78,452 pages) in under a minute on 32 CPU workers. That is a genuinely
different trade -- no tables, no figures, no formulas, no layout understanding -- and
the two sets of numbers should not be compared. It exists for when you need bulk text
and nothing else.

## Configuration

All settings are driven by environment variables with sensible defaults. Key settings:

| Variable | Default | Purpose |
|----------|---------|---------|
| `REDIS_URL` | `redis://localhost:6379/0` | Redis connection |
| `DOCLING_INSTANCES` | `http://localhost:8001,...,8008` | Docling instance URLs |
| `VLLM_API_URL` | `http://localhost:8000/v1/chat/completions` | vLLM endpoint |
| `VLLM_MODEL` | `Qwen/Qwen3-VL-8B-Instruct` | Vision-language model |
| `OLLAMA_URL` | `http://localhost:11434` | Ollama embedding service |
| `QDRANT_HOST` | `localhost` | Qdrant vector store |
| `VLM_BACKEND` | `vllm` | VLM backend: `vllm` (local GPU) or `bedrock` (AWS API) |
| `BEDROCK_MODEL_ID` | `us.amazon.nova-2-lite-v1:0` | Model used when `VLM_BACKEND=bedrock` |

See [docs/configuration.md](docs/configuration.md) for the complete reference.

## CLI Usage

Meridian provides a command-line interface for batch document processing:

```bash
# Submit a batch of PDFs
meridian submit <input_dir> [--collection NAME] [--batch-id ID]
meridian submit /data/pdfs --collection research_papers

# Optional flags for submit
meridian submit /data/pdfs --no-tables --no-figures --no-formulas --no-embeddings

# Check status (all batches or specific batch)
meridian status
meridian status batch_20250101_120000

# Resume an interrupted batch
meridian resume <batch_id>
meridian resume <batch_id> --force  # Force recover stuck docs

# Retry only failed documents
meridian retry <batch_id>

# Pause processing (running tasks complete, new ones held)
meridian pause <batch_id>

# View failed documents with error details
meridian failed <batch_id>
meridian failed <batch_id> --limit 50

# Clear all state for a batch
meridian clear <batch_id>
meridian clear <batch_id> --force  # Skip confirmation

# List all batches
meridian list
```

## Running Without a Local Vision Model

The VLM step -- reading tables, formulas and figures out of cropped page regions --
normally runs against a vLLM server you host yourself, which needs enough VRAM for a
vision model on top of Docling. If you do not have that, point the VLM step at AWS
Bedrock instead:

```bash
pip install -e ".[bedrock]"
export VLM_BACKEND=bedrock          # default is 'vllm'
export BEDROCK_REGION=us-east-1
export BEDROCK_MODEL_ID=us.amazon.nova-2-lite-v1:0
```

Both backends implement the same interface and return the same schema, so nothing else
in the pipeline changes. Credentials come from the standard boto3 chain -- environment,
shared config, or an instance role -- and are never read from Meridian's own config.

You trade GPU capacity for per-token cost and API latency, and you are subject to your
account's Bedrock throughput quota (`BEDROCK_MAX_CONCURRENCY` defaults to a conservative
5). To find out what that trade costs on your own documents before committing to it:

```bash
python examples/bedrock_cost_benchmark.py ./pdfs --project-to 100000
```

Docling still runs locally. It is much faster on a GPU, but
`DOCLING_ACCELERATOR_DEVICE=cpu` works -- that combination is the lowest-hardware way to
run the full pipeline.

## Docker Deployment

For containerized deployment:

```bash
# Start all services with Docker Compose
docker compose up -d

# Check logs
docker compose logs -f worker

# Stop services
docker compose down
```

The Docker Compose file includes Redis, Qdrant, vLLM (with GPU), Ollama, and Celery workers. See [docs/deployment.md](docs/deployment.md) for details.

## Project Structure

```
meridian/
├── meridian/                    # Python package
│   ├── config.py                # Central configuration (env-var driven)
│   ├── utils/
│   │   └── bbox.py              # Bounding box utilities
│   ├── services/
│   │   ├── docling_service/     # GPU service: PDF processing (FastAPI)
│   │   │   ├── main.py          # FastAPI app with /process and /health endpoints
│   │   │   ├── processor.py     # Docling pipeline configuration
│   │   │   ├── extractors.py    # Table, figure, formula extraction
│   │   │   └── models.py        # Request/response schemas
│   │   └── image_service/       # PDF page image rendering service
│   │       └── main.py
│   ├── clients/
│   │   ├── docling_client.py    # HTTP client with load balancing + failover
│   │   ├── vlm_backend.py       # Selects the VLM backend from VLM_BACKEND
│   │   ├── vlm_client.py        # Sync VLM client
│   │   ├── vlm_client_async.py  # Async VLM client (concurrent requests, vLLM)
│   │   ├── vlm_client_bedrock.py # Async VLM client (AWS Bedrock Converse)
│   │   ├── embedding_client.py  # Sync embedding client
│   │   └── embedding_client_async.py  # Async embedding + Qdrant storage
│   ├── workers/
│   │   ├── document_worker.py   # 6-step processing pipeline
│   │   ├── box_annotator.py     # Formula page annotation (PIL)
│   │   ├── chunker.py           # Document chunking with VLM injection
│   │   ├── celery_app.py        # Celery configuration
│   │   └── tasks.py             # Celery task definitions
│   └── orchestrator/
│       ├── batch_orchestrator.py # Batch submit/resume/retry
│       ├── state_manager.py     # Redis state tracking
│       ├── progress.py          # Progress display formatting
│       └── cli.py               # CLI entry point
├── scripts/
│   ├── start_all.sh             # Start all services
│   ├── stop_all.sh              # Stop all services
│   ├── start_docling_instances.sh  # Manage 8 Docling instances
│   ├── start_vllm.sh            # Start vLLM Docker container
│   ├── start_worker.sh          # Start Celery worker
│   └── docling_watchdog.sh      # Auto-restart crashed instances
├── gpu-monitoring/
│   ├── monitor.py               # Live GPU + service health dashboard
│   ├── fetch_ntrs.py            # Harvest PDFs from the public NASA NTRS archive
│   ├── parse_pypdfium2.py       # Text-only bulk parser (no Docling, no VLM)
│   └── raw_text_ab.py           # A/B raw-text extractors vs Docling output
├── docker/
│   └── worker.Dockerfile        # Worker container image
├── docker-compose.yml           # Full stack deployment
├── Makefile                     # Common operations
├── pyproject.toml               # Package configuration
├── tests/                       # Test suite
├── SETUP_NOTES.md               # Full self-hosting runbook + tuning tables
├── docs/                        # Documentation
│   ├── architecture.md
│   ├── deployment.md
│   └── configuration.md
└── examples/
    ├── process_single_doc.py    # Single document processing
    ├── batch_process.py         # Batch processing with progress
    └── bedrock_cost_benchmark.py # Measure Bedrock VLM cost/throughput
```

## Make Targets

```bash
make help              # Show all available targets
make install           # Install base package
make install-gpu       # Install with GPU support
make install-worker    # Install for worker nodes
make install-dev       # Install with dev tools
make start             # Start all services
make stop              # Stop all services
make status            # Check service status
make test              # Run tests
make lint              # Run linters
make docker-up         # Start Docker services
make docker-down       # Stop Docker services
```

## Documentation

- [Architecture](docs/architecture.md) -- System design, pipeline details, load balancing, failover
- [Deployment](docs/deployment.md) -- Prerequisites, installation, GPU requirements, scaling
- [Configuration](docs/configuration.md) -- Complete environment variable reference

## License

Apache License 2.0. See [LICENSE](LICENSE) for details.
