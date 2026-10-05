# Configuration Reference

All configuration is driven by environment variables with sensible defaults. Settings are defined in `meridian/config.py` and can be set via a `.env` file in the project root or exported in the shell.

## Redis

| Variable | Default | Description |
|----------|---------|-------------|
| `REDIS_URL` | `redis://localhost:6379/0` | Redis connection URL. Used for Celery broker/backend, batch state management, and load balancing counter. |

## Docling Service

### Server Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `DOCLING_SERVICE_HOST` | `0.0.0.0` | Host to bind the Docling FastAPI service. |
| `DOCLING_SERVICE_PORT` | `8001` | Port for the Docling FastAPI service. Each instance uses a unique port. |
| `DOCLING_MAX_CONCURRENT` | `10` | Maximum concurrent requests per instance. |
| `DOCLING_REQUEST_TIMEOUT` | `300` | Request timeout in seconds (5 minutes). |
| `DOCLING_TEMP_DIR` | `/tmp/docling_service` | Temporary directory for uploaded PDFs. Cleaned after each request. |

### Pipeline Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `DOCLING_OCR_ENABLED` | `false` | Enable OCR processing. Disable to save ~11GB GPU memory. VLMs handle visual content for most use cases. |
| `DOCLING_OCR_ENGINE` | `easyocr` | OCR engine to use. Only applies when `DOCLING_OCR_ENABLED=true`. |
| `DOCLING_OCR_BITMAP_THRESHOLD` | `0.74` | Bitmap detection threshold for OCR. Only applies when `DOCLING_OCR_ENABLED=true`. |
| `DOCLING_TABLE_STRUCTURE` | `true` | Enable table structure detection (row/column/cell boundaries). |
| `DOCLING_TABLE_MODE` | `accurate` | Table structure mode: `accurate` (slower, better quality) or `fast`. |
| `DOCLING_TABLE_CELL_MATCHING` | `true` | Enable table cell content matching. |
| `DOCLING_FORMULA_ENRICHMENT` | `false` | Enable Docling's built-in formula-to-LaTeX conversion (CodeFormulaModel). Disabled by default because it is extremely slow on scanned documents (46s/formula vs 0.7s on digital PDFs). Formula DETECTION still works via the layout model regardless of this setting. Meridian uses its own VLM-based formula processing instead. |
| `DOCLING_PICTURE_EXTRACTION` | `true` | Enable picture/figure region extraction. |

### GPU / Accelerator Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `DOCLING_ACCELERATOR_DEVICE` | `auto` | Device selection: `auto`, `cuda`, `mps`, or `cpu`. `auto` selects the best available. |
| `DOCLING_ACCELERATOR_THREADS` | `4` | Number of accelerator threads. |

### Batch Processing and Parallelism

These control Docling's internal parallelism for multi-document and multi-page processing within a single instance.

| Variable | Default | Description |
|----------|---------|-------------|
| `DOCLING_DOC_BATCH_SIZE` | `4` | Number of documents per batch for Docling's `convert_all()`. |
| `DOCLING_DOC_BATCH_CONCURRENCY` | `4` | Number of parallel threads for document processing within Docling. |
| `DOCLING_PAGE_BATCH_SIZE` | `4` | Page batch size for within-document parallelism. |

## Image Extraction Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `DOCLING_IMAGE_SCALE` | `2.0` | Scale factor for extracted images. 2.0 = 144 DPI. Workers may auto-adjust based on file size (2.0 for <30MB, 1.5 for <100MB, 1.0 for <200MB, 0.75 for larger). |
| `DOCLING_TABLE_PADDING` | `70` | Padding in points around table bounding boxes when cropping table images. |
| `DOCLING_FORMULA_PADDING` | `20` | Padding in points around formula bounding boxes. |
| `DOCLING_PICTURE_PADDING` | `10` | Padding in points around picture bounding boxes. |
| `DOCLING_PICTURE_CONTEXT_CHARS` | `300` | Number of characters of surrounding text to include as context for figure descriptions. |
| `DOCLING_FORMULA_MODE` | `bbox` | Formula extraction mode: `bbox` (crop formula bounding boxes) or `page` (extract full page for pages with formulas, better for complex layouts). |
| `DOCLING_FORMULA_VERTICAL_EXPANSION` | `0.75` | Vertical expansion for formula bounding box as a fraction of height. Adds padding above and below the detected formula region. |

## Content Block Filtering

**WARNING:** These thresholds are carefully tuned and are used by BOTH the box annotator and the chunker. They MUST remain identical in both components to ensure correct formula placement. Changing these values without understanding the coupling between box annotation and chunking will cause formulas to be injected at wrong positions.

| Variable | Default | Description |
|----------|---------|-------------|
| `DOCLING_MIN_WIDTH_PT` | `20` | Minimum block width in points (~0.28 inches). Blocks narrower than this AND shorter than min height are filtered. |
| `DOCLING_MIN_HEIGHT_PT` | `10` | Minimum block height in points (~0.14 inches). Blocks shorter than this AND narrower than min width are filtered. |
| `DOCLING_MIN_AREA_PT2` | `500` | Minimum block area in square points (~0.1 square inches). Blocks smaller than this are always filtered, regardless of width/height. Catches small reference numbers and dots. |
| `DOCLING_CONTAINMENT_THRESHOLD` | `0.8` | Nested box containment threshold. If a box is 80%+ contained within a larger box, the smaller box is removed. Eliminates duplicate boxes from nested layout elements (e.g., inline math inside paragraphs). |

**Filtering logic:** `remove_if (width < MIN_WIDTH AND height < MIN_HEIGHT) OR (area < MIN_AREA)`

This means wide single-line text (large width, small height) is kept, while small square-ish elements (small area) are filtered.

## Table Detection

| Variable | Default | Description |
|----------|---------|-------------|
| `DOCLING_DETECT_MISSED_TABLES` | `true` | Enable detection of tables missed by the layout model using heuristic methods. |
| `DOCLING_DETECT_CONTINUATION` | `true` | Enable detection of table continuation pages (gap pages between detected tables). |

## Performance and Profiling

| Variable | Default | Description |
|----------|---------|-------------|
| `DOCLING_PROFILE_TIMINGS` | `false` | Enable detailed pipeline timing profiling. |
| `DOCLING_PDF_BACKEND` | `v2` | PDF backend: `pypdfium2` (fastest), `v2` (default balanced), `v4` (best semantic extraction). |

## Logging

| Variable | Default | Description |
|----------|---------|-------------|
| `DOCLING_LOG_LEVEL` | `INFO` | Log level for the Docling service. Accepts: `DEBUG`, `INFO`, `WARNING`, `ERROR`. |

## VLM Service

| Variable | Default | Description |
|----------|---------|-------------|
| `VLLM_API_URL` | `http://localhost:8000/v1/chat/completions` | vLLM OpenAI-compatible API endpoint. |
| `VLLM_MODEL` | `Qwen/Qwen3-VL-8B-Instruct` | Vision-language model name. Must match the model loaded in vLLM. |

### Backend Selection

The VLM step can run against a locally hosted vLLM server or the AWS Bedrock Converse
API. Both backends return the same schema, so no other setting changes when you switch.

| Variable | Default | Purpose |
|----------|---------|---------|
| `VLM_BACKEND` | `vllm` | `vllm` for a local GPU server, `bedrock` for the AWS API. |

`VLM_BACKEND` is read once when `meridian.workers.document_worker` is imported, so set
it before starting workers rather than mid-run.

### Bedrock Backend

Only read when `VLM_BACKEND=bedrock`. Requires `pip install 'meridian[bedrock]'`.
AWS credentials come from the standard boto3 chain (environment variables, shared
config, or an instance role) and are never read from Meridian's own configuration.

| Variable | Default | Purpose |
|----------|---------|---------|
| `BEDROCK_REGION` | `us-east-1` | AWS region for the Bedrock runtime client. |
| `BEDROCK_MODEL_ID` | `us.amazon.nova-2-lite-v1:0` | Bedrock model identifier. Must support image input. |
| `BEDROCK_MAX_CONCURRENCY` | `5` | Cap on in-flight Converse calls per worker. Raise only as far as your account's TPS quota allows. |
| `BEDROCK_MAX_IMAGE_DIM` | `1536` | Longest edge, in pixels, that region crops are downscaled to before upload. Bounds token cost. |
| `BEDROCK_MAX_RETRIES` | `5` | Retry attempts, using botocore adaptive mode. |
| `BEDROCK_RETRY_DELAY` | `2.0` | Base delay in seconds between retries. |
| `BEDROCK_TEMPERATURE` | `0.2` | Sampling temperature for VLM calls. |
| `BEDROCK_MAX_TOKENS_TABLE` | `4000` | Output token cap for table reads. |
| `BEDROCK_MAX_TOKENS_FORMULA` | `4000` | Output token cap for formula reads. |
| `BEDROCK_MAX_TOKENS_PICTURE` | `1500` | Output token cap for figure descriptions. |

Use `examples/bedrock_cost_benchmark.py` to measure what this costs on your own
documents before committing to it.

## Embedding and Vector Store

### Ollama (Embeddings)

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_URL` | `http://localhost:11434` | Ollama API endpoint for embedding generation. |
| `EMBEDDING_MODEL` | `qwen3-embedding:4b-q8_0` | Embedding model name in Ollama. The `q8_0` suffix indicates 8-bit quantization. |
| `EMBEDDING_DIMENSIONS` | `1024` | Embedding vector dimensions. The model supports 32-4096; 1024 is the recommended tradeoff between quality and storage. |

### Qdrant (Vector Store)

| Variable | Default | Description |
|----------|---------|-------------|
| `QDRANT_HOST` | `localhost` | Qdrant server hostname. |
| `QDRANT_PORT` | `6333` | Qdrant HTTP REST API port. |
| `QDRANT_COLLECTION` | `meridian_documents` | Default Qdrant collection name. Can be overridden per batch via `--collection` flag. |

## Load Balancing

| Variable | Default | Description |
|----------|---------|-------------|
| `DOCLING_INSTANCES` | `http://localhost:8001,...,http://localhost:8008` | Comma-separated list of Docling instance URLs. Workers distribute requests across these using round-robin via a Redis atomic counter. |

The default value configures 8 instances on ports 8001-8008. Adjust to match your actual Docling instance deployment. If running fewer instances (e.g., on a smaller GPU), reduce this list accordingly.

## Image Service

| Variable | Default | Description |
|----------|---------|-------------|
| `IMAGE_SERVICE_PORT` | `9847` | Port for the PDF page image rendering service. |
| `PDF_DIR` | `/mnt/pdfs` | Directory where source PDFs are stored. Used by the image service for page rendering. |

## Example .env File

```bash
# Redis
REDIS_URL=redis://localhost:6379/0

# Docling service
DOCLING_SERVICE_HOST=0.0.0.0
DOCLING_SERVICE_PORT=8001
DOCLING_MAX_CONCURRENT=10
DOCLING_OCR_ENABLED=false
DOCLING_TABLE_STRUCTURE=true
DOCLING_PICTURE_EXTRACTION=true
DOCLING_ACCELERATOR_DEVICE=auto

# Load balancing (8 instances)
DOCLING_INSTANCES=http://localhost:8001,http://localhost:8002,http://localhost:8003,http://localhost:8004,http://localhost:8005,http://localhost:8006,http://localhost:8007,http://localhost:8008

# VLM (VLM_BACKEND=bedrock swaps this for the Bedrock settings above)
VLM_BACKEND=vllm
VLLM_API_URL=http://localhost:8000/v1/chat/completions
VLLM_MODEL=Qwen/Qwen3-VL-8B-Instruct

# Embeddings
OLLAMA_URL=http://localhost:11434
EMBEDDING_MODEL=qwen3-embedding:4b-q8_0
EMBEDDING_DIMENSIONS=1024

# Vector store
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_COLLECTION=meridian_documents

# Image service
IMAGE_SERVICE_PORT=9847
PDF_DIR=/mnt/pdfs

# Logging
DOCLING_LOG_LEVEL=INFO
```
