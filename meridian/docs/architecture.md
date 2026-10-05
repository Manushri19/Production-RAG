# Meridian Architecture

## System Overview

Meridian is a distributed document processing pipeline designed to process 10,000+ PDFs on GPU infrastructure. The architecture separates GPU-intensive model inference from stateless CPU processing, allowing each component to scale independently.

### Design Principles

1. **GPU services load once, serve many** -- Docling, vLLM, and Ollama load models into GPU memory at startup and serve requests over HTTP. No per-request model loading.
2. **Stateless workers** -- Celery workers have zero GPU dependencies. They call services via HTTP, making them trivially scalable.
3. **Redis as coordination backbone** -- Celery broker, batch state machine, load balancing counter, and progress tracking all use Redis.
4. **Fail gracefully** -- Instance failover, watchdog restart, batch resume, and failed document retry at every level.

### Component Map

```
+----------------------------------------------------------------------+
|                         Control Plane                                  |
|                                                                        |
|  +------------------+    +-------------------+    +------------------+ |
|  |   CLI (meridian) |    | BatchOrchestrator |    |  StateManager    | |
|  | submit/status/   |--->| submit_batch()    |--->| Redis sets:      | |
|  | resume/retry     |    | resume()          |    | pending          | |
|  +------------------+    | retry_failed()    |    | processing       | |
|                          +-------------------+    | complete/failed  | |
|                                                   +------------------+ |
+----------------------------------------------------------------------+
                                   |
                           Celery task queue
                                   |
+----------------------------------------------------------------------+
|                          Data Plane                                    |
|                                                                        |
|  +------------------------------------------------------------------+ |
|  |                    Document Worker                                | |
|  |                                                                   | |
|  |  1. DoclingClient.process_pdf()  ---HTTP-->  Docling Service      | |
|  |  2. annotate_formula_pages()     (local PIL)                      | |
|  |  3. process_vlm_tasks_async()    ---HTTP-->  vLLM Service         | |
|  |  4. UnifiedChunker.build_chunks() (local CPU)                     | |
|  |  5. AsyncEmbeddingClient()       ---HTTP-->  Ollama               | |
|  |  6. store_document_chunks()      ---HTTP-->  Qdrant               | |
|  +------------------------------------------------------------------+ |
+----------------------------------------------------------------------+
                                   |
+----------------------------------------------------------------------+
|                        GPU Services                                   |
|                                                                        |
|  +-----------+  +-----------+  +-----------+       +--------+         |
|  | Docling   |  | Docling   |  | Docling   | x8    | vLLM   |         |
|  | :8001     |  | :8002     |  | :8003     | ...   | :8000  |         |
|  +-----------+  +-----------+  +-----------+       +--------+         |
|                                                                        |
|  +----------+  +---------+                                            |
|  | Ollama   |  | Qdrant  |                                            |
|  | :11434   |  | :6333   |                                            |
|  +----------+  +---------+                                            |
+----------------------------------------------------------------------+
```

## Processing Pipeline

Each document flows through 6 sequential steps. Steps 1, 3, 5, and 6 are HTTP calls to GPU services. Steps 2 and 4 are local CPU operations.

### Step 1: Docling (PDF Processing)

The Docling service converts a PDF into a structured document with layout detection.

**Input:** Raw PDF file (sent as multipart upload)

**Processing:**
- Layout analysis detects paragraphs, headings, tables, figures, formulas, page headers/footers
- Table structure extraction identifies rows, columns, and cell boundaries
- Figure regions are cropped as images
- Formula pages are identified and returned as full-page images
- All bounding boxes use BOTTOMLEFT coordinate origin

**Output:**
- DoclingDocument JSON (full document structure with text and bounding boxes)
- Table images (cropped regions with padding)
- Figure images (cropped with surrounding context text)
- Formula page images (full pages where formulas were detected)

**Key detail:** Formula pages return the ENTIRE page image, not cropped formula regions. This is because formulas often span complex layouts and the VLM needs full-page context to produce accurate LaTeX.

### Step 2: Annotate (Formula Box Numbering)

Local PIL operation that draws numbered boxes on formula page images.

**Input:** DoclingDocument + formula page images from Step 1

**Processing:**
- For each formula page, extract all content blocks (formulas, text, figures)
- Filter tiny blocks using content block thresholds (min width, height, area)
- Remove nested boxes (containment threshold: 80% overlap)
- Sort blocks top-to-bottom (`-bbox.t` for BOTTOMLEFT coordinates)
- Draw numbered red boxes on the page image

**Output:** Annotated page images with numbered boxes + block metadata

**CRITICAL COUPLING:** The box numbering in Step 2 MUST match the block ordering in Step 4 (chunking). Both use identical filtering thresholds from `config.py`. If these get out of sync, formulas will be injected at wrong positions in the final document.

### Step 3: VLM (Vision-Language Model)

Async HTTP calls to vLLM for understanding visual content.

**Input:** Table images, figure images, annotated formula pages

**Processing:**
- ALL VLM requests are sent concurrently using `asyncio` and `aiohttp`
- Tables: "Describe this table in markdown format"
- Figures: "Describe what this figure shows" (with surrounding text context)
- Formulas: "For each numbered box, provide the LaTeX representation"
- vLLM handles batching internally via `--max-num-seqs 128`

**Output:** Markdown descriptions for tables, text descriptions for figures, LaTeX for formulas

**Performance:** Concurrent dispatch is a key optimization. A document with 5 tables, 3 figures, and 2 formula pages sends all 10 requests simultaneously rather than sequentially.

### Step 4: Chunk (Document Assembly)

Local CPU operation that builds ordered document chunks with injected VLM content.

**Input:** DoclingDocument + VLM results from Step 3

**Processing:**
- Traverse document in reading order (top-to-bottom, respecting multi-column layout)
- Replace table regions with VLM-generated markdown
- Replace figure regions with VLM-generated descriptions
- Inject formula LaTeX at numbered box positions (matched to Step 2 numbering)
- Merge small text chunks (below 150 characters) with adjacent chunks
- Split oversized chunks at 2000 characters

**Output:** List of chunks, each with text content, page number, chunk type (text/table/figure/formula), and source metadata

### Step 5: Embed (Vector Generation)

Async HTTP calls to Ollama for embedding generation.

**Input:** Chunk text from Step 4

**Processing:**
- Batch chunks into groups of 32
- Send embedding requests to Ollama asynchronously
- Model: `qwen3-embedding:4b-q8_0` at 1024 dimensions

**Output:** Vector embeddings for each chunk

### Step 6: Store (Vector Persistence)

HTTP calls to Qdrant for vector storage.

**Input:** Embeddings + chunk metadata

**Processing:**
- Upsert points into Qdrant collection
- Each point contains: embedding vector, chunk text, document ID, page number, chunk type, source metadata
- Collection uses cosine similarity distance

**Output:** Stored and searchable document chunks

## Load Balancing

### Round-Robin via Redis Atomic Counter

The Docling client distributes requests across 8 instances using a Redis-backed atomic counter:

```python
# In DoclingClient._get_next_instance():
counter = redis.incr("meridian:docling:instance_counter")
instance_index = (counter - 1) % num_instances
```

This ensures true round-robin distribution even across multiple Celery worker processes. Each worker increments the shared counter atomically, so no two concurrent requests go to the same instance unless all others are busy.

**Why Redis counter instead of local round-robin:** Multiple Celery workers run as separate processes. A local counter in each worker would cause all workers to hit instance 0 first, then instance 1, creating hot spots. The shared Redis counter ensures even distribution globally.

### Instance Configuration

Default: 8 instances on ports 8001-8008. Configurable via `DOCLING_INSTANCES` environment variable as a comma-separated list of URLs.

## Instance Failover

When a Docling instance crashes (segfault, OOM, etc.), the client automatically fails over to the next instance.

### Failure Detection

The client catches these specific errors during `process_pdf()`:

| Error Type | Meaning |
|------------|---------|
| `httpx.ConnectError` | Instance is down, port not listening |
| `httpx.TimeoutException` | Instance overloaded or hung |
| `ConnectionResetError` | Instance crashed mid-processing |
| `httpx.RemoteProtocolError` | Server disconnected mid-request (segfault) |
| `httpx.HTTPStatusError` (5xx) | Server-side error |

### Failover Flow

```
Worker sends request to instance :8003
  |
  +--> :8003 crashes (RemoteProtocolError)
  |
  +--> Client catches error, logs warning
  |
  +--> Gets next instance via round-robin (:8004)
  |
  +--> Re-reads PDF file (file handle consumed by first attempt)
  |
  +--> Sends request to :8004
  |
  +--> Success (or continues to :8005, :8006, etc.)
```

The client tries up to N instances (where N = number of configured instances) before raising `DoclingServiceUnavailable`. Client errors (4xx) are NOT retried on other instances since they indicate a problem with the request itself.

## Watchdog Auto-Restart

The watchdog script (`scripts/docling_watchdog.sh`) runs as a background process and monitors all Docling instances.

### Monitoring Loop

Every 30 seconds:
1. Send HTTP GET to `/health` on each instance port
2. If an instance does not respond within 5 seconds:
   - Check if the process (PID) is still running
   - If running but unresponsive: kill -9 the process
   - Start a new instance on the same port
   - Wait up to 60 seconds for the new instance to become healthy
   - Log the restart event

### Recovery Timing

| Event | Time |
|-------|------|
| Detection | Up to 30 seconds (check interval) |
| Kill + restart | ~2 seconds |
| Model loading | 30-60 seconds |
| Total recovery | ~60-90 seconds |

During recovery, the failed instance is simply skipped by the failover mechanism. Documents that were being processed on the crashed instance will fail and can be retried via `meridian retry <batch_id>`.

## GPU Memory Management

### Docling Service

Each Docling instance loads layout detection models at startup (~1.2GB GPU memory per instance). To prevent memory accumulation across documents:

- Temporary files are cleaned up after each request
- The service runs with a configured temp directory that is cleared per-request
- With 8 instances, total Docling GPU usage is approximately 10GB

### vLLM Service

vLLM manages its own GPU memory pool:
- `--gpu-memory-utilization 0.55` -- uses 55% of remaining GPU memory after Docling
- `--max-num-seqs 128` -- handles up to 128 concurrent inference sequences
- `--enable-chunked-prefill` -- overlaps prefill with decode for better throughput
- `--enable-prefix-caching` -- caches common prompt prefixes (system prompts)
- FP8 quantization reduces model memory footprint

### Memory Layout (H100 80GB Example)

```
|<--- Docling: ~10GB --->|<--- vLLM: ~38GB (55% of 70GB) --->|<-- Free -->|
```

## Batch Orchestration

### State Machine

Each document in a batch transitions through states tracked in Redis sets:

```
pending --> processing --> complete
                |
                +--> failed
```

State transitions are atomic via Redis `SMOVE` operations:
- **Submit:** All doc IDs added to `pending` set
- **Task pickup:** `SMOVE` from `pending` to `processing`
- **Success:** `SMOVE` from `processing` to `complete` + store timing data
- **Failure:** `SMOVE` from `processing` to `failed` + store error message

### Redis Key Structure

```
meridian:batch:{batch_id}:info        # Hash: batch metadata (created_at, collection, options)
meridian:batch:{batch_id}:pending     # Set: doc IDs waiting to be processed
meridian:batch:{batch_id}:processing  # Set: doc IDs currently being processed
meridian:batch:{batch_id}:complete    # Set: doc IDs successfully processed
meridian:batch:{batch_id}:failed      # Set: doc IDs that failed
meridian:batch:{batch_id}:paths       # Hash: doc_id -> pdf_path mapping
meridian:batch:{batch_id}:errors      # Hash: doc_id -> error message
meridian:batch:{batch_id}:timings     # Hash: doc_id -> timing JSON
```

### Resume and Recovery

**Resume** (`meridian resume`): Re-queues all documents in the `pending` set as new Celery tasks. Optionally recovers "stale" documents stuck in `processing` for longer than a configurable timeout (default: 600 seconds).

**Retry** (`meridian retry`): Moves all documents from the `failed` set back to `pending`, then calls resume.

**Force recovery** (`meridian resume --force`): Uses 0-second timeout to immediately recover ALL documents in `processing` state. Useful after a full system restart where no workers are running.

### Progress Tracking

Progress is computed from Redis set cardinalities:

```python
total = len(pending) + len(processing) + len(complete) + len(failed)
percent = complete / total * 100
docs_per_minute = complete / elapsed_minutes
estimated_remaining = (pending + processing) / docs_per_minute * 60
```

## Content Block Filtering and Formula Placement

### The Coupling Problem

Formula placement in the final document depends on box numbering being identical between:
1. **Box annotator** (Step 2) -- draws numbered boxes on page images for VLM
2. **Chunker** (Step 4) -- injects VLM formula results at the correct positions

If the annotator numbers boxes [1, 2, 3, 4] but the chunker sees a different set of blocks (due to different filtering), formula "Box 2" from the VLM would be placed at the wrong location.

### Filtering Thresholds

Both components use the same thresholds from `config.py`:

| Threshold | Value | Purpose |
|-----------|-------|---------|
| `CONTENT_BLOCK_MIN_WIDTH_PT` | 20 pt | Filter blocks narrower than ~0.28 inches |
| `CONTENT_BLOCK_MIN_HEIGHT_PT` | 10 pt | Filter blocks shorter than ~0.14 inches |
| `CONTENT_BLOCK_MIN_AREA_PT2` | 500 pt^2 | Filter blocks smaller than ~0.1 square inches |
| `CONTENT_BLOCK_CONTAINMENT_THRESHOLD` | 0.8 | Remove blocks 80%+ contained within another |

**Filtering logic:** A block is removed if `(width < MIN_WIDTH AND height < MIN_HEIGHT) OR (area < MIN_AREA)`. This keeps wide single-line text (large width) while filtering small reference numbers and dots (small area).

### Coordinate System

All bounding boxes use **BOTTOMLEFT** origin (standard PDF coordinates where y=0 is at the bottom of the page). To sort blocks top-to-bottom for reading order, sort by `-bbox.t` (negative top coordinate).

## Async VLM Processing

### Concurrency Model

Step 3 sends ALL VLM requests concurrently using Python's `asyncio`:

```python
# Simplified flow in process_vlm_tasks_async():
async with aiohttp.ClientSession() as session:
    tasks = []
    for table in tables:
        tasks.append(send_vlm_request(session, table_image, table_prompt))
    for figure in figures:
        tasks.append(send_vlm_request(session, figure_image, figure_prompt))
    for formula_page in formulas:
        tasks.append(send_vlm_request(session, page_image, formula_prompt))

    results = await asyncio.gather(*tasks)
```

This is critical for throughput. A document with 10 visual elements processes in roughly the time of the single longest VLM inference, not 10x that time.

### vLLM Server-Side Batching

vLLM's continuous batching (`--max-num-seqs 128`) handles the concurrent requests efficiently on the GPU side. Combined with `--enable-chunked-prefill`, the server overlaps prompt processing with token generation across requests.

### Max Token Limits

Each content type has different output length requirements to stay within the model's context window (`--max-model-len 8192`):

| Content Type | Max Tokens |
|-------------|------------|
| Tables | 6000 |
| Formulas | 4000 |
| Figures | 2000 |
