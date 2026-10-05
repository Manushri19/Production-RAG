# Meridian — Self-Contained Setup Runbook

Portable instructions to bring Meridian up from scratch on a fresh Linux VM.

Three configurations covered, ordered by simplicity:

- **§0A Ultra-fast raw text (pypdfium2)** — *recommended for bulk*. No GPU,
  no Docling, no vLLM. ~800 docs/min, ~1.5 hours for 100k docs. Output is
  flat per-page text in JSON. Use when you'll embed downstream and don't
  need heading/layout structure preserved.
- **§0B Docling text-only** — Structured chunks with headings + reading
  order, no VLM enrichment. ~10× slower than pypdfium2 but produces richer
  JSON. Needs GPU.
- **§1+ Full mode** — Adds vLLM (Qwen3-VL) for table/figure/formula
  enrichment and Ollama+Qdrant for embeddings + retrieval. Original H200
  configuration; reference for when you scale up.

§12+ is the H200 session log: lessons, benchmark numbers, non-obvious bugs.
§13 is the Blackwell session log: pypdfium2 vs Docling A/B and the pivot to
text-only ingestion.

---

## 0A. Ultra-fast raw-text setup (pypdfium2)

Use when you want PDF → text fast and will chunk + embed in a separate step
later. **Bypasses Docling entirely.** Output: one JSON per PDF with
`page_texts: [...]` ready to feed an embedder.

### 0A.1 What runs

Nothing as a service. Just a Python script + ProcessPoolExecutor. No GPU
required. No Docling, vLLM, Ollama, Qdrant, or Redis needed.

### 0A.2 Pre-flight + system packages

```bash
apt-get install -y poppler-utils python3.11-venv python3.11-dev curl
```

### 0A.3 Workdir + venv

```bash
WORK=/workspace/meridian
mkdir -p $WORK/{pdfs/ntrs,output_pypdfium2,logs}
python3.11 -m venv $WORK/venv
$WORK/venv/bin/pip install -U pip wheel setuptools
$WORK/venv/bin/pip install pypdfium2
# (Or use the full meridian install — pypdfium2 is already a transitive dep)
```

### 0A.4 Fetch a batch

```bash
python3 $REPO/gpu-monitoring/fetch_ntrs.py \
    --out $WORK/pdfs/ntrs --count 1000 --query apollo --max-mb 80
```

### 0A.5 Parse

```bash
$WORK/venv/bin/python $WORK/parse_pypdfium2.py --workers 32
```

`parse_pypdfium2.py` lives at `/workspace/meridian/parse_pypdfium2.py`.
It includes:
- Per-page extraction via `pypdfium2`
- Soft-hyphen cleanup (drops U+00AD and U+FFFE — the latter is a corruption
  pattern seen in Apollo PDFs at line-break hyphenation)
- Two-part low-quality detection:
    - *empty/sparse* page: <100 alphanumeric chars
    - *garbled-OCR* page: alphanum / non-whitespace ratio <0.7 (catches
      docs whose embedded text layer is OCR garbage like Apollo
      `19690026608`)
  Doc is flagged `is_low_quality=true` if >40% of pages match either.
  Flagged IDs go to `_low_quality_list.txt`.

### 0A.6 Output shape

One JSON per PDF at `$WORK/output_pypdfium2/<doc_id>.json`:

```json
{
  "document_id": "19720021182",
  "source": "/workspace/meridian/pdfs/ntrs/19720021182.pdf",
  "pages": 348,
  "extractor": "pypdfium2",
  "elapsed_s": 17.297,
  "is_low_quality": false,
  "quality_stats": {
    "low_quality_pages": 37,
    "total_pages": 348,
    "alphanum_chars": 410289
  },
  "page_texts": ["page 1 text...", "page 2 text...", ...]
}
```

Plus aggregate files in the same dir:
- `_run_meta.json` — wall time, throughput, totals
- `_low_quality_list.txt` — doc IDs flagged as scanned/garbled
- `_failed_list.txt` — doc IDs that errored (rare; pypdfium2 tolerates malformed PDFs well)

### 0A.7 Benchmark on this Blackwell pod

| Metric | Value |
|---|---|
| Docs | 1000 (Apollo NTRS) |
| Pages | 78,452 |
| Wall clock | **47–50 s** (32 workers) |
| Throughput | **~1,200 docs/min, ~95,000 pages/min** |
| Effective parallelism | 28–30× of 32 |
| Failures | 0 / 1000 |
| Low-quality flagged | 146 / 1000 (~14.6%) |
| **108k docs extrapolation** | **~1.5 hours** |

CPU-bound, not GPU-bound. The Blackwell GPU is unused — would run identically
on a 32-vCPU CPU-only pod at much lower hourly cost.

### 0A.8 What you give up vs Docling

- No paragraph/heading detection — text is page-flat. Use sliding-window
  chunks (e.g., 500 tokens, 50 overlap) at embed time.
- No reading-order normalization in multi-column pages. pypdfium2 generally
  gets multi-column right but isn't perfect.
- No table structure — table cells become a stream of words.
- No figure/caption association.

If any of those matter, use §0B (Docling text-only) instead.

---

## 0B. Docling text-only setup (Blackwell 96 GB, glibc 2.39)

Use when you want PDF → structured chunks (paragraphs, headings, reading
order) and can spare the time + GPU. Slower than 0A but produces richer JSON.

### 0B.1 What runs

| Service | Needed? | Why |
|---|---|---|
| Docling × 8 (ports 9001-9008) | yes | layout model + structured doc extraction |
| Redis | yes (tiny) | shared round-robin counter for the Docling client load balancer; without it, every request hits instance 0 and 7 of 8 instances sit idle |
| vLLM | **no** | only used for VLM enrichment of tables/figures/formulas |
| Ollama | **no** | embeddings only |
| Qdrant | **no** | vector storage only |

### 0B.2 Pre-flight

```bash
nvidia-smi --query-gpu=name,memory.total --format=csv
nproc; free -h; df -h /workspace
ldd --version | head -1
which docker && docker run --rm hello-world      # if "no docker": go native
```

This pod (RTX Pro 6000 Blackwell, 96 GB, glibc 2.39, 128 cores, 2 TB RAM,
685 TB MooseFS at /workspace) has no Docker — install everything natively.

### 0B.3 System packages

```bash
apt-get update
apt-get install -y poppler-utils python3.11-venv python3.11-dev curl redis-server
```

If the deadsnakes PPA mirror 503s, that's fine as long as `python3.11` and
`python3.11-dev` are already installed (check with `dpkg -l | grep python3.11`).
`python3.11 -m venv` works without the `python3.11-venv` apt package on
Blackwell pods because `ensurepip` is bundled.

### 0B.4 Workdir layout

```bash
WORK=/workspace/meridian
mkdir -p $WORK/{hf-cache,pdfs/ntrs,output,logs}
for p in 9001 9002 9003 9004 9005 9006 9007 9008; do
    mkdir -p $WORK/tmp/docling_$p
done
```

Per-port temp dirs avoid the same-millisecond upload collision bug
(see §12.2 #3).

### 0B.5 Python venv + Meridian

```bash
python3.11 -m venv $WORK/venv
$WORK/venv/bin/pip install -U pip wheel setuptools
HF_HOME=$WORK/hf-cache $WORK/venv/bin/pip install -e ".[gpu]"
```

We do **not** install vLLM in text-only mode.

### 0B.6 `.env` (text-only)

The repo's `.env` is already set up for this mode. Key lines:

```ini
DOCLING_ACCELERATOR_DEVICE=cuda
DOCLING_LAYOUT_MODEL=heron        # fastest; accuracy diff vs egret-large
                                  # only matters with a downstream VLM
DOCLING_TABLE_STRUCTURE=false     # text-only mode (option A)
DOCLING_PICTURE_EXTRACTION=false  # nothing downstream consumes pictures
DOCLING_FORMULA_ENRICHMENT=false
DOCLING_OCR_ENABLED=false
DOCLING_DOC_BATCH_SIZE=8
DOCLING_DOC_BATCH_CONCURRENCY=8
DOCLING_PAGE_BATCH_SIZE=8
DOCLING_INSTANCES=http://localhost:9001,...,http://localhost:9008
HF_HOME=/workspace/meridian/hf-cache
TRANSFORMERS_CACHE=/workspace/meridian/hf-cache
```

### 0B.7 Start / stop

```bash
# Redis (shared round-robin counter)
mkdir -p $WORK/redis
redis-server --daemonize yes \
    --dir $WORK/redis \
    --logfile $WORK/logs/redis.log \
    --bind 127.0.0.1 --port 6379
redis-cli ping   # PONG

# Docling
$WORK/start_docling.sh 8       # idempotent; waits for /health to report ready
$WORK/stop_docling.sh           # SIGTERM via tracked pidfiles
```

**Important:** without Redis, the docling client logs
`Redis counter failed, falling back to first instance` and routes every
request to port 9001 — silently negating the rest of the pool. Always
start Redis before parsing.

First start downloads layout-model weights (~500 MB) into `$HF_HOME` and
takes ~30-60s before the first instance reports `models_loaded:true`. After
that, restarts are seconds.

### 0B.8 Fetch a benchmark batch

```bash
python3 $REPO/gpu-monitoring/fetch_ntrs.py \
    --out $WORK/pdfs/ntrs --count 100 --query apollo --max-mb 80
```

Uses system Python (httpx is already there). NTRS rejects requests without a
User-Agent header — the script sets one.

### 0B.9 Parse

```bash
$WORK/venv/bin/meridian parse $WORK/pdfs/ntrs \
    --output $WORK/output \
    --no-tables --no-figures --no-formulas \
    --workers 8
```

The three `--no-*` flags collectively short-circuit the VLM step
(`total_vlm_tasks=0` in `document_worker.py`). No `--store` means no
Qdrant/embedding calls. Result: pure Docling + chunker, output as JSON per
document.

### 0B.10 Text-only benchmark (RTX Pro 6000 Blackwell, 100 NTRS PDFs)

*(filled in after the run — see §13.)*

---

## 1. Pre-flight: know your VM (full mode)

Run these and note the answers — most decisions below depend on them:

```bash
nvidia-smi --query-gpu=name,memory.total --format=csv     # GPU + VRAM
free -h                                                    # RAM
nproc                                                      # CPU count
df -h                                                      # disk; find a big one
ldd --version | head -1                                    # glibc version
which docker; systemctl is-active docker 2>/dev/null      # Docker available?
ss -tln                                                    # ports already in use
ls -la /                                                   # is this a container?
```

Tells you:
- **VRAM** drives `vllm.gpu-memory-utilization` and number of Docling instances.
- **glibc** drives Qdrant version compatibility (see §4.3).
- **Docker availability** — if `unshare: operation not permitted` when running
  containers, you're in an unprivileged container; do native install.
- **Big disk** — route the HF cache, model files, and logs there. Docling +
  vLLM models total ~12 GB; HF cache grows over time.
- **In-use ports** — RunPod-style proxies often grab 7270, 7861, 8001, 8081,
  9091, 3001. Pick a clear range for Docling (this guide uses 9001-9012).

Decide once at the start:
- `WORK_DIR` — base directory on the big disk. (This guide uses `/workspace/meridian`.)
- `VENV` — Python venv path. (`$WORK_DIR/venv`)
- `DOCLING_BASE_PORT` — first free port for Docling. (Default 9001)
- `VLLM_PORT` — vLLM port. (Default 8000)

---

## 2. System packages

```bash
sudo apt-get update
sudo apt-get install -y \
    redis-server \
    poppler-utils \
    zstd \
    python3.11 python3.11-venv python3.11-dev \
    curl
```

Notes:
- `redis-server` runs Redis natively (we don't use Docker for it).
- `poppler-utils` is a docling runtime dep.
- Python 3.11 is the safe pin; 3.14 is too new for torch/docling.

---

## 3. Layout

```bash
WORK=/workspace/meridian   # adjust to your big disk
mkdir -p $WORK/{hf-cache,ollama,qdrant,redis,pdfs,output,logs,tmp/docling_service}
```

Set persistent env vars (you will reuse these everywhere):

```bash
export HF_HOME=$WORK/hf-cache
export TRANSFORMERS_CACHE=$WORK/hf-cache
```

---

## 4. Components

### 4.1 Redis

```bash
redis-server --daemonize yes \
    --dir $WORK/redis \
    --logfile $WORK/logs/redis.log \
    --bind 127.0.0.1
redis-cli ping   # PONG
```

### 4.2 Ollama (embeddings)

Don't use `curl | sh`. Download the binary tarball and verify:

```bash
LATEST=$(curl -sL https://api.github.com/repos/ollama/ollama/releases/latest \
    | grep -E '"browser_download_url".*ollama-linux-amd64.tar.zst' \
    | head -1 | cut -d'"' -f4)
cd /tmp && curl -fL -o ollama.tar.zst "$LATEST"
zstd -d ollama.tar.zst -o ollama.tar
mkdir -p $WORK/ollama-install
tar --no-same-owner -xf ollama.tar -C $WORK/ollama-install
ln -sf $WORK/ollama-install/bin/ollama /usr/local/bin/ollama
rm /tmp/ollama.tar.zst /tmp/ollama.tar

# Daemon
OLLAMA_HOST=127.0.0.1:11434 OLLAMA_MODELS=$WORK/ollama \
    nohup ollama serve > $WORK/logs/ollama.log 2>&1 &

# Pull the embedding model used by Meridian
ollama pull qwen3-embedding:4b-q8_0
```

### 4.3 Qdrant (vector store)

**Pick a version that matches your glibc.** Run `ldd --version | head -1`:

| Your glibc | Use Qdrant version |
|---|---|
| ≥ 2.38 (Ubuntu 24.04+) | latest |
| 2.35 (Ubuntu 22.04) | **≤ 1.12.6** |
| 2.31 (Ubuntu 20.04) | ≤ 1.10.x |

For glibc 2.35:

```bash
cd /tmp && curl -fL -o qdrant.tar.gz \
    https://github.com/qdrant/qdrant/releases/download/v1.12.6/qdrant-x86_64-unknown-linux-gnu.tar.gz
tar --no-same-owner -xzf qdrant.tar.gz -C $WORK/qdrant

cat > $WORK/qdrant/config.yaml <<'EOF'
storage:
  storage_path: /workspace/meridian/qdrant/storage    # update to match $WORK
service:
  host: 127.0.0.1
  http_port: 6333
  grpc_port: 6334
log_level: INFO
EOF

cd $WORK/qdrant && nohup ./qdrant --config-path $WORK/qdrant/config.yaml \
    > $WORK/logs/qdrant.log 2>&1 &

curl -sf http://127.0.0.1:6333/collections   # should return ok
```

If the binary fails with `GLIBC_2.38 not found`, drop one minor Qdrant version
and retry.

### 4.4 Python venv + Meridian

```bash
python3.11 -m venv $WORK/venv
source $WORK/venv/bin/activate
pip install --upgrade pip wheel setuptools

cd /path/to/meridian-oss      # the cloned repo
HF_HOME=$WORK/hf-cache pip install -e ".[gpu]"
pip install vllm              # latest is fine; pulls a compatible torch
```

### 4.5 vLLM (vision-language model)

vLLM runs as a process — no Docker needed. Save this as `$WORK/start_vllm.sh`:

```bash
#!/bin/bash
export HF_HOME=/workspace/meridian/hf-cache
export TRANSFORMERS_CACHE=/workspace/meridian/hf-cache
exec /workspace/meridian/venv/bin/python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-VL-8B-Instruct \
    --quantization fp8 \
    --gpu-memory-utilization 0.40 \
    --max-num-seqs 256 \
    --max-model-len 16384 \
    --enable-chunked-prefill \
    --enable-prefix-caching \
    --host 0.0.0.0 \
    --port 8000
```

`chmod +x $WORK/start_vllm.sh && nohup $WORK/start_vllm.sh > $WORK/logs/vllm.log 2>&1 &`

First start downloads the model (~10 GB) and runs CUDA-graph compilation —
expect 1–3 minutes before `curl http://127.0.0.1:8000/v1/models` returns 200.

#### Tuning gpu-memory-utilization

`gpu-memory-utilization` is a fraction of *the whole GPU at vLLM startup*.
Pick it so total VRAM usage stays comfortably under capacity:

| GPU VRAM | Suggested util | Resulting KV cache |
|---|---|---|
| 80 GB (H100) | 0.55 | ~30 GB |
| 143 GB (H200) | 0.40 | ~43 GB |
| 24 GB (4090) | 0.30 | ~5 GB |

The 8B model itself takes ~8 GB; the rest of the budget becomes KV cache.
Bigger KV cache = more concurrent VLM requests.

### 4.6 Docling instances

Pick a port range with no conflicts. Save as `$WORK/start_docling.sh`:

```bash
#!/bin/bash
N="${1:-12}"
BASE_PORT="${BASE_PORT:-9001}"
LOG_DIR=/workspace/meridian/logs
mkdir -p "$LOG_DIR"

for i in $(seq 0 $((N-1))); do
    PORT=$((BASE_PORT + i))
    PIDFILE="$LOG_DIR/docling_${PORT}.pid"
    if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then continue; fi
    cd /root/code/meridian-oss          # path to your meridian repo
    DOCLING_SERVICE_PORT=$PORT \
    HF_HOME=/workspace/meridian/hf-cache \
    TRANSFORMERS_CACHE=/workspace/meridian/hf-cache \
    nohup /workspace/meridian/venv/bin/python -m uvicorn \
        meridian.services.docling_service.main:app \
        --host 0.0.0.0 --port $PORT \
        > "$LOG_DIR/docling_${PORT}.log" 2>&1 &
    echo $! > "$PIDFILE"
done
```

`chmod +x $WORK/start_docling.sh && $WORK/start_docling.sh 12`

#### How many Docling instances?

Each instance uses ~1.2 GB VRAM. Roughly:

| GPU VRAM | Instance count |
|---|---|
| 24 GB | 2 |
| 40 GB (A100) | 4 |
| 80 GB (H100) | 8 (the README's reference) |
| 143 GB (H200) | 12 |

More than 12 hits diminishing returns — the layout model becomes
compute-bound rather than memory-bound.

---

## 5. Configuration: `.env`

Drop this in the repo root as `.env`. Adjust paths and the docling instance
list to match your port range.

```ini
# Redis
REDIS_URL=redis://localhost:6379/0

# Docling service
DOCLING_SERVICE_HOST=0.0.0.0
DOCLING_SERVICE_PORT=9001
DOCLING_MAX_CONCURRENT=16
DOCLING_REQUEST_TIMEOUT=600
DOCLING_TEMP_DIR=/workspace/meridian/tmp/docling_service
DOCLING_OCR_ENABLED=false
DOCLING_TABLE_STRUCTURE=true
DOCLING_TABLE_MODE=accurate
DOCLING_ACCELERATOR_DEVICE=cuda
DOCLING_ACCELERATOR_THREADS=8
DOCLING_DOC_BATCH_SIZE=8
DOCLING_DOC_BATCH_CONCURRENCY=8
DOCLING_PAGE_BATCH_SIZE=8
DOCLING_IMAGE_SCALE=2.0
DOCLING_PDF_BACKEND=v2

# Docling load balancing — must list every running instance
DOCLING_INSTANCES=http://localhost:9001,http://localhost:9002,http://localhost:9003,http://localhost:9004,http://localhost:9005,http://localhost:9006,http://localhost:9007,http://localhost:9008,http://localhost:9009,http://localhost:9010,http://localhost:9011,http://localhost:9012

# vLLM
VLLM_API_URL=http://localhost:8000/v1/chat/completions
VLLM_MODEL=Qwen/Qwen3-VL-8B-Instruct
VLLM_API_KEY=EMPTY
VLLM_GPU_MEMORY_UTILIZATION=0.40
VLLM_MAX_NUM_SEQS=256
VLLM_MAX_MODEL_LEN=16384

# Ollama
OLLAMA_URL=http://localhost:11434
EMBEDDING_MODEL=qwen3-embedding:4b-q8_0
EMBEDDING_DIMENSIONS=1024

# Qdrant
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_COLLECTION=meridian_documents

# Workers
CELERY_CONCURRENCY=12

# HF cache (keep big files off the small root disk)
HF_HOME=/workspace/meridian/hf-cache
TRANSFORMERS_CACHE=/workspace/meridian/hf-cache
```

### Tuning by GPU class

| Setting | 24 GB | 40 GB | 80 GB (H100) | 143 GB (H200) |
|---|---|---|---|---|
| Docling instances | 2 | 4 | 8 | 12 |
| `VLLM_GPU_MEMORY_UTILIZATION` | 0.30 | 0.40 | 0.55 | 0.40 |
| `VLLM_MAX_NUM_SEQS` | 32 | 64 | 128 | 256 |
| `VLLM_MAX_MODEL_LEN` | 8192 | 8192 | 8192 | 16384 |
| `DOCLING_DOC_BATCH_SIZE` | 2 | 4 | 4 | 8 |
| `DOCLING_PAGE_BATCH_SIZE` | 2 | 4 | 4 | 8 |
| `CELERY_CONCURRENCY` | 2 | 4 | 3 | 12 |

---

## 6. Patch: DoclingClient must read `.env`

Upstream `meridian/clients/docling_client.py` hardcodes `DEFAULT_INSTANCES` to
`localhost:8001-8008`. If you run on different ports (e.g., 9001-9012), the
worker silently sends to the wrong ports and fails with **"Unknown processing
error"** with no log on any Docling instance.

Apply this patch (already applied in this repo as of 2026-04-26):

```python
# meridian/clients/docling_client.py — top of file
from meridian.config import (
    DOCLING_INSTANCES as _CONFIG_INSTANCES,
    REDIS_URL as _CONFIG_REDIS_URL,
)
DEFAULT_INSTANCES = list(_CONFIG_INSTANCES)
```

```python
# In DoclingClientConfig:
redis_url: str = field(default_factory=lambda: _CONFIG_REDIS_URL)
```

If the symptom returns ("Unknown processing error" with no docling log entry),
this is the first thing to check.

---

## 7. Start / stop scripts

`$WORK/start_all.sh` — idempotent boot of every service:

```bash
#!/bin/bash
set -e
LOG_DIR=/workspace/meridian/logs
mkdir -p "$LOG_DIR"

# Redis
if ! redis-cli ping > /dev/null 2>&1; then
    redis-server --daemonize yes --dir /workspace/meridian/redis \
        --logfile "$LOG_DIR/redis.log" --bind 127.0.0.1
fi

# Qdrant
if ! curl -sf http://127.0.0.1:6333/collections > /dev/null 2>&1; then
    cd /workspace/meridian/qdrant
    nohup ./qdrant --config-path /workspace/meridian/qdrant/config.yaml \
        > "$LOG_DIR/qdrant.log" 2>&1 &
    sleep 3
fi

# Ollama
if ! curl -sf http://127.0.0.1:11434/api/tags > /dev/null 2>&1; then
    OLLAMA_HOST=127.0.0.1:11434 OLLAMA_MODELS=/workspace/meridian/ollama \
        nohup ollama serve > "$LOG_DIR/ollama.log" 2>&1 &
    sleep 3
fi

# vLLM
if ! curl -sf http://127.0.0.1:8000/v1/models > /dev/null 2>&1; then
    nohup /workspace/meridian/start_vllm.sh > "$LOG_DIR/vllm.log" 2>&1 &
fi

# Docling
/workspace/meridian/start_docling.sh 12

# Wait
until curl -sf http://127.0.0.1:8000/v1/models 2>/dev/null | grep -q '"id"'; do sleep 5; done
for p in 9001 9002 9003 9004 9005 9006 9007 9008 9009 9010 9011 9012; do
    until curl -sf "http://localhost:$p/health" 2>/dev/null | grep -q '"models_loaded":true'; do
        sleep 2
    done
done
echo "All services up."
```

`$WORK/stop_all.sh`:

```bash
#!/bin/bash
for f in /workspace/meridian/logs/docling_*.pid; do
    [ -f "$f" ] || continue
    kill "$(cat "$f")" 2>/dev/null; rm -f "$f"
done
pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null
pkill -f "ollama serve" 2>/dev/null
pkill -f "/workspace/meridian/qdrant/qdrant" 2>/dev/null
redis-cli shutdown nosave 2>/dev/null
```

---

## 8. Validation

```bash
# Process a small PDF end-to-end
/workspace/meridian/venv/bin/meridian parse /workspace/meridian/pdfs/some.pdf \
    --output /workspace/meridian/output \
    --store --collection meridian_test

# Inspect Qdrant
curl -s http://localhost:6333/collections/meridian_test | python3 -m json.tool
```

You should see a JSON file in `/workspace/meridian/output/` with `chunks: [...]`
populated and a Qdrant collection with one vector per chunk.

Quick semantic-search smoke test:

```python
import asyncio
from meridian.clients.embedding_client_async import AsyncEmbeddingClient

async def go():
    async with AsyncEmbeddingClient(collection_name="meridian_test") as c:
        for r in await c.search_async("your query here", limit=3):
            print(f"{r['score']:.3f}  page={r.get('page_no')}  {r.get('text','')[:120]}")

asyncio.run(go())
```

---

## 9. Common pitfalls

| Symptom | Cause | Fix |
|---|---|---|
| `unshare: operation not permitted` when starting any container | Unprivileged container; nested namespaces blocked | Don't use Docker — install everything natively per this doc |
| Qdrant: `version GLIBC_2.38 not found` | New Qdrant on old Ubuntu | Use Qdrant ≤ 1.12.6 on glibc 2.35 |
| `meridian parse` says "Unknown processing error", no Docling log shows the request | Section 6 patch missing — client uses default ports 8001-8008 | Apply the patch; verify `python -c "from meridian.clients.docling_client import DEFAULT_INSTANCES; print(DEFAULT_INSTANCES)"` |
| `address already in use` on Docling startup | Port collision (RunPod nginx grabs 8001 etc.) | Move Docling to 9001+; update `.env` `DOCLING_INSTANCES` |
| HF model download exhausts root disk | `/` is small | Set `HF_HOME` and `TRANSFORMERS_CACHE` to the big disk *before* installing or starting any service |
| vLLM startup runs OOM mid-load | `gpu-memory-utilization` too high relative to Docling's pre-allocation | Start vLLM first, or lower `VLLM_GPU_MEMORY_UTILIZATION` |
| Single-doc throughput looks low (e.g., ~10 pg/min) | Embedding step is serial through Ollama | Expected. Throughput scales with batch size — meridian's design assumes many docs in flight via Celery |

---

## 10. What's running where (reference card)

| Service | Process | Port | Data on disk |
|---|---|---|---|
| Redis | `redis-server` | 6379 | `$WORK/redis/` |
| Qdrant | `$WORK/qdrant/qdrant` | 6333 (HTTP), 6334 (gRPC) | `$WORK/qdrant/storage/` |
| Ollama | `ollama serve` | 11434 | `$WORK/ollama/` |
| vLLM | `python -m vllm.entrypoints.openai.api_server` | 8000 | HF cache at `$HF_HOME` |
| Docling × 12 | `python -m uvicorn meridian.services.docling_service.main:app` | 9001-9012 | HF cache at `$HF_HOME` |
| Celery (optional) | `celery -A meridian.workers.celery_app worker` | n/a | uses Redis |

For most workflows you don't need Celery — `meridian parse` runs the pipeline
in-process. Use Celery (`meridian batch submit ...`) for very large batches.

---

## 11. Performance notes

Single document on H200 (15-page PDF, 4 tables, 6 figures): **85 s, 10.6 pg/min**.
Bottleneck is the embedding step (Ollama, single-threaded per request).

Throughput scales roughly with concurrent documents — the 180+ pg/min
in the README is for batches of thousands of PDFs going through Celery,
where multiple docs are in different pipeline stages simultaneously.

To benchmark: drop a folder of PDFs into `$WORK/pdfs/` and run
`meridian parse $WORK/pdfs --workers 12`.

---

## 12. Session log — H200 setup + tuning (2026-04-26)

### 12.1 Configuration evolution + benchmark results

Three benchmark runs over the same H200 RunPod, same software versions:

| Run | Layout | Docling # | vLLM util / max_seqs | Wall clock | Throughput | Failures |
|---|---|---|---|---|---|---|
| Initial 20 papers (academic, ~30pg avg) | heron | 12 | 0.40 / 256 | 4m 53s | 118.7 pg/min | 0 |
| **NTRS-25 v1** (Apollo reports) | heron | 12 | 0.40 / 256 | 25m 45s | **82.9 pg/min** | 5 retried (hit 9-min soft limit) |
| **NTRS-25 v2** (same 25 PDFs) | egret-large | 5 | 0.65 / 1024 | 27m 40s | **77.2 pg/min** | 0 |

#### What v2 actually showed

v2 had **zero retry storms** (the soft limit raise to 30 min worked) but the
cumulative throughput came out slightly *lower* than v1. Looking at per-doc
times reveals what actually happened:

- 24 of 25 docs in v2 completed by the 12-minute mark.
- The 25th doc — `19720021432` (355 pages, 329 tables + 348 figures detected,
  677 VLM requests) — took **20m 1s** by itself, dragging wall clock from
  ~13 min to ~28 min. Same doc in v1 took 8m 25s (505s).
- Excluding that one tail, v2's effective throughput on 24/25 was
  **~95-100 pg/min sustained** with no retries — a real improvement over v1's
  ~83 pg/min counting retry overhead.

Why the same doc took 2.4× longer in v2:
- 5 Docling instances vs v1's 12. Docling step queues more under load.
- `egret-large` is slower per-page than `heron` for the layout pass, even
  though it's more accurate.
- vLLM at 1024 max_seqs handles bursts better but still has to chew through
  677 sequential VLM requests for that doc.

Honest read: v2 is the better config for a real production batch (no retries,
predictable behavior) but for *this specific batch* the single pathological
doc dominated the wall-clock-based throughput metric. The improvement is
real on the bulk of docs and obscured by averaging.

Key VRAM observation: dropping Docling 12→5 freed ~9 GB. egret-large
per-instance VRAM turned out to be similar to heron (~1.2 GB), so the freed
VRAM came mostly from running fewer copies. That headroom went to vLLM's KV
cache.

### 12.2 Real bugs found and fixed

1. **`DEFAULT_INSTANCES` hardcoded to ports 8001-8008** in
   `meridian/clients/docling_client.py`. If you run Docling on different ports
   (we use 9001+), the worker silently fires at non-existent endpoints and
   reports `"Unknown processing error"` with no log on any Docling instance.
   Fix: import `DOCLING_INSTANCES` and `REDIS_URL` from `meridian.config` and
   use them as the dataclass defaults. Already applied in this repo.

2. **Qdrant `AsyncQdrantClient` defaults to a 5s httpx timeout.** Under
   concurrent write load (multiple workers each PUTting hundreds of vectors),
   Qdrant takes longer than 5s to ack and writes start failing. Symptom:
   `httpx.ReadTimeout` traceback, or — worse — collections silently storing
   **all-zero vectors** (we found 4 collections like this from the first
   attempts; cosine similarity to anything was always 0). Fix:
   `AsyncQdrantClient(... timeout=120)` in `meridian/clients/embedding_client_async.py`.

3. **Shared `DOCLING_TEMP_DIR` collision.** When 12 Docling instances
   parallel-warm, they all upload to the same `/tmp/docling_service` and use
   timestamp-suffixed filenames. Same-millisecond uploads stomp on each other,
   producing `ConversionError: Input document ... is not valid.` Fix: each
   port gets its own `tmp/docling_<port>/` (see `start_docling.sh`).

4. **Sourcing `.env` into a service launcher exports `VLLM_API_KEY` to vLLM,
   which then enables bearer-token auth.** Symptom: every request gets HTTP
   401. The `.env` carries `VLLM_API_KEY=EMPTY` for the *client*; the *server*
   should not see it. Fix: `unset VLLM_API_KEY` inside `start_vllm.sh` after
   sourcing `.env`.

5. **`kill -9` of vLLM leaks the CUDA context.** `nvidia-smi` keeps reporting
   the GPU memory as used by a host PID that no longer exists in the
   container's PID namespace. We cannot `nvidia-smi --gpu-reset` from inside
   an unprivileged container. Recovery: kill all GPU-using processes; the
   driver eventually reclaims the leaked memory once nothing else is holding
   the GPU. Lesson: prefer SIGTERM and let vLLM shut down cleanly.

### 12.3 Ollama embedding right-sizing

Default Ollama loads `qwen3-embedding:4b-q8_0` with the model's max context
(40,960 tokens) reserved for KV cache, even though we only embed chunks of a
few hundred tokens. Result: ~14 GB VRAM for what should be a 4 GB model.

Settings that brought it down to ~5.4 GB:
- `OLLAMA_CONTEXT_LENGTH=4096` (the env that controls the actual KV alloc)
- `OLLAMA_KV_CACHE_TYPE=q8_0` (8-bit KV instead of fp16)
- `OLLAMA_NUM_PARALLEL=12` (later raised to 24)
- Embedding client also sends `options: {"num_ctx": 4096}` per request

Q8 vs Q4: stuck with Q8. Q4 saves ~2 GB on weights but costs ~1-2% retrieval
quality on MTEB-style benchmarks. Not worth it given the VRAM headroom.

### 12.4 Celery soft time limit — what it is and why it matters

`meridian/workers/celery_app.py` has two timeouts per task:

- `task_soft_time_limit=540` (9 min) — Celery raises `SoftTimeLimitExceeded`
  *inside* the task. Pipeline catches it, marks failed, Celery retries
  (`max_retries=1` in `tasks.py`).
- `task_time_limit=600` (10 min) — hard kill of the worker process.

In NTRS-25 v1, **5 of the 25 docs** (the 200-355 page Apollo reports) took
longer than 9 min on first attempt because they were running concurrently with
11 other big docs all hitting vLLM. Soft limit fired → marked failed → retry
queued. Retries succeeded because the cluster was less loaded. Cost: ~30 min
of wasted compute, double-counted "Total: 30 / Failed: 5" in the batch state
manager (the state machine doesn't unwind a failed entry when its retry succeeds).

We raised soft to 1800s (30 min) and hard to 2100s (35 min) for v2.

### 12.5 GPU process labelling: the registry approach

`nvidia-smi --query-compute-apps=pid` returns **host PIDs** that don't exist
in this RunPod container's PID namespace. NSpid lookups, `lsof`, and
`process_name` from NVML all return nothing useful. Heuristics ("biggest is
vLLM, smallest 12 are Docling") break the moment a service restarts, dies, or
shrinks (e.g., Ollama dropped from 14 GB to 5.4 GB after tuning, then got
mislabeled as "Docling #01").

Solution: **a registry file at `/workspace/meridian/gpu_labels.json`** that
each service launcher writes to *after* CUDA context init.

- `start_vllm.sh`, `start_docling.sh`, `start_ollama.sh`, `start_watchdog.sh`
  all snapshot `nvidia-smi --query-compute-apps=pid` before/after launching
  their service and write the diff to the registry.
- The watchdog re-labels any Docling instance it auto-restarts and prunes
  dead host PIDs.
- `monitor.py` reads the registry — anything not in it shows as `?`.

### 12.6 RunPod container caveats

- **No Docker.** `dockerd` runs but `docker run` fails with
  `failed to register layer: unshare: operation not permitted`. The pod
  doesn't grant the namespace capabilities needed for nested containers.
  Native install of every component (Redis, Qdrant, Ollama, vLLM via pip) is
  the only working path.
- **Port collisions with RunPod's nginx proxy.** 8001, 8081, 9091, 7270, 7861,
  3001 are already bound. We moved Docling to **9001+**; vLLM stays on 8000
  (free).
- **Persistent storage at `/workspace`** (218 TB free, MooseFS). Root `/` is a
  20 GB overlay — anything model-sized goes to `/workspace`.

### 12.7 Patterns observed in the data

NTRS / Apollo-era technical reports are unusually VLM-heavy:

| Doc | Pages | Tables detected | Figures detected | VLM requests |
|---|---|---|---|---|
| 19720021432 | 355 | 329 | 348 | 677 |
| 19650025875 | 312 | 216 | 292 | 518 |
| 19660019789 | 107 | 89 | 81 + 29 formula pages | 199 |
| 19660025933 | 90 | 58 | 65 + 9 formula pages | 132 |

The **"large pictures as table candidates"** path in
`meridian/services/docling_service/extractors.py` (gated by
`PICTURE_TABLE_CANDIDATE_MIN_AREA = 50000` pt²) is the dominant source of
extra VLM calls for these scanned-PDF-style documents — one log line shows
*"Added 309 large pictures as table candidates"* on a single doc, even with
the more accurate egret-large layout model. The layout model only contributes
the 21 detections; the 309 are from the area-threshold heuristic in the
extractor, which is independent of the layout model choice.

### 12.8 Operational scripts (current state)

| Script | Purpose |
|---|---|
| `/workspace/meridian/start_all.sh` | Bring everything up from cold |
| `/workspace/meridian/start_ollama.sh` | Ollama with KV q8, ctx=4096, parallel=24 — registers in gpu_labels.json |
| `/workspace/meridian/start_docling.sh N [warm]` | N Docling instances with egret-large, per-port tmpdir, expandable_segments — registers per-port labels |
| `/workspace/meridian/start_vllm.sh` | Sources .env (without leaking `VLLM_API_KEY` to vLLM), launches with `gpu_util=0.65 max_seqs=1024` — registers as "vLLM" |
| `/workspace/meridian/start_watchdog.sh` | Monitors ports 9001-9005, restarts crashed Docling and re-labels in registry |
| `/workspace/meridian/stop_all.sh` | Best-effort stop |
| `/workspace/meridian/monitor.py` (`meridian-monitor`) | Live dashboard — reads gpu_labels.json + service health |
| `gpu-monitoring/fetch_ntrs.py` | Harvest N PDFs from NASA NTRS (needs User-Agent header) |

### 12.9 Open observations (not yet acted on)

- **Single pathological doc dominates wall clock.** `19720021432` (355 pages,
  677 VLM calls) took 20 min in v2 vs 8.5 min in v1. With 4 other workers
  idle, this doc ran serially through its 677 VLM requests against vLLM's
  1024-seq cap — but its requests were funneled through one celery worker's
  async event loop, so the per-doc parallelism limit was the bottleneck, not
  vLLM. Per-doc internal concurrency (semaphore in
  `clients/vlm_client_async.py`) caps in-flight requests per document
  regardless of vLLM's capacity.
- The bigger NTRS docs spend most of their time waiting on vLLM, not on
  Docling or embedding. Per-doc timing breakdowns (in chunks JSON) confirm
  this.
- vLLM at 0.65 leaves ~30 GB unused on a 143 GB H200.
- The Celery state manager records both the original failure and the retry
  attempt as separate entries in `Total`, which is why a 25-doc batch can
  show "Total: 30 / Failed: 5" when retries succeed (v1 hit this; v2 didn't).
- 4 collections from earlier runs in this session contain all-zero vectors
  (from the Qdrant timeout bug pre-fix). They've been deleted; future runs
  use the patched async client.
- The `PICTURE_TABLE_CANDIDATE_MIN_AREA = 50000` heuristic in
  `services/docling_service/extractors.py` adds large pictures as table
  candidates regardless of layout-model accuracy — independent of `heron` vs
  `egret-large`. One Apollo doc logged "Added 309 large pictures as table
  candidates" even on the more accurate model. This is a fixed-area filter,
  not a learned one.



---

## 13. Session log — Blackwell pivot to text-only (2026-04-28)

### 13.1 Why we pivoted

The original plan was to keep running on H200 with the full VLM + embedding
pipeline, but a 30-day continuous run was too expensive. Moved to a cheaper
RTX Pro 6000 Blackwell pod (96 GB VRAM, 128 vCPU, 2 TB RAM, glibc 2.39) and
scoped the pipeline down to **text-only ingestion** — parse now, embed later
when (and if) retrieval needs justify it.

### 13.2 Decisions made (record so we don't re-litigate them tomorrow)

| Decision | Why |
|---|---|
| Drop tables/figures/formulas | All require VLM downstream, which we're skipping. Output is pure prose. |
| Drop vLLM, Ollama, Qdrant for now | Not needed for parsing. Ollama+Qdrant brought up briefly to test embedding throughput; both abandoned for now. |
| Use **pypdfium2** instead of Docling | A/B'd against pdftotext, pymupdf, Docling on 30 representative Apollo NTRS docs. pypdfium2 wins on prose flow (clean paragraph reading, intact sentences, soft-hyphens preserved-and-cleanable). pymupdf was *worse* on Apollo PDFs — split words to one-per-line. pdftotext --layout was OK but bloated with whitespace padding. **87× faster than Docling end-to-end.** |
| Defer OCR for the ~3% of garbled docs | OCR recovers ~2-3% of corpus content from corrupted text layers. Top-K retrieval is robust to that loss. Decision: tag them via `needs_ocr=true`, run OCR retroactively only if a real query needs them. |
| Defer embedding | Pivoted Ollama+Qdrant up but realized I was tuning configs (dimensions, batch size, concurrency) without a baseline measurement. Backed out. **Next time: run with library defaults first, measure, then tune.** |
| Embed later, not during parse | Lets you re-chunk or swap embedding models without re-parsing PDFs. Saves cost. |

### 13.3 Benchmark numbers (1000 NTRS PDFs, Blackwell)

| Metric | pypdfium2 | Docling text-only |
|---|---|---|
| Wall clock (1000 docs) | **~50 s** (32 workers) | extrapolated 109 min (16 workers + GPU) |
| Throughput | ~1,200 docs/min, ~95k pages/min | ~9 docs/min, ~1,200 pages/min |
| GPU used | none | 16 instances + Redis |
| Failures | 0 / 1000 | 0 / 311 (we killed Docling early) |
| 108k extrapolation | **~1.5 h** | ~6-8 days |

Effective parallelism on pypdfium2: 28-30× of 32 (95% efficient, CPU-bound).

### 13.4 Quality flagging heuristic (pypdfium2)

Per-page classification:
- `empty`   = <100 alphanumeric chars (figure pages with brief captions)
- `garbled` = ≥100 alphanum AND alphanum/non-whitespace ratio <0.7 (corrupted text layer or OCR-noise)
- `ok`      = real prose

Per-doc flag `needs_ocr=true` when `garbled / non_empty > 0.30`. On 1000 NTRS:
- 30 docs flagged (3%)
- Stored in `output_pypdfium2/_needs_ocr_list.txt`

False-positive notes: TOC pages with dot-leaders and dense-table pages can
trigger `garbled` at the page level — not a problem because the doc-level
flag still identifies real text-layer-corruption cases (manually verified
on `19690026608`).

### 13.5 State at end of session

| Thing | Where |
|---|---|
| 1000 source PDFs | `/workspace/meridian/pdfs/ntrs/` (8 GB, MooseFS — survives pod deletion) |
| 1000 parsed JSONs | `/workspace/meridian/output_pypdfium2/` (99 MB, MooseFS) |
| Run metadata | `/workspace/meridian/output_pypdfium2/_run_meta.json` |
| Needs-OCR list | `/workspace/meridian/output_pypdfium2/_needs_ocr_list.txt` (30 docs) |
| Working scripts (versioned) | `gpu-monitoring/{parse_pypdfium2,raw_text_ab,start_docling,stop_docling}.{py,sh}` |
| `.env` (text-only Blackwell config) | repo root |
| 16 Docling instances | running but unused — kill before next pod boot |
| Qdrant | running, empty — no collections were created |
| Ollama | stopped (was running, freed 5 GB VRAM before close) |

### 13.6 Resume plan — tomorrow

**Goal:** pull 108k PDFs from S3, parse them, end up with a directory of JSONs ready to embed.

1. **Spin up a fresh Blackwell-class pod** mounted to the same `/workspace`
   network volume. The 1000 already-parsed PDFs and JSONs will be sitting
   there. Save time: don't re-parse them.

2. **Reproduce the environment** (most of this should already exist on the
   network volume):
   ```bash
   # If venv survived: source it. If not, recreate per §0A.3.
   apt-get install -y poppler-utils python3.11-venv python3.11-dev curl awscli
   ```

3. **AWS S3 pull.** User has the bucket details + CLI configured locally:
   ```bash
   export AWS_ACCESS_KEY_ID=...
   export AWS_SECRET_ACCESS_KEY=...
   export AWS_DEFAULT_REGION=...
   aws s3 ls s3://<bucket>/<prefix>/ --summarize --recursive | tail -3
   aws configure set default.s3.max_concurrent_requests 64
   aws configure set default.s3.multipart_chunksize 32MB
   mkdir -p /workspace/meridian/pdfs/ntrs_full
   aws s3 sync s3://<bucket>/<prefix>/ /workspace/meridian/pdfs/ntrs_full --quiet
   ```
   Expected: ~30-80 min for ~864 GB depending on RunPod-to-S3 bandwidth.

4. **Parse:**
   ```bash
   /workspace/meridian/venv/bin/python /workspace/meridian/parse_pypdfium2.py \
       --pdf-dir /workspace/meridian/pdfs/ntrs_full \
       --out-dir /workspace/meridian/output_pypdfium2_full \
       --workers 32
   ```
   Expected: ~1.5 h for 108k.

5. **Optional optimization:** pipeline download + parse with a watcher loop
   so parse runs concurrent with sync. Saves ~30 min, not yet implemented.

6. **After parse completes:** revisit embedding. **Run with library
   defaults first** — Ollama + qwen3-embedding:4b-q8_0 + Qdrant. Measure
   throughput. Only tune after seeing the baseline.

### 13.7 Lessons

- The H200's GPU was overkill for text-only ingestion. 99% of per-doc time
  was CPU-bound (PDF text extraction + post-processing). Should have
  spotted this earlier — would have saved a week of H200 rental.
- pypdfium2 is the right default for Apollo-era scanned PDFs. PyMuPDF
  surprisingly worse on the same docs.
- Don't change configs without a baseline measurement. Touched too many
  knobs at once during the embedding bring-up; backed out.
- Network volume (MooseFS at /workspace) is the right place for everything
  that needs to survive pod cycling. 685 TB available, 816 MB/s write speed.
