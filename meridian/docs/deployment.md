# Deployment Guide

## Prerequisites

### Hardware

- **GPU:** NVIDIA GPU with at least 40GB VRAM (H100 80GB recommended for full stack)
- **CPU:** 8+ cores recommended for Celery workers
- **RAM:** 32GB+ system memory
- **Storage:** SSD recommended for PDF I/O throughput

### Software

- Python 3.10 or newer (3.11+ recommended)
- NVIDIA GPU drivers (535+ for CUDA 12.x)
- Docker and Docker Compose (for vLLM, Qdrant, and optional containerized deployment)
- NVIDIA Container Toolkit (for Docker GPU access)
- Redis 7+
- Git

### GPU Memory Budget (H100 80GB)

| Component | Memory | Notes |
|-----------|--------|-------|
| Docling (8 instances) | ~10 GB | ~1.2 GB per instance |
| vLLM (Qwen3-VL-8B FP8) | ~38 GB | 55% of remaining ~70 GB |
| Ollama (embedding model) | ~4 GB | qwen3-embedding:4b-q8_0 |
| CUDA overhead | ~3 GB | Driver, context |
| **Total** | **~55 GB** | Leaves ~25 GB headroom |

For GPUs with less VRAM, reduce the number of Docling instances or lower `--gpu-memory-utilization` for vLLM.

## Bare Metal Installation

### 1. Clone and Install

```bash
git clone https://github.com/Rajsuthan/meridian.git
cd meridian

# Install the package with worker dependencies
pip install -e ".[worker]"

# If running Docling service on this machine, also install GPU dependencies
pip install -e ".[gpu]"
```

### 2. Install Redis

```bash
# Ubuntu/Debian
sudo apt update && sudo apt install redis-server

# Start Redis
sudo systemctl start redis-server
sudo systemctl enable redis-server

# Verify
redis-cli ping  # Should return PONG
```

### 3. Install Qdrant

```bash
# Docker is the simplest option
docker run -d --name qdrant \
    -p 6333:6333 -p 6334:6334 \
    -v qdrant_data:/qdrant/storage \
    qdrant/qdrant:latest

# Verify
curl http://localhost:6333/healthz
```

### 4. Install Ollama

```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull the embedding model
ollama pull qwen3-embedding:4b-q8_0

# Verify
curl http://localhost:11434/api/tags
```

### 5. Configure Environment

```bash
cp .env.example .env
```

Edit `.env` with your settings. At minimum, verify these defaults work for your setup:

```bash
REDIS_URL=redis://localhost:6379/0
VLLM_API_URL=http://localhost:8000/v1/chat/completions
VLLM_MODEL=Qwen/Qwen3-VL-8B-Instruct
OLLAMA_URL=http://localhost:11434
QDRANT_HOST=localhost
QDRANT_PORT=6333
```

### 6. Start Services

Start services in order. Each depends on the previous:

#### Start Docling Instances

```bash
./scripts/start_docling_instances.sh start
```

This starts 8 Docling service instances on ports 8001-8008. Model loading takes 30-60 seconds. The script waits and reports when all instances are healthy.

Verify:
```bash
curl http://localhost:8001/health
# {"status":"healthy","models_loaded":true,...}
```

#### Start vLLM

```bash
./scripts/start_vllm.sh
```

This starts a Docker container running vLLM with the Qwen3-VL-8B-Instruct model. First run downloads the model (~8GB). Model loading takes 1-3 minutes.

Verify:
```bash
curl http://localhost:8000/v1/models
# {"data":[{"id":"Qwen/Qwen3-VL-8B-Instruct",...}]}
```

#### Start Celery Worker

```bash
./scripts/start_worker.sh start
```

Verify:
```bash
./scripts/start_worker.sh status
```

#### Start Watchdog

```bash
# Run in background
nohup ./scripts/docling_watchdog.sh > logs/watchdog.log 2>&1 &

# Or run in tmux for easy monitoring
tmux new -s watchdog "./scripts/docling_watchdog.sh"
```

#### Start All at Once

Alternatively, start everything with a single command:

```bash
make start
# or
./scripts/start_all.sh
```

### 7. Verify Full Stack

```bash
# Check all Docling instances
for port in 8001 8002 8003 8004 8005 8006 8007 8008; do
    echo -n "Port $port: "
    curl -s http://localhost:$port/health | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['status'], '- models loaded' if d.get('models_loaded') else '- loading...')" 2>/dev/null || echo "unreachable"
done

# Check vLLM
curl -s http://localhost:8000/v1/models | python3 -c "import sys,json; print('vLLM:', json.load(sys.stdin)['data'][0]['id'])"

# Check Ollama
curl -s http://localhost:11434/api/tags | python3 -c "import sys,json; models=json.load(sys.stdin).get('models',[]); print('Ollama:', len(models), 'models')"

# Check Qdrant
curl -s http://localhost:6333/healthz && echo " Qdrant: OK"

# Check Redis
redis-cli ping
```

## Docker Compose Deployment

For a fully containerized deployment:

### 1. Configure

```bash
cp .env.example .env
# Edit .env as needed
```

### 2. Build and Start

```bash
# Build worker image
docker compose build

# Start all services
docker compose up -d

# Check status
docker compose ps

# Follow logs
docker compose logs -f
```

### 3. Service Endpoints (Docker)

When running in Docker, services communicate via Docker network names:

| Service | Internal URL | External Port |
|---------|-------------|---------------|
| Redis | `redis://redis:6379/0` | 6379 |
| Qdrant | `http://qdrant:6333` | 6333, 6334 |
| vLLM | `http://vllm:8000` | 8000 |
| Ollama | `http://ollama:11434` | 11434 |
| Docling | `http://docling:8001` | 8001 |

The worker container automatically uses Docker-internal URLs via environment variables set in `docker-compose.yml`.

### 4. GPU Requirements for Docker

Ensure NVIDIA Container Toolkit is installed:

```bash
# Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release; echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt update && sudo apt install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Verify
docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi
```

### 5. Stop Services

```bash
docker compose down

# To also remove volumes (deletes all data):
docker compose down -v
```

## Starting Individual Services

### Docling Instances

```bash
# Start all 8 instances
./scripts/start_docling_instances.sh start

# Stop all instances
./scripts/start_docling_instances.sh stop

# Check status
./scripts/start_docling_instances.sh status

# Restart all
./scripts/start_docling_instances.sh restart
```

### vLLM

```bash
# Start (Docker container)
./scripts/start_vllm.sh

# Stop
docker stop vllm

# Check logs
docker logs -f vllm

# Restart
docker restart vllm
```

### Celery Worker

```bash
# Start
./scripts/start_worker.sh start

# Stop
./scripts/start_worker.sh stop

# Status
./scripts/start_worker.sh status
```

### Watchdog

```bash
# Start in background
./scripts/docling_watchdog.sh &

# Or via make
make watchdog
```

## Health Endpoints

| Service | Endpoint | Healthy Response |
|---------|----------|-----------------|
| Docling | `GET /health` | `{"status": "healthy", "models_loaded": true}` |
| vLLM | `GET /v1/models` | `{"data": [{"id": "..."}]}` |
| Ollama | `GET /api/tags` | `{"models": [...]}` |
| Qdrant | `GET /healthz` | (empty 200 OK) |
| Redis | `redis-cli ping` | `PONG` |

## Scaling Considerations

### Scaling Docling Instances

More instances = higher throughput but more GPU memory. Adjust in `scripts/start_docling_instances.sh` by modifying the `PORTS` array:

```bash
# 4 instances (for smaller GPUs)
PORTS=(8001 8002 8003 8004)

# 8 instances (default, for H100)
PORTS=(8001 8002 8003 8004 8005 8006 8007 8008)
```

Update `DOCLING_INSTANCES` in `.env` to match:
```bash
DOCLING_INSTANCES=http://localhost:8001,http://localhost:8002,http://localhost:8003,http://localhost:8004
```

### Scaling Celery Workers

Increase concurrency (parallel document processing within one worker):

```bash
celery -A meridian.workers.celery_app worker --concurrency=6
```

Or run multiple worker processes:

```bash
# Terminal 1
celery -A meridian.workers.celery_app worker --concurrency=3

# Terminal 2
celery -A meridian.workers.celery_app worker --concurrency=3
```

### Scaling vLLM

Adjust GPU memory utilization and max sequences:

```bash
# More memory for larger batches
--gpu-memory-utilization 0.70

# More concurrent sequences
--max-num-seqs 256
```

### Multi-GPU Deployment

For machines with multiple GPUs, assign services to specific GPUs:

```bash
# Docling on GPU 0
CUDA_VISIBLE_DEVICES=0 ./scripts/start_docling_instances.sh start

# vLLM on GPU 1
docker run --gpus '"device=1"' ... vllm/vllm-openai:latest ...
```

### Multi-Node Deployment

Meridian supports multi-node deployment by pointing workers at remote services:

```bash
# On GPU node: run Docling, vLLM, Ollama
# On CPU node: run Celery workers pointing at GPU node

# CPU node .env:
REDIS_URL=redis://gpu-node:6379/0
DOCLING_INSTANCES=http://gpu-node:8001,http://gpu-node:8002,...
VLLM_API_URL=http://gpu-node:8000/v1/chat/completions
OLLAMA_URL=http://gpu-node:11434
QDRANT_HOST=gpu-node
```

## Troubleshooting

### Docling Instance Crashes

Symptoms: `RemoteProtocolError` or `ConnectionResetError` in worker logs.

The failover mechanism automatically routes to another instance. The watchdog will restart the crashed instance within ~90 seconds. If instances crash frequently:

- Check GPU memory with `nvidia-smi`
- Reduce instance count
- Check logs in `logs/docling_*.log`

### vLLM Out of Memory

Symptoms: vLLM container exits or returns 500 errors.

Reduce GPU memory utilization:
```bash
--gpu-memory-utilization 0.40
```

Or reduce max sequence length:
```bash
--max-model-len 4096
```

### Worker Not Processing

Check Celery worker is connected to Redis:
```bash
celery -A meridian.workers.celery_app inspect ping
```

Check for queued tasks:
```bash
celery -A meridian.workers.celery_app inspect active
```

### Batch Stuck in Processing

Documents stuck in `processing` state (worker crashed mid-task):
```bash
# Force recover after 10 minutes (default)
meridian resume <batch_id>

# Force recover immediately
meridian resume <batch_id> --force
```
