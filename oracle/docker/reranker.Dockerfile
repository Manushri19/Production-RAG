# Reranker Sidecar Dockerfile
# CPU-only is sufficient for BAAI/bge-reranker-v2-m3 at <30 candidates
# Add GPU reservation in docker-compose if you need sub-50ms reranking
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install requirements
COPY reranker/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy sidecar source
COPY reranker/main.py ./

# HuggingFace model cache mount point
ENV HF_HOME=/models
VOLUME ["/models"]

RUN useradd -m -u 1001 reranker && chown -R reranker:reranker /app
USER reranker

EXPOSE 8011

# Model download happens on first request (or /health call)
# Pre-download at build time by uncommenting:
# RUN python -c "from sentence_transformers import CrossEncoder; CrossEncoder('BAAI/bge-reranker-v2-m3')"

CMD ["python", "main.py"]
