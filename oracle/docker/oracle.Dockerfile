# Oracle API Dockerfile
# Lightweight Python image — no GPU needed for the API service itself
FROM python:3.11-slim

WORKDIR /app

# Install build tools for any compiled deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy and install dependencies first (layer caching)
COPY pyproject.toml ./
RUN pip install --no-cache-dir -e "."

# Copy source
COPY oracle/ ./oracle/
COPY .env.example .env.example

# Non-root user for security
RUN useradd -m -u 1001 oracle && chown -R oracle:oracle /app
USER oracle

EXPOSE 8010

CMD ["uvicorn", "oracle.api.app:app", "--host", "0.0.0.0", "--port", "8010", "--log-level", "info"]
