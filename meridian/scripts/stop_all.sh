#!/bin/bash
# Stop all Meridian services
#
# Usage: ./scripts/stop_all.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LOG_DIR="$PROJECT_DIR/logs"

echo "=== Stopping Meridian Services ==="
echo ""

# 1. Stop watchdog
if [ -f "$LOG_DIR/watchdog.pid" ]; then
    pid=$(cat "$LOG_DIR/watchdog.pid")
    if kill -0 "$pid" 2>/dev/null; then
        echo "Stopping watchdog (PID: $pid)..."
        kill "$pid" 2>/dev/null || true
        rm -f "$LOG_DIR/watchdog.pid"
    fi
fi

# 2. Stop Celery worker
echo "Stopping Celery worker..."
"$SCRIPT_DIR/start_worker.sh" stop 2>/dev/null || true

# 3. Stop Docling instances
echo "Stopping Docling instances..."
"$SCRIPT_DIR/start_docling_instances.sh" stop 2>/dev/null || true

# 4. Stop vLLM container
if docker ps -q -f name=vllm | grep -q . 2>/dev/null; then
    echo "Stopping vLLM container..."
    docker stop vllm 2>/dev/null || true
fi

echo ""
echo "=== All Services Stopped ==="
