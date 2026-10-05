#!/bin/bash
# Start all Meridian services
#
# Usage: ./scripts/start_all.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== Starting Meridian Services ==="
echo ""

# 1. Start Redis (if not running)
if ! redis-cli ping > /dev/null 2>&1; then
    echo "Starting Redis..."
    redis-server --daemonize yes
    sleep 1
fi
echo "Redis: OK"

# 2. Start Docling instances
echo ""
echo "Starting Docling instances..."
"$SCRIPT_DIR/start_docling_instances.sh" start

# 3. Start vLLM (if not running)
if ! curl -s http://localhost:8000/v1/models > /dev/null 2>&1; then
    echo ""
    echo "Starting vLLM..."
    "$SCRIPT_DIR/start_vllm.sh"
fi

# 4. Start Celery worker
echo ""
echo "Starting Celery worker..."
"$SCRIPT_DIR/start_worker.sh" start

# 5. Start watchdog
echo ""
echo "Starting Docling watchdog..."
nohup "$SCRIPT_DIR/docling_watchdog.sh" > "$(dirname "$SCRIPT_DIR")/logs/watchdog.log" 2>&1 &
echo $! > "$(dirname "$SCRIPT_DIR")/logs/watchdog.pid"
echo "Watchdog PID: $(cat "$(dirname "$SCRIPT_DIR")/logs/watchdog.pid")"

echo ""
echo "=== All Services Started ==="
echo ""
echo "Services:"
echo "  Redis:    localhost:6379"
echo "  Docling:  localhost:8001-8008"
echo "  vLLM:     localhost:8000"
echo "  Celery:   running"
echo "  Watchdog: running"
echo ""
echo "Submit a batch:"
echo "  meridian submit /path/to/pdfs --collection my_collection"
