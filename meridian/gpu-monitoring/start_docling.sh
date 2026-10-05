#!/bin/bash
# Start N Docling instances on ports BASE_PORT..BASE_PORT+N-1
# Idempotent: skips ports already running.
set -u

N="${1:-8}"
BASE_PORT="${BASE_PORT:-9001}"
REPO_DIR="${REPO_DIR:-/workspace/meridian-oss}"
WORK_DIR="${WORK_DIR:-/workspace/meridian}"
LOG_DIR="$WORK_DIR/logs"
mkdir -p "$LOG_DIR"

for i in $(seq 0 $((N-1))); do
    PORT=$((BASE_PORT + i))
    PIDFILE="$LOG_DIR/docling_${PORT}.pid"
    if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
        echo "port $PORT: already running (pid $(cat "$PIDFILE"))"
        continue
    fi

    # Per-port temp dir avoids the same-millisecond upload collision bug
    # documented in SETUP_NOTES §12.2
    TMP="$WORK_DIR/tmp/docling_${PORT}"
    mkdir -p "$TMP"

    cd "$REPO_DIR"
    DOCLING_SERVICE_PORT=$PORT \
    DOCLING_TEMP_DIR="$TMP" \
    HF_HOME="$WORK_DIR/hf-cache" \
    TRANSFORMERS_CACHE="$WORK_DIR/hf-cache" \
    nohup "$WORK_DIR/venv/bin/python" -m uvicorn \
        meridian.services.docling_service.main:app \
        --host 0.0.0.0 --port "$PORT" \
        > "$LOG_DIR/docling_${PORT}.log" 2>&1 &
    echo $! > "$PIDFILE"
    echo "port $PORT: started (pid $!)"
done

echo
echo "Waiting for instances to load models..."
for i in $(seq 0 $((N-1))); do
    PORT=$((BASE_PORT + i))
    until curl -sf "http://localhost:$PORT/health" 2>/dev/null | grep -q '"models_loaded":true'; do
        sleep 2
    done
    echo "  port $PORT: ready"
done
echo "All $N Docling instances ready on ports $BASE_PORT-$((BASE_PORT+N-1))."
