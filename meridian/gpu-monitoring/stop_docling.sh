#!/bin/bash
# Stop all Docling instances tracked by pidfiles in $WORK_DIR/logs
WORK_DIR="${WORK_DIR:-/workspace/meridian}"
LOG_DIR="$WORK_DIR/logs"

for f in "$LOG_DIR"/docling_*.pid; do
    [ -f "$f" ] || continue
    pid=$(cat "$f")
    if kill -0 "$pid" 2>/dev/null; then
        kill "$pid" && echo "stopped pid $pid ($(basename "$f"))"
    fi
    rm -f "$f"
done

# Belt-and-braces: kill any leftover uvicorn workers
pkill -f "meridian.services.docling_service.main:app" 2>/dev/null
echo "done."
