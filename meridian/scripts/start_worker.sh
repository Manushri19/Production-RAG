#!/bin/bash
# Start Celery worker with optimal settings
#
# Usage: ./start_worker.sh [start|stop|status]
#
# IMPORTANT: Concurrency is set to 3 to match Docling instances.
# Do NOT use default concurrency (CPU cores) as it causes contention.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LOG_DIR="$PROJECT_DIR/logs"
PID_FILE="$LOG_DIR/celery_worker.pid"
LOG_FILE="$LOG_DIR/celery_worker.log"

# Configuration
CONCURRENCY=3  # Match number of Docling instances
LOGLEVEL=info

# Create log directory
mkdir -p "$LOG_DIR"

start_worker() {
    echo "Starting Celery worker..."
    echo "  Concurrency: $CONCURRENCY"
    echo "  Log file: $LOG_FILE"

    # Check if already running
    if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
        echo "Worker already running (PID: $(cat "$PID_FILE"))"
        return 0
    fi

    cd "$PROJECT_DIR"

    # Start worker in background
    nohup celery -A meridian.workers.celery_app worker \
        --loglevel=$LOGLEVEL \
        --concurrency=$CONCURRENCY \
        > "$LOG_FILE" 2>&1 &

    echo $! > "$PID_FILE"
    echo "Started with PID: $(cat "$PID_FILE")"

    # Wait a moment and check if it started
    sleep 2
    if kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
        echo "Worker is running."
    else
        echo "ERROR: Worker failed to start. Check $LOG_FILE"
        rm -f "$PID_FILE"
        return 1
    fi
}

stop_worker() {
    echo "Stopping Celery worker..."

    if [ -f "$PID_FILE" ]; then
        pid=$(cat "$PID_FILE")
        if kill -0 "$pid" 2>/dev/null; then
            echo "Stopping worker (PID: $pid)..."
            kill "$pid"

            # Wait for graceful shutdown
            for i in {1..10}; do
                if ! kill -0 "$pid" 2>/dev/null; then
                    echo "Worker stopped."
                    rm -f "$PID_FILE"
                    return 0
                fi
                sleep 1
            done

            # Force kill if still running
            echo "Force killing worker..."
            kill -9 "$pid" 2>/dev/null
            rm -f "$PID_FILE"
        else
            echo "Worker not running (stale PID file)"
            rm -f "$PID_FILE"
        fi
    else
        echo "No PID file found. Checking for orphan processes..."
        pkill -f "celery.*meridian.workers.celery_app" 2>/dev/null || true
    fi

    echo "Done."
}

status_worker() {
    echo "Celery Worker Status:"
    echo "====================="

    if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
        pid=$(cat "$PID_FILE")
        echo "Status: RUNNING"
        echo "PID: $pid"
        echo "Concurrency: $CONCURRENCY"
        echo ""
        echo "Recent log:"
        tail -10 "$LOG_FILE" 2>/dev/null || echo "(no log file)"
    else
        echo "Status: STOPPED"
        if [ -f "$PID_FILE" ]; then
            echo "(stale PID file exists)"
        fi
    fi
}

restart_worker() {
    stop_worker
    sleep 2
    start_worker
}

# Main
case "${1:-start}" in
    start)
        start_worker
        ;;
    stop)
        stop_worker
        ;;
    status)
        status_worker
        ;;
    restart)
        restart_worker
        ;;
    *)
        echo "Usage: $0 [start|stop|status|restart]"
        exit 1
        ;;
esac
