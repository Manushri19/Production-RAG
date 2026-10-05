#!/bin/bash
# Docling Watchdog Script
# Monitors Docling instances and restarts them if they crash
# Run this in tmux for persistence: tmux new -s watchdog "./scripts/docling_watchdog.sh"

# NOTE: Do NOT use 'set -e' here - we want the watchdog to keep running
# even if individual instance restarts fail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LOG_DIR="$PROJECT_DIR/logs"
CHECK_INTERVAL=30  # Check every 30 seconds

# Ports to monitor
PORTS=(8001 8002 8003 8004 8005 8006 8007 8008)

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

check_and_restart() {
    local port=$1
    local pid_file="$LOG_DIR/docling_$port.pid"
    local log_file="$LOG_DIR/docling_$port.log"

    # Check if instance is responding
    if curl -s --max-time 5 "http://localhost:$port/health" > /dev/null 2>&1; then
        return 0  # Healthy
    fi

    log "WARNING: Instance on port $port not responding"

    # Check if PID exists and process is running
    if [ -f "$pid_file" ]; then
        local pid=$(cat "$pid_file")
        if kill -0 "$pid" 2>/dev/null; then
            log "Process $pid still running but not responding - killing it"
            kill -9 "$pid" 2>/dev/null || true
            sleep 2
        fi
        rm -f "$pid_file"
    fi

    # Restart the instance
    log "Restarting instance on port $port..."
    cd "$PROJECT_DIR"
    nohup python -m uvicorn meridian.services.docling_service.main:app \
        --host 0.0.0.0 --port "$port" > "$log_file" 2>&1 &
    echo $! > "$pid_file"
    log "Started new instance on port $port (PID: $(cat "$pid_file"))"

    # Wait for it to become healthy
    local attempts=0
    local max_attempts=60  # Wait up to 60 seconds
    while [ $attempts -lt $max_attempts ]; do
        if curl -s --max-time 2 "http://localhost:$port/health" > /dev/null 2>&1; then
            log "Instance on port $port is now healthy"
            return 0
        fi
        sleep 1
        attempts=$((attempts + 1))
    done

    log "ERROR: Instance on port $port failed to become healthy after restart"
    return 0  # Return 0 to not break the loop - watchdog will retry next cycle
}

main() {
    log "=== Docling Watchdog Started ==="
    log "Monitoring ports: ${PORTS[*]}"
    log "Check interval: ${CHECK_INTERVAL}s"
    log ""

    while true; do
        for port in "${PORTS[@]}"; do
            check_and_restart "$port" || true  # Continue even if one instance fails
        done
        sleep "$CHECK_INTERVAL"
    done
}

# Handle Ctrl+C gracefully
trap 'log "Watchdog stopped"; exit 0' INT TERM

main
