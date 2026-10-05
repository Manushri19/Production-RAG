#!/bin/bash
# Start Docling service instances for parallel processing
#
# Usage: ./start_docling_instances.sh [start|stop|status]
#
# Each instance uses ~1.2GB GPU memory (with memory cleanup fix)
# 8 instances = ~10GB total (sweet spot for H100)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LOG_DIR="$PROJECT_DIR/logs"

# Create log directory
mkdir -p "$LOG_DIR"

# Instance ports (8 instances - sweet spot for H100)
PORTS=(8001 8002 8003 8004 8005 8006 8007 8008)

start_instances() {
    echo "Starting ${#PORTS[@]} Docling service instances..."
    echo ""

    for port in "${PORTS[@]}"; do
        log_file="$LOG_DIR/docling_$port.log"
        pid_file="$LOG_DIR/docling_$port.pid"

        # Check if already running
        if [ -f "$pid_file" ] && kill -0 "$(cat "$pid_file")" 2>/dev/null; then
            echo "Instance on port $port already running (PID: $(cat "$pid_file"))"
            continue
        fi

        echo "Starting instance on port $port..."

        # Start in background - use full module path
        cd "$PROJECT_DIR"
        nohup python -m uvicorn meridian.services.docling_service.main:app --host 0.0.0.0 --port "$port" > "$log_file" 2>&1 &
        echo $! > "$pid_file"

        echo "  PID: $(cat "$pid_file")"
        echo "  Log: $log_file"
    done

    echo ""
    echo "Waiting for instances to load models (this may take 30-60 seconds)..."
    echo ""

    # Wait for all instances to be healthy
    max_wait=180  # 3 minutes
    start_time=$(date +%s)

    while true; do
        all_healthy=true
        healthy_count=0

        for port in "${PORTS[@]}"; do
            if curl -s "http://localhost:$port/health" | grep -q '"models_loaded":true'; then
                ((healthy_count++))
            else
                all_healthy=false
            fi
        done

        elapsed=$(($(date +%s) - start_time))

        if $all_healthy; then
            echo "All $healthy_count instances are ready!"
            break
        fi

        if [ $elapsed -gt $max_wait ]; then
            echo "Warning: Timeout waiting for instances. $healthy_count/${#PORTS[@]} ready."
            break
        fi

        echo "  Waiting... $healthy_count/${#PORTS[@]} instances ready ($elapsed s)"
        sleep 5
    done

    echo ""
    status_instances
}

stop_instances() {
    echo "Stopping Docling service instances..."

    for port in "${PORTS[@]}"; do
        pid_file="$LOG_DIR/docling_$port.pid"

        if [ -f "$pid_file" ]; then
            pid=$(cat "$pid_file")
            if kill -0 "$pid" 2>/dev/null; then
                echo "Stopping instance on port $port (PID: $pid)..."
                kill "$pid"
                rm -f "$pid_file"
            else
                echo "Instance on port $port not running (stale PID file)"
                rm -f "$pid_file"
            fi
        else
            echo "No PID file for port $port"
        fi
    done

    echo "Done."
}

status_instances() {
    echo "Docling Service Instance Status:"
    echo "================================="
    echo ""

    for port in "${PORTS[@]}"; do
        pid_file="$LOG_DIR/docling_$port.pid"

        printf "Port %s: " "$port"

        # Check PID
        if [ -f "$pid_file" ] && kill -0 "$(cat "$pid_file")" 2>/dev/null; then
            printf "PID %s, " "$(cat "$pid_file")"
        else
            printf "NOT RUNNING, "
            continue
        fi

        # Check health
        health=$(curl -s "http://localhost:$port/health" 2>/dev/null || echo '{"status":"unreachable"}')

        if echo "$health" | grep -q '"models_loaded":true'; then
            echo "HEALTHY (models loaded)"
        elif echo "$health" | grep -q '"status":"healthy"'; then
            echo "STARTING (loading models...)"
        else
            echo "UNHEALTHY"
        fi
    done

    echo ""

    # GPU memory usage
    if command -v nvidia-smi &> /dev/null; then
        echo "GPU Memory Usage:"
        nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits | \
            awk '{printf "  %d MB / %d MB (%.1f%%)\n", $1, $2, $1*100/$2}'
    fi
}

# Main
case "${1:-start}" in
    start)
        start_instances
        ;;
    stop)
        stop_instances
        ;;
    status)
        status_instances
        ;;
    restart)
        stop_instances
        sleep 2
        start_instances
        ;;
    *)
        echo "Usage: $0 [start|stop|status|restart]"
        exit 1
        ;;
esac
