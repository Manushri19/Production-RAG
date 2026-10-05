#!/bin/bash
# Helpers to maintain /workspace/meridian/gpu_labels.json — host PID → service label.
# Sourced by service start scripts.

REGISTRY=/workspace/meridian/gpu_labels.json
[ -f "$REGISTRY" ] || echo '{}' > "$REGISTRY"

# Snapshot current NVML host PIDs into a Bash array (PIDs only).
gpu_snapshot() {
    nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null \
        | tr -d ' ' | grep -E '^[0-9]+$' | sort -u
}

# Atomic JSON update via python (jq isn't installed everywhere).
# Args: $1=label  $2..N=host PIDs
gpu_label_pids() {
    local label="$1"; shift
    [ "$#" -eq 0 ] && return 0
    /workspace/meridian/venv/bin/python - "$REGISTRY" "$label" "$@" <<'PY'
import json, sys, os
path, label, *pids = sys.argv[1:]
try:
    with open(path) as f: reg = json.load(f)
except Exception:
    reg = {}
for p in pids:
    reg[str(p)] = label
tmp = path + ".tmp"
with open(tmp, "w") as f:
    json.dump(reg, f, indent=2, sort_keys=True)
os.replace(tmp, path)
PY
}

# Diff: capture pids before (in $1), then call gpu_label_diff <label> <before_array>...
# Usage:
#   before=$(gpu_snapshot)
#   ... start service, wait healthy ...
#   after=$(gpu_snapshot)
#   gpu_label_diff "vLLM" "$before" "$after"
gpu_label_diff() {
    local label="$1"
    local before="$2"
    local after="$3"
    local new
    new=$(comm -13 <(echo "$before") <(echo "$after"))
    [ -z "$new" ] && return 0
    gpu_label_pids "$label" $new
}

# Remove labels for PIDs that no longer have GPU allocations (called by monitor too).
gpu_label_prune() {
    /workspace/meridian/venv/bin/python - "$REGISTRY" <<'PY'
import json, subprocess, sys, os
path = sys.argv[1]
try:
    with open(path) as f:
        reg = json.load(f)
except Exception:
    sys.exit(0)
out = subprocess.run(
    ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
    capture_output=True, text=True, timeout=5,
)
live_pids = {p.strip() for p in out.stdout.splitlines() if p.strip().isdigit()}
new_reg = {k: v for k, v in reg.items() if k in live_pids}
if new_reg != reg:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(new_reg, f, indent=2, sort_keys=True)
    os.replace(tmp, path)
PY
}
