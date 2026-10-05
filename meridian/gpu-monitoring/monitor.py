#!/usr/bin/env python3
"""
Meridian live dashboard.

Replaces `watch nvidia-smi` with a view that actually reflects what Meridian
is doing: GPU + per-service status (PID, CPU, RSS, uptime), Docling instance
health, Qdrant collections, recent processing outputs, and live log tails.

Usage:
    /workspace/meridian/venv/bin/python /workspace/meridian/monitor.py
    # or: ./monitor.py once    (single-shot, no live refresh)
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import httpx
from rich.align import Align
from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TextColumn
from rich.table import Table
from rich.text import Text

WORK = Path("/workspace/meridian")
LOG_DIR = WORK / "logs"
OUTPUT_DIR = WORK / "output"
DOCLING_PORTS = list(range(9001, 9013))
DOCLING_HEALTH_CACHE: dict[int, tuple[float, bool]] = {}

# -- helpers ----------------------------------------------------------------


def _run(cmd: list[str], timeout: float = 2.0) -> Optional[str]:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if r.returncode == 0:
            return r.stdout
    except Exception:
        pass
    return None


def gpu_processes() -> list[dict]:
    """Per-process VRAM via nvidia-smi. PIDs are HOST PIDs and may not exist
    in this container's /proc — we still surface them so users can compare to
    GPU panel totals."""
    out = _run([
        "nvidia-smi",
        "--query-compute-apps=pid,used_memory",
        "--format=csv,noheader,nounits",
    ])
    if not out:
        return []
    procs = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2 and parts[0].isdigit():
            procs.append({"pid": int(parts[0]), "vram_mb": int(parts[1])})
    procs.sort(key=lambda x: -x["vram_mb"])
    return procs


REGISTRY_PATH = WORK / "gpu_labels.json"


def assign_gpu_proc_labels(procs: list[dict]) -> dict[int, str]:
    """Read /workspace/meridian/gpu_labels.json — a registry maintained by
    the launch scripts (start_vllm.sh, start_docling.sh, start_ollama.sh,
    start_watchdog.sh). Each service writes its host PID after CUDA init.
    Anything not in the registry shows as "?" — no guessing."""
    labels: dict[int, str] = {}
    try:
        import json as _json
        with open(REGISTRY_PATH) as f:
            raw = _json.load(f)
        for k, v in raw.items():
            try:
                labels[int(k)] = str(v)
            except (ValueError, TypeError):
                continue
    except Exception:
        pass
    return labels


def gpu_stats() -> dict:
    out = _run([
        "nvidia-smi",
        "--query-gpu=name,memory.used,memory.total,utilization.gpu,power.draw,power.limit,temperature.gpu",
        "--format=csv,noheader,nounits",
    ])
    if not out:
        return {}
    parts = [p.strip() for p in out.strip().split(",")]
    if len(parts) < 7:
        return {}
    return {
        "name": parts[0],
        "mem_used_mb": int(parts[1]),
        "mem_total_mb": int(parts[2]),
        "util_pct": int(parts[3]),
        "power_w": float(parts[4]) if parts[4] != "[N/A]" else 0.0,
        "power_limit_w": float(parts[5]) if parts[5] != "[N/A]" else 0.0,
        "temp_c": int(parts[6]) if parts[6] != "[N/A]" else 0,
    }


def proc_stats(pid: int) -> Optional[dict]:
    """psutil-free: read /proc/<pid>/stat and statm."""
    try:
        with open(f"/proc/{pid}/status") as f:
            status = dict(line.split(":\t", 1) for line in f if ":\t" in line)
        rss_kb = int(status.get("VmRSS", "0 kB").split()[0])
        with open(f"/proc/{pid}/stat") as f:
            stat = f.read().split()
        # fields: 14=utime, 15=stime, 22=starttime
        utime = int(stat[13])
        stime = int(stat[14])
        starttime = int(stat[21])
        with open("/proc/uptime") as f:
            uptime_s = float(f.read().split()[0])
        with open("/proc/stat") as f:
            btime_line = next(line for line in f if line.startswith("btime "))
        clk_tck = os.sysconf("SC_CLK_TCK")
        proc_age_s = uptime_s - (starttime / clk_tck)
        cpu_seconds = (utime + stime) / clk_tck
        cpu_pct = (cpu_seconds / proc_age_s * 100) if proc_age_s > 0 else 0
        return {
            "rss_mb": rss_kb / 1024,
            "cpu_pct": cpu_pct,
            "uptime_s": proc_age_s,
        }
    except Exception:
        return None


def fmt_uptime(s: float) -> str:
    s = int(s)
    if s < 60:
        return f"{s}s"
    m, s = divmod(s, 60)
    if m < 60:
        return f"{m}m{s:02d}s"
    h, m = divmod(m, 60)
    if h < 24:
        return f"{h}h{m:02d}m"
    d, h = divmod(h, 24)
    return f"{d}d{h:02d}h"


def fmt_mb(mb: float) -> str:
    if mb >= 1024:
        return f"{mb/1024:.1f} GB"
    return f"{int(mb)} MB"


def http_ok(url: str, want: Optional[str] = None, timeout: float = 1.0) -> Optional[str]:
    try:
        r = httpx.get(url, timeout=timeout)
        if r.status_code == 200:
            return r.text
        return None
    except Exception:
        return None


def find_pid(pattern: str, comm: Optional[str] = None) -> Optional[int]:
    """Match a pid via cmdline pattern. If `comm` is given, also require that
    /proc/<pid>/comm equals it — filters out shell wrappers that contain the
    pattern in their argv."""
    out = _run(["pgrep", "-f", pattern])
    if not out:
        return None
    pids = [int(x) for x in out.split() if x.strip().isdigit()]
    if comm:
        filtered = []
        for p in pids:
            try:
                with open(f"/proc/{p}/comm") as f:
                    if f.read().strip() == comm:
                        filtered.append(p)
            except OSError:
                continue
        pids = filtered
    if not pids:
        return None
    return min(pids)


def read_pidfile(path: Path) -> Optional[int]:
    try:
        pid = int(path.read_text().strip())
        os.kill(pid, 0)
        return pid
    except Exception:
        return None


# -- service descriptors ----------------------------------------------------


@dataclass
class Service:
    name: str
    port: Optional[int]
    pid: Optional[int]
    status: str
    note: str = ""


def collect_services() -> list[Service]:
    out: list[Service] = []

    # Redis
    pid = find_pid(r"redis-server", comm="redis-server")
    redis_ok = bool(_run(["redis-cli", "ping"]))
    out.append(Service("Redis", 6379, pid, "OK" if redis_ok else "DOWN"))

    # Qdrant — comm filter excludes the bash launcher
    pid = find_pid(r"qdrant", comm="qdrant")
    res = http_ok("http://127.0.0.1:6333/collections")
    out.append(Service("Qdrant", 6333, pid, "OK" if res else "DOWN"))

    # Ollama
    pid = find_pid(r"ollama serve", comm="ollama")
    res = http_ok("http://127.0.0.1:11434/api/tags")
    note = ""
    if res:
        try:
            import json as _json
            data = _json.loads(res)
            note = ", ".join(m["name"] for m in data.get("models", []))
        except Exception:
            pass
    out.append(Service("Ollama", 11434, pid, "OK" if res else "DOWN", note))

    # vLLM
    pid = find_pid(r"vllm.entrypoints.openai.api_server")
    res = http_ok("http://127.0.0.1:8000/v1/models")
    note = ""
    if res:
        m = re.search(r'"id":"([^"]+)"', res)
        if m:
            note = m.group(1)
    out.append(Service("vLLM", 8000, pid, "OK" if res else "DOWN", note))

    # Docling — count healthy and report aggregate
    healthy = 0
    docling_pids = []
    now = time.time()
    for port in DOCLING_PORTS:
        cached = DOCLING_HEALTH_CACHE.get(port)
        if cached and now - cached[0] < 5:  # cache 5s
            ok = cached[1]
        else:
            res = http_ok(f"http://127.0.0.1:{port}/health", timeout=0.5)
            ok = bool(res and '"models_loaded":true' in res)
            DOCLING_HEALTH_CACHE[port] = (now, ok)
        if ok:
            healthy += 1
        pf = LOG_DIR / f"docling_{port}.pid"
        p = read_pidfile(pf)
        if p:
            docling_pids.append(p)
    overall_pid = docling_pids[0] if docling_pids else None
    out.append(
        Service(
            "Docling",
            None,
            overall_pid,
            "OK" if healthy == len(DOCLING_PORTS) else f"{healthy}/{len(DOCLING_PORTS)}",
            f"{len(docling_pids)} procs, ports {DOCLING_PORTS[0]}-{DOCLING_PORTS[-1]}",
        )
    )

    return out


# -- panels -----------------------------------------------------------------


def render_gpu(stats: dict, procs: list[dict]) -> Panel:
    if not stats:
        return Panel(Text("nvidia-smi unavailable", style="red"), title="GPU")
    used = stats["mem_used_mb"]
    total = stats["mem_total_mb"]
    pct = used / total * 100 if total else 0
    bar = Progress(
        TextColumn("[bold cyan]VRAM"),
        BarColumn(bar_width=None),
        TextColumn(f"{used:>5,d} / {total:,d} MB ({pct:.1f}%)"),
        expand=True,
    )
    bar.add_task("", total=total, completed=used)

    util_color = "green" if stats["util_pct"] < 50 else "yellow" if stats["util_pct"] < 85 else "red"
    line2 = Text.assemble(
        ("util ", "dim"), (f"{stats['util_pct']}%", util_color),
        ("   power ", "dim"), (f"{stats['power_w']:.0f}W / {stats['power_limit_w']:.0f}W", "magenta"),
        ("   temp ", "dim"), (f"{stats['temp_c']}°C", "blue"),
        ("   procs ", "dim"), (f"{len(procs)}", "cyan"),
    )

    # Per-process VRAM table (host PIDs from NVML — show all)
    if procs:
        labels = assign_gpu_proc_labels(procs)
        proc_table = Table(box=None, show_header=True, expand=True, header_style="bold dim")
        proc_table.add_column("Process", style="cyan", overflow="ellipsis")
        proc_table.add_column("Host PID", justify="right", style="dim")
        proc_table.add_column("VRAM", justify="right")
        proc_table.add_column("% of GPU", justify="right", style="dim")
        # Display order: vLLM, Ollama, then Docling sorted by port (or label string)
        def sort_key(p):
            label = labels.get(p["pid"], "?")
            if label == "vLLM":
                return (0, "")
            if label.startswith("Ollama"):
                return (1, "")
            if label.startswith("Docling"):
                return (2, label)  # natural sort by port substring
            return (3, label)
        sorted_procs = sorted(procs, key=sort_key)
        for p in sorted_procs:
            vram_mb = p["vram_mb"]
            proc_table.add_row(
                labels.get(p["pid"], "?"),
                str(p["pid"]),
                fmt_mb(vram_mb),
                f"{vram_mb / total * 100:.1f}%" if total else "—",
            )
        body = Group(bar, line2, Text(""), proc_table)
    else:
        body = Group(bar, line2)

    return Panel(
        body,
        title=f"[bold]{stats['name']}[/bold]",
        border_style="cyan",
    )


STATUS_STYLE = {"OK": "bold green", "DOWN": "bold red"}


def render_services(services: Iterable[Service]) -> Panel:
    table = Table(box=None, expand=True, show_header=True, header_style="bold")
    table.add_column("Service", style="cyan", no_wrap=True)
    table.add_column("Port", justify="right")
    table.add_column("Status", justify="center")
    table.add_column("PID", justify="right")
    table.add_column("CPU%", justify="right")
    table.add_column("RAM", justify="right")
    table.add_column("Up", justify="right")
    table.add_column("Note", overflow="ellipsis", no_wrap=True)

    for s in services:
        style = STATUS_STYLE.get(s.status, "yellow")
        cpu = rss = up = "—"
        if s.pid:
            ps = proc_stats(s.pid)
            if ps:
                cpu = f"{ps['cpu_pct']:.1f}"
                rss = fmt_mb(ps["rss_mb"])
                up = fmt_uptime(ps["uptime_s"])
        table.add_row(
            s.name,
            str(s.port) if s.port else "—",
            Text(s.status, style=style),
            str(s.pid) if s.pid else "—",
            cpu,
            rss,
            up,
            s.note,
        )
    return Panel(table, title="Services", border_style="green")


def qdrant_summary() -> Optional[Panel]:
    res = http_ok("http://127.0.0.1:6333/collections")
    if not res:
        return None
    try:
        import json as _json
        cols = _json.loads(res).get("result", {}).get("collections", [])
    except Exception:
        return None
    if not cols:
        return Panel(Text("no collections yet", style="dim"), title="Qdrant", border_style="magenta")

    table = Table(box=None, expand=True)
    table.add_column("Collection", style="cyan")
    table.add_column("Points", justify="right")
    table.add_column("Status")
    for c in cols:
        name = c["name"]
        info = http_ok(f"http://127.0.0.1:6333/collections/{name}")
        pts = "?"
        st = "?"
        if info:
            try:
                d = _json.loads(info)["result"]
                pts = f"{d.get('points_count', 0):,}"
                st = d.get("status", "?")
            except Exception:
                pass
        st_style = "green" if st == "green" else "yellow"
        table.add_row(name, pts, Text(st, style=st_style))
    return Panel(table, title="Qdrant", border_style="magenta")


def output_summary() -> Panel:
    if not OUTPUT_DIR.exists():
        return Panel(Text("no output yet", style="dim"), title="Recent output", border_style="yellow")
    files = sorted(OUTPUT_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:8]
    if not files:
        return Panel(Text("no output yet", style="dim"), title="Recent output", border_style="yellow")
    table = Table(box=None, expand=True)
    table.add_column("File", style="cyan", overflow="ellipsis", no_wrap=True)
    table.add_column("Size", justify="right")
    table.add_column("Modified", justify="right", style="dim")
    for f in files:
        st = f.stat()
        table.add_row(
            f.name,
            fmt_mb(st.st_size / 1024 / 1024 * 1024 / 1024) if st.st_size > 1024 * 1024 else f"{st.st_size//1024} KB",
            time.strftime("%H:%M:%S", time.localtime(st.st_mtime)),
        )
    return Panel(table, title=f"Recent output ({len(list(OUTPUT_DIR.glob('*.json')))} total)", border_style="yellow")


# Log tails — keep the last N lines per source, drop noise


_LOG_BUFFERS: dict[str, deque] = {}
_LOG_OFFSETS: dict[str, int] = {}


def _log_sources() -> list[tuple[str, Path]]:
    sources: list[tuple[str, Path]] = []
    if (LOG_DIR / "vllm.log").exists():
        sources.append(("vllm", LOG_DIR / "vllm.log"))
    if (LOG_DIR / "qdrant.log").exists():
        sources.append(("qdrant", LOG_DIR / "qdrant.log"))
    if (LOG_DIR / "ollama.log").exists():
        sources.append(("ollama", LOG_DIR / "ollama.log"))
    # Docling: pick the most recently modified log so a single tail gives a feel
    docling_logs = sorted(LOG_DIR.glob("docling_*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
    if docling_logs:
        sources.append(("docling", docling_logs[0]))
    return sources


_NOISE_RE = re.compile(
    r"GET /health|GET /stats|GET /collections|GET /v1/models|GET /api/tags"
    r"|metrics|Capturing CUDA graph|loading plugin|^$",
    re.IGNORECASE,
)


def _tail_new(path: Path) -> list[str]:
    key = str(path)
    try:
        size = path.stat().st_size
    except OSError:
        return []
    last = _LOG_OFFSETS.get(key, max(0, size - 4096))
    if last > size:
        last = 0  # rotated
    if last == size:
        return []
    try:
        with open(path, "rb") as f:
            f.seek(last)
            chunk = f.read(size - last)
    except OSError:
        return []
    _LOG_OFFSETS[key] = size
    text = chunk.decode("utf-8", errors="replace")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return [ln for ln in lines if not _NOISE_RE.search(ln)]


def render_logs() -> Panel:
    for label, path in _log_sources():
        buf = _LOG_BUFFERS.setdefault(label, deque(maxlen=20))
        for line in _tail_new(path):
            buf.append((label, line))

    flat: list[tuple[str, str]] = []
    for buf in _LOG_BUFFERS.values():
        flat.extend(list(buf)[-5:])
    flat = flat[-15:]

    if not flat:
        body = Text("no recent log activity", style="dim")
    else:
        out = Text()
        for label, line in flat:
            color = {"vllm": "magenta", "docling": "cyan", "qdrant": "yellow", "ollama": "green"}.get(label, "white")
            out.append(f"[{label:>7}] ", style=f"bold {color}")
            # truncate long lines
            out.append(line[:160] + ("…" if len(line) > 160 else ""))
            out.append("\n")
        body = out
    return Panel(body, title="Recent log activity (filtered)", border_style="white")


# -- driver ----------------------------------------------------------------


def render_dashboard() -> Layout:
    procs = gpu_processes()
    # Top panel grows with the number of GPU procs (3 rows of header + ~1 per proc)
    top_size = max(7, 6 + min(len(procs), 16))
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="top", size=top_size),
        Layout(name="middle", size=14),
        Layout(name="bottom"),
    )
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    layout["header"].update(
        Panel(
            Align.center(Text(f"Meridian Live  •  {ts}  •  ctrl-c to exit", style="bold")),
            border_style="bright_blue",
        )
    )
    layout["top"].update(render_gpu(gpu_stats(), procs))
    layout["middle"].update(render_services(collect_services()))
    layout["bottom"].split_row(
        Layout(name="qd"),
        Layout(name="out"),
        Layout(name="logs", ratio=2),
    )
    qd = qdrant_summary() or Panel(Text("Qdrant down", style="red"), title="Qdrant")
    layout["bottom"]["qd"].update(qd)
    layout["bottom"]["out"].update(output_summary())
    layout["bottom"]["logs"].update(render_logs())
    return layout


def main():
    once = len(sys.argv) > 1 and sys.argv[1] == "once"
    console = Console()
    if once:
        console.print(render_dashboard())
        return

    refresh_hz = 2
    try:
        with Live(render_dashboard(), console=console, refresh_per_second=refresh_hz, screen=True) as live:
            while True:
                time.sleep(1.0 / refresh_hz)
                live.update(render_dashboard())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
