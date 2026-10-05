"""
Init Command - Auto-detect GPU and generate configuration.

Detects available NVIDIA GPUs, calculates optimal settings for
Docling instances, vLLM memory utilization, and worker concurrency.
Generates a .env file with the computed values.
"""

import os
import json
import subprocess
import sys
from pathlib import Path


# Memory budget constants (in GB)
CUDA_OVERHEAD_GB = 3.0
OLLAMA_EMBEDDING_GB = 4.0
VLLM_MODEL_BASE_GB = 8.0  # Qwen3-VL-8B FP8
DOCLING_PER_INSTANCE_GB = 1.2
MIN_VLLM_HEADROOM_GB = 4.0  # Minimum free VRAM for KV cache


def detect_gpus() -> list:
    """
    Detect NVIDIA GPUs using nvidia-smi.

    Returns:
        List of dicts with gpu_index, name, memory_total_mb
    """
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )

        if result.returncode != 0:
            return []

        gpus = []
        for line in result.stdout.strip().split("\n"):
            if not line.strip():
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3:
                gpus.append({
                    "index": int(parts[0]),
                    "name": parts[1],
                    "memory_total_mb": int(parts[2]),
                    "memory_total_gb": round(int(parts[2]) / 1024, 1),
                })

        return gpus

    except FileNotFoundError:
        return []
    except Exception:
        return []


def calculate_config(gpus: list) -> dict:
    """
    Calculate optimal configuration based on available GPUs.

    Strategy:
    - All GPU services run on GPU 0 (or single GPU)
    - Docling instances share GPU memory with vLLM
    - Memory budget: CUDA overhead + Ollama + Docling instances + vLLM

    Returns:
        Dict with configuration values
    """
    if not gpus:
        # CPU-only fallback
        return {
            "docling_instances": 1,
            "docling_ports": "8001",
            "vllm_gpu_utilization": 0.0,
            "celery_concurrency": 1,
            "estimated_pages_per_min": 5,
            "gpu_name": "None (CPU mode)",
            "gpu_vram_gb": 0,
            "notes": "No GPU detected. Running in CPU mode (slow).",
        }

    # Use the primary GPU (largest VRAM)
    primary_gpu = max(gpus, key=lambda g: g["memory_total_mb"])
    vram_gb = primary_gpu["memory_total_gb"]

    # Calculate available VRAM for services
    available_gb = vram_gb - CUDA_OVERHEAD_GB - OLLAMA_EMBEDDING_GB

    if available_gb < 10:
        # Very small GPU - minimal config
        docling_instances = 1
        vllm_util = 0.25
        concurrency = 1
        est_pages = 15
    elif available_gb < 25:
        # Consumer GPU (24GB class: RTX 3090/4090)
        # Budget: ~13GB available after overhead
        # Docling: 2 instances = 2.4GB
        # vLLM: ~10GB
        docling_instances = 2
        vllm_util = 0.30
        concurrency = 2
        est_pages = 45
    elif available_gb < 40:
        # Mid-range (A100 40GB)
        docling_instances = 4
        vllm_util = 0.40
        concurrency = 3
        est_pages = 90
    elif available_gb < 70:
        # High-end (A100 80GB)
        docling_instances = 6
        vllm_util = 0.50
        concurrency = 3
        est_pages = 140
    else:
        # Premium (H100 80GB)
        docling_instances = 8
        vllm_util = 0.55
        concurrency = 3
        est_pages = 180

    # Multi-GPU scaling
    total_gpus = len(gpus)
    if total_gpus > 1:
        # Scale Docling instances and concurrency with GPU count
        # (vLLM stays on one GPU for now)
        docling_instances = min(docling_instances * total_gpus, 32)
        concurrency = min(concurrency * total_gpus, 12)
        est_pages = est_pages * total_gpus

    # Build port list
    base_port = 8001
    ports = ",".join(
        f"http://localhost:{base_port + i}"
        for i in range(docling_instances)
    )

    return {
        "docling_instances": docling_instances,
        "docling_ports": ports,
        "vllm_gpu_utilization": vllm_util,
        "celery_concurrency": concurrency,
        "estimated_pages_per_min": est_pages,
        "gpu_name": primary_gpu["name"],
        "gpu_vram_gb": vram_gb,
        "gpu_count": total_gpus,
        "notes": None,
    }


def generate_env_file(config: dict, output_path: Path):
    """Generate .env file with computed configuration."""
    lines = [
        "# Meridian Configuration",
        f"# Auto-generated for: {config['gpu_name']} ({config.get('gpu_vram_gb', 0)}GB)",
        f"# Estimated throughput: ~{config['estimated_pages_per_min']} pages/min",
        "",
        "# --- Docling Service ---",
        f"DOCLING_INSTANCES={config['docling_ports']}",
        "",
        "# --- vLLM / Inference ---",
        "# For local GPU: use localhost vLLM",
        "# For API providers: set URL and key (e.g. OpenRouter, OpenAI, Together)",
        "VLLM_API_URL=http://localhost:8000/v1/chat/completions",
        "VLLM_MODEL=Qwen/Qwen3-VL-8B-Instruct",
        "VLLM_API_KEY=EMPTY",
        f"VLLM_GPU_UTILIZATION={config['vllm_gpu_utilization']}",
        "",
        "# --- Embedding ---",
        "OLLAMA_URL=http://localhost:11434",
        "EMBEDDING_MODEL=qwen3-embedding:4b-q8_0",
        "",
        "# --- Vector Store ---",
        "QDRANT_HOST=localhost",
        "QDRANT_PORT=6333",
        "",
        "# --- Redis ---",
        "REDIS_URL=redis://localhost:6379/0",
        "",
        "# --- Workers ---",
        f"CELERY_CONCURRENCY={config['celery_concurrency']}",
        "",
        "# --- Image Extraction ---",
        "DOCLING_IMAGE_SCALE=2.0",
        "",
    ]

    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")


def cmd_init(args):
    """Handle the init command."""
    print("Detecting GPU configuration...")
    print()

    gpus = detect_gpus()

    if not gpus:
        print("No NVIDIA GPU detected.")
        print("Meridian works best with a GPU but can run in CPU mode (slow).")
        print()
    else:
        print(f"Found {len(gpus)} GPU(s):")
        for gpu in gpus:
            print(f"  [{gpu['index']}] {gpu['name']} ({gpu['memory_total_gb']} GB)")
        print()

    config = calculate_config(gpus)

    print("Recommended configuration:")
    print(f"  Docling instances:  {config['docling_instances']}")
    print(f"  vLLM utilization:   {config['vllm_gpu_utilization']}")
    print(f"  Worker concurrency: {config['celery_concurrency']}")
    print(f"  Est. throughput:    ~{config['estimated_pages_per_min']} pages/min")
    print()

    if config.get("notes"):
        print(f"Note: {config['notes']}")
        print()

    # Write .env file
    env_path = Path(".env")
    if env_path.exists() and not args.force:
        print(f".env file already exists. Use --force to overwrite.")
        sys.exit(1)

    generate_env_file(config, env_path)
    print(f"Configuration written to {env_path}")
    print()
    print("Next steps:")
    print("  1. Review .env and adjust if needed")
    print("  2. meridian start    # Start all services")
    print("  3. meridian parse <file_or_dir>  # Process documents")
