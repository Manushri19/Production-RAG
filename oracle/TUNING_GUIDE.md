# Oracle Tuning & Benchmarking Guide

Oracle is designed for low-latency, synchronous query processing. Unlike Meridian, which optimizes for total throughput (pages per minute), Oracle optimizes for **Time-To-First-Token (TTFT)** and **Search Precision**. 

This document serves as a living diary to record your latency benchmarks, prompt engineering experiments, and hardware tuning decisions as you scale to production.

---

## 1. Search Precision vs. Latency Tradeoffs

The heaviest operation in the retrieval phase is the Cross-Encoder Reranker (`BAAI/bge-reranker-v2-m3`). The latency of this step scales linearly with `HYBRID_TOP_K` (the number of candidates Qdrant sends to the sidecar).

### Configuration Levers
- **`HYBRID_TOP_K`** (Default: 30): The number of raw chunks pulled from Qdrant via Reciprocal Rank Fusion (RRF). 
  - *Increase* (e.g., 50-100) to maximize recall (finding the needle in the haystack).
  - *Decrease* (e.g., 10-20) to aggressively drop API latency.
- **`FINAL_TOP_K`** (Default: 5): The absolute number of chunks injected into the vLLM prompt.
  - *Keep low* (<10) to prevent LLM context-window bloating and reduce generation latency.

### Benchmarking the Reranker (CPU vs GPU)
By default, the reranker sidecar runs on the CPU.
* **CPU Expectation**: ~150-250ms for 30 candidates.
* **GPU Expectation**: ~10-20ms for 30 candidates.

**To enable GPU for the Reranker:**
If sub-100ms retrieval latency is mandated, edit `docker/reranker.Dockerfile` to use a PyTorch CUDA base image (e.g., `pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime`), and update `docker-compose.yml` to attach the NVIDIA runtime.

---

## 2. LLM Latency & Generation (vLLM)

Because Oracle uses Server-Sent Events (SSE) streaming, the perceived speed for the end-user is dictated by **Time-To-First-Token (TTFT)**.

### Tuning Generation Speed
- **Prompt Size**: Passing 5 large chunks (e.g., 500 words each) results in a ~2,500 word prefix. vLLM is heavily optimized for prefix-caching, but massive prompts will still delay the TTFT.
- **Max Tokens (`VLLM_MAX_TOKENS`)**: Hard-cap this (e.g., 1024) to prevent runaway generations.
- **Temperature (`VLLM_TEMPERATURE`)**: Kept at `0.1`. Low temperature not only prevents hallucinations but can marginally improve sampling speed.

---

## 3. Experiment Log

Use this section to record your live benchmarks as you test Oracle against different hardware setups or user query sets.

### [Template] Experiment Name
* **Date**: YYYY-MM-DD
* **Hardware**: e.g., 1x AWS g5.xlarge (A10G) for vLLM, CPU for Oracle/Qdrant
* **Query Set**: 50 materials engineering questions
* **Config**: `HYBRID_TOP_K=30`, `FINAL_TOP_K=5`

**Results:**
- **Avg Retrieval Time (Qdrant + Rerank)**: ___ ms
- **Avg TTFT (Time-to-First-Token)**: ___ ms
- **Avg Total Response Time**: ___ seconds
- **Notes/Observations**: *e.g., Reranker on CPU bottlenecked concurrent requests; need to move to GPU or reduce HYBRID_TOP_K to 15.*

---

## 4. Prompt Evolution Diary

The system prompt in `oracle/generation/prompt_builder.py` is strict. If users report that the LLM is refusing to answer valid questions, or occasionally hallucinating, log the prompt tweaks here.

* **[YYYY-MM-DD] v1.0**: Initial strict prompt. Mandated `[N]` citations and exact verbatim extraction.
* *(Log future iterations here...)*
