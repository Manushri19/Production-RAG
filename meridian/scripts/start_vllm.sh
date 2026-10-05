#!/bin/bash
# Start vLLM with optimized configuration
#
# Changes from previous config:
#   gpu-memory-utilization: 0.35 -> 0.55
#   max-num-seqs: 16 -> 128
#   Added: enable-chunked-prefill, enable-prefix-caching

set -e

echo "=== Starting vLLM ==="
echo ""

# Stop existing container if running
if docker ps -q -f name=vllm | grep -q .; then
    echo "Stopping existing vLLM container..."
    docker stop vllm
fi

# Remove container if exists
if docker ps -aq -f name=vllm | grep -q .; then
    echo "Removing existing vLLM container..."
    docker rm vllm
fi

echo ""
echo "Starting vLLM with optimized config..."
echo "  - gpu-memory-utilization: 0.55 (was 0.35)"
echo "  - max-num-seqs: 128 (was 16)"
echo "  - enable-chunked-prefill: true"
echo "  - enable-prefix-caching: true"
echo ""

docker run -d --name vllm \
    --gpus all \
    -p 8000:8000 \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    vllm/vllm-openai:latest \
    --model Qwen/Qwen3-VL-8B-Instruct \
    --quantization fp8 \
    --gpu-memory-utilization 0.55 \
    --max-num-seqs 128 \
    --max-model-len 8192 \
    --enable-chunked-prefill \
    --enable-prefix-caching

echo ""
echo "vLLM container started. Waiting for model to load..."
echo "Check status with: docker logs -f vllm"
echo ""
echo "To verify ready, run:"
echo "  curl http://localhost:8000/v1/models"
