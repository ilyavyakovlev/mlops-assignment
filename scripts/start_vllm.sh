#!/usr/bin/env bash
#
# Start vLLM with Phase 1 configuration for Qwen3-30B-A3B on a single H100 80GB.
# Workload profile: 1.5–3K prompt tokens, short structured outputs, ~2-3 serial calls/request.
#
# Reference: https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html

set -euo pipefail

MODEL="${VLLM_MODEL:-Qwen/Qwen3-30B-A3B-Instruct-2507}"
UV="${HOME}/.local/bin/uv"

exec "$UV" run python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --host 0.0.0.0 \
    --port 8000 \
    --dtype bfloat16 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.92 \
    --max-num-seqs 64 \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    --disable-log-requests
#
# Flag rationale (for REPORT.md Phase 1):
#
#   --dtype bfloat16            H100 native dtype; better numerical stability than fp16
#   --max-model-len 8192        Covers 3K schema+prompt+output with headroom; smaller
#                               context = less KV cache pressure per slot
#   --gpu-memory-utilization 0.92  Leave ~6 GB buffer; MoE expert weights are large
#   --max-num-seqs 64           Starting point for concurrency; tune down if P95 climbs
#   --enable-prefix-caching     DB schemas repeat across requests → big KV cache hit rate
#   --enable-chunked-prefill    Better batching of variable-length prompts under load
#   --disable-log-requests      Reduces per-request logging overhead at high RPS
