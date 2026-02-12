#!/usr/bin/env bash
set -euo pipefail

MODEL="Qwen/Qwen2.5-14B-Instruct"
HOST="0.0.0.0"
BASE_PORT=8000

export TOKENIZERS_PARALLELISM=false
export HF_HOME="/home/jovyan/people/Glebov/synt_gen_2/hf_cache"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

mkdir -p "/home/jovyan/people/Glebov/synt_gen_2/logs"
mkdir -p "$HF_HOME"

for GPU in 0; do
  PORT=$((BASE_PORT + GPU))

  echo "Starting vLLM on GPU=$GPU PORT=$PORT ..."

  CUDA_VISIBLE_DEVICES=$GPU \
    vllm serve "$MODEL" \
      --host "$HOST" \
      --port "$PORT" \
      --tensor-parallel-size 1 \
      --pipeline-parallel-size 1 \
      --gpu-memory-utilization 0.90 \
      --max-model-len 4096 \
      --served-model-name "$MODEL" \
      > "/home/jovyan/people/Glebov/synt_gen_2/logs/vllm_gpu${GPU}.log" 2>&1 &
done

echo "All started. Tail logs: tail -f /home/jovyan/people/Glebov/synt_gen_2/logs/vllm_gpu*.log"