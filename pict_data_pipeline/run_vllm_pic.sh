#!/usr/bin/env bash
set -euo pipefail

MODEL="Qwen/Qwen2.5-14B-Instruct"
HOST="0.0.0.0"
BASE_PORT=8000
MAX_NUM_SEQS=2

export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

mkdir -p "/home/jovyan/people/Glebov/synt_gen_2/logs_for_pic"

for GPU in 0 1 2; do
  PORT=$((BASE_PORT + GPU))

  echo "Starting vLLM on GPU=$GPU PORT=$PORT ..."

  CUDA_VISIBLE_DEVICES=$GPU \
    vllm serve "$MODEL" \
      --host "$HOST" \
      --port "$PORT" \
      --tensor-parallel-size 1 \
      --pipeline-parallel-size 1 \
      --gpu-memory-utilization 0.97 \
      --max-model-len 5200 \
      --max-num-seqs "$MAX_NUM_SEQS" \
      --served-model-name "$MODEL" \
      > "/home/jovyan/people/Glebov/synt_gen_2/logs_for_pic/vllm_gpu${GPU}.log" 2>&1 &
done

echo "All started. Tail logs: tail -f /home/jovyan/people/Glebov/synt_gen_2/logs_for_pic/vllm_gpu*.log"