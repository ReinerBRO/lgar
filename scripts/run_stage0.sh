#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/gemini/space/private/zjc/goals/lgar}"
ENV_PATH="${ENV_PATH:-/gemini/space/private/zjc/envs/zjc_env}"
LOG_DIR="${LOG_DIR:-/gemini/space/private/zjc/logs/lgar}"

mkdir -p "${LOG_DIR}"
cd "${PROJECT_ROOT}"
set +u
source "${ENV_PATH}/bin/activate"
set -u

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export HF_HOME="/gemini/space/private/zjc/hf_cache"
export TRANSFORMERS_CACHE="/gemini/space/private/zjc/hf_cache/transformers"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

python -m lgar_cpt.train_stage0 \
  --model-path /gemini/space/private/zjc/models/Qwen2.5-0.5B \
  --raw-data-dir /gemini/space/private/zjc/data/fineweb_edu_100BT-shuffled \
  --cache-dir /gemini/space/private/zjc/goals/lgar/data/qwen_fineweb_stage0 \
  --output-dir /gemini/space/private/zjc/goals/lgar \
  --target-cache-tokens 64000000 \
  --max-shards 2 \
  --seq-len 8192 \
  --short-window 1024 \
  --local-window 1024 \
  --label-audit-sequences 10000 \
  --label-audit-batch-size "${LABEL_AUDIT_BATCH_SIZE:-4}" \
  --steps 100 \
  --batch-size 1 \
  --eval-batches 4 \
  --short-limit 16 \
  --long-limit 8 \
  --dtype bf16 \
  --attn-implementation eager \
  --skip-downstream-eval
