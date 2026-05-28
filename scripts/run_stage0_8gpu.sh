#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/gemini/space/private/zjc/goals/lgar}"
ENV_PATH="${ENV_PATH:-/gemini/space/private/zjc/envs/zjc_env}"
LOG_DIR="${LOG_DIR:-/gemini/space/private/zjc/logs/lgar}"
AUDIT_DIR="${AUDIT_DIR:-/gemini/space/private/zjc/goals/lgar/reports/label_audit_8gpu_shards}"
AUDIT_JSON="${AUDIT_JSON:-/gemini/space/private/zjc/goals/lgar/reports/label_audit_8gpu.json}"
TOTAL_SEQUENCES="${TOTAL_SEQUENCES:-10000}"
NUM_SHARDS="${NUM_SHARDS:-8}"
SEQ_PER_SHARD="$(( (TOTAL_SEQUENCES + NUM_SHARDS - 1) / NUM_SHARDS ))"

mkdir -p "${LOG_DIR}" "${AUDIT_DIR}"
rm -f "${AUDIT_DIR}"/shard_*.json "${LOG_DIR}"/label_audit_shard_*.log
cd "${PROJECT_ROOT}"
set +u
source "${ENV_PATH}/bin/activate"
set -u
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export HF_HOME="/gemini/space/private/zjc/hf_cache"
export TRANSFORMERS_CACHE="/gemini/space/private/zjc/hf_cache/transformers"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [[ "${REUSE_LABEL_AUDIT:-1}" == "1" && -s "${AUDIT_JSON}" ]]; then
  echo "{\"kind\":\"reuse_label_audit\",\"path\":\"${AUDIT_JSON}\"}"
else
  pids=()
  for shard in $(seq 0 $((NUM_SHARDS - 1))); do
    (
      export CUDA_VISIBLE_DEVICES="${shard}"
      python -m lgar_cpt.label_audit_shard \
        --model-path /gemini/space/private/zjc/models/Qwen2.5-0.5B \
        --cache-dir /gemini/space/private/zjc/goals/lgar/data/qwen_fineweb_stage0 \
        --output "${AUDIT_DIR}/shard_${shard}.json" \
        --shard-id "${shard}" \
        --num-shards "${NUM_SHARDS}" \
        --sequences "${SEQ_PER_SHARD}" \
        --batch-size "${LABEL_AUDIT_BATCH_SIZE:-4}" \
        --seq-len 8192 \
        --short-window 1024 \
        --local-window 1024 \
        --dtype bf16 \
        --attn-implementation eager \
        --seed 20260524
    ) > "${LOG_DIR}/label_audit_shard_${shard}.log" 2>&1 &
    pids+=("$!")
  done

  for pid in "${pids[@]}"; do
    wait "$pid"
  done

  python -m lgar_cpt.merge_label_audit \
    --input-dir "${AUDIT_DIR}" \
    --output "${AUDIT_JSON}" \
    --expected-shards "${NUM_SHARDS}"
fi

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
  --label-audit-json "${AUDIT_JSON}" \
  --steps 100 \
  --batch-size 1 \
  --eval-batches 4 \
  --short-limit 16 \
  --long-limit 8 \
  --dtype bf16 \
  --attn-implementation eager \
  --skip-downstream-eval
