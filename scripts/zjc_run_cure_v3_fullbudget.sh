#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/zjc_env.sh
source "${SCRIPT_DIR}/zjc_env.sh"
cd "${PROJECT_ROOT}"

TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_NAME="${RUN_NAME:-cure_v3_fullbudget_top24_fullhu10_alr1e4}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/runs/cure_v3_fullbudget_${TIMESTAMP}}"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/logs}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/zjc_cure_v3_fullbudget_${TIMESTAMP}.log}"
TOKENS_PER_RANK="${TOKENS_PER_RANK:-12500000}"
BATCH_SIZE="${BATCH_SIZE:-2}"
LOCAL_WINDOW="${LOCAL_WINDOW:-1024}"
SEQ_LEN="${SEQ_LEN:-4096}"

mkdir -p "${OUTPUT_ROOT}" "${LOG_DIR}"
exec > >(tee -a "${LOG_FILE}") 2>&1

if [[ -n "${GLOBAL_TOKENS:-}" ]]; then
  TOKENS_PER_RANK=$((GLOBAL_TOKENS / NPROC_PER_NODE))
fi
GLOBAL_TOKEN_ESTIMATE=$((TOKENS_PER_RANK * NPROC_PER_NODE))

echo "=== ZJC CURE-v3 full-budget train ==="
echo "timestamp=${TIMESTAMP}"
echo "project_root=${PROJECT_ROOT}"
echo "python=${PYTHON}"
echo "nproc_per_node=${NPROC_PER_NODE}"
echo "tokens_per_rank=${TOKENS_PER_RANK}"
echo "global_token_estimate=${GLOBAL_TOKEN_ESTIMATE}"
echo "run_name=${RUN_NAME}"
echo "output_root=${OUTPUT_ROOT}"
echo "log_file=${LOG_FILE}"
date -Is
if [[ "${SHOW_GPU_INFO:-0}" == "1" ]]; then
  nvidia-smi || true
fi

require_dir "model" "${MODEL_PATH}"
require_dir "token cache" "${CACHE_DIR}"
require_file "ce checkpoint" "${CE_CKPT}"
require_file "top heads json" "${TOP_HEADS_JSON}"

"${PYTHON}" -m torch.distributed.run --standalone --nproc_per_node="${NPROC_PER_NODE}" -m curcpt.train \
  --run-name "${RUN_NAME}" \
  --method cure_cpt \
  --model-path "${MODEL_PATH}" \
  --cache-dir "${CACHE_DIR}" \
  --output-dir "${OUTPUT_ROOT}" \
  --checkpoint-path "${CE_CKPT}" \
  --ablation-results "${TOP_HEADS_JSON}" \
  --seq-len "${SEQ_LEN}" \
  --local-window "${LOCAL_WINDOW}" \
  --batch-size "${BATCH_SIZE}" \
  --tokens-per-run "${TOKENS_PER_RANK}" \
  --lr 0.0 \
  --adapter-lr "${ADAPTER_LR:-1e-4}" \
  --adapter-weight-decay "${ADAPTER_WEIGHT_DECAY:-0.0}" \
  --freeze-base-model \
  --lambda-rh-ce "${LAMBDA_RH_CE:-0.0}" \
  --lambda-rh-kd "${LAMBDA_RH_KD:-0.0}" \
  --lambda-cov "${LAMBDA_COV:-0.0}" \
  --lambda-full-hu-ce "${LAMBDA_FULL_HU_CE:-1.0}" \
  --lora-rank "${LORA_RANK:-16}" \
  --lora-alpha "${LORA_ALPHA:-32.0}" \
  --eval-interval "${EVAL_INTERVAL:-50}" \
  --seed "${SEED:-1337}" \
  --dtype "${DTYPE}" \
  --attn-implementation "${ATTN_IMPLEMENTATION}"

CKPT="${OUTPUT_ROOT}/runs/cure/${RUN_NAME}/checkpoint.pt"
echo "checkpoint=${CKPT}"
echo "=== ZJC CURE-v3 train done ==="
date -Is
