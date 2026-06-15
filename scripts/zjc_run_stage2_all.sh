#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/zjc_env.sh
source "${SCRIPT_DIR}/zjc_env.sh"
cd "${PROJECT_ROOT}"

STAGE2_TAG="${STAGE2_TAG:-$(date +%Y%m%d_%H%M%S)}"
GLOBAL_TOKENS="${GLOBAL_TOKENS:-2000000000}"
TOKENS_PER_RANK="${TOKENS_PER_RANK:-$((GLOBAL_TOKENS / NPROC_PER_NODE))}"
SEQ_LEN="${SEQ_LEN:-8192}"
BATCH_SIZE="${BATCH_SIZE:-1}"
RUN_CE="${RUN_CE:-1}"
RUN_LONGCE="${RUN_LONGCE:-1}"
RUN_CURE="${RUN_CURE:-1}"
RUN_GATE_EVAL="${RUN_GATE_EVAL:-1}"
AUTO_PREPARE_CACHE="${AUTO_PREPARE_CACHE:-1}"
CACHE_TARGET_TOKENS="${CACHE_TARGET_TOKENS:-2200000000}"

LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/logs}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/zjc_stage2_all_${STAGE2_TAG}.log}"
mkdir -p "${LOG_DIR}"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "=== ZJC Stage2 all train ==="
echo "stage2_tag=${STAGE2_TAG}"
echo "global_tokens=${GLOBAL_TOKENS}"
echo "tokens_per_rank=${TOKENS_PER_RANK}"
echo "nproc_per_node=${NPROC_PER_NODE}"
echo "seq_len=${SEQ_LEN}"
echo "batch_size_per_rank=${BATCH_SIZE}"
echo "run_ce=${RUN_CE} run_longce=${RUN_LONGCE} run_cure=${RUN_CURE}"
echo "run_gate_eval=${RUN_GATE_EVAL}"
echo "raw_data_dir=${RAW_DATA_DIR}"
echo "cache_dir=${CACHE_DIR}"
echo "auto_prepare_cache=${AUTO_PREPARE_CACHE}"
echo "cache_target_tokens=${CACHE_TARGET_TOKENS}"
echo "log_file=${LOG_FILE}"
date -Is

if [[ ! -f "${CACHE_DIR}/cache_info.json" || ! -f "${CACHE_DIR}/tokens.npy" ]]; then
  if [[ "${AUTO_PREPARE_CACHE}" != "1" ]]; then
    echo "missing stage2 token cache: ${CACHE_DIR}" >&2
    echo "Run: TARGET_TOKENS=${CACHE_TARGET_TOKENS} OUTPUT_CACHE_DIR=${CACHE_DIR} bash scripts/zjc_prepare_fineweb_cache.sh" >&2
    exit 2
  fi
  TARGET_TOKENS="${CACHE_TARGET_TOKENS}" OUTPUT_CACHE_DIR="${CACHE_DIR}" \
    bash "${SCRIPT_DIR}/zjc_prepare_fineweb_cache.sh"
fi

if [[ "${RUN_CE}" == "1" ]]; then
  CE_RUN_NAME="stage2_ce_cpt_${STAGE2_TAG}"
  CE_OUTPUT_ROOT="${PROJECT_ROOT}/runs/stage2_ce_cpt_${STAGE2_TAG}"
  RUN_NAME="${CE_RUN_NAME}" OUTPUT_ROOT="${CE_OUTPUT_ROOT}" METHOD=ce_cpt \
    STAGE2_TAG="${STAGE2_TAG}" GLOBAL_TOKENS="${GLOBAL_TOKENS}" TOKENS_PER_RANK="${TOKENS_PER_RANK}" \
    SEQ_LEN="${SEQ_LEN}" BATCH_SIZE="${BATCH_SIZE}" \
    bash "${SCRIPT_DIR}/zjc_run_stage2_train_one.sh"
  export CE_CKPT="${CE_OUTPUT_ROOT}/runs/cure/${CE_RUN_NAME}/checkpoint.pt"
fi

if [[ "${RUN_LONGCE}" == "1" ]]; then
  LONGCE_RUN_NAME="stage2_longce_cpt_${STAGE2_TAG}"
  LONGCE_OUTPUT_ROOT="${PROJECT_ROOT}/runs/stage2_longce_cpt_${STAGE2_TAG}"
  RUN_NAME="${LONGCE_RUN_NAME}" OUTPUT_ROOT="${LONGCE_OUTPUT_ROOT}" METHOD=longce_cpt \
    STAGE2_TAG="${STAGE2_TAG}" GLOBAL_TOKENS="${GLOBAL_TOKENS}" TOKENS_PER_RANK="${TOKENS_PER_RANK}" \
    SEQ_LEN="${SEQ_LEN}" BATCH_SIZE="${BATCH_SIZE}" \
    bash "${SCRIPT_DIR}/zjc_run_stage2_train_one.sh"
fi

if [[ "${RUN_CURE}" == "1" ]]; then
  CURE_RUN_NAME="stage2_cure_v3_${STAGE2_TAG}"
  CURE_OUTPUT_ROOT="${PROJECT_ROOT}/runs/stage2_cure_v3_${STAGE2_TAG}"
  RUN_NAME="${CURE_RUN_NAME}" OUTPUT_ROOT="${CURE_OUTPUT_ROOT}" METHOD=cure_cpt \
    STAGE2_TAG="${STAGE2_TAG}" GLOBAL_TOKENS="${GLOBAL_TOKENS}" TOKENS_PER_RANK="${TOKENS_PER_RANK}" \
    SEQ_LEN="${SEQ_LEN}" BATCH_SIZE="${BATCH_SIZE}" CE_CKPT="${CE_CKPT}" \
    bash "${SCRIPT_DIR}/zjc_run_stage2_train_one.sh"
  export CURE_CKPT="${CURE_OUTPUT_ROOT}/runs/cure/${CURE_RUN_NAME}/checkpoint.pt"
fi

if [[ "${RUN_GATE_EVAL}" == "1" ]]; then
  if [[ -n "${LONGCE_OUTPUT_ROOT:-}" && -n "${LONGCE_RUN_NAME:-}" ]]; then
    export LONGCE_CKPT="${LONGCE_OUTPUT_ROOT}/runs/cure/${LONGCE_RUN_NAME}/checkpoint.pt"
  fi
  if [[ -n "${CURE_OUTPUT_ROOT:-}" && -n "${CURE_RUN_NAME:-}" ]]; then
    export CURE_CKPT="${CURE_OUTPUT_ROOT}/runs/cure/${CURE_RUN_NAME}/checkpoint.pt"
  fi
  REPORT_DIR="${PROJECT_ROOT}/reports/stage2_gate_eval_${STAGE2_TAG}" \
    TIMESTAMP="${STAGE2_TAG}" \
    bash "${SCRIPT_DIR}/zjc_run_stage2_gate_eval.sh"
fi

echo "=== ZJC Stage2 all train/eval done ==="
echo "ce_ckpt=${CE_CKPT:-}"
if [[ -n "${LONGCE_OUTPUT_ROOT:-}" && -n "${LONGCE_RUN_NAME:-}" ]]; then
  echo "longce_ckpt=${LONGCE_OUTPUT_ROOT}/runs/cure/${LONGCE_RUN_NAME}/checkpoint.pt"
fi
if [[ -n "${CURE_OUTPUT_ROOT:-}" && -n "${CURE_RUN_NAME:-}" ]]; then
  echo "cure_ckpt=${CURE_OUTPUT_ROOT}/runs/cure/${CURE_RUN_NAME}/checkpoint.pt"
fi
date -Is
