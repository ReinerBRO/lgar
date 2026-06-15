#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/zjc_env.sh
source "${SCRIPT_DIR}/zjc_env.sh"
cd "${PROJECT_ROOT}"

BASE_TAG="${BASE_TAG:-20260601_160711}"
GLOBAL_TOKENS="${GLOBAL_TOKENS:-2000000000}"
SEQ_LEN="${SEQ_LEN:-8192}"
BATCH_SIZE="${BATCH_SIZE:-1}"
RUN_EVAL="${RUN_EVAL:-0}"
SEQ_LENS="${SEQ_LENS:-8192}"
MAX_SAMPLES="${MAX_SAMPLES:-200}"

CE_CKPT="${CE_CKPT:-${PROJECT_ROOT}/runs/stage2_ce_cpt_${BASE_TAG}/runs/cure/stage2_ce_cpt_${BASE_TAG}/checkpoint.pt}"
TOP_HEADS_JSON="${TOP_HEADS_JSON:-${PROJECT_ROOT}/configs/cure_top24_heads.json}"

MATRIX_TAG="${MATRIX_TAG:-cure_hu_mask_${BASE_TAG}}"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/logs}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/zjc_${MATRIX_TAG}.log}"
REPORT_BASE="${REPORT_BASE:-${PROJECT_ROOT}/reports/${MATRIX_TAG}}"

mkdir -p "${LOG_DIR}" "${REPORT_BASE}"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "=== ZJC CURE HU/main-loss-mask ablation ==="
echo "base_tag=${BASE_TAG}"
echo "global_tokens=${GLOBAL_TOKENS}"
echo "seq_len=${SEQ_LEN}"
echo "batch_size_per_rank=${BATCH_SIZE}"
echo "run_eval=${RUN_EVAL}"
echo "seq_lens=${SEQ_LENS}"
echo "max_samples=${MAX_SAMPLES}"
echo "ce_ckpt=${CE_CKPT}"
echo "top_heads_json=${TOP_HEADS_JSON}"
echo "report_base=${REPORT_BASE}"
echo "log_file=${LOG_FILE}"
date -Is

require_file "CE checkpoint" "${CE_CKPT}"
require_file "top heads json" "${TOP_HEADS_JSON}"
require_dir "model" "${MODEL_PATH}"
require_dir "token cache" "${CACHE_DIR}"

VARIANTS=(
  "a1_hu_only|none|1.0|0.0"
  "a2_hu_only_preserve|none|1.0|0.02"
  "a3_valid_remote_hu_preserve|valid_remote|0.5|0.02"
  "a4_hu_only_stronger|none|2.0|0.05"
)

train_variant() {
  local name="$1"
  local main_mask="$2"
  local lambda_hu="$3"
  local lambda_nonhu="$4"

  local run_name="stage2_cure_${name}_${BASE_TAG}"
  local output_root="${PROJECT_ROOT}/runs/${run_name}"
  local ckpt="${output_root}/runs/cure/${run_name}/checkpoint.pt"
  local summary="${output_root}/runs/cure/${run_name}/summary.json"

  echo "=== train ${run_name} main_mask=${main_mask} lambda_hu=${lambda_hu} lambda_nonhu=${lambda_nonhu} ==="
  echo "checkpoint=${ckpt}"
  if [[ -s "${ckpt}" && -s "${summary}" ]]; then
    echo "[skip train] checkpoint and summary already exist"
    return 0
  fi

  env \
    STAGE2_TAG="${BASE_TAG}" \
    METHOD=cure_cpt \
    RUN_NAME="${run_name}" \
    OUTPUT_ROOT="${output_root}" \
    LOG_FILE="${LOG_DIR}/zjc_stage2_${run_name}.log" \
    GLOBAL_TOKENS="${GLOBAL_TOKENS}" \
    SEQ_LEN="${SEQ_LEN}" \
    BATCH_SIZE="${BATCH_SIZE}" \
    CE_CKPT="${CE_CKPT}" \
    TOP_HEADS_JSON="${TOP_HEADS_JSON}" \
    CURE_MAIN_LOSS_MASK="${main_mask}" \
    LAMBDA_FULL_HU_CE="${lambda_hu}" \
    LAMBDA_NONHU_LOGP="${lambda_nonhu}" \
    LAMBDA_RH_CE="${LAMBDA_RH_CE:-0.0}" \
    LAMBDA_RH_KD="${LAMBDA_RH_KD:-0.0}" \
    LAMBDA_COV="${LAMBDA_COV:-0.0}" \
    LORA_RANK="${LORA_RANK:-16}" \
    LORA_ALPHA="${LORA_ALPHA:-32.0}" \
    ADAPTER_LR="${ADAPTER_LR:-1e-4}" \
    ADAPTER_WEIGHT_DECAY="${ADAPTER_WEIGHT_DECAY:-0.0}" \
    UTILITY_TOP_FRACTION_TRAINING="${UTILITY_TOP_FRACTION_TRAINING:-0.10}" \
    bash "${SCRIPT_DIR}/zjc_run_stage2_train_one.sh"
}

if [[ "${RUN_TRAIN:-1}" == "1" ]]; then
  for row in "${VARIANTS[@]}"; do
    IFS='|' read -r name main_mask lambda_hu lambda_nonhu <<< "${row}"
    train_variant "${name}" "${main_mask}" "${lambda_hu}" "${lambda_nonhu}"
  done
fi

if [[ "${RUN_EVAL}" == "1" ]]; then
  require_file "LongCE checkpoint" "${LONGCE_CKPT:-${PROJECT_ROOT}/runs/stage2_longce_cpt_${BASE_TAG}/runs/cure/stage2_longce_cpt_${BASE_TAG}/checkpoint.pt}"
  for row in "${VARIANTS[@]}"; do
    IFS='|' read -r name _main_mask _lambda_hu _lambda_nonhu <<< "${row}"
    ckpt="${PROJECT_ROOT}/runs/stage2_cure_${name}_${BASE_TAG}/runs/cure/stage2_cure_${name}_${BASE_TAG}/checkpoint.pt"
    require_file "checkpoint ${name}" "${ckpt}"
    for seqlen in ${SEQ_LENS}; do
      out="${REPORT_BASE}/${name}/official${seqlen}_n${MAX_SAMPLES}"
      echo "=== eval ${name} seqlen=${seqlen} ==="
      CE_CKPT="${CE_CKPT}" \
      LONGCE_CKPT="${LONGCE_CKPT:-${PROJECT_ROOT}/runs/stage2_longce_cpt_${BASE_TAG}/runs/cure/stage2_longce_cpt_${BASE_TAG}/checkpoint.pt}" \
      CURE_CKPT="${ckpt}" \
      MODELS="cure_cpt" OUTPUT_ROOT="${out}" SEQ_LEN="${seqlen}" MAX_SAMPLES="${MAX_SAMPLES}" \
      RULER_FOLDER="official${seqlen}_n${MAX_SAMPLES}" \
      bash "${SCRIPT_DIR}/zjc_run_ruler13_fullbudget.sh"
    done
  done
fi

echo "=== ZJC CURE HU/main-loss-mask ablation done ==="
date -Is
