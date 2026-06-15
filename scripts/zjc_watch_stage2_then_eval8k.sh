#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
USER_CE_CKPT="${CE_CKPT:-}"
USER_LONGCE_CKPT="${LONGCE_CKPT:-}"
USER_CURE_CKPT="${CURE_CKPT:-}"
USER_TOP_HEADS_JSON="${TOP_HEADS_JSON:-}"
# shellcheck source=scripts/zjc_env.sh
source "${SCRIPT_DIR}/zjc_env.sh"
cd "${PROJECT_ROOT}"

export PATH="${ZJC_ENV_DIR}/bin:${PATH}"
export PYTHON="${PYTHON:-${ZJC_ENV_DIR}/bin/python}"

infer_stage2_tag() {
  local latest
  latest="$(ls -dt "${PROJECT_ROOT}"/runs/stage2_ce_cpt_* 2>/dev/null | head -1 || true)"
  if [[ -z "${latest}" ]]; then
    return 1
  fi
  basename "${latest}" | sed 's/^stage2_ce_cpt_//'
}

STAGE2_TAG="${STAGE2_TAG:-$(infer_stage2_tag)}"
if [[ -z "${STAGE2_TAG}" ]]; then
  echo "STAGE2_TAG is required and no stage2_ce_cpt run was found" >&2
  exit 2
fi

POLL_SECONDS="${POLL_SECONDS:-60}"
WAIT_4K_GATE="${WAIT_4K_GATE:-1}"
AUTO_PREPARE_RULER8K="${AUTO_PREPARE_RULER8K:-1}"
EVAL8_SEQ_LEN="${EVAL8_SEQ_LEN:-8192}"
EVAL8_MAX_SAMPLES="${EVAL8_MAX_SAMPLES:-200}"
EVAL8_RULER_FOLDER="${EVAL8_RULER_FOLDER:-official${EVAL8_SEQ_LEN}_n${EVAL8_MAX_SAMPLES}}"
EVAL8_REPORT_DIR="${EVAL8_REPORT_DIR:-${PROJECT_ROOT}/reports/stage2_gate_eval_${STAGE2_TAG}_8k}"

CE_CKPT="${USER_CE_CKPT:-${PROJECT_ROOT}/runs/stage2_ce_cpt_${STAGE2_TAG}/runs/cure/stage2_ce_cpt_${STAGE2_TAG}/checkpoint.pt}"
LONGCE_CKPT="${USER_LONGCE_CKPT:-${PROJECT_ROOT}/runs/stage2_longce_cpt_${STAGE2_TAG}/runs/cure/stage2_longce_cpt_${STAGE2_TAG}/checkpoint.pt}"
CURE_CKPT="${USER_CURE_CKPT:-${PROJECT_ROOT}/runs/stage2_cure_v3_${STAGE2_TAG}/runs/cure/stage2_cure_v3_${STAGE2_TAG}/checkpoint.pt}"
TOP_HEADS_JSON="${USER_TOP_HEADS_JSON:-${PROJECT_ROOT}/configs/cure_top24_heads.json}"

GATE4_REPORT_DIR="${GATE4_REPORT_DIR:-${PROJECT_ROOT}/reports/stage2_gate_eval_${STAGE2_TAG}}"
GATE4_DONE="${GATE4_DONE:-${GATE4_REPORT_DIR}/gate_eval_manifest.json}"
EVAL8_DONE="${EVAL8_REPORT_DIR}/gate_eval_manifest.json"

resolve_ruler_dir() {
  if [[ -f "${RULER_DIR}/_assets/RULER/scripts/data/prepare.py" ]]; then
    printf '%s\n' "${RULER_DIR}"
    return 0
  fi
  local found
  found="$(find "${ZJC_ROOT}" "${PROJECT_ROOT}" -path '*/_assets/RULER/scripts/data/prepare.py' -type f 2>/dev/null | head -1 || true)"
  if [[ -n "${found}" ]]; then
    printf '%s\n' "${found%/_assets/RULER/scripts/data/prepare.py}"
    return 0
  fi
  printf '%s\n' "${RULER_DIR}"
}
RULER_DIR="$(resolve_ruler_dir)"
export RULER_DIR

log() {
  printf '[%s] %s\n' "$(date -Is)" "$*" | tee -a "${PROJECT_ROOT}/logs/zjc_watch_stage2_then_eval8k_${STAGE2_TAG}.log"
}

wait_file() {
  local label="$1"
  local path="$2"
  while [[ ! -f "${path}" ]]; do
    log "waiting ${label}: ${path}"
    sleep "${POLL_SECONDS}"
  done
  log "ready ${label}: ${path}"
}

ruler8k_complete() {
  local data_dir="${RULER_DIR}/generated/${EVAL8_RULER_FOLDER}"
  [[ -d "${data_dir}" ]] || return 1
  local complete=0
  local f count
  shopt -s nullglob
  for f in "${data_dir}"/*/validation.jsonl; do
    count="$(wc -l < "${f}")"
    if [[ "${count}" -ge "${EVAL8_MAX_SAMPLES}" ]]; then
      complete=$((complete + 1))
    fi
  done
  shopt -u nullglob
  [[ "${complete}" -eq 13 ]]
}

log "stage2_tag=${STAGE2_TAG}"
log "wait_4k_gate=${WAIT_4K_GATE}"
log "ruler_dir=${RULER_DIR}"
log "eval8_report_dir=${EVAL8_REPORT_DIR}"
log "ce_ckpt=${CE_CKPT}"
log "longce_ckpt=${LONGCE_CKPT}"
log "cure_ckpt=${CURE_CKPT}"

wait_file "CE checkpoint" "${CE_CKPT}"
wait_file "LongCE checkpoint" "${LONGCE_CKPT}"
wait_file "CURE checkpoint" "${CURE_CKPT}"

if [[ "${WAIT_4K_GATE}" == "1" ]]; then
  wait_file "4k gate eval manifest" "${GATE4_DONE}"
fi

if [[ -f "${EVAL8_DONE}" ]]; then
  log "8k gate eval already done: ${EVAL8_DONE}"
  exit 0
fi

if ! ruler8k_complete; then
  if [[ "${AUTO_PREPARE_RULER8K}" != "1" ]]; then
    log "8k RULER data incomplete and AUTO_PREPARE_RULER8K=0"
    exit 2
  fi
  log "preparing ${EVAL8_RULER_FOLDER}"
  RULER_DIR="${RULER_DIR}" \
    SEQ_LEN="${EVAL8_SEQ_LEN}" \
    MAX_SAMPLES="${EVAL8_MAX_SAMPLES}" \
    RULER_FOLDER="${EVAL8_RULER_FOLDER}" \
    bash "${SCRIPT_DIR}/zjc_prepare_ruler13_data.sh"
fi

log "starting 8k gate eval"
REPORT_DIR="${EVAL8_REPORT_DIR}" \
TIMESTAMP="${STAGE2_TAG}_8k" \
EVAL_SEQ_LEN="${EVAL8_SEQ_LEN}" \
RULER_FOLDER="${EVAL8_RULER_FOLDER}" \
MAX_SAMPLES="${EVAL8_MAX_SAMPLES}" \
MODELS="${MODELS:-ce_cpt longce_cpt cure_v3_fullbudget}" \
CE_CKPT="${CE_CKPT}" \
LONGCE_CKPT="${LONGCE_CKPT}" \
CURE_CKPT="${CURE_CKPT}" \
TOP_HEADS_JSON="${TOP_HEADS_JSON}" \
RULER_DIR="${RULER_DIR}" \
bash "${SCRIPT_DIR}/zjc_run_stage2_gate_eval.sh"

log "8k gate eval done: ${EVAL8_DONE}"
