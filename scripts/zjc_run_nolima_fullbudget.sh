#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/zjc_env.sh
source "${SCRIPT_DIR}/zjc_env.sh"
cd "${PROJECT_ROOT}"

TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/reports/nolima_fullbudget_zjc_${TIMESTAMP}}"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/logs}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/zjc_nolima_fullbudget_${TIMESTAMP}.log}"
CONTEXT_LENGTHS="${NOLIMA_CONTEXT_LENGTHS:-1024 1536}"
MODELS="${MODELS:-ce_cpt adapter_ce longce_cpt cure_v3_fullbudget}"
read -r -a GPUS <<< "${GPU_LIST}"
GPU_COUNT="${GPU_COUNT:-${#GPUS[@]}}"
MAX_PARALLEL="${MAX_PARALLEL:-${GPU_COUNT}}"
SAMPLE_FRACTION="${NOLIMA_SAMPLE_FRACTION:-0.333333}"
EVAL_MODE="${EVAL_MODE:-full}"

mkdir -p "${OUTPUT_ROOT}" "${LOG_DIR}"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "=== ZJC NoLiMA fullbudget eval ==="
echo "output_root=${OUTPUT_ROOT}"
echo "contexts=${CONTEXT_LENGTHS}"
echo "models=${MODELS}"
echo "gpus=${GPU_LIST}"
echo "max_parallel=${MAX_PARALLEL}"
echo "sample_fraction=${SAMPLE_FRACTION}"
echo "eval_mode=${EVAL_MODE}"
date -Is
if [[ "${SHOW_GPU_INFO:-0}" == "1" ]]; then
  nvidia-smi || true
fi

require_dir "model" "${MODEL_PATH}"
require_file "NoLiMA needle set" "${NOLIMA_NEEDLE_SET}"
require_dir "NoLiMA haystack" "${NOLIMA_HAYSTACK_DIR}"
require_dir "prefix scripts" "${PREFIX_SCRIPTS_DIR}"

ckpt_for_model() {
  case "$1" in
    ce_cpt) printf '%s\n' "${CE_CKPT}" ;;
    adapter_ce) printf '%s\n' "${ADAPTER_CKPT}" ;;
    longce_cpt) printf '%s\n' "${LONGCE_CKPT}" ;;
    cure_v3_fullbudget|cure_cpt) printf '%s\n' "${CURE_CKPT}" ;;
    *) return 1 ;;
  esac
}

running=0
launch_one() {
  local model="$1"
  local ctx="$2"
  local gpu="$3"
  local ckpt
  ckpt="$(ckpt_for_model "${model}")"
  require_file "${model} checkpoint" "${ckpt}"
  local output="${OUTPUT_ROOT}/nolima_${model}_${ctx}_${EVAL_MODE}.json"
  local model_log="${LOG_DIR}/zjc_nolima_${model}_${ctx}_${EVAL_MODE}_${TIMESTAMP}.log"
  echo "[run] model=${model} ctx=${ctx} gpu=${gpu} output=${output}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" -m curcpt.eval_nolima \
    --model-path "${MODEL_PATH}" \
    --checkpoint-path "${ckpt}" \
    --output "${output}" \
    --prefix-scripts-dir "${PREFIX_SCRIPTS_DIR}" \
    --needle-set-path "${NOLIMA_NEEDLE_SET}" \
    --haystack-dir "${NOLIMA_HAYSTACK_DIR}" \
    --context-length "${ctx}" \
    --seq-len "${SEQ_LEN:-4096}" \
    --max-books "${NOLIMA_MAX_BOOKS:-5}" \
    --max-experiments "${NOLIMA_MAX_EXPERIMENTS:-8}" \
    --document-depth-percent-intervals "${NOLIMA_DEPTH_INTERVALS:-5}" \
    --sample-fraction "${SAMPLE_FRACTION}" \
    --sample-offset "${NOLIMA_SAMPLE_OFFSET:-0}" \
    --max-new-tokens "${NOLIMA_MAX_NEW_TOKENS:-12}" \
    --preview-limit "${NOLIMA_PREVIEW_LIMIT:-12}" \
    --eval-mode "${EVAL_MODE}" \
    --local-window "${LOCAL_WINDOW:-1024}" \
    --retrieval-heads-json "${TOP_HEADS_JSON}" \
    --reference-checkpoint-path "${CURE_CKPT}" \
    --dtype "${DTYPE}" \
    --attn-implementation "${ATTN_IMPLEMENTATION}" \
    > "${model_log}" 2>&1 &
}

slot=0
for ctx in ${CONTEXT_LENGTHS}; do
  for model in ${MODELS}; do
    gpu="${GPUS[$((slot % GPU_COUNT))]}"
    launch_one "${model}" "${ctx}" "${gpu}"
    slot=$((slot + 1))
    running=$((running + 1))
    if [[ "${running}" -ge "${MAX_PARALLEL}" ]]; then
      wait -n
      running=$((running - 1))
    fi
  done
done
wait

"${PYTHON}" - "${OUTPUT_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
summary = {}
for path in sorted(root.glob("nolima_*.json")):
    data = json.loads(path.read_text(encoding="utf-8"))
    summary[path.stem] = data.get("summary", data.get("metrics", data))
out = root / "compare_summary.json"
out.write_text(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")
print(out)
PY

echo "=== ZJC NoLiMA eval done ==="
echo "output_root=${OUTPUT_ROOT}"
date -Is
