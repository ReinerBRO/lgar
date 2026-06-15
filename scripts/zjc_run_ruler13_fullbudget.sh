#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
USER_RULER_FOLDER="${RULER_FOLDER:-}"
# shellcheck source=scripts/zjc_env.sh
source "${SCRIPT_DIR}/zjc_env.sh"
cd "${PROJECT_ROOT}"

TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/reports/ruler13_fullbudget_zjc_${TIMESTAMP}}"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/logs}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/zjc_ruler13_fullbudget_${TIMESTAMP}.log}"
MAX_SAMPLES="${MAX_SAMPLES:-200}"
RULER_SEED="${RULER_SEED:-39}"
SEQ_LEN="${SEQ_LEN:-4096}"
if [[ -z "${USER_RULER_FOLDER}" ]]; then
  RULER_FOLDER="official${SEQ_LEN}_n${MAX_SAMPLES}"
fi
DATA_DIR="${RULER_DIR}/generated/${RULER_FOLDER}"

TASKS=(
  niah_single_1
  niah_single_2
  niah_single_3
  niah_multikey_1
  niah_multikey_2
  niah_multikey_3
  niah_multivalue
  niah_multiquery
  vt
  cwe
  fwe
  qa_1
  qa_2
)
TASKS_CSV=$(IFS=,; echo "${TASKS[*]}")

mkdir -p "${OUTPUT_ROOT}" "${LOG_DIR}"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "=== ZJC RULER13 fullbudget eval ==="
echo "ruler_dir=${RULER_DIR}"
echo "data_dir=${DATA_DIR}"
echo "output_root=${OUTPUT_ROOT}"
echo "max_samples=${MAX_SAMPLES}"
echo "nproc_per_node=${NPROC_PER_NODE}"
echo "seq_len=${SEQ_LEN}"
echo "ruler_folder=${RULER_FOLDER}"
date -Is
if [[ "${SHOW_GPU_INFO:-0}" == "1" ]]; then
  nvidia-smi || true
fi

ASSET_PREPARE="${RULER_DIR}/_assets/RULER/scripts/data/prepare.py"
require_file "RULER prepare.py" "${ASSET_PREPARE}"
require_dir "model" "${MODEL_PATH}"
require_file "ce checkpoint" "${CE_CKPT}"
require_file "longce checkpoint" "${LONGCE_CKPT}"
require_file "cure checkpoint" "${CURE_CKPT}"

ln -sfn "_assets/RULER/scripts" "${RULER_DIR}/scripts"
export PYTHONPATH="${RULER_DIR}/_assets/RULER/_python_vendor:${RULER_DIR}/_assets/RULER/scripts:${RULER_DIR}/_assets/RULER/scripts/data:${PYTHONPATH:-}"

prepare_task() {
  local task="$1"
  local task_file="${DATA_DIR}/${task}/validation.jsonl"
  if [[ -f "${task_file}" ]]; then
    local line_count
    line_count=$(wc -l < "${task_file}")
    if [[ "${line_count}" -ge "${MAX_SAMPLES}" ]]; then
      echo "[prepare] reuse ${task}: ${line_count} rows"
      return 0
    fi
  fi
  echo "[prepare] build ${task}"
  "${PYTHON}" "${ASSET_PREPARE}" \
    --save_dir "${DATA_DIR}" \
    --benchmark synthetic \
    --task "${task}" \
    --subset validation \
    --tokenizer_path "${MODEL_PATH}" \
    --tokenizer_type hf \
    --max_seq_length "${SEQ_LEN}" \
    --model_template_type base \
    --num_samples "${MAX_SAMPLES}" \
    --random_seed "${RULER_SEED}"
}

run_model() {
  local model_name="$1"
  local ckpt="$2"
  local batch_size="$3"
  local pred_dir="${OUTPUT_ROOT}/${model_name}/${RULER_FOLDER}/pred"
  mkdir -p "${pred_dir}"
  echo "[generate] model=${model_name} ckpt=${ckpt}"
  "${PYTHON}" -m torch.distributed.run --standalone --nproc_per_node="${NPROC_PER_NODE}" -m lgar_cpt.ruler13_generate \
    --model-path "${MODEL_PATH}" \
    --checkpoint-path "${ckpt}" \
    --ruler-dir "${RULER_DIR}" \
    --data-dir "${DATA_DIR}" \
    --save-dir "${pred_dir}" \
    --tasks "${TASKS_CSV}" \
    --subset validation \
    --max-samples "${MAX_SAMPLES}" \
    --mode full \
    --seq-len "${SEQ_LEN}" \
    --batch-size "${batch_size}" \
    --append-answer-prefix \
    --attn-implementation "${ATTN_IMPLEMENTATION}" \
    --dtype "${DTYPE}"
  "${PYTHON}" -m lgar_cpt.ruler13_score --ruler-dir "${RULER_DIR}" --data-dir "${pred_dir}"
}

mkdir -p "${DATA_DIR}"
for task in "${TASKS[@]}"; do
  prepare_task "${task}"
done

MODELS="${MODELS:-ce_cpt longce_cpt cure_v3_fullbudget}"
for model in ${MODELS}; do
  case "${model}" in
    ce_cpt) run_model "ce_cpt" "${CE_CKPT}" "${CE_BATCH_SIZE:-4}" ;;
    adapter_ce) run_model "adapter_ce" "${ADAPTER_CKPT}" "${ADAPTER_BATCH_SIZE:-2}" ;;
    longce_cpt) run_model "longce_cpt" "${LONGCE_CKPT}" "${LONGCE_BATCH_SIZE:-4}" ;;
    cure_v3_fullbudget|cure_cpt) run_model "cure_v3_fullbudget" "${CURE_CKPT}" "${CURE_BATCH_SIZE:-2}" ;;
    *) echo "unsupported model in MODELS: ${model}" >&2; exit 2 ;;
  esac
done

"${PYTHON}" - "${OUTPUT_ROOT}" "${RULER_FOLDER}" ${MODELS} <<'PY'
import csv
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
folder = sys.argv[2]
models = sys.argv[3:]
summary = {"folder": folder, "models": {}}
for model in models:
    normalized = "cure_v3_fullbudget" if model == "cure_cpt" else model
    csv_path = root / normalized / folder / "pred" / "summary.csv"
    if not csv_path.exists():
        continue
    with csv_path.open("r", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    tasks = rows[0][1:]
    scores = [float(x) for x in rows[1][1:]]
    summary["models"][normalized] = {
        "mean_score": sum(scores) / len(scores) if scores else float("nan"),
        "task_scores": dict(zip(tasks, scores)),
    }
if "ce_cpt" in summary["models"]:
    ce = summary["models"]["ce_cpt"]["mean_score"]
    summary["deltas_vs_ce"] = {
        model: payload["mean_score"] - ce
        for model, payload in summary["models"].items()
        if model != "ce_cpt"
    }
out = root / "compare_summary.json"
out.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
print(out)
print(json.dumps(summary, indent=2, sort_keys=True))
PY

echo "=== ZJC RULER13 eval done ==="
echo "output_root=${OUTPUT_ROOT}"
date -Is
