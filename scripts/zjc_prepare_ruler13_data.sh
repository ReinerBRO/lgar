#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/zjc_env.sh
source "${SCRIPT_DIR}/zjc_env.sh"
cd "${PROJECT_ROOT}"

SEQ_LEN="${SEQ_LEN:-${EVAL_SEQ_LEN:-8192}}"
MAX_SAMPLES="${MAX_SAMPLES:-200}"
RULER_SEED="${RULER_SEED:-39}"
RULER_FOLDER="${RULER_FOLDER:-official${SEQ_LEN}_n${MAX_SAMPLES}}"
DATA_DIR="${RULER_DIR}/generated/${RULER_FOLDER}"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/logs}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/zjc_prepare_ruler13_${RULER_FOLDER}_$(date +%Y%m%d_%H%M%S).log}"

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

mkdir -p "${DATA_DIR}" "${LOG_DIR}"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "=== ZJC prepare RULER13 data ==="
echo "ruler_dir=${RULER_DIR}"
echo "data_dir=${DATA_DIR}"
echo "seq_len=${SEQ_LEN}"
echo "max_samples=${MAX_SAMPLES}"
echo "ruler_seed=${RULER_SEED}"
echo "log_file=${LOG_FILE}"
date -Is

ASSET_PREPARE="${RULER_DIR}/_assets/RULER/scripts/data/prepare.py"
if [[ ! -f "${ASSET_PREPARE}" ]]; then
  FOUND_PREPARE="$(
    find "${ZJC_ROOT}" "${PROJECT_ROOT}" -path '*/_assets/RULER/scripts/data/prepare.py' -type f 2>/dev/null | head -1
  )"
  if [[ -n "${FOUND_PREPARE}" ]]; then
    RULER_DIR="${FOUND_PREPARE%/_assets/RULER/scripts/data/prepare.py}"
    DATA_DIR="${RULER_DIR}/generated/${RULER_FOLDER}"
    mkdir -p "${DATA_DIR}"
    ASSET_PREPARE="${FOUND_PREPARE}"
    echo "[resolve] ruler_dir=${RULER_DIR}"
    echo "[resolve] data_dir=${DATA_DIR}"
  fi
fi
require_file "RULER prepare.py" "${ASSET_PREPARE}"
require_dir "model" "${MODEL_PATH}"

ln -sfn "_assets/RULER/scripts" "${RULER_DIR}/scripts"
export PYTHONPATH="${RULER_DIR}/_assets/RULER/_python_vendor:${RULER_DIR}/_assets/RULER/scripts:${RULER_DIR}/_assets/RULER/scripts/data:${PYTHONPATH:-}"

prepare_task() {
  local task="$1"
  local task_file="${DATA_DIR}/${task}/validation.jsonl"
  if [[ -f "${task_file}" ]]; then
    local line_count
    line_count=$(wc -l < "${task_file}")
    if [[ "${line_count}" -ge "${MAX_SAMPLES}" ]]; then
      echo "[reuse] ${task}: ${line_count} rows"
      return 0
    fi
    echo "[rebuild] ${task}: only ${line_count}/${MAX_SAMPLES} rows"
  else
    echo "[build] ${task}"
  fi
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

for task in "${TASKS[@]}"; do
  prepare_task "${task}"
done

"${PYTHON}" - "${DATA_DIR}" "${MAX_SAMPLES}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
need = int(sys.argv[2])
tasks = sorted(p.name for p in root.iterdir() if (p / "validation.jsonl").exists())
counts = {}
bad = {}
for task in tasks:
    path = root / task / "validation.jsonl"
    count = sum(1 for _ in path.open("r", encoding="utf-8"))
    counts[task] = count
    if count < need:
        bad[task] = count
summary = {
    "data_dir": str(root),
    "tasks": counts,
    "task_count": len(counts),
    "required_samples": need,
    "complete": len(counts) == 13 and not bad,
    "incomplete": bad,
}
out = root / "prepare_summary.json"
out.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
if not summary["complete"]:
    raise SystemExit(2)
PY

echo "=== ZJC prepare RULER13 data done ==="
echo "data_dir=${DATA_DIR}"
date -Is
