#!/bin/bash
#SBATCH -J cure_ruler13
#SBATCH -p acd_u
#SBATCH -n 8
#SBATCH --gres=gpu:4
#SBATCH -o /dev/null
#SBATCH -e /dev/null
#SBATCH -D /data/user/xnie012/pythonprojects/lgar

set -euo pipefail

mkdir -p logs
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="logs/standard_ruler13_${TIMESTAMP}_${SLURM_JOB_ID}.log"
exec > "${LOG_FILE}" 2>&1

module load cuda/12.2 2>/dev/null || module load cuda 2>/dev/null || true
set +u
source /data/user/xnie012/envs/memgen/bin/activate
set -u

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME=/data/user/xnie012/.cache/huggingface
export HF_DATASETS_CACHE=${HF_HOME}/datasets
export TRANSFORMERS_CACHE=${HF_HOME}/hub
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_P2P_LEVEL="${NCCL_P2P_LEVEL:-LOC}"
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

MODEL_PATH=${MODEL_PATH:-/data/user/xnie012/cache/Models/Qwen2.5-0.5B}
RULER_DIR=${RULER_DIR:-/data/user/xnie012/cache/RULER_official}
OUTPUT_ROOT=${OUTPUT_ROOT:-/data/user/xnie012/pythonprojects/lgar/reports/cure_standard_eval_20260531_200q/ruler13_official}
MAX_SAMPLES=${MAX_SAMPLES:-200}
RULER_SEED=${RULER_SEED:-39}
NPROC_PER_NODE=${NPROC_PER_NODE:-4}

CE_CKPT=${CE_CKPT:-/data/user/xnie012/pythonprojects/lgar/runs/ce_cpt_20260531_130512/runs/cure/ce_cpt_pilot/checkpoint.pt}
ADAPTER_CKPT=${ADAPTER_CKPT:-/data/user/xnie012/pythonprojects/lgar/runs/pilot_adapter_ce_20260531_154531/runs/cure/pilot_adapter_ce/checkpoint.pt}
LONGCE_CKPT=${LONGCE_CKPT:-/data/user/xnie012/pythonprojects/lgar/runs/pilot_longce_20260531_154639/runs/cure/pilot_longce/checkpoint.pt}
CURE_CKPT=${CURE_CKPT:-/data/user/xnie012/pythonprojects/lgar/runs/pilot_cure_20260531_154024/runs/cure/pilot_cure/checkpoint.pt}

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

LENGTHS=(
  "4096:official4096_n${MAX_SAMPLES}"
)

TASKS_CSV=$(IFS=,; echo "${TASKS[*]}")
ASSET_PREPARE="${RULER_DIR}/_assets/RULER/scripts/data/prepare.py"

if [[ ! -f "${ASSET_PREPARE}" ]]; then
  echo "Missing official RULER asset prepare.py at ${ASSET_PREPARE}" >&2
  exit 1
fi

ln -sfn "_assets/RULER/scripts" "${RULER_DIR}/scripts"
export PYTHONPATH="${RULER_DIR}/_assets/RULER/_python_vendor:${RULER_DIR}/_assets/RULER/scripts:${RULER_DIR}/_assets/RULER/scripts/data:${PYTHONPATH:-}"

echo "=== Standard RULER13 200-sample eval ==="
echo "model=${MODEL_PATH}"
echo "ruler_dir=${RULER_DIR}"
echo "output_root=${OUTPUT_ROOT}"
echo "max_samples=${MAX_SAMPLES}"
echo "nproc=${NPROC_PER_NODE}"
nvidia-smi || true

prepare_length() {
  local seq_len="$1"
  local folder_name="$2"
  local data_dir="${RULER_DIR}/generated/${folder_name}"
  mkdir -p "${data_dir}"
  for task in "${TASKS[@]}"; do
    local task_file="${data_dir}/${task}/validation.jsonl"
    if [[ -f "${task_file}" ]]; then
      local line_count
      line_count=$(wc -l < "${task_file}")
      if [[ "${line_count}" -ge "${MAX_SAMPLES}" ]]; then
        echo "[prepare] reuse ${folder_name}/${task}: ${line_count} rows"
        continue
      fi
    fi
    echo "[prepare] build ${folder_name}/${task}"
    python "${ASSET_PREPARE}" \
      --save_dir "${data_dir}" \
      --benchmark synthetic \
      --task "${task}" \
      --subset validation \
      --tokenizer_path "${MODEL_PATH}" \
      --tokenizer_type hf \
      --max_seq_length "${seq_len}" \
      --model_template_type base \
      --num_samples "${MAX_SAMPLES}" \
      --random_seed "${RULER_SEED}"
    local final_count
    final_count=$(wc -l < "${task_file}")
    if [[ "${final_count}" -lt "${MAX_SAMPLES}" ]]; then
      echo "RULER data generation incomplete for ${folder_name}/${task}: ${final_count}" >&2
      exit 1
    fi
  done
}

run_model() {
  local model_name="$1"
  local ckpt="$2"
  local batch_size="$3"
  if [[ ! -f "${ckpt}" ]]; then
    echo "Missing checkpoint for ${model_name}: ${ckpt}" >&2
    exit 1
  fi
  for pair in "${LENGTHS[@]}"; do
    local seq_len="${pair%%:*}"
    local folder_name="${pair##*:}"
    local data_dir="${RULER_DIR}/generated/${folder_name}"
    local pred_dir="${OUTPUT_ROOT}/${model_name}/${folder_name}/pred"
    mkdir -p "${pred_dir}"
    echo "[generate] model=${model_name} length=${seq_len} samples=${MAX_SAMPLES}"
    torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" -m lgar_cpt.ruler13_generate \
      --model-path "${MODEL_PATH}" \
      --checkpoint-path "${ckpt}" \
      --ruler-dir "${RULER_DIR}" \
      --data-dir "${data_dir}" \
      --save-dir "${pred_dir}" \
      --tasks "${TASKS_CSV}" \
      --subset validation \
      --max-samples "${MAX_SAMPLES}" \
      --mode full \
      --seq-len "${seq_len}" \
      --batch-size "${batch_size}" \
      --append-answer-prefix \
      --attn-implementation sdpa \
      --dtype bf16
    python -m lgar_cpt.ruler13_score --ruler-dir "${RULER_DIR}" --data-dir "${pred_dir}"
  done
}

for pair in "${LENGTHS[@]}"; do
  prepare_length "${pair%%:*}" "${pair##*:}"
done

run_model "ce_cpt" "${CE_CKPT}" 4
run_model "adapter_ce" "${ADAPTER_CKPT}" 2
run_model "longce_cpt" "${LONGCE_CKPT}" 4
run_model "cure_cpt" "${CURE_CKPT}" 2

python - "${OUTPUT_ROOT}" <<'PY'
import csv
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
models = ["ce_cpt", "adapter_ce", "longce_cpt", "cure_cpt"]
lengths = ["official4096_n200"]
summary = {"lengths": {}}

for length in lengths:
    summary["lengths"][length] = {}
    for model in models:
        csv_path = root / model / length / "pred" / "summary.csv"
        with csv_path.open("r", encoding="utf-8") as handle:
            rows = list(csv.reader(handle))
        tasks = rows[0][1:]
        scores = [float(x) for x in rows[1][1:]]
        nulls = rows[2][1:]
        summary["lengths"][length][model] = {
            "task_scores": dict(zip(tasks, scores)),
            "task_nulls": dict(zip(tasks, nulls)),
            "mean_score": sum(scores) / len(scores) if scores else float("nan"),
        }
    ce = summary["lengths"][length]["ce_cpt"]["mean_score"]
    summary["lengths"][length]["deltas_vs_ce"] = {
        model: summary["lengths"][length][model]["mean_score"] - ce
        for model in models
        if model != "ce_cpt"
    }
    summary["lengths"][length]["cure_vs_adapter"] = (
        summary["lengths"][length]["cure_cpt"]["mean_score"]
        - summary["lengths"][length]["adapter_ce"]["mean_score"]
    )
    summary["lengths"][length]["cure_vs_longce"] = (
        summary["lengths"][length]["cure_cpt"]["mean_score"]
        - summary["lengths"][length]["longce_cpt"]["mean_score"]
    )

out_path = root / "compare_summary.json"
out_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
print(out_path)
print(json.dumps(summary, indent=2, sort_keys=True))
PY

echo "=== Standard RULER13 done ==="
