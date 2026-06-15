#!/bin/bash
#SBATCH -J cv1_ruler
#SBATCH -p acd_u
#SBATCH -n 8
#SBATCH --gres=gpu:4
#SBATCH -o /dev/null
#SBATCH -e /dev/null
#SBATCH -D /data/user/xnie012/pythonprojects/lgar

set -euo pipefail

mkdir -p logs
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="logs/cure_v1_ruler13_rh_${TIMESTAMP}_${SLURM_JOB_ID}.log"
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
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODEL_PATH=${MODEL_PATH:-/data/user/xnie012/cache/Models/Qwen2.5-0.5B}
RULER_DIR=${RULER_DIR:-/data/user/xnie012/cache/RULER_official}
OUTPUT_ROOT=${OUTPUT_ROOT:-/data/user/xnie012/pythonprojects/lgar/reports/cure_v1_matrix_20260531_215846/ruler13_rh_official4096_n200}
MAX_SAMPLES=${MAX_SAMPLES:-200}
NPROC_PER_NODE=${NPROC_PER_NODE:-4}
FOLDER_NAME=${FOLDER_NAME:-official4096_n200}
SEQ_LEN=${SEQ_LEN:-4096}
BATCH_SIZE=${BATCH_SIZE:-1}

CE_CKPT=${CE_CKPT:-/data/user/xnie012/pythonprojects/lgar/runs/ce_cpt_20260531_130512/runs/cure/ce_cpt_pilot/checkpoint.pt}
CURE_V1_CKPT=${CURE_V1_CKPT:-/data/user/xnie012/pythonprojects/lgar/runs/cure_v1_matrix_20260531_215846/cure_v1_top12_r16_lam10_alr1e4/runs/cure/cure_v1_top12_r16_lam10_alr1e4/checkpoint.pt}
ABLATE=${ABLATE:-/data/user/xnie012/pythonprojects/lgar/runs/ablate_full_20260531_131608/ablation_results_top12.json}

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

ln -sfn "_assets/RULER/scripts" "${RULER_DIR}/scripts"
export PYTHONPATH="/data/user/xnie012/pythonprojects/lgar:${RULER_DIR}/_assets/RULER/_python_vendor:${RULER_DIR}/_assets/RULER/scripts:${RULER_DIR}/_assets/RULER/scripts/data:${PYTHONPATH:-}"

DATA_DIR="${RULER_DIR}/generated/${FOLDER_NAME}"
mkdir -p "${OUTPUT_ROOT}"

echo "=== CURE-v1 RULER13 RH-bottleneck official4096_n200 ==="
echo "output_root=${OUTPUT_ROOT}"
date
nvidia-smi || true

for task in "${TASKS[@]}"; do
  task_file="${DATA_DIR}/${task}/validation.jsonl"
  if [[ ! -f "${task_file}" ]]; then
    echo "Missing shared RULER data: ${task_file}" >&2
    exit 1
  fi
  count=$(wc -l < "${task_file}")
  if [[ "${count}" -lt "${MAX_SAMPLES}" ]]; then
    echo "Incomplete shared RULER data: ${task_file} has ${count}" >&2
    exit 1
  fi
done

run_model() {
  local model_name="$1"
  local ckpt="$2"
  local pred_dir="${OUTPUT_ROOT}/${model_name}/${FOLDER_NAME}/pred"
  mkdir -p "${pred_dir}"
  echo "=== generate ${model_name} rh_bottleneck ==="
  torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" -m lgar_cpt.ruler13_generate \
    --model-path "${MODEL_PATH}" \
    --checkpoint-path "${ckpt}" \
    --ruler-dir "${RULER_DIR}" \
    --data-dir "${DATA_DIR}" \
    --save-dir "${pred_dir}" \
    --tasks "${TASKS_CSV}" \
    --subset validation \
    --max-samples "${MAX_SAMPLES}" \
    --mode rh_bottleneck \
    --retrieval-heads-json "${ABLATE}" \
    --reference-checkpoint-path "${CURE_V1_CKPT}" \
    --seq-len "${SEQ_LEN}" \
    --local-window 1024 \
    --batch-size "${BATCH_SIZE}" \
    --append-answer-prefix \
    --attn-implementation sdpa \
    --dtype bf16
  python -m lgar_cpt.ruler13_score --ruler-dir "${RULER_DIR}" --data-dir "${pred_dir}"
}

run_model "ce_cpt_rh" "${CE_CKPT}"
run_model "cure_v1_rh" "${CURE_V1_CKPT}"

python - "${OUTPUT_ROOT}" <<'PY'
import csv
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
models = ["ce_cpt_rh", "cure_v1_rh"]
length = "official4096_n200"
summary = {"length": length, "models": {}}
for model in models:
    csv_path = root / model / length / "pred" / "summary.csv"
    with csv_path.open("r", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    tasks = rows[0][1:]
    scores = [float(x) for x in rows[1][1:]]
    summary["models"][model] = {
        "task_scores": dict(zip(tasks, scores)),
        "mean_score": sum(scores) / len(scores) if scores else float("nan"),
    }
summary["cure_v1_minus_ce"] = (
    summary["models"]["cure_v1_rh"]["mean_score"]
    - summary["models"]["ce_cpt_rh"]["mean_score"]
)
out = root / "compare_summary.json"
out.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
PY

date
echo "output=${OUTPUT_ROOT}/compare_summary.json"
