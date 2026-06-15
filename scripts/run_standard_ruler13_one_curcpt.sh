#!/bin/bash
#SBATCH -J cure_ruler1
#SBATCH -p acd_u
#SBATCH -n 8
#SBATCH --gres=gpu:4
#SBATCH -o /dev/null
#SBATCH -e /dev/null
#SBATCH -D /data/user/xnie012/pythonprojects/lgar

set -euo pipefail

mkdir -p logs
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="logs/standard_ruler13_one_${MODEL_NAME:-model}_${TIMESTAMP}_${SLURM_JOB_ID}.log"
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

MODEL_NAME=${MODEL_NAME:?MODEL_NAME is required}
CHECKPOINT_PATH=${CHECKPOINT_PATH:?CHECKPOINT_PATH is required}
BATCH_SIZE=${BATCH_SIZE:-2}
MODEL_PATH=${MODEL_PATH:-/data/user/xnie012/cache/Models/Qwen2.5-0.5B}
RULER_DIR=${RULER_DIR:-/data/user/xnie012/cache/RULER_official}
OUTPUT_ROOT=${OUTPUT_ROOT:-/data/user/xnie012/pythonprojects/lgar/reports/cure_standard_eval_20260531_200q/ruler13_official}
MAX_SAMPLES=${MAX_SAMPLES:-200}
NPROC_PER_NODE=${NPROC_PER_NODE:-4}
FOLDER_NAME=${FOLDER_NAME:-official4096_n200}
SEQ_LEN=${SEQ_LEN:-4096}

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
export PYTHONPATH="${RULER_DIR}/_assets/RULER/_python_vendor:${RULER_DIR}/_assets/RULER/scripts:${RULER_DIR}/_assets/RULER/scripts/data:${PYTHONPATH:-}"

DATA_DIR="${RULER_DIR}/generated/${FOLDER_NAME}"
PRED_DIR="${OUTPUT_ROOT}/${MODEL_NAME}/${FOLDER_NAME}/pred"
mkdir -p "${PRED_DIR}"

echo "=== Standard RULER13 single model ==="
echo "model_name=${MODEL_NAME}"
echo "checkpoint=${CHECKPOINT_PATH}"
echo "data_dir=${DATA_DIR}"
echo "pred_dir=${PRED_DIR}"
echo "seq_len=${SEQ_LEN} max_samples=${MAX_SAMPLES} nproc=${NPROC_PER_NODE}"
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

torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" -m lgar_cpt.ruler13_generate \
  --model-path "${MODEL_PATH}" \
  --checkpoint-path "${CHECKPOINT_PATH}" \
  --ruler-dir "${RULER_DIR}" \
  --data-dir "${DATA_DIR}" \
  --save-dir "${PRED_DIR}" \
  --tasks "${TASKS_CSV}" \
  --subset validation \
  --max-samples "${MAX_SAMPLES}" \
  --mode full \
  --seq-len "${SEQ_LEN}" \
  --batch-size "${BATCH_SIZE}" \
  --append-answer-prefix \
  --attn-implementation sdpa \
  --dtype bf16

python -m lgar_cpt.ruler13_score --ruler-dir "${RULER_DIR}" --data-dir "${PRED_DIR}"
echo "=== Single-model RULER13 done: ${MODEL_NAME} ==="
