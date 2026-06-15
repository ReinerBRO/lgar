#!/bin/bash
#SBATCH -J cv1_rul1
#SBATCH -p acd_u
#SBATCH -n 8
#SBATCH --gres=gpu:4
#SBATCH -o /dev/null
#SBATCH -e /dev/null
#SBATCH -D /data/user/xnie012/pythonprojects/lgar

set -euo pipefail

MODEL_NAME=${MODEL_NAME:?MODEL_NAME is required}
CHECKPOINT_PATH=${CHECKPOINT_PATH:?CHECKPOINT_PATH is required}

mkdir -p logs
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="logs/cure_v1_ruler13_one_${MODEL_NAME}_${TIMESTAMP}_${SLURM_JOB_ID}.log"
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
BATCH_SIZE=${BATCH_SIZE:-4}
ABLATE=${ABLATE:-/data/user/xnie012/pythonprojects/lgar/runs/ablate_full_20260531_131608/ablation_results_top12.json}
REFERENCE_CHECKPOINT_PATH=${REFERENCE_CHECKPOINT_PATH:-/data/user/xnie012/pythonprojects/lgar/runs/cure_v1_matrix_20260531_215846/cure_v1_top12_r16_lam10_alr1e4/runs/cure/cure_v1_top12_r16_lam10_alr1e4/checkpoint.pt}

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
PRED_DIR="${OUTPUT_ROOT}/${MODEL_NAME}/${FOLDER_NAME}/pred"
mkdir -p "${PRED_DIR}"

echo "=== CURE-v1 RULER13 one RH model ==="
echo "model_name=${MODEL_NAME}"
echo "checkpoint=${CHECKPOINT_PATH}"
echo "pred_dir=${PRED_DIR}"
date
nvidia-smi || true

torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" -m lgar_cpt.ruler13_generate \
  --model-path "${MODEL_PATH}" \
  --checkpoint-path "${CHECKPOINT_PATH}" \
  --ruler-dir "${RULER_DIR}" \
  --data-dir "${DATA_DIR}" \
  --save-dir "${PRED_DIR}" \
  --tasks "${TASKS_CSV}" \
  --subset validation \
  --max-samples "${MAX_SAMPLES}" \
  --mode rh_bottleneck \
  --retrieval-heads-json "${ABLATE}" \
  --reference-checkpoint-path "${REFERENCE_CHECKPOINT_PATH}" \
  --seq-len "${SEQ_LEN}" \
  --local-window 1024 \
  --batch-size "${BATCH_SIZE}" \
  --append-answer-prefix \
  --attn-implementation sdpa \
  --dtype bf16

python -m lgar_cpt.ruler13_score --ruler-dir "${RULER_DIR}" --data-dir "${PRED_DIR}"
echo "=== done ${MODEL_NAME} ==="
