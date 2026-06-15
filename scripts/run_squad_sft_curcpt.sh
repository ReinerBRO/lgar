#!/bin/bash
#SBATCH -J squad_sft
#SBATCH -p acd_u
#SBATCH -n 8
#SBATCH --gres=gpu:4
#SBATCH -o /dev/null
#SBATCH -e /dev/null
#SBATCH -D /data/user/xnie012/pythonprojects/lgar

set -euo pipefail

RUN_NAME=${RUN_NAME:?RUN_NAME is required}
CHECKPOINT_PATH=${CHECKPOINT_PATH:?CHECKPOINT_PATH is required}
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

mkdir -p logs
LOG_FILE="logs/squad_sft_${RUN_NAME}_${TIMESTAMP}_${SLURM_JOB_ID}.log"
exec > "${LOG_FILE}" 2>&1

set +u
source /data/user/xnie012/envs/memgen/bin/activate
set -u

export PYTHONUNBUFFERED=1
export PYTHONPATH="/data/user/xnie012/pythonprojects/lgar:${PYTHONPATH:-}"
export HF_HOME=/data/user/xnie012/cache
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export NCCL_IB_DISABLE=1
export NCCL_P2P_LEVEL="${NCCL_P2P_LEVEL:-NVL}"
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODEL_PATH=${MODEL_PATH:-/data/user/xnie012/cache/Models/Qwen2.5-0.5B}
SQUAD_PATH=${SQUAD_PATH:-/data/user/xnie012/cache/data/chatqa/squad1.1/train.json}
OUTPUT_ROOT=${OUTPUT_ROOT:-/data/user/xnie012/pythonprojects/lgar/runs/squad_sft_${TIMESTAMP}}
NPROC_PER_NODE=${NPROC_PER_NODE:-4}
SEQ_LEN=${SEQ_LEN:-1024}
BATCH_SIZE=${BATCH_SIZE:-2}
EPOCHS=${EPOCHS:-1}
MAX_STEPS=${MAX_STEPS:-0}
MAX_TRAIN_EXAMPLES=${MAX_TRAIN_EXAMPLES:-0}
LR=${LR:-1e-5}
MIN_LR=${MIN_LR:-1e-6}
LOG_INTERVAL=${LOG_INTERVAL:-50}

mkdir -p "${OUTPUT_ROOT}"

echo "=== SQuAD short SFT ==="
echo "run_name=${RUN_NAME}"
echo "checkpoint=${CHECKPOINT_PATH}"
echo "output_root=${OUTPUT_ROOT}"
echo "seq_len=${SEQ_LEN} batch_size=${BATCH_SIZE} epochs=${EPOCHS} max_steps=${MAX_STEPS}"
date
nvidia-smi || true

torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" -m curcpt.squad_sft \
  --run-name "${RUN_NAME}" \
  --model-path "${MODEL_PATH}" \
  --checkpoint-path "${CHECKPOINT_PATH}" \
  --squad-path "${SQUAD_PATH}" \
  --output-dir "${OUTPUT_ROOT}" \
  --seq-len "${SEQ_LEN}" \
  --batch-size "${BATCH_SIZE}" \
  --epochs "${EPOCHS}" \
  --max-steps "${MAX_STEPS}" \
  --max-train-examples "${MAX_TRAIN_EXAMPLES}" \
  --lr "${LR}" \
  --min-lr "${MIN_LR}" \
  --weight-decay "${WEIGHT_DECAY:-0.0}" \
  --log-interval "${LOG_INTERVAL}" \
  --seed "${SEED:-1337}" \
  --dtype "${DTYPE:-bf16}" \
  --attn-implementation "${ATTN_IMPLEMENTATION:-sdpa}"

date
echo "=== SQuAD short SFT done ==="
cat "${OUTPUT_ROOT}/${RUN_NAME}/summary.json"
