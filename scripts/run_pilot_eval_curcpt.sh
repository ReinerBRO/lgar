#!/bin/bash
#SBATCH -J cure_eval
#SBATCH -p acd_u
#SBATCH -n 8
#SBATCH --gres=gpu:4
#SBATCH -o /dev/null
#SBATCH -e /dev/null
#SBATCH -D /data/user/xnie012/pythonprojects/lgar

set -euo pipefail

mkdir -p ./logs
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
MODEL_TAG=${MODEL_TAG:?MODEL_TAG is required, e.g. ce_cpt, adapter_ce, longce, cure}
CHECKPOINT_PATH=${CHECKPOINT_PATH:?CHECKPOINT_PATH is required}

LOG_FILE="./logs/pilot_eval_${MODEL_TAG}_${TIMESTAMP}_${SLURM_JOB_ID}.log"
exec > "$LOG_FILE" 2>&1

set +u
source /data/user/xnie012/envs/memgen/bin/activate
set -u
export PYTHONUNBUFFERED=1
export PYTHONPATH="/data/user/xnie012/pythonprojects/lgar:${PYTHONPATH:-}"
export HF_HOME=/data/user/xnie012/cache
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export NCCL_IB_DISABLE=1 NCCL_P2P_LEVEL=NVL
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODEL_PATH=${MODEL_PATH:-/data/user/xnie012/cache/Models/Qwen2.5-0.5B}
CACHE_DIR=${CACHE_DIR:-/data/user/xnie012/pythonprojects/lgar/data/qwen_fineweb_pilot}
SHORT_MC_DIR=${SHORT_MC_DIR:-/data/user/xnie012/cache/data}
OUTPUT_ROOT=${OUTPUT_ROOT:-/data/user/xnie012/pythonprojects/lgar/reports/cure_pilot_eval_${TIMESTAMP}}
OUTPUT=${OUTPUT_ROOT}/${MODEL_TAG}.json

echo "=== CURE pilot eval: ${MODEL_TAG} ==="
date
nvidia-smi --query-gpu=index,memory.used --format=csv || true

torchrun --standalone --nproc_per_node=4 -m curcpt.evaluate \
  --model-path "${MODEL_PATH}" \
  --cache-dir "${CACHE_DIR}" \
  --short-mc-dir "${SHORT_MC_DIR}" \
  --checkpoint-path "${CHECKPOINT_PATH}" \
  --output "${OUTPUT}" \
  --seq-len 4096 \
  --short-window 1024 \
  --local-window 1024 \
  --eval-batches "${EVAL_BATCHES:-8}" \
  --batch-size "${BATCH_SIZE:-1}" \
  --short-limit "${SHORT_LIMIT:-32}" \
  --long-limit "${LONG_LIMIT:-16}" \
  --long-lengths "${LONG_LENGTHS:-2048,4096}" \
  --seed "${SEED:-1337}" \
  --dtype "${DTYPE:-bf16}" \
  --attn-implementation "${ATTN_IMPLEMENTATION:-sdpa}"

date
echo "=== done ==="
cat "${OUTPUT}"
