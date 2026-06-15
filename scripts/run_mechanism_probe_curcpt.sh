#!/bin/bash
#SBATCH -J cure_probe
#SBATCH -p acd_u
#SBATCH -n 4
#SBATCH --gres=gpu:1
#SBATCH -o /dev/null
#SBATCH -e /dev/null
#SBATCH -D /data/user/xnie012/pythonprojects/lgar

set -euo pipefail

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
mkdir -p ./logs
LOG_FILE="./logs/mechanism_probe_${TIMESTAMP}_${SLURM_JOB_ID}.log"
exec > "$LOG_FILE" 2>&1

set +u
source /data/user/xnie012/envs/memgen/bin/activate
set -u
export PYTHONUNBUFFERED=1
export PYTHONPATH="/data/user/xnie012/pythonprojects/lgar:${PYTHONPATH:-}"
export HF_HOME=/data/user/xnie012/cache
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODEL_PATH=${MODEL_PATH:-/data/user/xnie012/cache/Models/Qwen2.5-0.5B}
CACHE_DIR=${CACHE_DIR:-/data/user/xnie012/pythonprojects/lgar/data/qwen_fineweb_pilot}
OUTPUT_ROOT=${OUTPUT_ROOT:-/data/user/xnie012/pythonprojects/lgar/reports/cure_mechanism_probe_${TIMESTAMP}}
OUTPUT=${OUTPUT_ROOT}/mechanism_probe.json

CE_CKPT=${CE_CKPT:-/data/user/xnie012/pythonprojects/lgar/runs/ce_cpt_20260531_130512/runs/cure/ce_cpt_pilot/checkpoint.pt}
ADAPTER_CKPT=${ADAPTER_CKPT:-/data/user/xnie012/pythonprojects/lgar/runs/pilot_adapter_ce_20260531_154531/runs/cure/pilot_adapter_ce/checkpoint.pt}
LONGCE_CKPT=${LONGCE_CKPT:-/data/user/xnie012/pythonprojects/lgar/runs/pilot_longce_20260531_154639/runs/cure/pilot_longce/checkpoint.pt}
CURE_CKPT=${CURE_CKPT:-/data/user/xnie012/pythonprojects/lgar/runs/pilot_cure_20260531_154024/runs/cure/pilot_cure/checkpoint.pt}

echo "=== CURE mechanism probe ==="
date
nvidia-smi --query-gpu=index,memory.used --format=csv || true

python -m curcpt.mechanism_probe \
  --model-path "${MODEL_PATH}" \
  --cache-dir "${CACHE_DIR}" \
  --output "${OUTPUT}" \
  --checkpoint "ce_cpt=${CE_CKPT}" \
  --checkpoint "adapter_ce=${ADAPTER_CKPT}" \
  --checkpoint "longce_cpt=${LONGCE_CKPT}" \
  --checkpoint "cure_cpt=${CURE_CKPT}" \
  --reference-name ce_cpt \
  --seq-len "${SEQ_LEN:-4096}" \
  --short-window "${SHORT_WINDOW:-1024}" \
  --num-sequences "${NUM_SEQUENCES:-32}" \
  --batch-size "${BATCH_SIZE:-1}" \
  --top-fraction "${TOP_FRACTION:-0.10}" \
  --seed "${SEED:-20260531}" \
  --dtype "${DTYPE:-bf16}" \
  --attn-implementation "${ATTN_IMPLEMENTATION:-sdpa}"

date
echo "output=${OUTPUT}"
echo "=== done ==="
