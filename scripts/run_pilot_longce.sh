#!/bin/bash
#SBATCH -J cure_c
#SBATCH -p acd_u
#SBATCH -n 8
#SBATCH --gres=gpu:4
#SBATCH -o /dev/null
#SBATCH -e /dev/null
#SBATCH -D /data/user/xnie012/pythonprojects/lgar

mkdir -p ./logs
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="./logs/pilot_longce_${TIMESTAMP}_${SLURM_JOB_ID}.log"
exec > "$LOG_FILE" 2>&1

source /data/user/xnie012/envs/memgen/bin/activate
export PYTHONUNBUFFERED=1
export PYTHONPATH="/data/user/xnie012/pythonprojects/lgar:${PYTHONPATH:-}"
export HF_HOME=/data/user/xnie012/cache
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export NCCL_IB_DISABLE=1 NCCL_P2P_LEVEL=NVL
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODEL_PATH=/data/user/xnie012/cache/Models/Qwen2.5-0.5B
CACHE_DIR=/data/user/xnie012/pythonprojects/lgar/data/qwen_fineweb_pilot
OUTPUT_DIR=/data/user/xnie012/pythonprojects/lgar/runs/pilot_longce_${TIMESTAMP}

echo "=== Pilot C: LongCE-CPT (4 GPU, 50M global, online utility) ==="
date; nvidia-smi --query-gpu=index,memory.used --format=csv || true

torchrun --standalone --nproc_per_node=4 -m curcpt.train \
  --run-name pilot_longce \
  --method longce_cpt \
  --model-path "${MODEL_PATH}" \
  --cache-dir "${CACHE_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --seq-len 4096 \
  --local-window 1024 \
  --batch-size 4 \
  --tokens-per-run 12500000 \
  --lr 1e-5 \
  --eval-interval 50 \
  --seed 1337 \
  --dtype bf16 \
  --attn-implementation sdpa

date; echo "=== done ==="
cat "${OUTPUT_DIR}/runs/cure/pilot_longce/summary.json" 2>/dev/null
