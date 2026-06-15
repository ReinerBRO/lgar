#!/bin/bash
#SBATCH -J cure_babi1k
#SBATCH -p acd_u
#SBATCH -n 8
#SBATCH --gres=gpu:4
#SBATCH -o /dev/null
#SBATCH -e /dev/null
#SBATCH -D /data/user/xnie012/pythonprojects/lgar

set -eo pipefail

mkdir -p logs
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="logs/babilong_official1k_fullbudget_${TIMESTAMP}_${SLURM_JOB_ID}.log"
exec > "${LOG_FILE}" 2>&1

module load cuda/12.2 2>/dev/null || module load cuda 2>/dev/null || true
source /data/user/xnie012/envs/memgen/bin/activate

export PYTHONUNBUFFERED=1
export HF_HOME=/data/user/xnie012/.cache/huggingface
export HF_DATASETS_CACHE=${HF_HOME}/datasets
export TRANSFORMERS_CACHE=${HF_HOME}/hub
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export BABILONG_ROOT=/data/user/xnie012/cache/data/babilong_official_1k

MODEL_PATH=/data/user/xnie012/cache/Models/Qwen2.5-0.5B
REPORT_DIR=/data/user/xnie012/pythonprojects/lgar/reports/long_context_public_fullbudget_20260601
mkdir -p "${REPORT_DIR}"

CE_CKPT=/data/user/xnie012/pythonprojects/lgar/runs/ce_cpt_20260531_130512/runs/cure/ce_cpt_pilot/checkpoint.pt
LONGCE_CKPT=/data/user/xnie012/pythonprojects/lgar/runs/pilot_longce_20260531_154639/runs/cure/pilot_longce/checkpoint.pt
CURE_CKPT=/data/user/xnie012/pythonprojects/lgar/runs/cure_v3_fullbudget_20260601_081709/runs/cure/cure_v3_fullbudget_top24_fullhu10_alr1e4/checkpoint.pt

echo "=== BABILong official1k fullbudget eval ==="
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<slurm-default>}"
nvidia-smi || true

torchrun --standalone --nproc_per_node=4 -m curcpt.eval_long_context_public \
  --model-path "${MODEL_PATH}" \
  --benchmark babilong \
  --tasks qa1,qa2,qa3,qa4,qa5 \
  --output "${REPORT_DIR}/babilong_official1k_fullbudget.json" \
  --checkpoint "ce_cpt=${CE_CKPT}" \
  --checkpoint "longce_cpt=${LONGCE_CKPT}" \
  --checkpoint "cure_v3_fullbudget=${CURE_CKPT}" \
  --max-examples 200 \
  --seq-len 4096 \
  --dtype bf16 \
  --attn-implementation sdpa

echo "=== BABILong official1k fullbudget eval done ==="
