#!/bin/bash
#SBATCH -J cure_ansprobe
#SBATCH -p acd_u
#SBATCH -n 8
#SBATCH --gres=gpu:4
#SBATCH -o /dev/null
#SBATCH -e /dev/null
#SBATCH -D /data/user/xnie012/pythonprojects/lgar

set -eo pipefail

mkdir -p logs
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="logs/ruler_answer_probe_${TIMESTAMP}_${SLURM_JOB_ID}.log"
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

MODEL_PATH=/data/user/xnie012/cache/Models/Qwen2.5-0.5B
DATA_DIR=/data/user/xnie012/cache/RULER_official/generated/official4096_n200
REPORT_DIR=/data/user/xnie012/pythonprojects/lgar/reports/cure_answer_probe_20260601
mkdir -p "${REPORT_DIR}"

CE_CKPT=/data/user/xnie012/pythonprojects/lgar/runs/ce_cpt_20260531_130512/runs/cure/ce_cpt_pilot/checkpoint.pt
ADAPTER_CKPT=/data/user/xnie012/pythonprojects/lgar/runs/adapter_freeze_control_20260601_005605/adapter_freeze_top24_r16_alr1e4/runs/cure/adapter_freeze_top24_r16_alr1e4/checkpoint.pt
CURE_V3_CKPT=/data/user/xnie012/pythonprojects/lgar/runs/cure_v3_freeze_gate_20260601_003410/cure_v3_freeze_top24_fullhu10_alr1e4/runs/cure/cure_v3_freeze_top24_fullhu10_alr1e4/checkpoint.pt
CURE_V5_CKPT=/data/user/xnie012/pythonprojects/lgar/runs/cure_v5_freeze_lambda_gate_20260601_011805/cure_v5_freeze_top24_frac010_fullhu20_alr1e4/runs/cure/cure_v5_freeze_top24_frac010_fullhu20_alr1e4/checkpoint.pt

echo "=== RULER answer probe ==="
nvidia-smi || true

torchrun --standalone --nproc_per_node=4 -m curcpt.ruler_answer_probe \
  --model-path "${MODEL_PATH}" \
  --data-dir "${DATA_DIR}" \
  --output "${REPORT_DIR}/ruler_answer_probe.json" \
  --reference-name ce_cpt \
  --checkpoint "ce_cpt=${CE_CKPT}" \
  --checkpoint "adapter_freeze=${ADAPTER_CKPT}" \
  --checkpoint "cure_v3=${CURE_V3_CKPT}" \
  --checkpoint "cure_v5_lambda2=${CURE_V5_CKPT}" \
  --max-samples 200 \
  --seq-len 4096 \
  --batch-size 2 \
  --dtype bf16 \
  --attn-implementation sdpa

echo "=== RULER answer probe done ==="
