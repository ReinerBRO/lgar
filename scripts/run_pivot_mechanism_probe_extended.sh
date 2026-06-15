#!/bin/bash
#SBATCH -J cure_xprobe
#SBATCH -p acd_u
#SBATCH -n 4
#SBATCH --gres=gpu:1
#SBATCH -o /dev/null
#SBATCH -e /dev/null
#SBATCH -D /data/user/xnie012/pythonprojects/lgar

set -euo pipefail

mkdir -p logs
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="logs/pivot_mechanism_probe_extended_${TIMESTAMP}_${SLURM_JOB_ID}.log"
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
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODEL_PATH=${MODEL_PATH:-/data/user/xnie012/cache/Models/Qwen2.5-0.5B}
CACHE_DIR=${CACHE_DIR:-/data/user/xnie012/pythonprojects/lgar/data/qwen_fineweb_pilot}
REPORT_DIR=${REPORT_DIR:-/data/user/xnie012/pythonprojects/lgar/reports/cure_after_fixlora_pivot_20260531}
OUTPUT=${OUTPUT:-${REPORT_DIR}/mechanism_probe_extended.json}
NUM_SEQUENCES=${NUM_SEQUENCES:-200}

CE_CKPT=${CE_CKPT:-/data/user/xnie012/pythonprojects/lgar/runs/ce_cpt_20260531_130512/runs/cure/ce_cpt_pilot/checkpoint.pt}
ADAPTER_CKPT=${ADAPTER_CKPT:-/data/user/xnie012/pythonprojects/lgar/runs/pilot_adapter_ce_20260531_173436/runs/cure/pilot_adapter_ce/checkpoint.pt}
LONGCE_CKPT=${LONGCE_CKPT:-/data/user/xnie012/pythonprojects/lgar/runs/pilot_longce_20260531_154639/runs/cure/pilot_longce/checkpoint.pt}
CURE_CKPT=${CURE_CKPT:-/data/user/xnie012/pythonprojects/lgar/runs/pilot_cure_20260531_173436/runs/cure/pilot_cure/checkpoint.pt}
ABLATE=${ABLATE:-/data/user/xnie012/pythonprojects/lgar/runs/ablate_full_20260531_131608/ablation_results_top6.json}

mkdir -p "${REPORT_DIR}"

python -m curcpt.mechanism_probe_extended \
  --model-path "${MODEL_PATH}" \
  --cache-dir "${CACHE_DIR}" \
  --output "${OUTPUT}" \
  --checkpoint "ce_cpt=${CE_CKPT}" \
  --checkpoint "adapter_ce=${ADAPTER_CKPT}" \
  --checkpoint "longce_cpt=${LONGCE_CKPT}" \
  --checkpoint "cure_cpt=${CURE_CKPT}" \
  --reference-name ce_cpt \
  --reference-checkpoint-path "${CURE_CKPT}" \
  --retrieval-heads-json "${ABLATE}" \
  --seq-len 4096 \
  --short-window 1024 \
  --num-sequences "${NUM_SEQUENCES}" \
  --batch-size 1 \
  --dtype bf16 \
  --attn-implementation sdpa

echo "output=${OUTPUT}"
