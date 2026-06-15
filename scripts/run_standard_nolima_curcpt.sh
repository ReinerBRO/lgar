#!/bin/bash
#SBATCH -J cure_nolima
#SBATCH -p acd_u
#SBATCH -n 4
#SBATCH --gres=gpu:1
#SBATCH -o /dev/null
#SBATCH -e /dev/null
#SBATCH -D /data/user/xnie012/pythonprojects/lgar

set -euo pipefail

MODEL_TAG=${MODEL_TAG:?MODEL_TAG is required}
CHECKPOINT_PATH=${CHECKPOINT_PATH:?CHECKPOINT_PATH is required}
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

mkdir -p ./logs
LOG_FILE="./logs/standard_nolima_${MODEL_TAG}_${TIMESTAMP}_${SLURM_JOB_ID}.log"
exec > "$LOG_FILE" 2>&1

set +u
source /data/user/xnie012/envs/memgen/bin/activate
set -u
export PYTHONUNBUFFERED=1
export PYTHONPATH="/data/user/xnie012/pythonprojects/lgar:/data/user/xnie012/pythonprojects/prefix/scripts:${PYTHONPATH:-}"
export HF_HOME=/data/user/xnie012/cache
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODEL_PATH=${MODEL_PATH:-/data/user/xnie012/cache/Models/Qwen2.5-0.5B}
OUTPUT_ROOT=${OUTPUT_ROOT:-/data/user/xnie012/pythonprojects/lgar/reports/cure_standard_eval_${TIMESTAMP}}
CONTEXT_LENGTHS=${NOLIMA_CONTEXT_LENGTHS:-"2048 4096"}

echo "=== standard NoLiMa: ${MODEL_TAG} ==="
date
nvidia-smi --query-gpu=index,memory.used --format=csv || true

for CTX in ${CONTEXT_LENGTHS}; do
  OUTPUT="${OUTPUT_ROOT}/nolima_${MODEL_TAG}_${CTX}.json"
  python -m curcpt.eval_nolima \
    --model-path "${MODEL_PATH}" \
    --checkpoint-path "${CHECKPOINT_PATH}" \
    --output "${OUTPUT}" \
    --context-length "${CTX}" \
    --seq-len "${SEQ_LEN:-4096}" \
    --needle-set-path "${NOLIMA_NEEDLE_SET:-/data/user/xnie012/cache/nolima/needlesets/needle_set_hard.json}" \
    --haystack-dir "${NOLIMA_HAYSTACK_DIR:-/data/user/xnie012/cache/nolima/haystack/rand_shuffle}" \
    --max-books "${NOLIMA_MAX_BOOKS:-5}" \
    --max-experiments "${NOLIMA_MAX_EXPERIMENTS:-8}" \
    --document-depth-percent-intervals "${NOLIMA_DEPTH_INTERVALS:-5}" \
    --max-new-tokens "${NOLIMA_MAX_NEW_TOKENS:-12}" \
    --preview-limit "${NOLIMA_PREVIEW_LIMIT:-12}" \
    --dtype "${DTYPE:-bf16}" \
    --attn-implementation "${ATTN_IMPLEMENTATION:-sdpa}"
done

date
echo "=== done ==="
ls -lah "${OUTPUT_ROOT}"/nolima_"${MODEL_TAG}"_*.json
