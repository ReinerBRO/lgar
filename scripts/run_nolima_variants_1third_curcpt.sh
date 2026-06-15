#!/bin/bash
#SBATCH -J cure_nolima_1third
#SBATCH -p acd_u
#SBATCH -n 4
#SBATCH --gres=gpu:1
#SBATCH -o /dev/null
#SBATCH -e /dev/null
#SBATCH -D /data/user/xnie012/pythonprojects/lgar

set -euo pipefail

MODEL_TAG=${MODEL_TAG:?MODEL_TAG is required}
CHECKPOINT_PATH=${CHECKPOINT_PATH:?CHECKPOINT_PATH is required}
NOLIMA_VARIANT=${NOLIMA_VARIANT:?NOLIMA_VARIANT is required}
CONTEXT_LENGTH=${CONTEXT_LENGTH:?CONTEXT_LENGTH is required}

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
mkdir -p ./logs
LOG_FILE="./logs/nolima_1third_${MODEL_TAG}_${NOLIMA_VARIANT}_${CONTEXT_LENGTH}_${TIMESTAMP}_${SLURM_JOB_ID}.log"
exec > "$LOG_FILE" 2>&1

set +u
source /data/user/xnie012/envs/memgen/bin/activate
set -u

export PYTHONUNBUFFERED=1
export PYTHONPATH="/data/user/xnie012/pythonprojects/lgar:/data/user/xnie012/pythonprojects/prefix:/data/user/xnie012/pythonprojects/prefix/scripts:${PYTHONPATH:-}"
export HF_HOME=/data/user/xnie012/cache
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODEL_PATH=${MODEL_PATH:-/data/user/xnie012/cache/Models/Qwen2.5-0.5B}
OUTPUT_ROOT=${OUTPUT_ROOT:-/data/user/xnie012/pythonprojects/lgar/reports/nolima_variants_1third_short_sft_20260601}
NEEDLE_DIR=${NOLIMA_NEEDLE_DIR:-/data/user/xnie012/cache/nolima/needlesets}
HAYSTACK_DIR=${NOLIMA_HAYSTACK_DIR:-/data/user/xnie012/cache/nolima/haystack/rand_shuffle}

case "${NOLIMA_VARIANT}" in
  standard) NEEDLE_FILE="needle_set.json" ;;
  w_distractor) NEEDLE_FILE="needle_set_w_Distractor.json" ;;
  hard) NEEDLE_FILE="needle_set_hard.json" ;;
  mc) NEEDLE_FILE="needle_set_MC.json" ;;
  onlydirect) NEEDLE_FILE="needle_set_ONLYDirect.json" ;;
  w_cot) NEEDLE_FILE="needle_set_w_CoT.json" ;;
  *)
    echo "Unsupported cached NoLiMA variant: ${NOLIMA_VARIANT}" >&2
    exit 2
    ;;
esac

mkdir -p "${OUTPUT_ROOT}"
OUTPUT="${OUTPUT_ROOT}/nolima_${MODEL_TAG}_${NOLIMA_VARIANT}_${CONTEXT_LENGTH}.json"

echo "=== NoLiMA 1/3 ${MODEL_TAG} ${NOLIMA_VARIANT} cl${CONTEXT_LENGTH} ==="
date
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv || true

python -m curcpt.eval_nolima \
  --model-path "${MODEL_PATH}" \
  --checkpoint-path "${CHECKPOINT_PATH}" \
  --output "${OUTPUT}" \
  --context-length "${CONTEXT_LENGTH}" \
  --seq-len "${SEQ_LEN:-2048}" \
  --needle-set-path "${NEEDLE_DIR}/${NEEDLE_FILE}" \
  --haystack-dir "${HAYSTACK_DIR}" \
  --max-books "${NOLIMA_MAX_BOOKS:-5}" \
  --max-experiments "${NOLIMA_MAX_EXPERIMENTS:-10}" \
  --document-depth-percent-intervals "${NOLIMA_DEPTH_INTERVALS:-26}" \
  --sample-fraction "${NOLIMA_SAMPLE_FRACTION:-0.3333333333333333}" \
  --sample-offset "${NOLIMA_SAMPLE_OFFSET:-0}" \
  --max-new-tokens "${NOLIMA_MAX_NEW_TOKENS:-12}" \
  --preview-limit "${NOLIMA_PREVIEW_LIMIT:-12}" \
  --dtype "${DTYPE:-bf16}" \
  --attn-implementation "${ATTN_IMPLEMENTATION:-sdpa}"

date
echo "=== done ==="
ls -lah "${OUTPUT}"
