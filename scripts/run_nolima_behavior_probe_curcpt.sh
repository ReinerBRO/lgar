#!/bin/bash
#SBATCH -J nolima_probe
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
LOG_FILE="./logs/nolima_behavior_probe_${MODEL_TAG}_${NOLIMA_VARIANT}_${CONTEXT_LENGTH}_${TIMESTAMP}_${SLURM_JOB_ID}.log"
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
OUTPUT_ROOT=${OUTPUT_ROOT:-/data/user/xnie012/pythonprojects/lgar/reports/nolima_behavior_probe_short_sft_20260601}
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
    echo "Unsupported NoLiMA variant: ${NOLIMA_VARIANT}" >&2
    exit 2
    ;;
esac

mkdir -p "${OUTPUT_ROOT}"
OUTPUT="${OUTPUT_ROOT}/probe_${MODEL_TAG}_${NOLIMA_VARIANT}_${CONTEXT_LENGTH}.json"

echo "=== NoLiMA behavior probe ${MODEL_TAG} ${NOLIMA_VARIANT} cl${CONTEXT_LENGTH} ==="
date
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv || true

python -m curcpt.nolima_behavior_probe \
  --model-path "${MODEL_PATH}" \
  --checkpoint-path "${CHECKPOINT_PATH}" \
  --model-tag "${MODEL_TAG}" \
  --variant "${NOLIMA_VARIANT}" \
  --context-length "${CONTEXT_LENGTH}" \
  --output "${OUTPUT}" \
  --seq-len "${SEQ_LEN:-2048}" \
  --needle-set-path "${NEEDLE_DIR}/${NEEDLE_FILE}" \
  --haystack-dir "${HAYSTACK_DIR}" \
  --max-books "${NOLIMA_MAX_BOOKS:-5}" \
  --max-experiments "${NOLIMA_MAX_EXPERIMENTS:-10}" \
  --document-depth-percent-intervals "${NOLIMA_DEPTH_INTERVALS:-26}" \
  --sample-fraction "${NOLIMA_SAMPLE_FRACTION:-0.3333333333333333}" \
  --sample-offset "${NOLIMA_SAMPLE_OFFSET:-0}" \
  --batch-size "${PROBE_BATCH_SIZE:-8}" \
  --preview-limit "${PROBE_PREVIEW_LIMIT:-24}" \
  --dtype "${DTYPE:-bf16}" \
  --attn-implementation "${ATTN_IMPLEMENTATION:-sdpa}"

date
echo "=== done ==="
ls -lah "${OUTPUT}"
