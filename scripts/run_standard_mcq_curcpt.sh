#!/bin/bash
#SBATCH -J cure_mcq
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
LOG_FILE="./logs/standard_mcq_${MODEL_TAG}_${TIMESTAMP}_${SLURM_JOB_ID}.log"
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
OUTPUT_ROOT=${OUTPUT_ROOT:-/data/user/xnie012/pythonprojects/lgar/reports/cure_standard_eval_${TIMESTAMP}}
OUTPUT=${OUTPUT_ROOT}/mcq_${MODEL_TAG}.json

echo "=== standard MCQ: ${MODEL_TAG} ==="
date
nvidia-smi --query-gpu=index,memory.used --format=csv || true

python -m curcpt.eval_public_mcq \
  --model-path "${MODEL_PATH}" \
  --checkpoint-path "${CHECKPOINT_PATH}" \
  --output "${OUTPUT}" \
  --tasks ${MCQ_TASKS:-arc_easy arc_challenge hellaswag winogrande} \
  --conditions ${MCQ_CONDITIONS:-clean irrelevant_demo} \
  --max-examples "${MCQ_MAX_EXAMPLES:-200}" \
  --batch-size "${MCQ_BATCH_SIZE:-8}" \
  --num-distractors "${MCQ_NUM_DISTRACTORS:-2}" \
  --max-prefix-tokens "${MCQ_MAX_PREFIX_TOKENS:-1800}" \
  --seq-len "${SEQ_LEN:-4096}" \
  --dtype "${DTYPE:-bf16}" \
  --attn-implementation "${ATTN_IMPLEMENTATION:-sdpa}"

date
echo "=== done ==="
cat "${OUTPUT}"
