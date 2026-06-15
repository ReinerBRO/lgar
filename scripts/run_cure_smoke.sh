#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/gemini/space/private/zjc/goals/lgar}"
ENV_PATH="${ENV_PATH:-/gemini/space/private/zjc/envs/zjc_env}"

cd "${PROJECT_ROOT}"
set +u
source "${ENV_PATH}/bin/activate"
set -u
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

export HF_HOME="/gemini/space/private/zjc/hf_cache"
export TRANSFORMERS_CACHE="/gemini/space/private/zjc/hf_cache/transformers"
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_P2P_LEVEL="${NCCL_P2P_LEVEL:-LOC}"
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

OUTPUT_DIR="${OUTPUT_DIR:-/gemini/space/private/zjc/goals/lgar/runs/cure_smoke_20260528}"

ARGS=(
  --run-name cure_smoke
  --method cure_cpt
  --model-path /gemini/space/private/zjc/models/Qwen2.5-0.5B
  --cache-dir /gemini/space/private/zjc/goals/lgar/data/qwen_fineweb_stage0
  --output-dir "${OUTPUT_DIR}"
  --seq-len 2048
  --local-window 1024
  --batch-size 4
  --steps 100
  --tokens-per-run 819200
  --lr 1e-5
  --eval-interval 10
  --seed 1337
  --dtype bf16
  --attn-implementation sdpa
)

# Optional: use offline signal cache if available
SIGNAL_DIR="${SIGNAL_DIR:-}"
if [[ -n "${SIGNAL_DIR}" ]]; then
  ARGS+=(--offline-signal-dir "${SIGNAL_DIR}")
fi

# Optional: use ablation results if available
ABLATION_RESULTS="${ABLATION_RESULTS:-}"
if [[ -n "${ABLATION_RESULTS}" ]]; then
  ARGS+=(--ablation-results "${ABLATION_RESULTS}")
fi

NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" -m curcpt.train "${ARGS[@]}"
