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

SIGNAL_DIR="${OUTPUT_DIR:-}"
if [[ -z "${SIGNAL_DIR}" ]]; then
  echo "OUTPUT_DIR is required, e.g. /gemini/space/private/zjc/goals/lgar/data/offline_signal_full_2k" >&2
  exit 2
fi

ARGS=(
  --model-path "${MODEL_PATH:-/gemini/space/private/zjc/models/Qwen2.5-0.5B}"
  --cache-dir "${CACHE_DIR:-/gemini/space/private/zjc/goals/lgar/data/qwen_fineweb_stage2_2b_2k}"
  --output-dir "${SIGNAL_DIR}"
  --split "${SPLIT:-train}"
  --target-tokens "${TARGET_TOKENS:-2200000000}"
  --layout-batch-size "${LAYOUT_BATCH_SIZE:-8192}"
  --mine-batch-size "${MINE_BATCH_SIZE:-8}"
  --seq-len "${SEQ_LEN:-2048}"
  --short-window "${SHORT_WINDOW:-1024}"
  --local-window "${LOCAL_WINDOW:-1024}"
  --min-remote-margin "${MIN_REMOTE_MARGIN:-256}"
  --long-nll-max-quantile "${LONG_NLL_MAX_QUANTILE:-0.60}"
  --final-global-budget "${FINAL_GLOBAL_BUDGET:-0.25}"
  --min-doc-tokens "${MIN_DOC_TOKENS:-256}"
  --val-docs "${VAL_DOCS:-1024}"
  --dtype "${DTYPE:-bf16}"
  --attn-implementation "${ATTN_IMPLEMENTATION:-sdpa}"
  --seed "${SEED:-20260524}"
)

if [[ -n "${CHECKPOINT_PATH:-}" ]]; then
  ARGS+=(--checkpoint-path "${CHECKPOINT_PATH}")
fi
if [[ -n "${NUM_SEQUENCES:-}" ]]; then
  ARGS+=(--num-sequences "${NUM_SEQUENCES}")
fi
if [[ "${SAVE_LSD:-0}" == "1" ]]; then
  ARGS+=(--save-lsd)
fi

NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" -m lgar_cpt.mine_lsd_signal_cache "${ARGS[@]}"
