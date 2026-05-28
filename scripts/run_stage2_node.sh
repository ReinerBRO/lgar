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

OUT_DIR="${OUTPUT_DIR:-/gemini/space/private/zjc/goals/lgar/stage2_2k}"
LOG_DIR="${LOG_DIR:-/gemini/space/private/zjc/logs/lgar}"
mkdir -p "${LOG_DIR}" "${OUT_DIR}/reports"

if [[ -z "${RUN_SELECTION:-}" ]]; then
  echo "RUN_SELECTION is required, e.g. Qwen0.5B_CE_CPT_2B=ce" >&2
  exit 2
fi

ARGS=(
  --model-path /gemini/space/private/zjc/models/Qwen2.5-0.5B
  --raw-data-dir /gemini/space/private/zjc/data/fineweb_edu_100BT-shuffled
  --cache-dir "${CACHE_DIR:-/gemini/space/private/zjc/goals/lgar/data/qwen_fineweb_stage0}"
  --output-dir "${OUT_DIR}"
  --stage1-review "${STAGE1_REVIEW:-/gemini/space/private/zjc/goals/lgar/reports/stage1_review.json}"
  --tokens-per-run "${TOKENS_PER_RUN:-2000000000}"
  --target-cache-tokens "${TARGET_CACHE_TOKENS:-2200000000}"
  --max-shards "${MAX_SHARDS:-80}"
  --seq-len "${SEQ_LEN:-2048}"
  --short-window "${SHORT_WINDOW:-1024}"
  --local-window "${LOCAL_WINDOW:-1024}"
  --lsd-top-fraction "${LSD_TOP_FRACTION:-0.05}"
  --long-nll-max-quantile "${LONG_NLL_MAX_QUANTILE:-0.60}"
  --router-target-budget "${ROUTER_TARGET_BUDGET:-0.10}"
  --final-global-budget "${FINAL_GLOBAL_BUDGET:-0.25}"
  --routed-layer-fraction "${ROUTED_LAYER_FRACTION:-0.3333333333333333}"
  --lambda-router "${LAMBDA_ROUTER:-0.02}"
  --lambda-budget "${LAMBDA_BUDGET:-0.005}"
  --batch-size "${BATCH_SIZE:-4}"
  --eval-batches "${EVAL_BATCHES:-4}"
  --eval-interval "${EVAL_INTERVAL:-800}"
  --save-interval "${SAVE_INTERVAL:-800}"
  --short-limit "${SHORT_LIMIT:-16}"
  --long-limit "${LONG_LIMIT:-8}"
  --dtype bf16
  --attn-implementation "${ATTN_IMPLEMENTATION:-sdpa}"
)

if [[ "${GRADIENT_CHECKPOINTING:-0}" == "1" ]]; then
  ARGS+=(--gradient-checkpointing)
fi
if [[ "${SKIP_COMPLETED:-1}" == "1" ]]; then
  ARGS+=(--skip-completed)
fi
if [[ "${RESUME_IF_AVAILABLE:-1}" == "1" ]]; then
  ARGS+=(--resume-if-available)
fi
if [[ -n "${OFFLINE_SIGNAL_DIR:-}" ]]; then
  ARGS+=(--offline-signal-dir "${OFFLINE_SIGNAL_DIR}")
fi
IFS=',' read -r -a SELECTED_RUNS <<< "${RUN_SELECTION}"
for selected_run in "${SELECTED_RUNS[@]}"; do
  ARGS+=(--run "${selected_run}")
done

NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" -m lgar_cpt.train_stage2 "${ARGS[@]}"
