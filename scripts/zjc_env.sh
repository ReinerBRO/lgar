#!/usr/bin/env bash
# Shared ZJC runtime defaults. Source this from scripts executed under
# /gemini/space/private/zjc/goals/lgar.

if [[ -n "${LGAR_ZJC_ENV_SOURCED:-}" ]] && declare -F require_dir >/dev/null 2>&1; then
  return 0 2>/dev/null || exit 0
fi
LGAR_ZJC_ENV_SOURCED=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
export ZJC_ROOT="${ZJC_ROOT:-/gemini/space/private/zjc}"
export CACHE_ROOT="${CACHE_ROOT:-${ZJC_ROOT}/cache}"
export DATA_ROOT="${DATA_ROOT:-${ZJC_ROOT}/data}"
export ZJC_ENV_NAME="${ZJC_ENV_NAME:-zjc_env}"
export ZJC_ENV_DIR="${ZJC_ENV_DIR:-${ZJC_ROOT}/envs/${ZJC_ENV_NAME}}"

_lgar_first_existing() {
  local first="$1"
  local path
  for path in "$@"; do
    if [[ -e "$path" ]]; then
      printf '%s\n' "$path"
      return 0
    fi
  done
  printf '%s\n' "$first"
}

_lgar_first_ruler_dir() {
  local first="$1"
  local path
  for path in "$@"; do
    if [[ -f "${path}/_assets/RULER/scripts/data/prepare.py" ]]; then
      printf '%s\n' "$path"
      return 0
    fi
  done
  for path in "$@"; do
    if [[ -d "$path" ]]; then
      printf '%s\n' "$path"
      return 0
    fi
  done
  printf '%s\n' "$first"
}

require_file() {
  local label="$1"
  local path="$2"
  if [[ ! -f "$path" ]]; then
    echo "missing ${label}: ${path}" >&2
    exit 1
  fi
}

require_dir() {
  local label="$1"
  local path="$2"
  if [[ ! -d "$path" ]]; then
    echo "missing ${label}: ${path}" >&2
    exit 1
  fi
}

if [[ -z "${PYTHON:-}" ]]; then
  if [[ -x "${ZJC_ENV_DIR}/bin/python" ]]; then
    export PYTHON="${ZJC_ENV_DIR}/bin/python"
  else
    export PYTHON="python"
  fi
fi

if [[ -z "${MODEL_PATH:-}" ]]; then
  export MODEL_PATH="$(_lgar_first_existing \
    "${CACHE_ROOT}/Models/Qwen2.5-0.5B" \
    "${CACHE_ROOT}/models/Qwen2.5-0.5B" \
    "${ZJC_ROOT}/models/Qwen2.5-0.5B" \
    "${PROJECT_ROOT}/models/Qwen2.5-0.5B")"
fi

export HF_HOME="${HF_HOME:-${CACHE_ROOT}/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/hub}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

export PREFIX_ROOT="${PREFIX_ROOT:-${ZJC_ROOT}/prefix}"
export PREFIX_SCRIPTS_DIR="${PREFIX_SCRIPTS_DIR:-${PREFIX_ROOT}/scripts}"
export PYTHONPATH="${PROJECT_ROOT}:${PREFIX_SCRIPTS_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

export RAW_DATA_DIR="${RAW_DATA_DIR:-$(_lgar_first_existing \
  "${DATA_ROOT}/fineweb_edu_100BT-shuffled" \
  "${CACHE_ROOT}/data/fineweb_edu_100BT-shuffled" \
  "${PROJECT_ROOT}/data/fineweb_edu_100BT-shuffled")}"
export STAGE2_CACHE_DIR="${STAGE2_CACHE_DIR:-${DATA_ROOT}/qwen_fineweb_stage2}"
export CACHE_DIR="${CACHE_DIR:-${STAGE2_CACHE_DIR}}"
export TOP_HEADS_JSON="${TOP_HEADS_JSON:-$(_lgar_first_existing \
  "${PROJECT_ROOT}/runs/ablate_full_20260531_131608/ablation_results_top24.json" \
  "${PROJECT_ROOT}/configs/cure_top24_heads.json")}"

export CE_CKPT="${CE_CKPT:-${PROJECT_ROOT}/runs/ce_cpt_20260531_130512/runs/cure/ce_cpt_pilot/checkpoint.pt}"
export ADAPTER_CKPT="${ADAPTER_CKPT:-${PROJECT_ROOT}/runs/pilot_adapter_ce_20260531_154531/runs/cure/pilot_adapter_ce/checkpoint.pt}"
export LONGCE_CKPT="${LONGCE_CKPT:-${PROJECT_ROOT}/runs/pilot_longce_20260531_154639/runs/cure/pilot_longce/checkpoint.pt}"
export CURE_CKPT="${CURE_CKPT:-${PROJECT_ROOT}/runs/cure_v3_fullbudget_20260601_081709/runs/cure/cure_v3_fullbudget_top24_fullhu10_alr1e4/checkpoint.pt}"

export RULER_DIR="${RULER_DIR:-$(_lgar_first_ruler_dir \
  "${CACHE_ROOT}/RULER_official" \
  "${DATA_ROOT}/RULER_official" \
  "${CACHE_ROOT}/RULER" \
  "${DATA_ROOT}/RULER" \
  "${PROJECT_ROOT}/data/RULER_official" \
  "${PROJECT_ROOT}/data/RULER")}"
export RULER_FOLDER="${RULER_FOLDER:-official4096_n200}"

export LONGBENCH_ROOT="${LONGBENCH_ROOT:-$(_lgar_first_existing \
  "${CACHE_ROOT}/data/LongBench" \
  "${DATA_ROOT}/LongBench" \
  "${PROJECT_ROOT}/data/LongBench")}"
export LONGEVAL_ROOT="${LONGEVAL_ROOT:-$(_lgar_first_existing \
  "${CACHE_ROOT}/data/longeval/evaluation" \
  "${DATA_ROOT}/longeval/evaluation" \
  "${PROJECT_ROOT}/data/longeval/evaluation")}"
export BABILONG_ROOT="${BABILONG_ROOT:-$(_lgar_first_existing \
  "${CACHE_ROOT}/data/babilong_official_1k" \
  "${DATA_ROOT}/babilong_official_1k" \
  "${CACHE_ROOT}/data/babilong" \
  "${DATA_ROOT}/babilong" \
  "${PROJECT_ROOT}/data/babilong_official_1k")}"
export NOLIMA_NEEDLE_SET="${NOLIMA_NEEDLE_SET:-$(_lgar_first_existing \
  "${CACHE_ROOT}/nolima/needlesets/needle_set_hard.json" \
  "${DATA_ROOT}/nolima/needlesets/needle_set_hard.json" \
  "${PROJECT_ROOT}/data/nolima/needlesets/needle_set_hard.json")}"
export NOLIMA_HAYSTACK_DIR="${NOLIMA_HAYSTACK_DIR:-$(_lgar_first_existing \
  "${CACHE_ROOT}/nolima/haystack/rand_shuffle" \
  "${DATA_ROOT}/nolima/haystack/rand_shuffle" \
  "${PROJECT_ROOT}/data/nolima/haystack/rand_shuffle")}"

export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
export GPU_LIST="${GPU_LIST:-0 1 2 3 4 5 6 7}"
export DTYPE="${DTYPE:-bf16}"
export ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"
