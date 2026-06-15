#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/zjc_env.sh
source "${SCRIPT_DIR}/zjc_env.sh"
cd "${PROJECT_ROOT}"

TARGET_TOKENS="${TARGET_TOKENS:-2200000000}"
MAX_SHARDS="${MAX_SHARDS:-}"
MIN_DOC_TOKENS="${MIN_DOC_TOKENS:-256}"
PREPARE_WORKERS="${PREPARE_WORKERS:-64}"
TOKENIZE_BATCH_SIZE="${TOKENIZE_BATCH_SIZE:-512}"
TOKENIZER_THREADS_PER_WORKER="${TOKENIZER_THREADS_PER_WORKER:-1}"
OUTPUT_CACHE_DIR="${OUTPUT_CACHE_DIR:-${STAGE2_CACHE_DIR}}"
OVERWRITE_CACHE="${OVERWRITE_CACHE:-0}"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/logs}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/zjc_prepare_fineweb_cache_$(date +%Y%m%d_%H%M%S).log}"

export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export RAYON_NUM_THREADS="${RAYON_NUM_THREADS:-${TOKENIZER_THREADS_PER_WORKER}}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export ARROW_NUM_THREADS="${ARROW_NUM_THREADS:-1}"

mkdir -p "${LOG_DIR}" "$(dirname "${OUTPUT_CACHE_DIR}")"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "=== ZJC prepare Qwen FineWeb token cache ==="
echo "raw_data_dir=${RAW_DATA_DIR}"
echo "model_path=${MODEL_PATH}"
echo "cache_dir=${OUTPUT_CACHE_DIR}"
echo "target_tokens=${TARGET_TOKENS}"
echo "max_shards=${MAX_SHARDS:-all}"
echo "prepare_workers=${PREPARE_WORKERS}"
echo "tokenize_batch_size=${TOKENIZE_BATCH_SIZE}"
echo "tokenizer_threads_per_worker=${TOKENIZER_THREADS_PER_WORKER}"
echo "log_file=${LOG_FILE}"
date -Is

require_dir "raw FineWeb data" "${RAW_DATA_DIR}"
require_dir "model" "${MODEL_PATH}"

if [[ -f "${OUTPUT_CACHE_DIR}/cache_info.json" && -f "${OUTPUT_CACHE_DIR}/tokens.npy" ]]; then
  echo "cache already exists: ${OUTPUT_CACHE_DIR}"
  echo "=== cache ready ==="
  ls -lh "${OUTPUT_CACHE_DIR}"
  date -Is
  exit 0
fi

if [[ -e "${OUTPUT_CACHE_DIR}" ]] && [[ "$(find "${OUTPUT_CACHE_DIR}" -mindepth 1 -maxdepth 1 | head -1)" != "" ]]; then
  if [[ "${OVERWRITE_CACHE}" == "1" ]]; then
    echo "removing incomplete cache because OVERWRITE_CACHE=1: ${OUTPUT_CACHE_DIR}"
    rm -rf "${OUTPUT_CACHE_DIR}"
  else
    echo "cache dir exists but is incomplete: ${OUTPUT_CACHE_DIR}" >&2
    echo "Set OVERWRITE_CACHE=1 to rebuild it, or choose OUTPUT_CACHE_DIR." >&2
    exit 2
  fi
fi

ARGS=(
  --raw-data-dir "${RAW_DATA_DIR}"
  --model-path "${MODEL_PATH}"
  --cache-dir "${OUTPUT_CACHE_DIR}"
  --target-tokens "${TARGET_TOKENS}"
  --min-doc-tokens "${MIN_DOC_TOKENS}"
  --batch-size "${TOKENIZE_BATCH_SIZE}"
  --workers "${PREPARE_WORKERS}"
)
if [[ -n "${MAX_SHARDS}" ]]; then
  ARGS+=(--max-shards "${MAX_SHARDS}")
fi

"${PYTHON}" -m lgar_cpt.data "${ARGS[@]}"

echo "=== cache ready ==="
ls -lh "${OUTPUT_CACHE_DIR}"
date -Is
