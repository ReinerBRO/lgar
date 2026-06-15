#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: scripts/zjc_launch_tmux.sh SESSION_NAME COMMAND [ARGS...]" >&2
  echo "Example: scripts/zjc_launch_tmux.sh cure_ruler bash scripts/zjc_run_ruler13_fullbudget.sh" >&2
  exit 2
fi

SESSION="$1"
shift

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/zjc_env.sh
source "${SCRIPT_DIR}/zjc_env.sh"
cd "${PROJECT_ROOT}"

LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/logs/zjc_tmux}"
LOG_PATH="${LOG_PATH:-${LOG_DIR}/${SESSION}_$(date +%Y%m%d_%H%M%S).log}"
mkdir -p "${LOG_DIR}"

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION}" >&2
  exit 2
fi

ENV_NAMES=(
  PROJECT_ROOT ZJC_ROOT CACHE_ROOT DATA_ROOT ZJC_ENV_NAME ZJC_ENV_DIR PYTHON
  CUDA_VISIBLE_DEVICES NPROC_PER_NODE GPU_LIST
  GLOBAL_TOKENS TOKENS_PER_RANK SEQ_LEN SEQ_LENS BATCH_SIZE BASE_TAG STAGE2_TAG
  RUN_GATE_EVAL RUN_EVAL RUN_TRAIN MATRIX_TAG REPORT_BASE LOG_FILE
  RAW_DATA_DIR STAGE2_CACHE_DIR CACHE_TARGET_TOKENS AUTO_PREPARE_CACHE OVERWRITE_CACHE
  PREPARE_WORKERS TOKENIZE_BATCH_SIZE TOKENIZER_THREADS_PER_WORKER
  MAX_SAMPLES NUM_PROBE_SEQS PUBLIC_BENCHMARKS NOLIMA_SAMPLE_FRACTION
  MODEL_PATH CACHE_DIR TOP_HEADS_JSON CE_CKPT LONGCE_CKPT CURE_CKPT ADAPTER_CKPT
  CURE_MAIN_LOSS_MASK LAMBDA_FULL_HU_CE LAMBDA_NONHU_LOGP LAMBDA_RH_CE LAMBDA_RH_KD
  LAMBDA_COV LORA_RANK LORA_ALPHA ADAPTER_LR ADAPTER_WEIGHT_DECAY
  UTILITY_TOP_FRACTION_TRAINING
  RULER_DIR RULER_FOLDER LONGBENCH_ROOT LONGEVAL_ROOT BABILONG_ROOT
  NOLIMA_NEEDLE_SET NOLIMA_HAYSTACK_DIR PREFIX_ROOT PREFIX_SCRIPTS_DIR
)
ENV_ARGS=()
for name in "${ENV_NAMES[@]}"; do
  if [[ -n "${!name+x}" ]]; then
    ENV_ARGS+=("${name}=${!name}")
  fi
done

printf -v CMD '%q ' env "${ENV_ARGS[@]}" "$@"
tmux new-session -d -s "${SESSION}" "cd '${PROJECT_ROOT}' && ${CMD} 2>&1 | tee -a '${LOG_PATH}'"

echo "session=${SESSION}"
echo "log=${LOG_PATH}"
