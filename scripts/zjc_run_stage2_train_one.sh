#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/zjc_env.sh
source "${SCRIPT_DIR}/zjc_env.sh"
cd "${PROJECT_ROOT}"

METHOD="${METHOD:?METHOD is required: ce_cpt|longce_cpt|cure_cpt|adapter_ce}"
STAGE2_TAG="${STAGE2_TAG:-$(date +%Y%m%d_%H%M%S)}"
SEQ_LEN="${SEQ_LEN:-8192}"
LOCAL_WINDOW="${LOCAL_WINDOW:-1024}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GLOBAL_TOKENS="${GLOBAL_TOKENS:-2000000000}"
TOKENS_PER_RANK="${TOKENS_PER_RANK:-$((GLOBAL_TOKENS / NPROC_PER_NODE))}"
EVAL_INTERVAL="${EVAL_INTERVAL:-50}"
SAVE_INTERVAL="${SAVE_INTERVAL:-500}"
RESUME_IF_AVAILABLE="${RESUME_IF_AVAILABLE:-1}"
SEED="${SEED:-1337}"
CURE_FROM_BASE="${CURE_FROM_BASE:-0}"
LR="${LR:-1e-5}"
if [[ "${METHOD}" == "cure_cpt" ]]; then
  if [[ "${CURE_FROM_BASE}" == "1" ]]; then
    LR="${CURE_BASE_LR:-1e-5}"
  else
    LR="${CURE_BASE_LR:-0.0}"
  fi
fi
RUN_NAME="${RUN_NAME:-stage2_${METHOD}_${STAGE2_TAG}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/runs/stage2_${METHOD}_${STAGE2_TAG}}"
TENSORBOARD_ROOT="${TENSORBOARD_ROOT:-${PROJECT_ROOT}/runs/tensorboard/${STAGE2_TAG}}"
TENSORBOARD_DIR="${TENSORBOARD_DIR:-${TENSORBOARD_ROOT}/${RUN_NAME}}"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/logs}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/zjc_stage2_${METHOD}_${STAGE2_TAG}.log}"

mkdir -p "${OUTPUT_ROOT}" "${LOG_DIR}"
exec > >(tee -a "${LOG_FILE}") 2>&1

GLOBAL_TOKEN_ESTIMATE=$((TOKENS_PER_RANK * NPROC_PER_NODE))
EXPECTED_STEPS=$((TOKENS_PER_RANK / (SEQ_LEN * BATCH_SIZE) ))
if [[ "${EXPECTED_STEPS}" -lt 1 ]]; then
  EXPECTED_STEPS=1
fi

echo "=== ZJC Stage2 train-one ==="
echo "timestamp=${STAGE2_TAG}"
echo "method=${METHOD}"
echo "run_name=${RUN_NAME}"
echo "project_root=${PROJECT_ROOT}"
echo "zjc_env_dir=${ZJC_ENV_DIR}"
echo "python=${PYTHON}"
echo "nproc_per_node=${NPROC_PER_NODE}"
echo "seq_len=${SEQ_LEN}"
echo "batch_size_per_rank=${BATCH_SIZE}"
echo "tokens_per_rank=${TOKENS_PER_RANK}"
echo "global_token_estimate=${GLOBAL_TOKEN_ESTIMATE}"
echo "expected_steps=${EXPECTED_STEPS}"
echo "output_root=${OUTPUT_ROOT}"
echo "log_file=${LOG_FILE}"
echo "tensorboard_dir=${TENSORBOARD_DIR}"
echo "save_interval=${SAVE_INTERVAL}"
echo "resume_if_available=${RESUME_IF_AVAILABLE}"
echo "gradient_checkpointing=${GRADIENT_CHECKPOINTING:-1}"
echo "fsdp=${FSDP:-0}"
if [[ "${FSDP:-0}" == "1" ]]; then
  echo "fsdp_limit_all_gathers=${FSDP_LIMIT_ALL_GATHERS:-1}"
  echo "fsdp_forward_prefetch=${FSDP_FORWARD_PREFETCH:-0}"
  echo "fsdp_backward_prefetch=${FSDP_BACKWARD_PREFETCH:-pre}"
fi
if [[ "${METHOD}" == "cure_cpt" ]]; then
  echo "cure_main_loss_mask=${CURE_MAIN_LOSS_MASK:-all}"
  echo "cure_from_base=${CURE_FROM_BASE}"
  echo "freeze_base_model=${FREEZE_BASE_MODEL:-auto}"
  echo "cure_require_freeze_base=${CURE_REQUIRE_FREEZE_BASE:-0}"
  if [[ "${CURE_REQUIRE_FREEZE_BASE:-0}" == "1" && "${FREEZE_BASE_MODEL:-0}" != "1" ]]; then
    echo "CURE_REQUIRE_FREEZE_BASE=1 but FREEZE_BASE_MODEL is not 1" >&2
    exit 2
  fi
fi
date -Is
if [[ "${SHOW_GPU_INFO:-0}" == "1" ]]; then
  nvidia-smi || true
fi

require_dir "model" "${MODEL_PATH}"
require_dir "token cache" "${CACHE_DIR}"
"${PYTHON}" - <<'PY'
import sys
print(f"python_executable={sys.executable}", flush=True)
PY

COMMON_ARGS=(
  --run-name "${RUN_NAME}"
  --method "${METHOD}"
  --model-path "${MODEL_PATH}"
  --cache-dir "${CACHE_DIR}"
  --output-dir "${OUTPUT_ROOT}"
  --seq-len "${SEQ_LEN}"
  --local-window "${LOCAL_WINDOW}"
  --batch-size "${BATCH_SIZE}"
  --tokens-per-run "${TOKENS_PER_RANK}"
  --lr "${LR}"
  --eval-interval "${EVAL_INTERVAL}"
  --save-interval "${SAVE_INTERVAL}"
  --tensorboard-dir "${TENSORBOARD_DIR}"
  --seed "${SEED}"
  --dtype "${DTYPE}"
  --attn-implementation "${ATTN_IMPLEMENTATION}"
  --attention-mask-mode "${ATTENTION_MASK_MODE:-document}"
)
if [[ "${RESUME_IF_AVAILABLE}" == "1" ]]; then
  COMMON_ARGS+=(--resume-if-available)
fi
if [[ "${NO_TENSORBOARD:-0}" == "1" ]]; then
  COMMON_ARGS+=(--no-tensorboard)
fi
if [[ "${GRADIENT_CHECKPOINTING:-1}" == "0" ]]; then
  COMMON_ARGS+=(--no-gradient-checkpointing)
fi
if [[ "${FSDP:-0}" == "1" ]]; then
  COMMON_ARGS+=(--fsdp)
fi

case "${METHOD}" in
  ce_cpt)
    ;;
  longce_cpt)
    COMMON_ARGS+=(--utility-top-fraction-training "${UTILITY_TOP_FRACTION_TRAINING:-0.10}")
    if [[ -n "${OFFLINE_SIGNAL_DIR:-}" ]]; then
      COMMON_ARGS+=(--offline-signal-dir "${OFFLINE_SIGNAL_DIR}")
    fi
    ;;
  adapter_ce)
    require_file "top heads json" "${TOP_HEADS_JSON}"
    COMMON_ARGS+=(
      --ablation-results "${TOP_HEADS_JSON}"
      --lora-rank "${LORA_RANK:-16}"
      --lora-alpha "${LORA_ALPHA:-32.0}"
      --adapter-lr "${ADAPTER_LR:-${LR}}"
      --adapter-weight-decay "${ADAPTER_WEIGHT_DECAY:-0.0}"
    )
    if [[ "${FREEZE_BASE_MODEL:-0}" == "1" ]]; then
      COMMON_ARGS+=(--freeze-base-model)
    fi
    ;;
  cure_cpt)
    require_file "top heads json" "${TOP_HEADS_JSON}"
    COMMON_ARGS+=(
      --ablation-results "${TOP_HEADS_JSON}"
      --adapter-lr "${ADAPTER_LR:-1e-4}"
      --adapter-weight-decay "${ADAPTER_WEIGHT_DECAY:-0.0}"
      --lambda-rh-ce "${LAMBDA_RH_CE:-0.0}"
      --lambda-rh-kd "${LAMBDA_RH_KD:-0.0}"
      --lambda-cov "${LAMBDA_COV:-0.0}"
      --lambda-full-hu-ce "${LAMBDA_FULL_HU_CE:-1.0}"
      --lambda-nonhu-logp "${LAMBDA_NONHU_LOGP:-0.0}"
      --cure-main-loss-mask "${CURE_MAIN_LOSS_MASK:-all}"
      --utility-top-fraction-training "${UTILITY_TOP_FRACTION_TRAINING:-0.10}"
      --lora-rank "${LORA_RANK:-16}"
      --lora-alpha "${LORA_ALPHA:-32.0}"
    )
    if [[ "${CURE_FROM_BASE}" == "1" ]]; then
      echo "cure_source=base_model"
      if [[ "${FREEZE_BASE_MODEL:-0}" == "1" ]]; then
        COMMON_ARGS+=(--freeze-base-model)
      fi
    else
      require_file "ce checkpoint" "${CE_CKPT}"
      echo "cure_source=ce_checkpoint:${CE_CKPT}"
      COMMON_ARGS+=(--checkpoint-path "${CE_CKPT}")
      if [[ "${FREEZE_BASE_MODEL:-1}" == "1" ]]; then
        COMMON_ARGS+=(--freeze-base-model)
      fi
    fi
    if [[ -n "${OFFLINE_SIGNAL_DIR:-}" ]]; then
      COMMON_ARGS+=(--offline-signal-dir "${OFFLINE_SIGNAL_DIR}")
    fi
    ;;
  *)
    echo "unsupported METHOD=${METHOD}" >&2
    exit 2
    ;;
esac

"${PYTHON}" -m torch.distributed.run --standalone --nproc_per_node="${NPROC_PER_NODE}" -m curcpt.train "${COMMON_ARGS[@]}"

CKPT="${OUTPUT_ROOT}/runs/cure/${RUN_NAME}/checkpoint.pt"
LATEST_CKPT="${OUTPUT_ROOT}/runs/cure/${RUN_NAME}/checkpoint_latest.pt"
SUMMARY="${OUTPUT_ROOT}/runs/cure/${RUN_NAME}/summary.json"
echo "checkpoint=${CKPT}"
echo "latest_checkpoint=${LATEST_CKPT}"
echo "summary=${SUMMARY}"
if [[ -f "${SUMMARY}" ]]; then
  cat "${SUMMARY}"
fi
echo "=== ZJC Stage2 train-one done ==="
date -Is
