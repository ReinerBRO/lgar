#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
USER_MODEL_PATH="${MODEL_PATH:-}"
USER_CACHE_DIR="${CACHE_DIR:-}"
USER_OFFLINE_SIGNAL_DIR="${OFFLINE_SIGNAL_DIR:-}"
USER_TOP_HEADS_JSON="${TOP_HEADS_JSON:-}"
unset LGAR_ZJC_ENV_SOURCED
unset MODEL_PATH CACHE_DIR OFFLINE_SIGNAL_DIR TOP_HEADS_JSON
export PROJECT_ROOT="${SCRIPT_PROJECT_ROOT}"
# shellcheck source=scripts/zjc_env.sh
source "${SCRIPT_DIR}/zjc_env.sh"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PREFIX_SCRIPTS_DIR:-}${PYTHONPATH:+:${PYTHONPATH}}"

TAG="${TAG:-20260609_qwen25_0p5b_16k_0p5b_random24_seed${RANDOM_HEAD_SEED:-1337}}"
RANDOM_HEAD_SEED="${RANDOM_HEAD_SEED:-1337}"
RANDOM_HEAD_COUNT="${RANDOM_HEAD_COUNT:-24}"
REFERENCE_HEADS_JSON="${REFERENCE_HEADS_JSON:-${PROJECT_ROOT}/configs/cure_top24_heads.json}"
EXCLUDE_TOP_HEADS="${EXCLUDE_TOP_HEADS:-0}"

if [[ -n "${USER_MODEL_PATH}" && "${USER_MODEL_PATH}" != *"/goals/ls_delta/"* ]]; then
  MODEL_PATH="${USER_MODEL_PATH}"
else
  MODEL_PATH="${ZJC_ROOT}/models/Qwen2.5-0.5B"
fi
if [[ -n "${USER_CACHE_DIR}" && "${USER_CACHE_DIR}" != *"/goals/ls_delta/"* ]]; then
  CACHE_DIR="${USER_CACHE_DIR}"
else
  CACHE_DIR="${DATA_ROOT}/qwen_fineweb_stage2"
fi
if [[ -n "${USER_OFFLINE_SIGNAL_DIR}" && "${USER_OFFLINE_SIGNAL_DIR}" != *"/goals/ls_delta/"* ]]; then
  OFFLINE_SIGNAL_DIR="${USER_OFFLINE_SIGNAL_DIR}"
else
  for candidate in \
    "${PROJECT_ROOT}/data/offline_signal_16k_1b_sw2048_lw2560" \
    "${PROJECT_ROOT}/data/offline_signal_16k_1b_sw2048_lw2048" \
    "${PROJECT_ROOT}/data/offline_signal_16k_0p5b_sw2048_lw2048" \
    "${PROJECT_ROOT}/data/offline_signal_16k_1b"; do
    if [[ -d "${candidate}" ]]; then
      OFFLINE_SIGNAL_DIR="${candidate}"
      break
    fi
  done
fi

if [[ -n "${USER_TOP_HEADS_JSON}" && "${USER_TOP_HEADS_JSON}" != *"/goals/ls_delta/"* ]]; then
  TOP_HEADS_JSON="${USER_TOP_HEADS_JSON}"
else
  TOP_HEADS_JSON="${PROJECT_ROOT}/configs/cure_qwen25_0p5b_16k_random${RANDOM_HEAD_COUNT}_seed${RANDOM_HEAD_SEED}_heads.json"
fi

require_dir "Qwen2.5-0.5B model" "${MODEL_PATH}"
require_dir "Qwen token cache" "${CACHE_DIR}"
require_dir "offline signal" "${OFFLINE_SIGNAL_DIR}"
require_file "reference top24 heads json" "${REFERENCE_HEADS_JSON}"

echo "=== Qwen2.5-0.5B 16k CURE random-head baseline ==="
echo "tag=${TAG}"
echo "model_path=${MODEL_PATH}"
echo "cache_dir=${CACHE_DIR}"
echo "offline_signal_dir=${OFFLINE_SIGNAL_DIR}"
echo "reference_heads_json=${REFERENCE_HEADS_JSON}"
echo "random_head_count=${RANDOM_HEAD_COUNT}"
echo "random_head_seed=${RANDOM_HEAD_SEED}"
echo "exclude_top_heads=${EXCLUDE_TOP_HEADS}"
echo "top_heads_json=${TOP_HEADS_JSON}"
date -Is

MAKE_RANDOM_ARGS=(
  --model-path "${MODEL_PATH}"
  --output "${TOP_HEADS_JSON}"
  --seed "${RANDOM_HEAD_SEED}"
  --num-selected "${RANDOM_HEAD_COUNT}"
  --reference-heads-json "${REFERENCE_HEADS_JSON}"
  --selection-name "random${RANDOM_HEAD_COUNT}_same_pool_as_top24"
)
if [[ "${EXCLUDE_TOP_HEADS}" == "1" ]]; then
  MAKE_RANDOM_ARGS+=(--exclude-heads-json "${REFERENCE_HEADS_JSON}" --selection-name "random${RANDOM_HEAD_COUNT}_excluding_top24")
fi
if [[ "${REGENERATE_RANDOM_HEADS:-0}" == "1" || ! -f "${TOP_HEADS_JSON}" ]]; then
  "${PYTHON}" scripts/zjc_make_random_heads.py "${MAKE_RANDOM_ARGS[@]}"
else
  echo "[reuse] random heads json already exists: ${TOP_HEADS_JSON}"
fi

"${PYTHON}" - "${MODEL_PATH}" "${TOP_HEADS_JSON}" "${RANDOM_HEAD_COUNT}" <<'PY'
import json
import sys
from pathlib import Path

model_dir = Path(sys.argv[1])
heads_path = Path(sys.argv[2])
expected = int(sys.argv[3])
cfg = json.loads((model_dir / "config.json").read_text())
data = json.loads(heads_path.read_text())
heads = data.get("retrieval_heads") or []
if len(heads) != expected:
    raise SystemExit(f"{heads_path} has {len(heads)} retrieval_heads, expected {expected}")
n_layers = int(cfg["num_hidden_layers"])
n_heads = int(cfg["num_attention_heads"])
bad = [h for h in heads if len(h) != 2 or not (0 <= int(h[0]) < n_layers) or not (0 <= int(h[1]) < n_heads)]
if bad:
    raise SystemExit(f"{heads_path} has out-of-range heads for model layers={n_layers} heads={n_heads}: {bad[:8]}")
print(f"model_layers={n_layers} model_q_heads={n_heads} num_selected={len(heads)}")
print("random_heads=" + json.dumps(heads))
PY

MODEL_PATH="${MODEL_PATH}" \
CACHE_DIR="${CACHE_DIR}" \
OFFLINE_SIGNAL_DIR="${OFFLINE_SIGNAL_DIR}" \
METHOD=cure_cpt \
RUN_NAME="stage2_cure_random${RANDOM_HEAD_COUNT}_a3_from_base_frozen_${TAG}" \
OUTPUT_ROOT="${PROJECT_ROOT}/runs/stage2_cure_random${RANDOM_HEAD_COUNT}_a3_from_base_frozen_${TAG}" \
SEQ_LEN=16384 \
LOCAL_WINDOW=2048 \
BATCH_SIZE=1 \
NPROC_PER_NODE=8 \
GLOBAL_TOKENS=500000000 \
SAVE_INTERVAL=763 \
EVAL_INTERVAL=50 \
ATTENTION_MASK_MODE=causal \
ATTN_IMPLEMENTATION=sdpa \
DTYPE=bf16 \
GRADIENT_CHECKPOINTING=1 \
CURE_FROM_BASE=1 \
FREEZE_BASE_MODEL=1 \
CURE_REQUIRE_FREEZE_BASE=1 \
TOP_HEADS_JSON="${TOP_HEADS_JSON}" \
CURE_MAIN_LOSS_MASK=valid_remote \
LAMBDA_FULL_HU_CE="${LAMBDA_FULL_HU_CE:-0.5}" \
LAMBDA_NONHU_LOGP="${LAMBDA_NONHU_LOGP:-0.02}" \
LAMBDA_RH_CE="${LAMBDA_RH_CE:-0.0}" \
LAMBDA_RH_KD="${LAMBDA_RH_KD:-0.0}" \
LAMBDA_COV="${LAMBDA_COV:-0.0}" \
LORA_RANK="${LORA_RANK:-16}" \
LORA_ALPHA="${LORA_ALPHA:-32.0}" \
ADAPTER_LR="${ADAPTER_LR:-1e-4}" \
ADAPTER_WEIGHT_DECAY="${ADAPTER_WEIGHT_DECAY:-0.0}" \
RESUME_IF_AVAILABLE="${RESUME_IF_AVAILABLE:-1}" \
bash "${SCRIPT_DIR}/zjc_run_stage2_train_one.sh"

echo "=== Qwen2.5-0.5B 16k CURE random-head baseline done ==="
date -Is
