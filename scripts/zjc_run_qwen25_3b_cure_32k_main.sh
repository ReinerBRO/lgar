#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
USER_MODEL_PATH="${MODEL_PATH:-}"
USER_CACHE_DIR="${CACHE_DIR:-}"
USER_OFFLINE_SIGNAL_DIR="${OFFLINE_SIGNAL_DIR:-}"
USER_TOP_HEADS_JSON="${TOP_HEADS_JSON:-}"
USER_ABLATION_RESULTS_JSON="${ABLATION_RESULTS_JSON:-}"
unset LGAR_ZJC_ENV_SOURCED
unset MODEL_PATH CACHE_DIR OFFLINE_SIGNAL_DIR TOP_HEADS_JSON ABLATION_RESULTS_JSON
export PROJECT_ROOT="${SCRIPT_PROJECT_ROOT}"
# shellcheck source=scripts/zjc_env.sh
source "${SCRIPT_DIR}/zjc_env.sh"
cd "${PROJECT_ROOT}"

TAG="${TAG:-20260606_qwen25_3b_32k_0p5b_cure_a3_top48}"
if [[ -n "${USER_MODEL_PATH}" && "${USER_MODEL_PATH}" != *"/goals/ls_delta/"* ]]; then
  MODEL_PATH="${USER_MODEL_PATH}"
else
  MODEL_PATH="${ZJC_ROOT}/models/Qwen2.5-3B"
fi
if [[ -n "${USER_CACHE_DIR}" && "${USER_CACHE_DIR}" != *"/goals/ls_delta/"* ]]; then
  CACHE_DIR="${USER_CACHE_DIR}"
else
  CACHE_DIR="${DATA_ROOT}/qwen_fineweb_stage2"
fi
if [[ -n "${USER_OFFLINE_SIGNAL_DIR}" && "${USER_OFFLINE_SIGNAL_DIR}" != *"/goals/ls_delta/"* ]]; then
  OFFLINE_SIGNAL_DIR="${USER_OFFLINE_SIGNAL_DIR}"
else
  OFFLINE_SIGNAL_DIR="${PROJECT_ROOT}/data/offline_signal_qwen25_3b_32k_0p5b_sw4096_lw4096"
fi

if [[ -n "${USER_TOP_HEADS_JSON}" && "${USER_TOP_HEADS_JSON}" != *"/goals/ls_delta/"* && "${USER_TOP_HEADS_JSON}" != *"top24"* ]]; then
  TOP_HEADS_JSON="${USER_TOP_HEADS_JSON}"
else
  TOP_HEADS_JSON=""
fi
ABLATION_RESULTS_JSON="${USER_ABLATION_RESULTS_JSON}"
DEFAULT_TOP_HEADS_JSON="${PROJECT_ROOT}/configs/cure_qwen25_3b_32k_top48_heads.json"

if [[ -z "${TOP_HEADS_JSON}" ]]; then
  for candidate in \
    "${DEFAULT_TOP_HEADS_JSON}" \
    "${PROJECT_ROOT}/configs/qwen25_3b_32k_top48_heads.json" \
    "${PROJECT_ROOT}/configs/cure_3b_32k_top48_heads.json"; do
    if [[ -f "${candidate}" ]]; then
      TOP_HEADS_JSON="${candidate}"
      break
    fi
  done
fi

if [[ -z "${TOP_HEADS_JSON}" && -z "${ABLATION_RESULTS_JSON}" ]]; then
  ABLATION_RESULTS_JSON="$(
    find "${PROJECT_ROOT}/runs" "${PROJECT_ROOT}/data" "${PROJECT_ROOT}/reports" -type f -name 'ablation_results*.json' 2>/dev/null \
      | grep -Ei 'qwen|qwen25' \
      | grep -Ei '3b' \
      | grep -Ei '32k' \
      | head -n 1 || true
  )"
fi

if [[ -z "${TOP_HEADS_JSON}" && -n "${ABLATION_RESULTS_JSON}" ]]; then
  require_file "3B 32k ablation results json" "${ABLATION_RESULTS_JSON}"
  "${PYTHON}" - "${ABLATION_RESULTS_JSON}" "${DEFAULT_TOP_HEADS_JSON}" <<'PY'
import json
import sys
from pathlib import Path

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
data = json.loads(src.read_text())
scores = data.get("head_scores") or {}
if not scores:
    raise SystemExit(f"missing head_scores in {src}")
ranked = sorted(scores.items(), key=lambda item: float(item[1]), reverse=True)[:48]
heads = [[int(layer), int(head)] for key, _ in ranked for layer, head in [key.split("_", 1)]]
dst.parent.mkdir(parents=True, exist_ok=True)
dst.write_text(json.dumps({
    "retrieval_heads": heads,
    "num_selected": len(heads),
    "selection": "top48_by_head_scores",
    "source": str(src),
    "head_scores": {key: float(value) for key, value in ranked},
}, indent=2) + "\n")
print(f"wrote {dst} from {src}")
PY
  TOP_HEADS_JSON="${DEFAULT_TOP_HEADS_JSON}"
fi

if [[ -z "${TOP_HEADS_JSON}" ]]; then
  echo "missing 3B 32k top48 heads json." >&2
  echo "Run first:" >&2
  echo "  bash scripts/zjc_run_qwen25_3b_32k_head_ablation_top48.sh" >&2
  echo "Or pass an existing raw ablation file:" >&2
  echo "  ABLATION_RESULTS_JSON=/path/to/ablation_results.json bash scripts/zjc_run_qwen25_3b_cure_32k_main.sh" >&2
  echo "Recent ablation candidates:" >&2
  find "${PROJECT_ROOT}/runs" "${PROJECT_ROOT}/data" "${PROJECT_ROOT}/reports" -type f -name 'ablation_results*.json' -printf '  %TY-%Tm-%Td %TH:%TM %p\n' 2>/dev/null \
    | sort -r \
    | head -n 20 >&2 || true
  exit 1
fi

require_file "3B 32k top48 heads json" "${TOP_HEADS_JSON}"
require_dir "Qwen2.5-3B model" "${MODEL_PATH}"
require_dir "Qwen token cache" "${CACHE_DIR}"
require_dir "offline signal" "${OFFLINE_SIGNAL_DIR}"
CURE_LOGP_CHUNK_SIZE="${CURE_LOGP_CHUNK_SIZE:-512}"
LGAR_CE_CHUNK_SIZE="${LGAR_CE_CHUNK_SIZE:-512}"
export CURE_LOGP_CHUNK_SIZE
export LGAR_CE_CHUNK_SIZE
echo "cure_logp_chunk_size=${CURE_LOGP_CHUNK_SIZE}"
echo "lgar_ce_chunk_size=${LGAR_CE_CHUNK_SIZE}"

"${PYTHON}" - "${MODEL_PATH}" "${TOP_HEADS_JSON}" <<'PY'
import json
import sys
from pathlib import Path

model_dir = Path(sys.argv[1])
heads_path = Path(sys.argv[2])
cfg = json.loads((model_dir / "config.json").read_text())
data = json.loads(heads_path.read_text())
heads = data.get("retrieval_heads") or []
if len(heads) != 48:
    raise SystemExit(f"{heads_path} has {len(heads)} retrieval_heads, expected 48")
n_layers = int(cfg["num_hidden_layers"])
n_heads = int(cfg["num_attention_heads"])
bad = [h for h in heads if len(h) != 2 or not (0 <= int(h[0]) < n_layers) or not (0 <= int(h[1]) < n_heads)]
if bad:
    raise SystemExit(f"{heads_path} has out-of-range heads for model layers={n_layers} heads={n_heads}: {bad[:8]}")
source = str(data.get("source", ""))
print(f"top_heads_json={heads_path}")
print(f"top_heads_source={source}")
print(f"model_layers={n_layers} model_q_heads={n_heads} num_selected={len(heads)}")
PY

MODEL_PATH="${MODEL_PATH}" \
CACHE_DIR="${CACHE_DIR}" \
OFFLINE_SIGNAL_DIR="${OFFLINE_SIGNAL_DIR}" \
METHOD=cure_cpt \
RUN_NAME="stage2_qwen25_3b_cure_a3_top48_32k_0p5b_${TAG}" \
OUTPUT_ROOT="${PROJECT_ROOT}/runs/stage2_qwen25_3b_cure_a3_top48_32k_0p5b_${TAG}" \
SEQ_LEN=32768 \
LOCAL_WINDOW=4096 \
BATCH_SIZE=1 \
NPROC_PER_NODE=8 \
GLOBAL_TOKENS=500000000 \
SAVE_INTERVAL=381 \
EVAL_INTERVAL=50 \
ATTENTION_MASK_MODE=causal \
ATTN_IMPLEMENTATION=sdpa \
DTYPE=bf16 \
FSDP=1 \
FSDP_LIMIT_ALL_GATHERS="${FSDP_LIMIT_ALL_GATHERS:-1}" \
FSDP_FORWARD_PREFETCH="${FSDP_FORWARD_PREFETCH:-0}" \
FSDP_BACKWARD_PREFETCH="${FSDP_BACKWARD_PREFETCH:-pre}" \
GRADIENT_CHECKPOINTING=1 \
CURE_FROM_BASE=1 \
CURE_REQUIRE_FREEZE_BASE=1 \
FREEZE_BASE_MODEL=1 \
CURE_BASE_LR=0.0 \
TOP_HEADS_JSON="${TOP_HEADS_JSON}" \
CURE_MAIN_LOSS_MASK=valid_remote \
CURE_LOGP_CHUNK_SIZE="${CURE_LOGP_CHUNK_SIZE}" \
LGAR_CE_CHUNK_SIZE="${LGAR_CE_CHUNK_SIZE}" \
LAMBDA_FULL_HU_CE="${LAMBDA_FULL_HU_CE:-0.5}" \
LAMBDA_NONHU_LOGP="${LAMBDA_NONHU_LOGP:-0.02}" \
LAMBDA_RH_CE=0.0 \
LAMBDA_RH_KD=0.0 \
LAMBDA_COV=0.0 \
UTILITY_TOP_FRACTION_TRAINING="${UTILITY_TOP_FRACTION_TRAINING:-0.10}" \
LORA_RANK=16 \
LORA_ALPHA=32.0 \
ADAPTER_LR=1e-4 \
ADAPTER_WEIGHT_DECAY=0.0 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
bash "${SCRIPT_DIR}/zjc_run_stage2_train_one.sh"
