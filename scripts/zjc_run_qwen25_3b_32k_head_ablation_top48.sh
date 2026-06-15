#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
USER_MODEL_PATH="${MODEL_PATH:-}"
USER_CACHE_DIR="${CACHE_DIR:-}"
USER_TOP_HEADS_JSON="${TOP_HEADS_JSON:-}"
USER_OFFLINE_SIGNAL_DIR="${OFFLINE_SIGNAL_DIR:-}"
unset LGAR_ZJC_ENV_SOURCED
unset MODEL_PATH CACHE_DIR TOP_HEADS_JSON OFFLINE_SIGNAL_DIR
export PROJECT_ROOT="${SCRIPT_PROJECT_ROOT}"
# shellcheck source=scripts/zjc_env.sh
source "${SCRIPT_DIR}/zjc_env.sh"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PREFIX_SCRIPTS_DIR:-}${PYTHONPATH:+:${PYTHONPATH}}"

TAG="${TAG:-20260606_qwen25_3b_32k_head_ablation}"
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
OUT_DIR="${OUT_DIR:-${PROJECT_ROOT}/runs/ablate_qwen25_3b_32k_top48_${TAG}}"
if [[ -n "${USER_TOP_HEADS_JSON}" && "${USER_TOP_HEADS_JSON}" != *"/goals/ls_delta/"* && "${USER_TOP_HEADS_JSON}" != *"top24"* ]]; then
  TOP_HEADS_JSON="${USER_TOP_HEADS_JSON}"
else
  TOP_HEADS_JSON="${PROJECT_ROOT}/configs/cure_qwen25_3b_32k_top48_heads.json"
fi

SEQ_LEN="${SEQ_LEN:-32768}"
LOCAL_WINDOW="${LOCAL_WINDOW:-4096}"
BATCH_SIZE="${BATCH_SIZE:-1}"
ABLATION_BATCHES="${ABLATION_BATCHES:-16}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
DTYPE="${DTYPE:-bf16}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"
CURE_NLL_CHUNK_SIZE="${CURE_NLL_CHUNK_SIZE:-512}"
export CURE_NLL_CHUNK_SIZE

require_dir "Qwen2.5-3B model" "${MODEL_PATH}"
require_dir "Qwen token cache" "${CACHE_DIR}"
require_dir "Qwen2.5-3B 32k offline signal" "${OFFLINE_SIGNAL_DIR}"
mkdir -p "${OUT_DIR}" "$(dirname "${TOP_HEADS_JSON}")"

echo "=== Qwen2.5-3B 32k head ablation ==="
echo "tag=${TAG}"
echo "model_path=${MODEL_PATH}"
echo "cache_dir=${CACHE_DIR}"
echo "offline_signal_dir=${OFFLINE_SIGNAL_DIR}"
echo "seq_len=${SEQ_LEN}"
echo "local_window=${LOCAL_WINDOW}"
echo "batch_size=${BATCH_SIZE}"
echo "ablation_batches_per_half=${ABLATION_BATCHES}"
echo "nproc_per_node=${NPROC_PER_NODE}"
echo "cure_nll_chunk_size=${CURE_NLL_CHUNK_SIZE}"
echo "out_dir=${OUT_DIR}"
echo "top_heads_json=${TOP_HEADS_JSON}"
date -Is

"${PYTHON}" - <<'PY'
import curcpt.run_head_ablation
print("import_check=curcpt.run_head_ablation ok", flush=True)
PY

"${PYTHON}" -m torch.distributed.run --standalone --nproc_per_node="${NPROC_PER_NODE}" -m curcpt.run_head_ablation \
  --model-path "${MODEL_PATH}" \
  --cache-dir "${CACHE_DIR}" \
  --offline-signal-dir "${OFFLINE_SIGNAL_DIR}" \
  --output-dir "${OUT_DIR}" \
  --seq-len "${SEQ_LEN}" \
  --local-window "${LOCAL_WINDOW}" \
  --batch-size "${BATCH_SIZE}" \
  --ablation-sequences "${ABLATION_BATCHES}" \
  --top-k-fraction 1.0 \
  --dtype "${DTYPE}" \
  --attn-implementation "${ATTN_IMPLEMENTATION}"

require_file "ablation results" "${OUT_DIR}/ablation_results.json"

"${PYTHON}" - "${OUT_DIR}/ablation_results.json" "${TOP_HEADS_JSON}" <<'PY'
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
print(f"wrote {dst}")
print("top10=" + json.dumps(heads[:10]))
PY

echo "=== Qwen2.5-3B 32k head ablation done ==="
date -Is
