#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
USER_RULER_FOLDER="${RULER_FOLDER:-}"
# shellcheck source=scripts/zjc_env.sh
source "${SCRIPT_DIR}/zjc_env.sh"
cd "${PROJECT_ROOT}"

TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
REPORT_DIR="${REPORT_DIR:-${PROJECT_ROOT}/reports/stage2_gate_eval_${TIMESTAMP}}"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/logs}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/zjc_stage2_gate_eval_${TIMESTAMP}.log}"
MODELS="${MODELS:-ce_cpt longce_cpt cure_v3_fullbudget}"
SEQ_LEN="${EVAL_SEQ_LEN:-4096}"
SHORT_WINDOW="${SHORT_WINDOW:-1024}"
MAX_SAMPLES="${MAX_SAMPLES:-200}"
NUM_PROBE_SEQS="${NUM_PROBE_SEQS:-128}"
if [[ -z "${USER_RULER_FOLDER}" ]]; then
  RULER_FOLDER="official${SEQ_LEN}_n${MAX_SAMPLES}"
fi

mkdir -p "${REPORT_DIR}" "${LOG_DIR}"
exec > >(tee -a "${LOG_FILE}") 2>&1

checkpoint_args=()
for model in ${MODELS}; do
  case "${model}" in
    ce_cpt)
      require_file "ce checkpoint" "${CE_CKPT}"
      checkpoint_args+=(--checkpoint "ce_cpt=${CE_CKPT}")
      ;;
    adapter_ce)
      require_file "adapter checkpoint" "${ADAPTER_CKPT}"
      checkpoint_args+=(--checkpoint "adapter_ce=${ADAPTER_CKPT}")
      ;;
    longce_cpt)
      require_file "longce checkpoint" "${LONGCE_CKPT}"
      checkpoint_args+=(--checkpoint "longce_cpt=${LONGCE_CKPT}")
      ;;
    cure_v3_fullbudget|cure_cpt)
      require_file "cure checkpoint" "${CURE_CKPT}"
      checkpoint_args+=(--checkpoint "cure_v3_fullbudget=${CURE_CKPT}")
      ;;
    *)
      echo "unsupported model in MODELS: ${model}" >&2
      exit 2
      ;;
  esac
done

echo "=== ZJC Stage2 gate eval, sampled only ==="
echo "report_dir=${REPORT_DIR}"
echo "models=${MODELS}"
echo "max_samples=${MAX_SAMPLES}"
echo "num_probe_seqs=${NUM_PROBE_SEQS}"
echo "seq_len=${SEQ_LEN}"
echo "ruler_folder=${RULER_FOLDER}"
date -Is
if [[ "${SHOW_GPU_INFO:-0}" == "1" ]]; then
  nvidia-smi || true
fi

require_dir "model" "${MODEL_PATH}"
require_dir "token cache" "${CACHE_DIR}"

echo "=== mechanism probe extended ==="
"${PYTHON}" -m curcpt.mechanism_probe_extended \
  --model-path "${MODEL_PATH}" \
  --cache-dir "${CACHE_DIR}" \
  --output "${REPORT_DIR}/mechanism_probe_extended.json" \
  "${checkpoint_args[@]}" \
  --reference-name ce_cpt \
  --reference-checkpoint-path "${CE_CKPT}" \
  --retrieval-heads-json "${TOP_HEADS_JSON}" \
  --seq-len "${SEQ_LEN}" \
  --short-window "${SHORT_WINDOW}" \
  --num-sequences "${NUM_PROBE_SEQS}" \
  --batch-size 1 \
  --dtype "${DTYPE}" \
  --attn-implementation "${ATTN_IMPLEMENTATION}"

echo "=== RULER answer probe n=${MAX_SAMPLES} ==="
RULER_DATA_DIR="${RULER_DIR}/generated/${RULER_FOLDER}"
if [[ ! -d "${RULER_DATA_DIR}" ]]; then
  echo "[prepare] missing RULER data, building ${RULER_FOLDER}"
  RULER_FOLDER="${RULER_FOLDER}" SEQ_LEN="${SEQ_LEN}" MAX_SAMPLES="${MAX_SAMPLES}" \
    bash "${SCRIPT_DIR}/zjc_prepare_ruler13_data.sh"
fi
require_dir "RULER generated data" "${RULER_DATA_DIR}"
"${PYTHON}" -m torch.distributed.run --standalone --nproc_per_node="${NPROC_PER_NODE}" -m curcpt.ruler_answer_probe \
  --model-path "${MODEL_PATH}" \
  --data-dir "${RULER_DATA_DIR}" \
  --output "${REPORT_DIR}/ruler_answer_probe_n${MAX_SAMPLES}.json" \
  "${checkpoint_args[@]}" \
  --reference-name ce_cpt \
  --max-samples "${MAX_SAMPLES}" \
  --seq-len "${SEQ_LEN}" \
  --batch-size "${RULER_PROBE_BATCH_SIZE:-2}" \
  --dtype "${DTYPE}" \
  --attn-implementation "${ATTN_IMPLEMENTATION}"

if [[ "${RUN_RULER_GENERATION:-1}" == "1" ]]; then
  echo "=== RULER13 generation n=${MAX_SAMPLES}; not full benchmark ==="
  MODELS="${MODELS}" OUTPUT_ROOT="${REPORT_DIR}/ruler13_generation" MAX_SAMPLES="${MAX_SAMPLES}" SEQ_LEN="${SEQ_LEN}" \
    bash "${SCRIPT_DIR}/zjc_run_ruler13_fullbudget.sh"
fi

if [[ "${RUN_PUBLIC_LONG_CONTEXT:-1}" == "1" ]]; then
  echo "=== sampled public long-context eval n=${MAX_SAMPLES}; default LongBench only ==="
  MODELS="${MODELS}" REPORT_DIR="${REPORT_DIR}/public_long_context" MAX_EXAMPLES="${MAX_SAMPLES}" SEQ_LEN="${SEQ_LEN}" \
    BENCHMARKS="${PUBLIC_BENCHMARKS:-longbench}" \
    bash "${SCRIPT_DIR}/zjc_run_public_long_context_fullbudget.sh"
fi

if [[ "${RUN_NOLIMA:-1}" == "1" ]]; then
  echo "=== NoLiMA sampled 1/3; not full ==="
  MODELS="${MODELS}" OUTPUT_ROOT="${REPORT_DIR}/nolima" NOLIMA_SAMPLE_FRACTION="${NOLIMA_SAMPLE_FRACTION:-0.333333}" \
    bash "${SCRIPT_DIR}/zjc_run_nolima_fullbudget.sh"
fi

"${PYTHON}" - "${REPORT_DIR}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
summary = {
    "report_dir": str(root),
    "files": sorted(str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()),
}
out = root / "gate_eval_manifest.json"
out.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
print(out)
PY

echo "=== ZJC Stage2 gate eval done ==="
echo "report_dir=${REPORT_DIR}"
date -Is
