#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/zjc_env.sh
source "${SCRIPT_DIR}/zjc_env.sh"
cd "${PROJECT_ROOT}"

TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
REPORT_DIR="${REPORT_DIR:-${PROJECT_ROOT}/reports/long_context_public_fullbudget_zjc_${TIMESTAMP}}"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/logs}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/zjc_public_long_context_${TIMESTAMP}.log}"
BENCHMARKS="${BENCHMARKS:-longbench longeval babilong}"
MAX_EXAMPLES="${MAX_EXAMPLES:-200}"
SEQ_LEN="${SEQ_LEN:-4096}"
MODELS="${MODELS:-ce_cpt longce_cpt cure_v3_fullbudget}"

mkdir -p "${REPORT_DIR}" "${LOG_DIR}"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "=== ZJC public long-context fullbudget eval ==="
echo "project_root=${PROJECT_ROOT}"
echo "report_dir=${REPORT_DIR}"
echo "benchmarks=${BENCHMARKS}"
echo "models=${MODELS}"
echo "max_examples=${MAX_EXAMPLES}"
echo "nproc_per_node=${NPROC_PER_NODE}"
date -Is
if [[ "${SHOW_GPU_INFO:-0}" == "1" ]]; then
  nvidia-smi || true
fi

require_dir "model" "${MODEL_PATH}"
CHECKPOINT_ARGS=()
for model in ${MODELS}; do
  case "${model}" in
    ce_cpt)
      require_file "ce checkpoint" "${CE_CKPT}"
      CHECKPOINT_ARGS+=(--checkpoint "ce_cpt=${CE_CKPT}")
      ;;
    adapter_ce)
      require_file "adapter checkpoint" "${ADAPTER_CKPT}"
      CHECKPOINT_ARGS+=(--checkpoint "adapter_ce=${ADAPTER_CKPT}")
      ;;
    longce_cpt)
      require_file "longce checkpoint" "${LONGCE_CKPT}"
      CHECKPOINT_ARGS+=(--checkpoint "longce_cpt=${LONGCE_CKPT}")
      ;;
    cure_v3_fullbudget|cure_cpt)
      require_file "cure checkpoint" "${CURE_CKPT}"
      CHECKPOINT_ARGS+=(--checkpoint "cure_v3_fullbudget=${CURE_CKPT}")
      ;;
    *)
      echo "unsupported model in MODELS: ${model}" >&2
      exit 2
      ;;
  esac
done

run_benchmark() {
  local benchmark="$1"
  local tasks output
  case "${benchmark}" in
    longbench)
      require_dir "LongBench root" "${LONGBENCH_ROOT}"
      tasks="${LONGBENCH_TASKS:-narrativeqa,hotpotqa,musique,triviaqa,dureader,qmsum}"
      output="${REPORT_DIR}/longbench_fullbudget.json"
      ;;
    longeval)
      require_dir "LongEval root" "${LONGEVAL_ROOT}"
      tasks="${LONGEVAL_TASKS:-topics,lines}"
      output="${REPORT_DIR}/longeval_fullbudget.json"
      ;;
    babilong)
      require_dir "BABILong root" "${BABILONG_ROOT}"
      tasks="${BABILONG_TASKS:-qa1,qa2,qa3,qa4,qa5}"
      output="${REPORT_DIR}/babilong_official1k_fullbudget.json"
      ;;
    *)
      echo "unsupported benchmark: ${benchmark}" >&2
      exit 2
      ;;
  esac

  echo "[run] benchmark=${benchmark} tasks=${tasks} output=${output}"
  "${PYTHON}" -m torch.distributed.run --standalone --nproc_per_node="${NPROC_PER_NODE}" -m curcpt.eval_long_context_public \
    --model-path "${MODEL_PATH}" \
    --benchmark "${benchmark}" \
    --tasks "${tasks}" \
    --output "${output}" \
    "${CHECKPOINT_ARGS[@]}" \
    --max-examples "${MAX_EXAMPLES}" \
    --seq-len "${SEQ_LEN}" \
    --dtype "${DTYPE}" \
    --attn-implementation "${ATTN_IMPLEMENTATION}"
}

for benchmark in ${BENCHMARKS}; do
  run_benchmark "${benchmark}"
done

echo "=== ZJC public long-context eval done ==="
echo "report_dir=${REPORT_DIR}"
date -Is
