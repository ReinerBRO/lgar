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
export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
BASE_SUMMARY="${BASE_SUMMARY:-${PROJECT_ROOT}/reports/stage1_summary.json}"
AGGREGATE_SUMMARY="${PROJECT_ROOT}/reports/fix_rounds_summary.json"
MAX_FIX_ROUNDS="${MAX_FIX_ROUNDS:-5}"

if [[ ! -f "${BASE_SUMMARY}" ]]; then
  echo "Stage1 base summary not found: ${BASE_SUMMARY}" >&2
  exit 2
fi
if [[ ! -f "${PROJECT_ROOT}/reports/stage1_summary_before_fix_rounds.json" ]]; then
  cp "${BASE_SUMMARY}" "${PROJECT_ROOT}/reports/stage1_summary_before_fix_rounds.json"
fi

run_round() {
  local round="$1"
  shift
  echo "==== L-GAR fix round ${round} $(date -Is) ===="
  local round_dir="${PROJECT_ROOT}/fix_rounds/round_${round}"
  case "${round_dir}" in
    "${PROJECT_ROOT}/fix_rounds/round_"*) rm -rf "${round_dir}" ;;
    *) echo "Refusing to clean unexpected round dir: ${round_dir}" >&2; exit 2 ;;
  esac
  mkdir -p "${round_dir}/reports"
  OUTPUT_DIR="${round_dir}" RUN_SELECTION="Qwen_LGAR_CPT=lgar_routed" "$@" bash scripts/run_stage1.sh
  python - <<PY
import json
from pathlib import Path

project = Path(${PROJECT_ROOT@Q})
base_summary = Path(${BASE_SUMMARY@Q})
round_dir = Path(${round_dir@Q})
combined_path = round_dir / "reports" / "stage1_summary.json"
single_path = round_dir / "reports" / "stage1_summary_Qwen_LGAR_CPT.json"
fallback_path = round_dir / "reports" / "stage1_summary.json"

base = json.loads(base_summary.read_text(encoding="utf-8"))
single = json.loads((single_path if single_path.exists() else fallback_path).read_text(encoding="utf-8"))
if "Qwen_LGAR_CPT" not in single:
    raise SystemExit(f"Round {round} did not produce Qwen_LGAR_CPT summary")
combined = dict(base)
combined["Qwen_LGAR_CPT"] = single["Qwen_LGAR_CPT"]
combined_path.write_text(json.dumps(combined, indent=2, sort_keys=True) + "\\n", encoding="utf-8")
PY
  python -m lgar_cpt.review_stage1_results \
    --summary "${round_dir}/reports/stage1_summary.json" \
    --output-json "${round_dir}/reports/stage1_review.json" \
    --output-md "${round_dir}/reports/stage1_review.md"
  python - <<PY
import json
from pathlib import Path

project = Path(${PROJECT_ROOT@Q})
round_dir = Path(${round_dir@Q})
aggregate_path = Path(${AGGREGATE_SUMMARY@Q})
review = json.loads((round_dir / "reports" / "stage1_review.json").read_text(encoding="utf-8"))
entry = {
    "round": int(${round@Q}),
    "round_dir": str(round_dir),
    "status": review.get("status"),
    "blocking_checks": review.get("blocking_checks", []),
    "mechanism": review.get("mechanism", {}),
}
if aggregate_path.exists():
    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
else:
    aggregate = {"rounds": []}
aggregate["rounds"] = [item for item in aggregate.get("rounds", []) if item.get("round") != entry["round"]]
aggregate["rounds"].append(entry)
aggregate["latest_status"] = entry["status"]
aggregate["passed"] = entry["status"] == "pass"
aggregate_path.write_text(json.dumps(aggregate, indent=2, sort_keys=True) + "\\n", encoding="utf-8")
if entry["status"] == "pass":
    (project / "reports" / "stage1_summary.json").write_text(
        (round_dir / "reports" / "stage1_summary.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (project / "reports" / "stage1_review.json").write_text(
        (round_dir / "reports" / "stage1_review.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (project / "reports" / "stage1_review.md").write_text(
        (round_dir / "reports" / "stage1_review.md").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
PY
  python - <<PY
import json
from pathlib import Path

review = json.loads((Path(${round_dir@Q}) / "reports" / "stage1_review.json").read_text(encoding="utf-8"))
raise SystemExit(0 if review.get("status") == "pass" else 1)
PY
}

COMMON_ENV=(
  env
  TOKENS_PER_RUN="${TOKENS_PER_RUN:-150000000}"
  TARGET_CACHE_TOKENS="${TARGET_CACHE_TOKENS:-220000000}"
  MAX_SHARDS="${MAX_SHARDS:-8}"
  SEQ_LEN="${SEQ_LEN:-8192}"
  SHORT_WINDOW="${SHORT_WINDOW:-1024}"
  BATCH_SIZE="${BATCH_SIZE:-1}"
  EVAL_BATCHES="${EVAL_BATCHES:-4}"
)

round_failed=0

if (( MAX_FIX_ROUNDS >= 1 )); then
  # Round 1: stricter labels.
  run_round 1 "${COMMON_ENV[@]}" LSD_TOP_FRACTION=0.05 LONG_NLL_MAX_QUANTILE=0.60 || round_failed=1
  [[ "${round_failed}" == "0" ]] && exit 0
fi
if (( MAX_FIX_ROUNDS >= 2 )); then
  # Round 2: reduce routing harm while preserving the 25% global budget requirement.
  round_failed=0
  run_round 2 "${COMMON_ENV[@]}" LOCAL_WINDOW=2048 FINAL_GLOBAL_BUDGET=0.25 || round_failed=1
  [[ "${round_failed}" == "0" ]] && exit 0
fi
if (( MAX_FIX_ROUNDS >= 3 )); then
  # Round 3: fewer routed layers with the required 25% global budget.
  round_failed=0
  run_round 3 "${COMMON_ENV[@]}" ROUTED_LAYER_FRACTION=0.25 FINAL_GLOBAL_BUDGET=0.25 || round_failed=1
  [[ "${round_failed}" == "0" ]] && exit 0
fi
if (( MAX_FIX_ROUNDS >= 4 )); then
  # Round 4: lower router pressure.
  round_failed=0
  run_round 4 "${COMMON_ENV[@]}" LAMBDA_ROUTER=0.01 LAMBDA_BUDGET=0.005 FINAL_GLOBAL_BUDGET=0.25 || round_failed=1
  [[ "${round_failed}" == "0" ]] && exit 0
fi
if (( MAX_FIX_ROUNDS >= 5 )); then
  # Round 5: conservative labels plus wider local context.
  round_failed=0
  run_round 5 "${COMMON_ENV[@]}" LSD_TOP_FRACTION=0.05 LONG_NLL_MAX_QUANTILE=0.60 LOCAL_WINDOW=2048 ROUTED_LAYER_FRACTION=0.25 FINAL_GLOBAL_BUDGET=0.25 || round_failed=1
  [[ "${round_failed}" == "0" ]] && exit 0
fi

exit 1
