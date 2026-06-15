#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/zjc_env.sh
source "${SCRIPT_DIR}/zjc_env.sh"
cd "${PROJECT_ROOT}"

BASE_TAG="${BASE_TAG:-20260601_160711}"
GLOBAL_TOKENS="${GLOBAL_TOKENS:-2000000000}"
MAX_SAMPLES="${MAX_SAMPLES:-200}"
RULER_SEED="${RULER_SEED:-39}"
SEQ_LENS="${SEQ_LENS:-4096 8192}"
RUN_TRAIN="${RUN_TRAIN:-1}"
RUN_EVAL="${RUN_EVAL:-1}"

CE_CKPT="${CE_CKPT:-${PROJECT_ROOT}/runs/stage2_ce_cpt_${BASE_TAG}/runs/cure/stage2_ce_cpt_${BASE_TAG}/checkpoint.pt}"
LONGCE_CKPT="${LONGCE_CKPT:-${PROJECT_ROOT}/runs/stage2_longce_cpt_${BASE_TAG}/runs/cure/stage2_longce_cpt_${BASE_TAG}/checkpoint.pt}"
ABLATION_JSON="${ABLATION_JSON:-${PROJECT_ROOT}/runs/ablate_full_20260531_131608/ablation_results.json}"
TOP48_HEADS_JSON="${TOP48_HEADS_JSON:-${PROJECT_ROOT}/configs/cure_top48_heads.json}"
TOP96_HEADS_JSON="${TOP96_HEADS_JSON:-${PROJECT_ROOT}/configs/cure_top96_heads.json}"

MATRIX_TAG="${MATRIX_TAG:-cure_expand_${BASE_TAG}}"
REPORT_BASE="${REPORT_BASE:-${PROJECT_ROOT}/reports/${MATRIX_TAG}}"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/logs}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/zjc_${MATRIX_TAG}.log}"

mkdir -p "${LOG_DIR}" "${REPORT_BASE}" "${PROJECT_ROOT}/configs"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "=== ZJC CURE expand matrix ==="
echo "project_root=${PROJECT_ROOT}"
echo "base_tag=${BASE_TAG}"
echo "matrix_tag=${MATRIX_TAG}"
echo "global_tokens=${GLOBAL_TOKENS}"
echo "seq_lens=${SEQ_LENS}"
echo "max_samples=${MAX_SAMPLES}"
echo "run_train=${RUN_TRAIN}"
echo "run_eval=${RUN_EVAL}"
echo "ce_ckpt=${CE_CKPT}"
echo "longce_ckpt=${LONGCE_CKPT}"
echo "ablation_json=${ABLATION_JSON}"
echo "top48_heads_json=${TOP48_HEADS_JSON}"
echo "top96_heads_json=${TOP96_HEADS_JSON}"
echo "report_base=${REPORT_BASE}"
echo "log_file=${LOG_FILE}"
date -Is

require_file "CE checkpoint" "${CE_CKPT}"
require_file "LongCE checkpoint" "${LONGCE_CKPT}"
require_dir "RULER root" "${RULER_DIR}"
require_dir "model" "${MODEL_PATH}"
require_dir "token cache" "${CACHE_DIR}"

export PATH="${ZJC_ENV_DIR}/bin:${PATH}"

find_ablation_json() {
  local candidate
  for candidate in \
    "${ABLATION_JSON}" \
    "${PROJECT_ROOT}/runs/ablate_full_20260531_131608/ablation_results.json" \
    "${PROJECT_ROOT}/runs/ablate_full_20260531_131608/ablation_results_top96.json" \
    "${PROJECT_ROOT}/runs/ablate_full_20260531_131608/ablation_results_top48.json"
  do
    if [[ -s "${candidate}" ]]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done
  return 1
}

if [[ -s "${TOP48_HEADS_JSON}" && -s "${TOP96_HEADS_JSON}" ]]; then
  echo "[heads] reuse existing ${TOP48_HEADS_JSON} and ${TOP96_HEADS_JSON}"
else
  if [[ -s "${TOP96_HEADS_JSON}" && ! -s "${TOP48_HEADS_JSON}" ]]; then
    echo "[heads] derive top48 from existing top96 config"
    "${PYTHON}" - "${TOP96_HEADS_JSON}" "${TOP48_HEADS_JSON}" <<'PY'
import json
import sys
from pathlib import Path

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
data = json.loads(src.read_text(encoding="utf-8"))
heads = data.get("retrieval_heads", [])
if len(heads) < 48:
    raise SystemExit(f"{src} has only {len(heads)} heads, need >=48")
payload = dict(data)
payload["retrieval_heads"] = heads[:48]
payload["num_selected"] = 48
payload["source"] = f"first 48 heads from {src}"
dst.parent.mkdir(parents=True, exist_ok=True)
dst.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
print(f"wrote {dst}")
PY
  fi
fi

if [[ ! -s "${TOP48_HEADS_JSON}" || ! -s "${TOP96_HEADS_JSON}" ]]; then
  RESOLVED_ABLATION_JSON="$(find_ablation_json || true)"
  if [[ -z "${RESOLVED_ABLATION_JSON}" ]]; then
    echo "missing top48/top96 head configs and no usable ablation json found." >&2
    echo "Need one of:" >&2
    echo "  ${TOP48_HEADS_JSON} and ${TOP96_HEADS_JSON}" >&2
    echo "  ${ABLATION_JSON}" >&2
    echo "  ${PROJECT_ROOT}/runs/ablate_full_20260531_131608/ablation_results*.json" >&2
    echo "Existing related files:" >&2
    find "${PROJECT_ROOT}/configs" "${PROJECT_ROOT}/runs" -maxdepth 4 -type f \
      \( -name 'cure_top*_heads.json' -o -name 'ablation_results*.json' \) \
      -print 2>/dev/null | sort >&2 || true
    exit 1
  fi
  echo "[heads] generate top48/top96 from ${RESOLVED_ABLATION_JSON}"
  "${PYTHON}" - "${RESOLVED_ABLATION_JSON}" "${TOP48_HEADS_JSON}" "${TOP96_HEADS_JSON}" <<'PY'
import json
import sys
from pathlib import Path

src = Path(sys.argv[1])
top48_path = Path(sys.argv[2])
top96_path = Path(sys.argv[3])
data = json.loads(src.read_text(encoding="utf-8"))

scores = data.get("head_scores")
ranked = None
if scores:
    ranked = sorted(scores.items(), key=lambda item: float(item[1]), reverse=True)
else:
    h1 = data.get("head_scores_half1", {})
    h2 = data.get("head_scores_half2", {})
    keys = sorted(set(h1) & set(h2))
    scores = {k: (float(h1[k]) + float(h2[k])) / 2.0 for k in keys}
    if scores:
        ranked = sorted(scores.items(), key=lambda item: float(item[1]), reverse=True)

if ranked is None:
    heads = data.get("retrieval_heads", [])
    if len(heads) < 96:
        raise SystemExit(f"{src} has no head_scores and only {len(heads)} retrieval_heads, need >=96")
    ranked = [(f"{int(layer)}_{int(head)}", float(len(heads) - idx)) for idx, (layer, head) in enumerate(heads)]

if len(ranked) < 96:
    raise SystemExit(f"not enough ranked heads in {src}: {len(ranked)} < 96")

for top_k, out in ((48, top48_path), (96, top96_path)):
    heads = [[int(x) for x in key.split("_")] for key, _ in ranked[:top_k]]
    payload = {
        "retrieval_heads": heads,
        "num_selected": top_k,
        "source": str(src),
        "selection": f"top{top_k}_by_head_scores",
        "head_scores": {key: float(value) for key, value in ranked[:top_k]},
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote {out}")
PY
fi

require_file "top48 heads json" "${TOP48_HEADS_JSON}"
require_file "top96 heads json" "${TOP96_HEADS_JSON}"

VARIANTS=(
  "top48_r16|48|16|32|${TOP48_HEADS_JSON}"
  "top48_r32|48|32|64|${TOP48_HEADS_JSON}"
  "top96_r32|96|32|64|${TOP96_HEADS_JSON}"
  "top96_r64|96|64|128|${TOP96_HEADS_JSON}"
)

train_variant() {
  local name="$1"
  local top_k="$2"
  local rank="$3"
  local alpha="$4"
  local heads_json="$5"

  local run_name="stage2_cure_${name}_${BASE_TAG}"
  local output_root="${PROJECT_ROOT}/runs/${run_name}"
  local ckpt="${output_root}/runs/cure/${run_name}/checkpoint.pt"
  local summary="${output_root}/runs/cure/${run_name}/summary.json"

  echo "=== train ${run_name} top=${top_k} rank=${rank} alpha=${alpha} ==="
  echo "checkpoint=${ckpt}"
  if [[ -s "${ckpt}" && -s "${summary}" ]]; then
    echo "[skip train] checkpoint and summary already exist"
    return 0
  fi

  env \
    STAGE2_TAG="${BASE_TAG}" \
    METHOD=cure_cpt \
    RUN_NAME="${run_name}" \
    OUTPUT_ROOT="${output_root}" \
    LOG_FILE="${LOG_DIR}/zjc_stage2_${run_name}.log" \
    GLOBAL_TOKENS="${GLOBAL_TOKENS}" \
    TOP_HEADS_JSON="${heads_json}" \
    CE_CKPT="${CE_CKPT}" \
    LORA_RANK="${rank}" \
    LORA_ALPHA="${alpha}" \
    ADAPTER_LR="${ADAPTER_LR:-1e-4}" \
    ADAPTER_WEIGHT_DECAY="${ADAPTER_WEIGHT_DECAY:-0.0}" \
    LAMBDA_FULL_HU_CE="${LAMBDA_FULL_HU_CE:-1.0}" \
    LAMBDA_RH_CE="${LAMBDA_RH_CE:-0.0}" \
    LAMBDA_RH_KD="${LAMBDA_RH_KD:-0.0}" \
    LAMBDA_COV="${LAMBDA_COV:-0.0}" \
    UTILITY_TOP_FRACTION_TRAINING="${UTILITY_TOP_FRACTION_TRAINING:-0.10}" \
    bash "${SCRIPT_DIR}/zjc_run_stage2_train_one.sh"
}

if [[ "${RUN_TRAIN}" == "1" ]]; then
  for row in "${VARIANTS[@]}"; do
    IFS='|' read -r name top_k rank alpha heads_json <<< "${row}"
    train_variant "${name}" "${top_k}" "${rank}" "${alpha}" "${heads_json}"
  done
fi

TASKS=(
  niah_single_1
  niah_single_2
  niah_single_3
  niah_multikey_1
  niah_multikey_2
  niah_multikey_3
  niah_multivalue
  niah_multiquery
  vt
  cwe
  fwe
  qa_1
  qa_2
)
TASKS_CSV=$(IFS=,; echo "${TASKS[*]}")

score_and_summarize() {
  local output_root="$1"
  local folder="$2"
  local seq_len="$3"

  "${PYTHON}" - "${output_root}" "${folder}" "${seq_len}" "${BASE_TAG}" <<'PY'
import csv
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
folder = sys.argv[2]
seq_len = sys.argv[3]
base_tag = sys.argv[4]

models = ["top48_r16", "top48_r32", "top96_r32", "top96_r64"]
summary = {"folder": folder, "models": {}}

base_report = Path(
    f"/gemini/space/private/zjc/goals/lgar/reports/stage2_gate_eval_{base_tag}"
    f"{'_8k' if seq_len == '8192' else ''}/ruler13_generation/compare_summary.json"
)
if base_report.exists():
    baseline = json.loads(base_report.read_text(encoding="utf-8"))
    for key in ("ce_cpt", "longce_cpt", "cure_v3_fullbudget"):
        if key in baseline.get("models", {}):
            summary["models"][key] = baseline["models"][key]

for model in models:
    csv_path = root / model / folder / "pred" / "summary.csv"
    if not csv_path.exists():
        continue
    with csv_path.open("r", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    tasks = rows[0][1:]
    scores = [float(x) for x in rows[1][1:]]
    summary["models"][model] = {
        "mean_score": sum(scores) / len(scores) if scores else float("nan"),
        "task_scores": dict(zip(tasks, scores)),
    }

if "ce_cpt" in summary["models"]:
    ce = summary["models"]["ce_cpt"]["mean_score"]
    summary["deltas_vs_ce"] = {
        key: value["mean_score"] - ce
        for key, value in summary["models"].items()
        if key != "ce_cpt"
    }

out = root / "compare_summary.json"
out.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
print(out)
print(json.dumps(summary, indent=2, sort_keys=True))
PY
}

run_eval_seq() {
  local seq_len="$1"
  local folder="official${seq_len}_n${MAX_SAMPLES}"
  local data_dir="${RULER_DIR}/generated/${folder}"
  local output_root="${REPORT_BASE}_${seq_len}/ruler13_generation"

  echo "=== eval RULER13 seq=${seq_len} n=${MAX_SAMPLES} ==="
  echo "folder=${folder}"
  echo "output_root=${output_root}"

  RULER_FOLDER="${folder}" SEQ_LEN="${seq_len}" MAX_SAMPLES="${MAX_SAMPLES}" RULER_SEED="${RULER_SEED}" \
    bash "${SCRIPT_DIR}/zjc_prepare_ruler13_data.sh"
  require_dir "RULER generated data" "${data_dir}"

  mkdir -p "${output_root}"
  for row in "${VARIANTS[@]}"; do
    IFS='|' read -r name top_k rank alpha heads_json <<< "${row}"
    local run_name="stage2_cure_${name}_${BASE_TAG}"
    local ckpt="${PROJECT_ROOT}/runs/${run_name}/runs/cure/${run_name}/checkpoint.pt"
    local pred_dir="${output_root}/${name}/${folder}/pred"

    require_file "checkpoint ${name}" "${ckpt}"
    echo "=== generate ${name} seq=${seq_len} ==="
    if [[ -s "${pred_dir}/summary.csv" ]]; then
      echo "[skip eval] existing summary: ${pred_dir}/summary.csv"
      continue
    fi

    mkdir -p "${pred_dir}"
    "${PYTHON}" -m torch.distributed.run --standalone --nproc_per_node="${NPROC_PER_NODE}" -m lgar_cpt.ruler13_generate \
      --model-path "${MODEL_PATH}" \
      --checkpoint-path "${ckpt}" \
      --ruler-dir "${RULER_DIR}" \
      --data-dir "${data_dir}" \
      --save-dir "${pred_dir}" \
      --tasks "${TASKS_CSV}" \
      --subset validation \
      --max-samples "${MAX_SAMPLES}" \
      --mode full \
      --seq-len "${seq_len}" \
      --batch-size "${CURE_EVAL_BATCH_SIZE:-2}" \
      --append-answer-prefix \
      --attn-implementation "${ATTN_IMPLEMENTATION}" \
      --dtype "${DTYPE}"
    "${PYTHON}" -m lgar_cpt.ruler13_score --ruler-dir "${RULER_DIR}" --data-dir "${pred_dir}"
  done

  score_and_summarize "${output_root}" "${folder}" "${seq_len}"
}

if [[ "${RUN_EVAL}" == "1" ]]; then
  for seq_len in ${SEQ_LENS}; do
    run_eval_seq "${seq_len}"
  done
fi

echo "=== ZJC CURE expand matrix done ==="
for seq_len in ${SEQ_LENS}; do
  echo "${REPORT_BASE}_${seq_len}/ruler13_generation/compare_summary.json"
done
date -Is
