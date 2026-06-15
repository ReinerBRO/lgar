#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
USER_CE_CKPT="${CE_CKPT:-}"
USER_LONGCE_CKPT="${LONGCE_CKPT:-}"
USER_CURE_CKPT="${CURE_CKPT:-}"
export MODEL_PATH="${MODEL_PATH:-/gemini/space/private/zjc/models/Qwen2.5-3B}"
# shellcheck source=scripts/zjc_env.sh
source "${SCRIPT_DIR}/zjc_env.sh"
cd "${PROJECT_ROOT}"

export PATH="${ZJC_ENV_DIR}/bin:${PATH}"
export PYTHON="${PYTHON:-${ZJC_ENV_DIR}/bin/python}"
export PYTHONPATH="${RULER_DIR}/_assets/RULER/_python_vendor:${RULER_DIR}/_assets/RULER/scripts:${RULER_DIR}/_assets/RULER/scripts/data:${PYTHONPATH:-}"

TAG="${TAG:-20260607_qwen25_3b_32k_0p5b}"
SEQ_LENS="${SEQ_LENS:-4096 8192 12288 16384 24576 32768}"
MAX_SAMPLES="${MAX_SAMPLES:-200}"
RULER_SEED="${RULER_SEED:-39}"
MODEL_ORDER="${MODEL_ORDER:-CE LongCE CURE}"
CKPT_VARIANTS="${CKPT_VARIANTS:-final numbered}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
FORCE_EVAL="${FORCE_EVAL:-0}"
PREPARE_RULER="${PREPARE_RULER:-1}"
RULER_REUSE_PLAIN_FOLDERS="${RULER_REUSE_PLAIN_FOLDERS:-1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/reports/ruler13_${TAG}_alltasks_4k_32k_n${MAX_SAMPLES}}"
OLD_SPLIT_OUTPUT_ROOT="${OLD_SPLIT_OUTPUT_ROOT:-${PROJECT_ROOT}/reports/ruler13_${TAG}_ruler12_4k_32k_n${MAX_SAMPLES}}"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/logs}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/zjc_ruler13_${TAG}_alltasks_4k_32k_n${MAX_SAMPLES}.log}"

TASKS_CSV="${TASKS_CSV:-niah_single_1,niah_single_2,niah_single_3,niah_multikey_1,niah_multikey_2,niah_multikey_3,niah_multivalue,niah_multiquery,vt,cwe,fwe,qa_1,qa_2}"
ASSET_PREPARE="${RULER_DIR}/_assets/RULER/scripts/data/prepare.py"

mkdir -p "${OUTPUT_ROOT}" "${LOG_DIR}"
exec > >(tee -a "${LOG_FILE}") 2>&1

find_final_ckpt() {
  local label="$1"
  local pattern="$2"
  local ckpt
  ckpt="$(
    find "${PROJECT_ROOT}/runs" -type f -name checkpoint.pt -path "*${pattern}*" 2>/dev/null \
      | sort \
      | tail -n 1
  )"
  if [[ -z "${ckpt}" ]]; then
    echo "missing ${label} final checkpoint under ${PROJECT_ROOT}/runs pattern=${pattern}" >&2
    exit 1
  fi
  printf '%s\n' "${ckpt}"
}

CE_CKPT="${USER_CE_CKPT:-$(find_final_ckpt CE "stage2_qwen25_3b_ce_32k_0p5b")}"
LONGCE_CKPT="${USER_LONGCE_CKPT:-$(find_final_ckpt LongCE "stage2_qwen25_3b_longce_32k_0p5b")}"
CURE_CKPT="${USER_CURE_CKPT:-$(find_final_ckpt CURE "stage2_qwen25_3b_cure_a3_top48_32k_0p5b")}"
CE_RUN_DIR="${CE_RUN_DIR:-$(dirname "${CE_CKPT}")}"
LONGCE_RUN_DIR="${LONGCE_RUN_DIR:-$(dirname "${LONGCE_CKPT}")}"
CURE_RUN_DIR="${CURE_RUN_DIR:-$(dirname "${CURE_CKPT}")}"

EVAL_LABELS=()
EVAL_CKPTS=()

append_eval_ckpt() {
  local label="$1"
  local ckpt="$2"
  if [[ ! -f "${ckpt}" ]]; then
    echo "missing checkpoint ${label}: ${ckpt}" >&2
    exit 1
  fi
  EVAL_LABELS+=("${label}")
  EVAL_CKPTS+=("${ckpt}")
}

add_model_ckpt_variants() {
  local base_label="$1"
  local run_dir="$2"
  local final_ckpt="$3"
  local latest_ckpt="${run_dir}/checkpoint_latest.pt"
  local variant ckpt name label
  for variant in ${CKPT_VARIANTS}; do
    case "${variant}" in
      final)
        append_eval_ckpt "${base_label}_final" "${final_ckpt}"
        ;;
      latest)
        append_eval_ckpt "${base_label}_latest" "${latest_ckpt}"
        ;;
      numbered)
        while IFS= read -r ckpt; do
          name="$(basename "${ckpt}" .pt)"
          label="${base_label}_${name#checkpoint_}"
          append_eval_ckpt "${label}" "${ckpt}"
        done < <(find "${run_dir}" -maxdepth 1 -type f \( -name 'checkpoint_step*.pt' -o -name 'checkpoint_tokens*.pt' -o -name 'checkpoint_gtokens*.pt' \) 2>/dev/null | sort)
        ;;
      *)
        echo "unsupported CKPT_VARIANTS entry=${variant}; use final/latest/numbered" >&2
        exit 2
        ;;
    esac
  done
}

for model_label in ${MODEL_ORDER}; do
  case "${model_label}" in
    CE) add_model_ckpt_variants "CE" "${CE_RUN_DIR}" "${CE_CKPT}" ;;
    LongCE) add_model_ckpt_variants "LongCE" "${LONGCE_RUN_DIR}" "${LONGCE_CKPT}" ;;
    CURE) add_model_ckpt_variants "CURE" "${CURE_RUN_DIR}" "${CURE_CKPT}" ;;
    *) echo "unsupported model label=${model_label}; use CE LongCE CURE" >&2; exit 2 ;;
  esac
done

tasks_array() {
  local csv="$1"
  IFS=',' read -r -a _TASK_ARRAY <<< "${csv}"
}

ruler_folder_complete() {
  local folder="$1"
  local tasks_csv="$2"
  local task file count
  tasks_array "${tasks_csv}"
  for task in "${_TASK_ARRAY[@]}"; do
    file="${RULER_DIR}/generated/${folder}/${task}/validation.jsonl"
    [[ -f "${file}" ]] || return 1
    count="$(wc -l < "${file}")"
    [[ "${count}" -ge "${MAX_SAMPLES}" ]] || return 1
  done
  return 0
}

resolve_data_folder() {
  local seqlen="$1"
  local tasks_csv="$2"
  local suffixed="official${seqlen}_n${MAX_SAMPLES}"
  local plain="official${seqlen}"
  if ruler_folder_complete "${suffixed}" "${tasks_csv}"; then
    printf '%s\n' "${suffixed}"
  elif [[ "${RULER_REUSE_PLAIN_FOLDERS}" == "1" ]] && ruler_folder_complete "${plain}" "${tasks_csv}"; then
    printf '%s\n' "${plain}"
  else
    printf '%s\n' "${suffixed}"
  fi
}

prepare_len() {
  local seqlen="$1"
  local folder
  folder="$(resolve_data_folder "${seqlen}" "${TASKS_CSV}")"
  if [[ "${PREPARE_RULER}" != "1" ]]; then
    echo "[prepare disabled] seqlen=${seqlen} folder=${folder}"
    return 0
  fi
  if ruler_folder_complete "${folder}" "${TASKS_CSV}"; then
    echo "[prepare complete] ${folder}"
    return 0
  fi

  local lock_dir="${RULER_DIR}/generated/.prepare_${folder}.lock"
  mkdir -p "${RULER_DIR}/generated"
  while ! mkdir "${lock_dir}" 2>/dev/null; do
    echo "[prepare wait] ${folder} lock=${lock_dir}"
    sleep 30
    folder="$(resolve_data_folder "${seqlen}" "${TASKS_CSV}")"
    if ruler_folder_complete "${folder}" "${TASKS_CSV}"; then
      echo "[prepare complete after wait] ${folder}"
      return 0
    fi
  done

  trap 'rm -rf "${lock_dir}"' RETURN
  local task file line_count
  tasks_array "${TASKS_CSV}"
  for task in "${_TASK_ARRAY[@]}"; do
    file="${RULER_DIR}/generated/${folder}/${task}/validation.jsonl"
    if [[ -f "${file}" ]]; then
      line_count="$(wc -l < "${file}")"
      if [[ "${line_count}" -ge "${MAX_SAMPLES}" ]]; then
        echo "[prepare skip] ${folder}/${task}: ${line_count} rows"
        continue
      fi
    fi
    echo "[prepare] ${folder}/${task}"
    "${PYTHON}" "${ASSET_PREPARE}" \
      --save_dir "${RULER_DIR}/generated/${folder}" \
      --benchmark synthetic \
      --task "${task}" \
      --subset validation \
      --tokenizer_path "${MODEL_PATH}" \
      --tokenizer_type hf \
      --max_seq_length "${seqlen}" \
      --model_template_type base \
      --num_samples "${MAX_SAMPLES}" \
      --random_seed "${RULER_SEED}"
  done
  rm -rf "${lock_dir}"
  trap - RETURN
}

batch_for_len() {
  case "$1" in
    4096) printf '%s\n' "${EVAL_BATCH_4K:-16}" ;;
    8192) printf '%s\n' "${EVAL_BATCH_8K:-8}" ;;
    12288) printf '%s\n' "${EVAL_BATCH_12K:-4}" ;;
    16384) printf '%s\n' "${EVAL_BATCH_16K:-4}" ;;
    24576) printf '%s\n' "${EVAL_BATCH_24K:-2}" ;;
    32768) printf '%s\n' "${EVAL_BATCH_32K:-1}" ;;
    *) printf '%s\n' "${EVAL_BATCH_SIZE:-1}" ;;
  esac
}

batch_map_arg() {
  local items=()
  local seqlen
  for seqlen in ${SEQ_LENS}; do
    items+=("${seqlen}=$(batch_for_len "${seqlen}")")
  done
  printf '%s\n' "${items[*]}"
}

score_label() {
  local label="$1"
  local seqlen folder pred_dir summary_csv
  for seqlen in ${SEQ_LENS}; do
    folder="$(resolve_data_folder "${seqlen}" "${TASKS_CSV}")"
    pred_dir="${OUTPUT_ROOT}/${label}/${folder}/pred"
    summary_csv="${pred_dir}/summary.csv"
    if [[ "${FORCE_EVAL}" != "1" && -s "${summary_csv}" ]]; then
      echo "[score skip] ${label} ${folder}: ${summary_csv}"
      continue
    fi
    echo "[score] ${label} ${folder}"
    "${PYTHON}" -m lgar_cpt.ruler13_score --ruler-dir "${RULER_DIR}" --data-dir "${pred_dir}"
  done
}

reuse_old_split_outputs() {
  local label="$1"
  local seqlen folder new_pred new_summary part1_pred part2_pred
  [[ -d "${OLD_SPLIT_OUTPUT_ROOT}" ]] || return 0
  for seqlen in ${SEQ_LENS}; do
    folder="$(resolve_data_folder "${seqlen}" "${TASKS_CSV}")"
    new_pred="${OUTPUT_ROOT}/${label}/${folder}/pred"
    new_summary="${new_pred}/summary.csv"
    if [[ "${FORCE_EVAL}" != "1" && -s "${new_summary}" ]]; then
      continue
    fi
    part1_pred="${OLD_SPLIT_OUTPUT_ROOT}/${label}/ruler1/${folder}/pred"
    part2_pred="${OLD_SPLIT_OUTPUT_ROOT}/${label}/ruler2/${folder}/pred"
    if [[ -s "${part1_pred}/summary.csv" && -s "${part2_pred}/summary.csv" ]]; then
      echo "[reuse old split] ${label} ${folder}: ${part1_pred} + ${part2_pred} -> ${new_pred}"
      mkdir -p "${new_pred}"
      find "${part1_pred}" "${part2_pred}" -maxdepth 1 -type f -name '*.jsonl' -exec cp -f {} "${new_pred}/" \;
      "${PYTHON}" -m lgar_cpt.ruler13_score --ruler-dir "${RULER_DIR}" --data-dir "${new_pred}"
    fi
  done
}

echo "=== ZJC Qwen2.5-3B RULER13 multi-length eval ==="
echo "tag=${TAG}"
echo "model_path=${MODEL_PATH}"
echo "seq_lens=${SEQ_LENS}"
echo "tasks=${TASKS_CSV}"
echo "model_order=${MODEL_ORDER}"
echo "ckpt_variants=${CKPT_VARIANTS}"
echo "max_samples=${MAX_SAMPLES}"
echo "nproc_per_node=${NPROC_PER_NODE}"
echo "output_root=${OUTPUT_ROOT}"
echo "old_split_output_root=${OLD_SPLIT_OUTPUT_ROOT}"
echo "log_file=${LOG_FILE}"
echo "CE_CKPT=${CE_CKPT}"
echo "LONGCE_CKPT=${LONGCE_CKPT}"
echo "CURE_CKPT=${CURE_CKPT}"
echo "CE_RUN_DIR=${CE_RUN_DIR}"
echo "LONGCE_RUN_DIR=${LONGCE_RUN_DIR}"
echo "CURE_RUN_DIR=${CURE_RUN_DIR}"
for idx in "${!EVAL_LABELS[@]}"; do
  echo "eval_${idx}=${EVAL_LABELS[$idx]} ckpt=${EVAL_CKPTS[$idx]}"
done
date -Is

require_dir "model" "${MODEL_PATH}"
require_file "RULER prepare.py" "${ASSET_PREPARE}"
for idx in "${!EVAL_LABELS[@]}"; do
  require_file "checkpoint ${EVAL_LABELS[$idx]}" "${EVAL_CKPTS[$idx]}"
done
ln -sfn "_assets/RULER/scripts" "${RULER_DIR}/scripts"

for seqlen in ${SEQ_LENS}; do
  prepare_len "${seqlen}"
done

reuse_args=()
if [[ "${RULER_REUSE_PLAIN_FOLDERS}" == "1" ]]; then
  reuse_args+=(--reuse-plain-folders)
fi
force_args=()
if [[ "${FORCE_EVAL}" == "1" ]]; then
  force_args+=(--force-eval)
fi

for idx in "${!EVAL_LABELS[@]}"; do
  label="${EVAL_LABELS[$idx]}"
  ckpt="${EVAL_CKPTS[$idx]}"
  reuse_old_split_outputs "${label}"
  echo "=== eval checkpoint label=${label} ckpt=${ckpt} ==="
  "${PYTHON}" -m torch.distributed.run --standalone --nproc_per_node="${NPROC_PER_NODE}" -m lgar_cpt.ruler13_generate_multi \
    --model-path "${MODEL_PATH}" \
    --checkpoint-path "${ckpt}" \
    --ruler-dir "${RULER_DIR}" \
    --output-root "${OUTPUT_ROOT}" \
    --label "${label}" \
    --seq-lens "${SEQ_LENS}" \
    --tasks "${TASKS_CSV}" \
    --subset validation \
    --max-samples "${MAX_SAMPLES}" \
    --mode full \
    --batch-size-map "$(batch_map_arg)" \
    --append-answer-prefix \
    --attn-implementation "${ATTN_IMPLEMENTATION}" \
    --dtype "${DTYPE}" \
    "${reuse_args[@]}" \
    "${force_args[@]}"
  score_label "${label}"
done

"${PYTHON}" - "${OUTPUT_ROOT}" "${MAX_SAMPLES}" "${SEQ_LENS}" -- "${EVAL_LABELS[@]}" <<'PY'
import csv
import json
import math
import sys
from pathlib import Path

root = Path(sys.argv[1])
max_samples = int(sys.argv[2])
seq_lens = [int(x) for x in sys.argv[3].split()]
sep = sys.argv.index("--")
labels = sys.argv[sep + 1 :]

summary = {"max_samples": max_samples, "lengths": {}, "models": labels}
rows = []
for seqlen in seq_lens:
    summary["lengths"][str(seqlen)] = {}
    row = {"seq_len": seqlen}
    for label in labels:
        csv_path = None
        folder = None
        for candidate in (f"official{seqlen}_n{max_samples}", f"official{seqlen}"):
            candidate_csv = root / label / candidate / "pred" / "summary.csv"
            if candidate_csv.exists():
                csv_path = candidate_csv
                folder = candidate
                break
        if csv_path is None:
            row[label] = math.nan
            summary["lengths"][str(seqlen)][label] = {"mean_score": math.nan, "task_scores": {}, "summary_csv": None}
            continue
        with csv_path.open("r", encoding="utf-8") as handle:
            csv_rows = list(csv.reader(handle))
        tasks = csv_rows[0][1:]
        scores = [float(x) for x in csv_rows[1][1:]]
        mean = sum(scores) / len(scores) if scores else math.nan
        row[label] = mean
        summary["lengths"][str(seqlen)][label] = {
            "mean_score": mean,
            "task_scores": dict(zip(tasks, scores)),
            "folder": folder,
            "summary_csv": str(csv_path),
        }
    rows.append(row)

csv_path = root / "compare_mean.csv"
with csv_path.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=["seq_len", *labels])
    writer.writeheader()
    for row in rows:
        writer.writerow(row)

json_path = root / "compare_summary.json"
json_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
print(f"wrote {csv_path}")
print(f"wrote {json_path}")
print("seq_len," + ",".join(labels))
for row in rows:
    vals = [str(row["seq_len"])]
    for label in labels:
        val = row.get(label, math.nan)
        vals.append("MISSING" if math.isnan(val) else f"{val:.3f}")
    print(",".join(vals))
PY

echo "=== ZJC Qwen2.5-3B RULER13 multi-length eval done ==="
echo "output_root=${OUTPUT_ROOT}"
date -Is
