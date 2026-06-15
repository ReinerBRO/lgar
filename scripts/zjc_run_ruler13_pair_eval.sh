#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
USER_CE_CKPT="${CE_CKPT:-}"
USER_LONGCE_CKPT="${LONGCE_CKPT:-}"
USER_CURE_BASE_CKPT="${CURE_BASE_CKPT:-}"
USER_CURE_CE_CKPT="${CURE_CE_CKPT:-}"
USER_CE_LATEST_CKPT="${CE_LATEST_CKPT:-}"
USER_LONGCE_LATEST_CKPT="${LONGCE_LATEST_CKPT:-}"
USER_CURE_BASE_LATEST_CKPT="${CURE_BASE_LATEST_CKPT:-}"
USER_CURE_CE_LATEST_CKPT="${CURE_CE_LATEST_CKPT:-}"
# shellcheck source=scripts/zjc_env.sh
source "${SCRIPT_DIR}/zjc_env.sh"
cd "${PROJECT_ROOT}"

export PATH="${ZJC_ENV_DIR}/bin:${PATH}"
export PYTHON="${PYTHON:-${ZJC_ENV_DIR}/bin/python}"

PAIR="${PAIR:?PAIR is required: ce_longce|cure_pair}"
TAG="${TAG:-20260604_16k_0p5b}"
SEQ_LENS="${SEQ_LENS:-4096 8192 16384}"
MAX_SAMPLES="${MAX_SAMPLES:-200}"
RULER_SEED="${RULER_SEED:-39}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/reports/ruler13_${TAG}_${PAIR}_n${MAX_SAMPLES}}"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/logs}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/zjc_ruler13_${TAG}_${PAIR}_n${MAX_SAMPLES}.log}"
FORCE_EVAL="${FORCE_EVAL:-0}"
PREPARE_RULER="${PREPARE_RULER:-1}"
CKPT_VARIANTS="${CKPT_VARIANTS:-final latest numbered}"
SKIP_MISSING_CKPT_VARIANTS="${SKIP_MISSING_CKPT_VARIANTS:-0}"
RULER_REUSE_PLAIN_FOLDERS="${RULER_REUSE_PLAIN_FOLDERS:-1}"

CE_RUN_DIR="${PROJECT_ROOT}/runs/stage2_ce_cpt_${TAG}/runs/cure/stage2_ce_cpt_${TAG}"
LONGCE_RUN_DIR="${PROJECT_ROOT}/runs/stage2_longce_cpt_${TAG}/runs/cure/stage2_longce_cpt_${TAG}"
CURE_BASE_RUN_DIR="${PROJECT_ROOT}/runs/stage2_cure_a3_from_base_valid_remote_hu_preserve_${TAG}/runs/cure/stage2_cure_a3_from_base_valid_remote_hu_preserve_${TAG}"
CURE_CE_RUN_DIR="${PROJECT_ROOT}/runs/stage2_cure_a3_from_ce_valid_remote_hu_preserve_${TAG}/runs/cure/stage2_cure_a3_from_ce_valid_remote_hu_preserve_${TAG}"
CE_CKPT="${USER_CE_CKPT:-${CE_RUN_DIR}/checkpoint.pt}"
LONGCE_CKPT="${USER_LONGCE_CKPT:-${LONGCE_RUN_DIR}/checkpoint.pt}"
CURE_BASE_CKPT="${USER_CURE_BASE_CKPT:-${CURE_BASE_RUN_DIR}/checkpoint.pt}"
CURE_CE_CKPT="${USER_CURE_CE_CKPT:-${CURE_CE_RUN_DIR}/checkpoint.pt}"
CE_LATEST_CKPT="${USER_CE_LATEST_CKPT:-${CE_RUN_DIR}/checkpoint_latest.pt}"
LONGCE_LATEST_CKPT="${USER_LONGCE_LATEST_CKPT:-${LONGCE_RUN_DIR}/checkpoint_latest.pt}"
CURE_BASE_LATEST_CKPT="${USER_CURE_BASE_LATEST_CKPT:-${CURE_BASE_RUN_DIR}/checkpoint_latest.pt}"
CURE_CE_LATEST_CKPT="${USER_CURE_CE_LATEST_CKPT:-${CURE_CE_RUN_DIR}/checkpoint_latest.pt}"

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
ASSET_PREPARE="${RULER_DIR}/_assets/RULER/scripts/data/prepare.py"

mkdir -p "${OUTPUT_ROOT}" "${LOG_DIR}"
exec > >(tee -a "${LOG_FILE}") 2>&1

MODEL_LABELS=()
MODEL_CKPTS=()
MODEL_BATCH_4K=()
MODEL_BATCH_8K=()
MODEL_BATCH_16K=()

append_model() {
  local label="$1"
  local ckpt="$2"
  local batch_4k="$3"
  local batch_8k="$4"
  local batch_16k="$5"
  MODEL_LABELS+=("${label}")
  MODEL_CKPTS+=("${ckpt}")
  MODEL_BATCH_4K+=("${batch_4k}")
  MODEL_BATCH_8K+=("${batch_8k}")
  MODEL_BATCH_16K+=("${batch_16k}")
}

append_selected_ckpt() {
  local label="$1"
  local ckpt="$2"
  local batch_4k="$3"
  local batch_8k="$4"
  local batch_16k="$5"
  if [[ -f "${ckpt}" || "${SKIP_MISSING_CKPT_VARIANTS}" != "1" ]]; then
    append_model "${label}" "${ckpt}" "${batch_4k}" "${batch_8k}" "${batch_16k}"
  else
    echo "[checkpoint skip missing] ${label}: ${ckpt}"
  fi
}

add_model_variants() {
  local base_label="$1"
  local run_dir="$2"
  local final_ckpt="$3"
  local latest_ckpt="$4"
  local batch_4k="$5"
  local batch_8k="$6"
  local batch_16k="$7"
  local variant ckpt name label

  for variant in ${CKPT_VARIANTS}; do
    case "${variant}" in
      final)
        append_selected_ckpt "${base_label}_final" "${final_ckpt}" "${batch_4k}" "${batch_8k}" "${batch_16k}"
        ;;
      latest|mid|middle)
        append_selected_ckpt "${base_label}_mid_latest" "${latest_ckpt}" "${batch_4k}" "${batch_8k}" "${batch_16k}"
        ;;
      numbered)
        while IFS= read -r ckpt; do
          name="$(basename "${ckpt}" .pt)"
          label="${base_label}_${name#checkpoint_}"
          append_selected_ckpt "${label}" "${ckpt}" "${batch_4k}" "${batch_8k}" "${batch_16k}"
        done < <(find "${run_dir}" -maxdepth 1 -type f \( -name 'checkpoint_step*.pt' -o -name 'checkpoint_tokens*.pt' -o -name 'checkpoint_gtokens*.pt' \) 2>/dev/null | sort)
        ;;
      *)
        echo "unsupported CKPT_VARIANTS entry=${variant}; expected final/latest/numbered" >&2
        exit 2
        ;;
    esac
  done
}

case "${PAIR}" in
  ce_longce)
    add_model_variants "CE" "${CE_RUN_DIR}" "${CE_CKPT}" "${CE_LATEST_CKPT}" "${CE_BATCH_SIZE:-4}" "${CE_BATCH_SIZE:-4}" "${CE_16K_BATCH_SIZE:-1}"
    add_model_variants "LongCE" "${LONGCE_RUN_DIR}" "${LONGCE_CKPT}" "${LONGCE_LATEST_CKPT}" "${LONGCE_BATCH_SIZE:-4}" "${LONGCE_BATCH_SIZE:-4}" "${LONGCE_16K_BATCH_SIZE:-1}"
    ;;
  cure_pair)
    add_model_variants "CURE_base" "${CURE_BASE_RUN_DIR}" "${CURE_BASE_CKPT}" "${CURE_BASE_LATEST_CKPT}" "${CURE_BATCH_SIZE:-2}" "${CURE_BATCH_SIZE:-2}" "${CURE_16K_BATCH_SIZE:-1}"
    add_model_variants "CURE_from_CE" "${CURE_CE_RUN_DIR}" "${CURE_CE_CKPT}" "${CURE_CE_LATEST_CKPT}" "${CURE_FROM_CE_BATCH_SIZE:-2}" "${CURE_FROM_CE_BATCH_SIZE:-2}" "${CURE_FROM_CE_16K_BATCH_SIZE:-1}"
    ;;
  *)
    echo "unsupported PAIR=${PAIR}; expected ce_longce or cure_pair" >&2
    exit 2
    ;;
esac

echo "=== ZJC RULER13 pair eval ==="
echo "pair=${PAIR}"
echo "tag=${TAG}"
echo "seq_lens=${SEQ_LENS}"
echo "max_samples=${MAX_SAMPLES}"
echo "nproc_per_node=${NPROC_PER_NODE}"
echo "output_root=${OUTPUT_ROOT}"
echo "log_file=${LOG_FILE}"
echo "force_eval=${FORCE_EVAL}"
echo "prepare_ruler=${PREPARE_RULER}"
echo "ckpt_variants=${CKPT_VARIANTS}"
for idx in "${!MODEL_LABELS[@]}"; do
  echo "model_${idx}=${MODEL_LABELS[$idx]} ckpt=${MODEL_CKPTS[$idx]}"
done
date -Is

require_file "RULER prepare.py" "${ASSET_PREPARE}"
require_dir "model" "${MODEL_PATH}"
for idx in "${!MODEL_LABELS[@]}"; do
  require_file "checkpoint ${MODEL_LABELS[$idx]}" "${MODEL_CKPTS[$idx]}"
done

ln -sfn "_assets/RULER/scripts" "${RULER_DIR}/scripts"
export PYTHONPATH="${RULER_DIR}/_assets/RULER/_python_vendor:${RULER_DIR}/_assets/RULER/scripts:${RULER_DIR}/_assets/RULER/scripts/data:${PYTHONPATH:-}"

ruler_folder_complete() {
  local folder="$1"
  local data_dir="${RULER_DIR}/generated/${folder}"
  local complete=0
  local f count
  shopt -s nullglob
  for f in "${data_dir}"/*/validation.jsonl; do
    count="$(wc -l < "${f}")"
    if [[ "${count}" -ge "${MAX_SAMPLES}" ]]; then
      complete=$((complete + 1))
    fi
  done
  shopt -u nullglob
  [[ "${complete}" -eq "${#TASKS[@]}" ]]
}

resolve_data_folder() {
  local seqlen="$1"
  local suffixed="official${seqlen}_n${MAX_SAMPLES}"
  local plain="official${seqlen}"
  if ruler_folder_complete "${suffixed}"; then
    echo "${suffixed}"
    return 0
  fi
  if [[ "${RULER_REUSE_PLAIN_FOLDERS}" == "1" ]] && ruler_folder_complete "${plain}"; then
    echo "${plain}"
    return 0
  fi
  echo "${suffixed}"
}

prepare_len_unlocked() {
  local seqlen="$1"
  local folder="official${seqlen}_n${MAX_SAMPLES}"
  local data_dir="${RULER_DIR}/generated/${folder}"
  mkdir -p "${data_dir}"
  for task in "${TASKS[@]}"; do
    local task_file="${data_dir}/${task}/validation.jsonl"
    if [[ -f "${task_file}" && "$(wc -l < "${task_file}")" -ge "${MAX_SAMPLES}" ]]; then
      echo "[prepare skip] ${folder}/${task}"
      continue
    fi
    echo "[prepare] ${folder}/${task}"
    "${PYTHON}" "${ASSET_PREPARE}" \
      --save_dir "${data_dir}" \
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
}

prepare_len() {
  local seqlen="$1"
  local folder
  folder="$(resolve_data_folder "${seqlen}")"
  if [[ "${PREPARE_RULER}" != "1" ]]; then
    echo "[prepare disabled] seqlen=${seqlen} folder=${folder}"
    return 0
  fi
  if ruler_folder_complete "${folder}"; then
    echo "[prepare complete] ${folder}"
    return 0
  fi

  local lock_dir="${RULER_DIR}/generated/.prepare_official${seqlen}_n${MAX_SAMPLES}.lock"
  mkdir -p "${RULER_DIR}/generated"
  while ! mkdir "${lock_dir}" 2>/dev/null; do
    echo "[prepare wait] official${seqlen}_n${MAX_SAMPLES} lock=${lock_dir}"
    sleep 30
    folder="$(resolve_data_folder "${seqlen}")"
    if ruler_folder_complete "${folder}"; then
      echo "[prepare complete after wait] ${folder}"
      return 0
    fi
  done

  trap 'rm -rf "${lock_dir}"' RETURN
  prepare_len_unlocked "${seqlen}"
  rm -rf "${lock_dir}"
  trap - RETURN
}

batch_for_model() {
  local idx="$1"
  local seqlen="$2"
  case "${seqlen}" in
    4096) echo "${MODEL_BATCH_4K[$idx]}" ;;
    8192) echo "${MODEL_BATCH_8K[$idx]}" ;;
    16384) echo "${MODEL_BATCH_16K[$idx]}" ;;
    *) echo "${EVAL_BATCH_SIZE:-1}" ;;
  esac
}

run_one() {
  local idx="$1"
  local seqlen="$2"
  local label="${MODEL_LABELS[$idx]}"
  local ckpt="${MODEL_CKPTS[$idx]}"
  local batch_size
  batch_size="$(batch_for_model "${idx}" "${seqlen}")"

  local folder
  folder="$(resolve_data_folder "${seqlen}")"
  local data_dir="${RULER_DIR}/generated/${folder}"
  local pred_dir="${OUTPUT_ROOT}/${label}/${folder}/pred"
  local summary_csv="${pred_dir}/summary.csv"
  mkdir -p "${pred_dir}"

  if [[ "${FORCE_EVAL}" != "1" && -s "${summary_csv}" ]]; then
    echo "[eval skip] ${label} ${folder}: ${summary_csv}"
    return 0
  fi

  echo "=== eval label=${label} seqlen=${seqlen} folder=${folder} batch=${batch_size} ==="
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
    --seq-len "${seqlen}" \
    --batch-size "${batch_size}" \
    --append-answer-prefix \
    --attn-implementation "${ATTN_IMPLEMENTATION}" \
    --dtype "${DTYPE}"
  "${PYTHON}" -m lgar_cpt.ruler13_score --ruler-dir "${RULER_DIR}" --data-dir "${pred_dir}"
}

for seqlen in ${SEQ_LENS}; do
  prepare_len "${seqlen}"
  for idx in "${!MODEL_LABELS[@]}"; do
    run_one "${idx}" "${seqlen}"
  done
done

"${PYTHON}" - "${OUTPUT_ROOT}" "${MAX_SAMPLES}" ${SEQ_LENS} -- "${MODEL_LABELS[@]}" <<'PY'
import csv
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
max_samples = int(sys.argv[2])
sep = sys.argv.index("--")
seq_lens = [int(x) for x in sys.argv[3:sep]]
labels = sys.argv[sep + 1 :]

summary = {"max_samples": max_samples, "lengths": {}, "models": labels}
for seqlen in seq_lens:
    candidates = [f"official{seqlen}_n{max_samples}", f"official{seqlen}"]
    section = {}
    for label in labels:
        csv_path = None
        folder = None
        for candidate in candidates:
            candidate_csv = root / label / candidate / "pred" / "summary.csv"
            if candidate_csv.exists():
                csv_path = candidate_csv
                folder = candidate
                break
        if csv_path is None or folder is None:
            continue
        with csv_path.open("r", encoding="utf-8") as handle:
            rows = list(csv.reader(handle))
        tasks = rows[0][1:]
        scores = [float(x) for x in rows[1][1:]]
        section[label] = {
            "mean_score": sum(scores) / len(scores) if scores else float("nan"),
            "task_scores": dict(zip(tasks, scores)),
            "folder": folder,
            "summary_csv": str(csv_path),
        }
    summary["lengths"][str(seqlen)] = section

out = root / "compare_summary.json"
out.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
print(out)
print(json.dumps(summary, indent=2, sort_keys=True))
PY

echo "=== ZJC RULER13 pair eval done ==="
echo "output_root=${OUTPUT_ROOT}"
date -Is
