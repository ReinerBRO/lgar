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

TAG="${TAG:-20260608_qwen25_3b_32k_0p5b_public}"
SEQ_LENS="${SEQ_LENS:-4096 8192 12288 16384 24576 32768}"
BENCHMARKS="${BENCHMARKS:-longbench babilong longeval}"
PART="${PART:-1}"
MODEL_ORDER="${MODEL_ORDER:-CE LongCE CURE}"
CKPT_VARIANTS="${CKPT_VARIANTS:-final numbered}"
MAX_EXAMPLES="${MAX_EXAMPLES:-0}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
FORCE_EVAL="${FORCE_EVAL:-0}"
DTYPE="${DTYPE:-bf16}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"
LONGBENCH_E="${LONGBENCH_E:-0}"
USE_KV_CACHE="${USE_KV_CACHE:-0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/reports/public_${TAG}_part${PART}_allckpt_4k_32k_n${MAX_EXAMPLES}}"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/logs}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/zjc_public_${TAG}_part${PART}_allckpt_4k_32k_n${MAX_EXAMPLES}.log}"

if [[ ! -d "${LONGBENCH_ROOT}" && -d "${DATA_ROOT}/longbench" ]]; then
  export LONGBENCH_ROOT="${DATA_ROOT}/longbench"
fi
if [[ ! -d "${BABILONG_ROOT}" && -d "${DATA_ROOT}/babilong" ]]; then
  export BABILONG_ROOT="${DATA_ROOT}/babilong"
fi
if [[ ! -d "${LONGEVAL_ROOT}" && -d "${DATA_ROOT}/longeval/evaluation" ]]; then
  export LONGEVAL_ROOT="${DATA_ROOT}/longeval/evaluation"
fi

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
          if [[ "${name}" == *gtokens0499m ]]; then
            echo "[skip-ckpt] ${base_label}_${name#checkpoint_}: covered by ${base_label}_final"
            continue
          fi
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

tasks_for_benchmark() {
  local benchmark="$1"
  case "${benchmark}:${LONGBENCH_E}:${PART}" in
    longbench:0:1) printf '%s\n' "narrativeqa,qasper,multifieldqa_en,multifieldqa_zh,hotpotqa,2wikimqa,musique,dureader,gov_report,qmsum,multi_news" ;;
    longbench:0:2) printf '%s\n' "vcsum,trec,triviaqa,samsum,lsht,passage_count,passage_retrieval_en,passage_retrieval_zh,lcc,repobench-p" ;;
    longbench:0:all) printf '%s\n' "narrativeqa,qasper,multifieldqa_en,multifieldqa_zh,hotpotqa,2wikimqa,musique,dureader,gov_report,qmsum,multi_news,vcsum,trec,triviaqa,samsum,lsht,passage_count,passage_retrieval_en,passage_retrieval_zh,lcc,repobench-p" ;;
    longbench:1:1) printf '%s\n' "qasper,multifieldqa_en,hotpotqa,2wikimqa,gov_report,multi_news,trec" ;;
    longbench:1:2) printf '%s\n' "triviaqa,samsum,passage_count,passage_retrieval_en,lcc,repobench-p" ;;
    longbench:1:all) printf '%s\n' "qasper,multifieldqa_en,hotpotqa,2wikimqa,gov_report,multi_news,trec,triviaqa,samsum,passage_count,passage_retrieval_en,lcc,repobench-p" ;;
    babilong:*:1) printf '%s\n' "qa1,qa2,qa3,qa4,qa5" ;;
    babilong:*:2) printf '%s\n' "qa6,qa7,qa8,qa9,qa10" ;;
    babilong:*:all) printf '%s\n' "qa1,qa2,qa3,qa4,qa5,qa6,qa7,qa8,qa9,qa10" ;;
    longeval:*:1) printf '%s\n' "topics" ;;
    longeval:*:2) printf '%s\n' "lines" ;;
    longeval:*:all) printf '%s\n' "topics,lines" ;;
    *)
      echo "unsupported PART=${PART}; use 1, 2, or all" >&2
      exit 2
      ;;
  esac
}

length_key_for_seq() {
  case "$1" in
    4096) printf '%s\n' "4k" ;;
    8192) printf '%s\n' "8k" ;;
    12288) printf '%s\n' "12k" ;;
    16384) printf '%s\n' "16k" ;;
    24576) printf '%s\n' "24k" ;;
    32768) printf '%s\n' "32k" ;;
    *) echo "unsupported seq len for BABILong length key: $1" >&2; exit 2 ;;
  esac
}

batch_for_len() {
  case "$1" in
    4096) printf '%s\n' "${EVAL_BATCH_4K:-16}" ;;
    8192) printf '%s\n' "${EVAL_BATCH_8K:-8}" ;;
    12288) printf '%s\n' "${EVAL_BATCH_12K:-6}" ;;
    16384) printf '%s\n' "${EVAL_BATCH_16K:-4}" ;;
    24576) printf '%s\n' "${EVAL_BATCH_24K:-2}" ;;
    32768) printf '%s\n' "${EVAL_BATCH_32K:-2}" ;;
    *) printf '%s\n' "${EVAL_BATCH_SIZE:-1}" ;;
  esac
}

check_babilong_inputs() {
  local length_key="$1"
  local tasks_csv="$2"
  local missing=0 task found
  IFS=',' read -r -a _BABI_TASKS <<< "${tasks_csv}"
  for task in "${_BABI_TASKS[@]}"; do
    found=0
    for path in \
      "${BABILONG_ROOT}/${task}/${length_key}.json" \
      "${BABILONG_ROOT}/${task}/${length_key}.jsonl" \
      "${BABILONG_ROOT}/${task}_${length_key}.json" \
      "${BABILONG_ROOT}/${task}_${length_key}.jsonl"; do
      if [[ -f "${path}" ]]; then
        found=1
      fi
    done
    if [[ "${found}" != "1" ]]; then
      echo "missing BABILong ${task} ${length_key}; expected ${BABILONG_ROOT}/${task}/${length_key}.json or .jsonl" >&2
      missing=1
    fi
  done
  [[ "${missing}" == "0" ]]
}

echo "=== ZJC Qwen2.5-3B public multilen eval ==="
echo "tag=${TAG}"
echo "part=${PART}"
echo "benchmarks=${BENCHMARKS}"
echo "seq_lens=${SEQ_LENS}"
echo "max_examples=${MAX_EXAMPLES} (0 means full cached set)"
echo "model_order=${MODEL_ORDER}"
echo "ckpt_variants=${CKPT_VARIANTS}"
echo "use_kv_cache=${USE_KV_CACHE}"
echo "model_path=${MODEL_PATH}"
echo "longbench_root=${LONGBENCH_ROOT}"
echo "babilong_root=${BABILONG_ROOT}"
echo "longeval_root=${LONGEVAL_ROOT}"
echo "output_root=${OUTPUT_ROOT}"
echo "log_file=${LOG_FILE}"
echo "CE_CKPT=${CE_CKPT}"
echo "LONGCE_CKPT=${LONGCE_CKPT}"
echo "CURE_CKPT=${CURE_CKPT}"
date -Is

require_dir "model" "${MODEL_PATH}"
for benchmark in ${BENCHMARKS}; do
  case "${benchmark}" in
    longbench) require_dir "LongBench root" "${LONGBENCH_ROOT}" ;;
    babilong) require_dir "BABILong root" "${BABILONG_ROOT}" ;;
    longeval) require_dir "LongEval root" "${LONGEVAL_ROOT}" ;;
    *) echo "unsupported benchmark=${benchmark}; use longbench babilong longeval" >&2; exit 2 ;;
  esac
done

CHECKPOINT_ARGS=()
for idx in "${!EVAL_LABELS[@]}"; do
  CHECKPOINT_ARGS+=(--checkpoint "${EVAL_LABELS[$idx]}=${EVAL_CKPTS[$idx]}")
done

for benchmark in ${BENCHMARKS}; do
  tasks="$(tasks_for_benchmark "${benchmark}")"
  for seq_len in ${SEQ_LENS}; do
    length_key="$(length_key_for_seq "${seq_len}")"
    eval_batch_size="$(batch_for_len "${seq_len}")"
    output="${OUTPUT_ROOT}/${benchmark}_${length_key}_part${PART}.json"
    if [[ "${FORCE_EVAL}" != "1" && -s "${output}" ]]; then
      echo "[skip] ${benchmark} ${length_key} part=${PART}: ${output}"
      continue
    fi

    extra_args=()
    if [[ "${USE_KV_CACHE}" == "1" ]]; then
      extra_args+=(--use-cache)
    fi
    if [[ "${benchmark}" == "longbench" && "${LONGBENCH_E}" == "1" ]]; then
      extra_args+=(--longbench-e)
    fi
    if [[ "${benchmark}" == "babilong" ]]; then
      check_babilong_inputs "${length_key}" "${tasks}"
      extra_args+=(--babilong-length "${length_key}")
    fi

    echo "[run] benchmark=${benchmark} length=${length_key} seq_len=${seq_len} batch_size=${eval_batch_size} part=${PART} tasks=${tasks}"
    "${PYTHON}" -m torch.distributed.run --standalone --nproc_per_node="${NPROC_PER_NODE}" -m curcpt.eval_long_context_public \
      --model-path "${MODEL_PATH}" \
      --benchmark "${benchmark}" \
      --tasks "${tasks}" \
      --output "${output}" \
      "${CHECKPOINT_ARGS[@]}" \
      --max-examples "${MAX_EXAMPLES}" \
      --seq-len "${seq_len}" \
      --batch-size "${eval_batch_size}" \
      --dtype "${DTYPE}" \
      --attn-implementation "${ATTN_IMPLEMENTATION}" \
      "${extra_args[@]}"
  done
done

echo "=== ZJC public multilen eval done ==="
echo "output_root=${OUTPUT_ROOT}"
date -Is
