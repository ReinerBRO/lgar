#!/bin/bash
#SBATCH -J cure_v3fb
#SBATCH -p acd_u
#SBATCH -n 8
#SBATCH --gres=gpu:4
#SBATCH -o /dev/null
#SBATCH -e /dev/null
#SBATCH -D /data/user/xnie012/pythonprojects/lgar

set -euo pipefail

mkdir -p logs
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="logs/cure_v3_fullbudget_${TIMESTAMP}_${SLURM_JOB_ID}.log"
exec > "${LOG_FILE}" 2>&1

set +u
source /data/user/xnie012/envs/memgen/bin/activate
set -u

export PYTHONUNBUFFERED=1
export PYTHONPATH="/data/user/xnie012/pythonprojects/lgar:${PYTHONPATH:-}"
export HF_HOME=/data/user/xnie012/cache
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export NCCL_IB_DISABLE=1
export NCCL_P2P_LEVEL=NVL
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODEL_PATH=${MODEL_PATH:-/data/user/xnie012/cache/Models/Qwen2.5-0.5B}
CACHE_DIR=${CACHE_DIR:-/data/user/xnie012/pythonprojects/lgar/data/qwen_fineweb_pilot}
CE_CKPT=${CE_CKPT:-/data/user/xnie012/pythonprojects/lgar/runs/ce_cpt_20260531_130512/runs/cure/ce_cpt_pilot/checkpoint.pt}
RULER_DIR=${RULER_DIR:-/data/user/xnie012/cache/RULER_official}
RULER_FOLDER=${RULER_FOLDER:-official4096_n200}
DATA_DIR="${RULER_DIR}/generated/${RULER_FOLDER}"

RUN_NAME=${RUN_NAME:-cure_v3_fullbudget_top24_fullhu10_alr1e4}
OUTPUT_ROOT=${OUTPUT_ROOT:-/data/user/xnie012/pythonprojects/lgar/runs/cure_v3_fullbudget_${TIMESTAMP}}
REPORT_DIR=${REPORT_DIR:-/data/user/xnie012/pythonprojects/lgar/reports/cure_v3_fullbudget_${TIMESTAMP}}
TOKENS_PER_RANK=${TOKENS_PER_RANK:-12500000}
BATCH_SIZE=${BATCH_SIZE:-2}
NUM_PROBE_SEQS=${NUM_PROBE_SEQS:-128}
MAX_SAMPLES=${MAX_SAMPLES:-200}
NPROC_PER_NODE=${NPROC_PER_NODE:-4}

TOP24=/data/user/xnie012/pythonprojects/lgar/runs/ablate_full_20260531_131608/ablation_results_top24.json

mkdir -p "${OUTPUT_ROOT}" "${REPORT_DIR}"

echo "=== CURE-v3 full-budget train/eval ==="
echo "timestamp=${TIMESTAMP}"
echo "run_name=${RUN_NAME}"
echo "tokens_per_rank=${TOKENS_PER_RANK}"
echo "output_root=${OUTPUT_ROOT}"
echo "report_dir=${REPORT_DIR}"
date
nvidia-smi || true

echo "=== train ${RUN_NAME} ==="
torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" -m curcpt.train \
  --run-name "${RUN_NAME}" \
  --method cure_cpt \
  --model-path "${MODEL_PATH}" \
  --cache-dir "${CACHE_DIR}" \
  --output-dir "${OUTPUT_ROOT}" \
  --checkpoint-path "${CE_CKPT}" \
  --ablation-results "${TOP24}" \
  --seq-len 4096 \
  --local-window 1024 \
  --batch-size "${BATCH_SIZE}" \
  --tokens-per-run "${TOKENS_PER_RANK}" \
  --lr 0.0 \
  --adapter-lr 1e-4 \
  --adapter-weight-decay 0.0 \
  --freeze-base-model \
  --lambda-rh-ce 0.0 \
  --lambda-rh-kd 0.0 \
  --lambda-cov 0.0 \
  --lambda-full-hu-ce 1.0 \
  --lora-rank 16 \
  --lora-alpha 32.0 \
  --eval-interval 50 \
  --seed 1337 \
  --dtype bf16 \
  --attn-implementation sdpa

CKPT="${OUTPUT_ROOT}/runs/cure/${RUN_NAME}/checkpoint.pt"
echo "checkpoint=${CKPT}"

echo "=== mechanism probe ==="
python -m curcpt.mechanism_probe \
  --model-path "${MODEL_PATH}" \
  --cache-dir "${CACHE_DIR}" \
  --output "${REPORT_DIR}/${RUN_NAME}_mechanism_probe.json" \
  --checkpoint "ce_cpt=${CE_CKPT}" \
  --checkpoint "${RUN_NAME}=${CKPT}" \
  --reference-name ce_cpt \
  --seq-len 4096 \
  --short-window 1024 \
  --num-sequences "${NUM_PROBE_SEQS}" \
  --batch-size 1 \
  --dtype bf16 \
  --attn-implementation sdpa

python - "${REPORT_DIR}/${RUN_NAME}_mechanism_probe.json" "${RUN_NAME}" <<'PY'
import json
import sys

path, name = sys.argv[1], sys.argv[2]
d = json.load(open(path))
m = d["models"][name]["metrics"]
delta = d.get("deltas_vs_reference", {}).get(name, {})
print(json.dumps({
    "event": "cure_v3_fullbudget_mechanism_summary",
    "variant": name,
    "normal_ce": m.get("normal_ce"),
    "delta_normal_ce_vs_ce": delta.get("normal_ce"),
    "ref_high_ce": m.get("ref_high_ce"),
    "delta_ref_high_ce_vs_ce": delta.get("ref_high_ce"),
    "ref_nonhigh_ce": m.get("ref_nonhigh_ce"),
    "delta_ref_nonhigh_ce_vs_ce": delta.get("ref_nonhigh_ce"),
    "adapter_high_active_minus_off_nll": m.get("adapter_ref_high_active_minus_off_delta_nll"),
    "adapter_all_active_minus_off_nll": m.get("adapter_active_minus_off_delta_nll"),
}, sort_keys=True), flush=True)
PY

echo "=== RULER answer probe ==="
torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" -m curcpt.ruler_answer_probe \
  --model-path "${MODEL_PATH}" \
  --data-dir "${DATA_DIR}" \
  --output "${REPORT_DIR}/${RUN_NAME}_ruler_answer_probe.json" \
  --reference-name ce_cpt \
  --checkpoint "ce_cpt=${CE_CKPT}" \
  --checkpoint "${RUN_NAME}=${CKPT}" \
  --max-samples "${MAX_SAMPLES}" \
  --seq-len 4096 \
  --batch-size 2 \
  --dtype bf16 \
  --attn-implementation sdpa

echo "=== RULER13 official generation/score ==="
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
ln -sfn "_assets/RULER/scripts" "${RULER_DIR}/scripts"
export PYTHONPATH="${RULER_DIR}/_assets/RULER/_python_vendor:${RULER_DIR}/_assets/RULER/scripts:${RULER_DIR}/_assets/RULER/scripts/data:${PYTHONPATH:-}"

PRED_DIR="${REPORT_DIR}/ruler13_official/${RUN_NAME}/${RULER_FOLDER}/pred"
mkdir -p "${PRED_DIR}"

torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" -m lgar_cpt.ruler13_generate \
  --model-path "${MODEL_PATH}" \
  --checkpoint-path "${CKPT}" \
  --ruler-dir "${RULER_DIR}" \
  --data-dir "${DATA_DIR}" \
  --save-dir "${PRED_DIR}" \
  --tasks "${TASKS_CSV}" \
  --subset validation \
  --max-samples "${MAX_SAMPLES}" \
  --mode full \
  --seq-len 4096 \
  --batch-size 2 \
  --append-answer-prefix \
  --attn-implementation sdpa \
  --dtype bf16

python -m lgar_cpt.ruler13_score --ruler-dir "${RULER_DIR}" --data-dir "${PRED_DIR}"

echo "=== CURE-v3 full-budget train/eval done ==="
date
echo "checkpoint=${CKPT}"
echo "report_dir=${REPORT_DIR}"
