#!/bin/bash
#SBATCH -J cure_v1
#SBATCH -p acd_u
#SBATCH -n 8
#SBATCH --gres=gpu:4
#SBATCH -o /dev/null
#SBATCH -e /dev/null
#SBATCH -D /data/user/xnie012/pythonprojects/lgar

set -euo pipefail

mkdir -p logs
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="logs/cure_v1_matrix_${TIMESTAMP}_${SLURM_JOB_ID}.log"
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
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODEL_PATH=${MODEL_PATH:-/data/user/xnie012/cache/Models/Qwen2.5-0.5B}
CACHE_DIR=${CACHE_DIR:-/data/user/xnie012/pythonprojects/lgar/data/qwen_fineweb_pilot}
CE_CKPT=${CE_CKPT:-/data/user/xnie012/pythonprojects/lgar/runs/ce_cpt_20260531_130512/runs/cure/ce_cpt_pilot/checkpoint.pt}
OUTPUT_ROOT=${OUTPUT_ROOT:-/data/user/xnie012/pythonprojects/lgar/runs/cure_v1_matrix_${TIMESTAMP}}
REPORT_DIR=${REPORT_DIR:-/data/user/xnie012/pythonprojects/lgar/reports/cure_v1_matrix_${TIMESTAMP}}
NUM_PROBE_SEQS=${NUM_PROBE_SEQS:-96}
BATCH_SIZE=${BATCH_SIZE:-1}
TOKENS_PER_RANK=${TOKENS_PER_RANK:-3125000}

TOP12=/data/user/xnie012/pythonprojects/lgar/runs/ablate_full_20260531_131608/ablation_results_top12.json
TOP24=/data/user/xnie012/pythonprojects/lgar/runs/ablate_full_20260531_131608/ablation_results_top24.json

mkdir -p "${OUTPUT_ROOT}" "${REPORT_DIR}"

echo "=== CURE-v1 mechanism matrix ==="
echo "timestamp=${TIMESTAMP}"
echo "output_root=${OUTPUT_ROOT}"
echo "report_dir=${REPORT_DIR}"
date
nvidia-smi --query-gpu=index,memory.used --format=csv || true

run_variant() {
  local name="$1"
  local ablate="$2"
  local lambda_rh_ce="$3"
  local adapter_lr="$4"
  local base_lr="$5"
  local tokens_per_rank="$6"

  local out_dir="${OUTPUT_ROOT}/${name}"
  local ckpt="${out_dir}/runs/cure/${name}/checkpoint.pt"
  local probe_json="${REPORT_DIR}/${name}_mechanism_probe.json"

  echo "=== train ${name} ==="
  echo "ablate=${ablate} lambda_rh_ce=${lambda_rh_ce} adapter_lr=${adapter_lr} base_lr=${base_lr} tokens_per_rank=${tokens_per_rank}"
  date

  torchrun --standalone --nproc_per_node=4 -m curcpt.train \
    --run-name "${name}" \
    --method cure_cpt \
    --model-path "${MODEL_PATH}" \
    --cache-dir "${CACHE_DIR}" \
    --output-dir "${out_dir}" \
    --checkpoint-path "${CE_CKPT}" \
    --ablation-results "${ablate}" \
    --seq-len 4096 \
    --local-window 1024 \
    --batch-size "${BATCH_SIZE}" \
    --tokens-per-run "${tokens_per_rank}" \
    --lr "${base_lr}" \
    --adapter-lr "${adapter_lr}" \
    --adapter-weight-decay 0.0 \
    --lambda-rh-ce "${lambda_rh_ce}" \
    --lambda-rh-kd 0.05 \
    --lambda-cov 0.0 \
    --lora-rank 16 \
    --lora-alpha 32.0 \
    --eval-interval 50 \
    --seed 1337 \
    --dtype bf16 \
    --attn-implementation sdpa

  echo "=== probe ${name} ==="
  date
  python -m curcpt.mechanism_probe_extended \
    --model-path "${MODEL_PATH}" \
    --cache-dir "${CACHE_DIR}" \
    --output "${probe_json}" \
    --checkpoint "ce_cpt=${CE_CKPT}" \
    --checkpoint "${name}=${ckpt}" \
    --reference-name ce_cpt \
    --retrieval-heads-json "${ablate}" \
    --seq-len 4096 \
    --short-window 1024 \
    --num-sequences "${NUM_PROBE_SEQS}" \
    --batch-size 1 \
    --dtype bf16 \
    --attn-implementation sdpa

  python - "${probe_json}" "${name}" <<'PY'
import json
import sys

path, name = sys.argv[1], sys.argv[2]
d = json.load(open(path))
m = d["models"][name]["buckets"]
summary = {
    "variant": name,
    "high_rh_adapter_active_minus_off_nll": m["high_utility"]["rh_adapter_active_minus_off_nll"],
    "all_rh_adapter_active_minus_off_nll": m["all_valid"]["rh_adapter_active_minus_off_nll"],
    "high_rh_minus_full_nll": m["high_utility"]["rh_minus_full_nll"],
    "all_rh_minus_full_nll": m["all_valid"]["rh_minus_full_nll"],
    "high_valid_count": m["high_utility"]["valid_count"],
    "all_valid_count": m["all_valid"]["valid_count"],
}
print(json.dumps({"event": "cure_v1_gate_summary", **summary}, sort_keys=True), flush=True)
PY
}

# Start from CE-CPT and spend 12.5M global tokens per variant by default
# (3.125M/rank). Batch 1 avoids the all-layer RH-bottleneck mask OOM risk.
# This is a mechanism gate, not an official downstream benchmark rerun.
run_variant "cure_v1_top12_r16_lam05_alr5e5" "${TOP12}" "0.5" "5e-5" "5e-6" "${TOKENS_PER_RANK}"
run_variant "cure_v1_top24_r16_lam05_alr5e5" "${TOP24}" "0.5" "5e-5" "5e-6" "${TOKENS_PER_RANK}"
run_variant "cure_v1_top12_r16_lam10_alr1e4" "${TOP12}" "1.0" "1e-4" "5e-6" "${TOKENS_PER_RANK}"

echo "=== CURE-v1 matrix done ==="
date
echo "report_dir=${REPORT_DIR}"
