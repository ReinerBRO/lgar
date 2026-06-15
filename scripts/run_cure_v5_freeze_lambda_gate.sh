#!/bin/bash
#SBATCH -J cure_v5l
#SBATCH -p acd_u
#SBATCH -n 8
#SBATCH --gres=gpu:4
#SBATCH -o /dev/null
#SBATCH -e /dev/null
#SBATCH -D /data/user/xnie012/pythonprojects/lgar

set -euo pipefail

mkdir -p logs
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="logs/cure_v5_freeze_lambda_gate_${TIMESTAMP}_${SLURM_JOB_ID}.log"
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
OUTPUT_ROOT=${OUTPUT_ROOT:-/data/user/xnie012/pythonprojects/lgar/runs/cure_v5_freeze_lambda_gate_${TIMESTAMP}}
REPORT_DIR=${REPORT_DIR:-/data/user/xnie012/pythonprojects/lgar/reports/cure_v5_freeze_lambda_gate_${TIMESTAMP}}
NUM_PROBE_SEQS=${NUM_PROBE_SEQS:-128}
BATCH_SIZE=${BATCH_SIZE:-2}
TOKENS_PER_RANK=${TOKENS_PER_RANK:-3125000}
TOP24=/data/user/xnie012/pythonprojects/lgar/runs/ablate_full_20260531_131608/ablation_results_top24.json

mkdir -p "${OUTPUT_ROOT}" "${REPORT_DIR}"

echo "=== CURE-v5 freeze-base lambda gate ==="
date

run_variant() {
  local lambda_full_hu="$1"
  local name="cure_v5_freeze_top24_frac010_fullhu${lambda_full_hu/./}_alr1e4"
  local out_dir="${OUTPUT_ROOT}/${name}"
  local ckpt="${out_dir}/runs/cure/${name}/checkpoint.pt"
  local probe_json="${REPORT_DIR}/${name}_mechanism_probe.json"

  echo "=== train ${name} ==="
  torchrun --standalone --nproc_per_node=4 -m curcpt.train \
    --run-name "${name}" \
    --method cure_cpt \
    --model-path "${MODEL_PATH}" \
    --cache-dir "${CACHE_DIR}" \
    --output-dir "${out_dir}" \
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
    --lambda-full-hu-ce "${lambda_full_hu}" \
    --utility-top-fraction-training 0.10 \
    --lora-rank 16 \
    --lora-alpha 32.0 \
    --eval-interval 25 \
    --seed 1337 \
    --dtype bf16 \
    --attn-implementation sdpa

  echo "=== full-path probe ${name} ==="
  python -m curcpt.mechanism_probe \
    --model-path "${MODEL_PATH}" \
    --cache-dir "${CACHE_DIR}" \
    --output "${probe_json}" \
    --checkpoint "ce_cpt=${CE_CKPT}" \
    --checkpoint "${name}=${ckpt}" \
    --reference-name ce_cpt \
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
m = d["models"][name]["metrics"]
delta = d.get("deltas_vs_reference", {}).get(name, {})
print(json.dumps({
    "event": "cure_v5_lambda_gate_summary",
    "variant": name,
    "checkpoint": d["models"][name]["checkpoint"],
    "normal_ce": m.get("normal_ce"),
    "delta_normal_ce_vs_ce": delta.get("normal_ce"),
    "ref_high_ce": m.get("ref_high_ce"),
    "delta_ref_high_ce_vs_ce": delta.get("ref_high_ce"),
    "adapter_high_active_minus_off_nll": m.get("adapter_ref_high_active_minus_off_delta_nll"),
    "adapter_high_logit_l2": m.get("adapter_ref_high_logit_l2"),
    "adapter_high_top1_flip": m.get("adapter_ref_high_top1_flip_rate"),
}, sort_keys=True), flush=True)
PY
}

run_variant "1.5"
run_variant "2.0"

echo "=== CURE-v5 lambda gate done ==="
date
echo "report_dir=${REPORT_DIR}"
