#!/bin/bash
#SBATCH -J cv1_noli
#SBATCH -p acd_u
#SBATCH -n 4
#SBATCH --gres=gpu:1
#SBATCH -o /dev/null
#SBATCH -e /dev/null
#SBATCH -D /data/user/xnie012/pythonprojects/lgar

set -euo pipefail

mkdir -p logs
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="logs/cure_v1_rh_nolima200_${TIMESTAMP}_${SLURM_JOB_ID}.log"
exec > "${LOG_FILE}" 2>&1

set +u
source /data/user/xnie012/envs/memgen/bin/activate
set -u

export PYTHONUNBUFFERED=1
export PYTHONPATH="/data/user/xnie012/pythonprojects/lgar:/data/user/xnie012/pythonprojects/prefix/scripts:${PYTHONPATH:-}"
export HF_HOME=/data/user/xnie012/cache
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODEL_PATH=${MODEL_PATH:-/data/user/xnie012/cache/Models/Qwen2.5-0.5B}
CE_CKPT=${CE_CKPT:-/data/user/xnie012/pythonprojects/lgar/runs/ce_cpt_20260531_130512/runs/cure/ce_cpt_pilot/checkpoint.pt}
CURE_V1_CKPT=${CURE_V1_CKPT:-/data/user/xnie012/pythonprojects/lgar/runs/cure_v1_matrix_20260531_215846/cure_v1_top12_r16_lam10_alr1e4/runs/cure/cure_v1_top12_r16_lam10_alr1e4/checkpoint.pt}
ABLATE=${ABLATE:-/data/user/xnie012/pythonprojects/lgar/runs/ablate_full_20260531_131608/ablation_results_top12.json}
REPORT_DIR=${REPORT_DIR:-/data/user/xnie012/pythonprojects/lgar/reports/cure_v1_matrix_20260531_215846/downstream_nolima200}
mkdir -p "${REPORT_DIR}"

echo "=== CURE-v1 RH NoLiMA200 ==="
date
nvidia-smi --query-gpu=index,memory.used --format=csv || true

run_eval() {
  local name="$1"
  local ckpt="$2"
  local mode="$3"
  local ctx="$4"
  local seq_len="$5"
  local out="${REPORT_DIR}/nolima_${name}_${mode}_${ctx}.json"
  echo "=== ${name} ${mode} ctx=${ctx} ==="
  python -m curcpt.eval_nolima \
    --model-path "${MODEL_PATH}" \
    --checkpoint-path "${ckpt}" \
    --output "${out}" \
    --context-length "${ctx}" \
    --seq-len "${seq_len}" \
    --needle-set-path /data/user/xnie012/cache/nolima/needlesets/needle_set_hard.json \
    --haystack-dir /data/user/xnie012/cache/nolima/haystack/rand_shuffle \
    --max-books 5 \
    --max-experiments 8 \
    --document-depth-percent-intervals 5 \
    --max-new-tokens 12 \
    --eval-mode "${mode}" \
    --local-window 1024 \
    --retrieval-heads-json "${ABLATE}" \
    --reference-checkpoint-path "${CURE_V1_CKPT}" \
    --dtype bf16 \
    --attn-implementation sdpa
}

# 5 books * 8 tests * 5 depths = 200 examples per run.
run_eval ce_cpt "${CE_CKPT}" full 2048 4096
run_eval ce_cpt "${CE_CKPT}" rh_bottleneck 2048 4096
run_eval cure_v1 "${CURE_V1_CKPT}" full 2048 4096
run_eval cure_v1 "${CURE_V1_CKPT}" rh_bottleneck 2048 4096

python - "${REPORT_DIR}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
summary = {}
for path in sorted(root.glob("nolima_*.json")):
    data = json.loads(path.read_text(encoding="utf-8"))
    key = path.stem.removeprefix("nolima_")
    summary[key] = {
        "official_accuracy": data["official_accuracy"],
        "normalized_accuracy": data["normalized_accuracy"],
        "num_examples": data["num_examples"],
        "overlength_examples": data["overlength_examples"],
        "eval_mode": data["eval_mode"],
        "context_length": data["context_length"],
    }
out = root / "summary.json"
out.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
PY

date
echo "output=${REPORT_DIR}/summary.json"
