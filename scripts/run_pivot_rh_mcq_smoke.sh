#!/bin/bash
#SBATCH -J cure_rhmcq
#SBATCH -p acd_u
#SBATCH -n 4
#SBATCH --gres=gpu:1
#SBATCH -o /dev/null
#SBATCH -e /dev/null
#SBATCH -D /data/user/xnie012/pythonprojects/lgar

set -euo pipefail

mkdir -p logs
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="logs/pivot_rh_mcq_smoke_${TIMESTAMP}_${SLURM_JOB_ID}.log"
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
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODEL_PATH=${MODEL_PATH:-/data/user/xnie012/cache/Models/Qwen2.5-0.5B}
REPORT_DIR=${REPORT_DIR:-/data/user/xnie012/pythonprojects/lgar/reports/cure_after_fixlora_pivot_20260531/rh_mcq_smoke}
mkdir -p "${REPORT_DIR}"

CE_CKPT=${CE_CKPT:-/data/user/xnie012/pythonprojects/lgar/runs/ce_cpt_20260531_130512/runs/cure/ce_cpt_pilot/checkpoint.pt}
ADAPTER_CKPT=${ADAPTER_CKPT:-/data/user/xnie012/pythonprojects/lgar/runs/pilot_adapter_ce_20260531_173436/runs/cure/pilot_adapter_ce/checkpoint.pt}
LONGCE_CKPT=${LONGCE_CKPT:-/data/user/xnie012/pythonprojects/lgar/runs/pilot_longce_20260531_154639/runs/cure/pilot_longce/checkpoint.pt}
CURE_CKPT=${CURE_CKPT:-/data/user/xnie012/pythonprojects/lgar/runs/pilot_cure_20260531_173436/runs/cure/pilot_cure/checkpoint.pt}
ABLATE=${ABLATE:-/data/user/xnie012/pythonprojects/lgar/runs/ablate_full_20260531_131608/ablation_results_top6.json}

for item in \
  "ce_cpt:${CE_CKPT}" \
  "adapter_ce:${ADAPTER_CKPT}" \
  "longce_cpt:${LONGCE_CKPT}" \
  "cure_cpt:${CURE_CKPT}"
do
  name=${item%%:*}
  ckpt=${item#*:}
  for mode in full rh_bottleneck
  do
    python -m curcpt.eval_public_mcq \
      --model-path "${MODEL_PATH}" \
      --checkpoint-path "${ckpt}" \
      --output "${REPORT_DIR}/mcq_${name}_${mode}.json" \
      --tasks arc_easy \
      --conditions clean \
      --max-examples 50 \
      --batch-size 8 \
      --seq-len 4096 \
      --eval-mode "${mode}" \
      --local-window 1024 \
      --reference-checkpoint-path "${CURE_CKPT}" \
      --retrieval-heads-json "${ABLATE}" \
      --dtype bf16 \
      --attn-implementation sdpa
  done
done

python - "${REPORT_DIR}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
summary = {}
for path in sorted(root.glob("mcq_*.json")):
    data = json.loads(path.read_text(encoding="utf-8"))
    key = path.stem.removeprefix("mcq_")
    task = data["tasks"]["arc_easy"]["conditions"]["clean"]
    summary[key] = {
        "accuracy_avg": task["accuracy_avg"],
        "avg_margin_avg": task["avg_margin_avg"],
        "eval_mode": data["eval_mode"],
        "num_examples": task["num_examples"],
    }
(root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
print(json.dumps(summary, indent=2, sort_keys=True))
PY

echo "output=${REPORT_DIR}/summary.json"
