#!/bin/bash
#SBATCH -J cv1_mcq
#SBATCH -p acd_u
#SBATCH -n 4
#SBATCH --gres=gpu:1
#SBATCH -o /dev/null
#SBATCH -e /dev/null
#SBATCH -D /data/user/xnie012/pythonprojects/lgar

set -euo pipefail

mkdir -p logs
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="logs/cure_v1_rh_mcq200_${TIMESTAMP}_${SLURM_JOB_ID}.log"
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
CE_CKPT=${CE_CKPT:-/data/user/xnie012/pythonprojects/lgar/runs/ce_cpt_20260531_130512/runs/cure/ce_cpt_pilot/checkpoint.pt}
CURE_V1_CKPT=${CURE_V1_CKPT:-/data/user/xnie012/pythonprojects/lgar/runs/cure_v1_matrix_20260531_215846/cure_v1_top12_r16_lam10_alr1e4/runs/cure/cure_v1_top12_r16_lam10_alr1e4/checkpoint.pt}
ABLATE=${ABLATE:-/data/user/xnie012/pythonprojects/lgar/runs/ablate_full_20260531_131608/ablation_results_top12.json}
REPORT_DIR=${REPORT_DIR:-/data/user/xnie012/pythonprojects/lgar/reports/cure_v1_matrix_20260531_215846/downstream_mcq200}
mkdir -p "${REPORT_DIR}"

echo "=== CURE-v1 RH MCQ200 ==="
date
nvidia-smi --query-gpu=index,memory.used --format=csv || true

run_eval() {
  local name="$1"
  local ckpt="$2"
  local mode="$3"
  local out="${REPORT_DIR}/mcq_${name}_${mode}.json"
  echo "=== ${name} ${mode} ==="
  python -m curcpt.eval_public_mcq \
    --model-path "${MODEL_PATH}" \
    --checkpoint-path "${ckpt}" \
    --output "${out}" \
    --tasks piqa arc_easy arc_challenge hellaswag winogrande openbookqa \
    --conditions clean irrelevant_demo \
    --max-examples 200 \
    --batch-size 4 \
    --num-distractors 2 \
    --max-prefix-tokens 1800 \
    --seq-len 4096 \
    --eval-mode "${mode}" \
    --local-window 1024 \
    --retrieval-heads-json "${ABLATE}" \
    --reference-checkpoint-path "${CURE_V1_CKPT}" \
    --dtype bf16 \
    --attn-implementation sdpa
}

run_eval ce_cpt "${CE_CKPT}" full
run_eval ce_cpt "${CE_CKPT}" rh_bottleneck
run_eval cure_v1 "${CURE_V1_CKPT}" full
run_eval cure_v1 "${CURE_V1_CKPT}" rh_bottleneck

python - "${REPORT_DIR}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
summary = {}
for path in sorted(root.glob("mcq_*.json")):
    data = json.loads(path.read_text(encoding="utf-8"))
    key = path.stem.removeprefix("mcq_")
    rows = {}
    accs = []
    for task, task_data in data["tasks"].items():
        for condition, metrics in task_data["conditions"].items():
            label = f"{task}/{condition}"
            rows[label] = {
                "accuracy_avg": metrics["accuracy_avg"],
                "avg_margin_avg": metrics["avg_margin_avg"],
                "num_examples": metrics["num_examples"],
            }
            accs.append(float(metrics["accuracy_avg"]))
    summary[key] = {
        "macro_accuracy": sum(accs) / max(len(accs), 1),
        "eval_mode": data["eval_mode"],
        "rows": rows,
    }
out = root / "summary.json"
out.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
PY

date
echo "output=${REPORT_DIR}/summary.json"
