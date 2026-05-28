#!/usr/bin/env bash
set -euo pipefail

cd /gemini/space/private/zjc/goals/lgar
set +u
source /gemini/space/private/zjc/envs/zjc_env/bin/activate
set -u

export NLTK_DATA=/gemini/space/private/zjc/goals/lgar/.nltk_data
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_P2P_LEVEL="${NCCL_P2P_LEVEL:-LOC}"
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

python - <<'PY'
import nltk
from pathlib import Path

target = Path("/gemini/space/private/zjc/goals/lgar/.nltk_data")
target.mkdir(parents=True, exist_ok=True)
for pkg in ("punkt", "punkt_tab"):
    try:
        nltk.data.find(f"tokenizers/{pkg}")
    except LookupError:
        nltk.download(pkg, download_dir=str(target), quiet=True)
PY

RULER_DIR=${RULER_DIR:-/gemini/space/private/zjc/data/RULER}
MODEL_PATH=${MODEL_PATH:-/gemini/space/private/zjc/models/Qwen2.5-0.5B}
OUTPUT_ROOT=${OUTPUT_ROOT:-/gemini/space/private/zjc/goals/lgar/reports/ruler13_official_stage2_mahiro_half_20260528}
CE_CKPT=${CE_CKPT:-/gemini/space/private/zjc/goals/lgar/stage2_2k_sdpa_full_bool_eren_madoka/eren_ce/runs/stage2/Qwen0.5B_CE_CPT_2B/checkpoint.pt}
LGAR_CKPT=${LGAR_CKPT:-/gemini/space/private/zjc/goals/lgar/stage2_offline_full_2k_sdpa_full_bool_mahiro_20260528/runs/stage2/Qwen0.5B_LGAR_CPT_2B/checkpoint.pt}
MAX_SAMPLES=${MAX_SAMPLES:-250}

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

LENGTHS=(
  "1024:official1k"
  "1536:official1536"
  "2048:official2048"
)

TASKS_CSV=$(IFS=,; echo "${TASKS[*]}")

run_model() {
  local model_name="$1"
  local ckpt="$2"
  local mode="$3"
  local batch_size="$4"
  for pair in "${LENGTHS[@]}"; do
    local seq_len="${pair%%:*}"
    local folder_name="${pair##*:}"
    local data_dir="$RULER_DIR/generated/$folder_name"
    local pred_dir="$OUTPUT_ROOT/$model_name/$folder_name/pred"
    mkdir -p "$pred_dir"
    echo "[ruler13] model=$model_name length=$seq_len max_samples=$MAX_SAMPLES"
    torchrun --nproc_per_node=8 --standalone -m lgar_cpt.ruler13_generate \
      --model-path "$MODEL_PATH" \
      --checkpoint-path "$ckpt" \
      --ruler-dir "$RULER_DIR" \
      --data-dir "$data_dir" \
      --save-dir "$pred_dir" \
      --tasks "$TASKS_CSV" \
      --subset validation \
      --max-samples "$MAX_SAMPLES" \
      --mode "$mode" \
      --batch-size "$batch_size" \
      --append-answer-prefix \
      --attn-implementation sdpa \
      --dtype bf16
    python -m lgar_cpt.ruler13_score --ruler-dir "$RULER_DIR" --data-dir "$pred_dir"
  done
}

run_model "ce" "$CE_CKPT" "full" 8
run_model "lgar" "$LGAR_CKPT" "routed" 2

python - <<'PY'
import json
from pathlib import Path

root = Path("/gemini/space/private/zjc/goals/lgar/reports/ruler13_official_stage2_mahiro_half_20260528")
lengths = ["official1k", "official1536", "official2048"]
models = ["ce", "lgar"]
summary = {"lengths": {}}

for folder_name in lengths:
    summary["lengths"][folder_name] = {}
    for model_name in models:
        summary_path = root / model_name / folder_name / "pred" / "summary.json"
        data = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["lengths"][folder_name][model_name] = data
    summary["lengths"][folder_name]["delta_mean_score"] = (
        summary["lengths"][folder_name]["lgar"]["mean_score"]
        - summary["lengths"][folder_name]["ce"]["mean_score"]
    )

out_path = root / "compare_summary.json"
out_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
print(out_path)
PY
