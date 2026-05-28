#!/usr/bin/env bash
set -euo pipefail

cd /gemini/space/private/zjc/goals/lgar
source /gemini/space/private/zjc/envs/zjc_env/bin/activate

export NLTK_DATA=/gemini/space/private/zjc/goals/lgar/.nltk_data

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

RULER_DIR=/gemini/space/private/zjc/data/RULER
MODEL_PATH=/gemini/space/private/zjc/models/Qwen2.5-0.5B
ASSET_PREPARE=$RULER_DIR/_assets/RULER/scripts/data/prepare.py
OUTPUT_ROOT=${OUTPUT_ROOT:-/gemini/space/private/zjc/goals/lgar/reports/ruler13_official_stage2_madoka}
CE_CKPT=${CE_CKPT:-/gemini/space/private/zjc/goals/lgar/stage2_2k_sdpa_full_bool_eren_madoka/eren_ce/runs/stage2/Qwen0.5B_CE_CPT_2B/checkpoint.pt}
LGAR_CKPT=${LGAR_CKPT:-/gemini/space/private/zjc/goals/lgar/stage2_2k_sdpa_full_bool_eren_madoka/madoka_lgar/runs/stage2/Qwen0.5B_LGAR_CPT_2B/checkpoint.pt}

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

prepare_length() {
  local seq_len="$1"
  local folder_name="$2"
  local data_dir="$RULER_DIR/generated/$folder_name"
  mkdir -p "$data_dir"
  for task in "${TASKS[@]}"; do
    local task_file="$data_dir/$task/validation.jsonl"
    if [[ -f "$task_file" ]]; then
      local line_count
      line_count=$(wc -l < "$task_file")
      if [[ "$line_count" -ge 500 ]]; then
        continue
      fi
      rm -f "$task_file"
    fi
    python "$ASSET_PREPARE" \
      --save_dir "$data_dir" \
      --benchmark synthetic \
      --task "$task" \
      --subset validation \
      --tokenizer_path "$MODEL_PATH" \
      --tokenizer_type hf \
      --max_seq_length "$seq_len" \
      --model_template_type base \
      --num_samples 500
    if [[ ! -f "$task_file" ]]; then
      echo "RULER data generation failed for $folder_name/$task" >&2
      exit 1
    fi
    local final_line_count
    final_line_count=$(wc -l < "$task_file")
    if [[ "$final_line_count" -lt 500 ]]; then
      echo "RULER data generation incomplete for $folder_name/$task: $final_line_count lines" >&2
      exit 1
    fi
  done
}

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
    torchrun --nproc_per_node=8 --standalone -m lgar_cpt.ruler13_generate \
      --model-path "$MODEL_PATH" \
      --checkpoint-path "$ckpt" \
      --ruler-dir "$RULER_DIR" \
      --data-dir "$data_dir" \
      --save-dir "$pred_dir" \
      --subset validation \
      --mode "$mode" \
      --batch-size "$batch_size" \
      --append-answer-prefix \
      --attn-implementation sdpa \
      --dtype bf16
    python -m lgar_cpt.ruler13_score --ruler-dir "$RULER_DIR" --data-dir "$pred_dir"
  done
}

python - <<'PY'
from pathlib import Path
Path("/gemini/space/private/zjc/goals/lgar/reports/ruler13_official_stage2_madoka").mkdir(parents=True, exist_ok=True)
PY

for pair in "${LENGTHS[@]}"; do
  prepare_length "${pair%%:*}" "${pair##*:}"
done

run_model "ce" "$CE_CKPT" "full" 8
run_model "lgar" "$LGAR_CKPT" "routed" 2

python - <<'PY'
import csv
import json
from pathlib import Path

root = Path("/gemini/space/private/zjc/goals/lgar/reports/ruler13_official_stage2_madoka")
lengths = ["official1k", "official1536", "official2048"]
models = ["ce", "lgar"]
summary = {"lengths": {}}

for folder_name in lengths:
    summary["lengths"][folder_name] = {}
    for model_name in models:
        csv_path = root / model_name / folder_name / "pred" / "summary.csv"
        with csv_path.open("r", encoding="utf-8") as f:
            rows = list(csv.reader(f))
        tasks = rows[0][1:]
        scores = [float(x) for x in rows[1][1:]]
        nulls = rows[2][1:]
        summary["lengths"][folder_name][model_name] = {
            "task_scores": dict(zip(tasks, scores)),
            "task_nulls": dict(zip(tasks, nulls)),
            "mean_score": sum(scores) / len(scores),
        }
    summary["lengths"][folder_name]["delta_mean_score"] = (
        summary["lengths"][folder_name]["lgar"]["mean_score"]
        - summary["lengths"][folder_name]["ce"]["mean_score"]
    )

out_path = root / "compare_summary.json"
out_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
print(out_path)
PY
