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

RULER_DIR=${RULER_DIR:-/gemini/space/private/zjc/data/RULER}
MODEL_PATH=${MODEL_PATH:-/gemini/space/private/zjc/models/Qwen2.5-0.5B}
OUTPUT_ROOT=${OUTPUT_ROOT:-/gemini/space/private/zjc/goals/lgar/reports/ruler13_official_stage2_mahiro_force_last_20260528}
BASE_COMPARE=${BASE_COMPARE:-/gemini/space/private/zjc/goals/lgar/reports/ruler13_official_stage2_mahiro_half_20260528/compare_summary.json}
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

run_lgar_force_last() {
  local batch_size="$1"
  for pair in "${LENGTHS[@]}"; do
    local seq_len="${pair%%:*}"
    local folder_name="${pair##*:}"
    local data_dir="$RULER_DIR/generated/$folder_name"
    local pred_dir="$OUTPUT_ROOT/lgar_force_last/$folder_name/pred"
    mkdir -p "$pred_dir"
    echo "[ruler13-force-last] length=$seq_len max_samples=$MAX_SAMPLES"
    torchrun --nproc_per_node=8 --standalone -m lgar_cpt.ruler13_generate \
      --model-path "$MODEL_PATH" \
      --checkpoint-path "$LGAR_CKPT" \
      --ruler-dir "$RULER_DIR" \
      --data-dir "$data_dir" \
      --save-dir "$pred_dir" \
      --tasks "$TASKS_CSV" \
      --subset validation \
      --max-samples "$MAX_SAMPLES" \
      --mode routed \
      --target-budget 0.25 \
      --force-last-query-global \
      --batch-size "$batch_size" \
      --append-answer-prefix \
      --attn-implementation sdpa \
      --dtype bf16
    python -m lgar_cpt.ruler13_score --ruler-dir "$RULER_DIR" --data-dir "$pred_dir"
  done
}

run_router_probe() {
  local batch_size="$1"
  for pair in "${LENGTHS[@]}"; do
    local seq_len="${pair%%:*}"
    local folder_name="${pair##*:}"
    local data_dir="$RULER_DIR/generated/$folder_name"
    local probe_dir="$OUTPUT_ROOT/router_probe/$folder_name"
    mkdir -p "$probe_dir"
    echo "[router-probe] length=$seq_len max_samples=$MAX_SAMPLES"
    torchrun --nproc_per_node=8 --standalone -m lgar_cpt.ruler13_router_probe \
      --model-path "$MODEL_PATH" \
      --checkpoint-path "$LGAR_CKPT" \
      --ruler-dir "$RULER_DIR" \
      --data-dir "$data_dir" \
      --save-dir "$probe_dir" \
      --tasks "$TASKS_CSV" \
      --subset validation \
      --max-samples "$MAX_SAMPLES" \
      --batch-size "$batch_size" \
      --target-budget 0.25 \
      --append-answer-prefix \
      --attn-implementation sdpa \
      --dtype bf16
    python - "$probe_dir" <<'PY'
import json
import sys
from pathlib import Path

probe_dir = Path(sys.argv[1])
items = []
for path in sorted(probe_dir.glob("router_probe_rank*.jsonl")):
    if path.name.endswith("_summary.json"):
        continue
    with path.open(encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if line:
                items.append(json.loads(line))

def avg(rows, key):
    return sum(float(row[key]) for row in rows) / max(1, len(rows))

by_task = {}
for item in items:
    by_task.setdefault(item["task"], []).append(item)

summary = {
    "count": len(items),
    "last_selected_rate": avg(items, "last_selected"),
    "last_rank_frac_mean": avg(items, "last_rank_frac"),
    "last_score_mean": avg(items, "last_score"),
    "tasks": {
        task: {
            "count": len(rows),
            "last_selected_rate": avg(rows, "last_selected"),
            "last_rank_frac_mean": avg(rows, "last_rank_frac"),
            "last_score_mean": avg(rows, "last_score"),
        }
        for task, rows in sorted(by_task.items())
    },
}
(probe_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
print(probe_dir / "summary.json")
PY
  done
}

run_router_probe 8
run_lgar_force_last 2

python - "$OUTPUT_ROOT" "$BASE_COMPARE" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
base_compare = Path(sys.argv[2])
base = json.loads(base_compare.read_text(encoding="utf-8"))
lengths = ["official1k", "official1536", "official2048"]
summary = {"lengths": {}}

for folder_name in lengths:
    old_row = base["lengths"][folder_name]
    ce = old_row["ce"]
    old_lgar = old_row["lgar"]
    force_path = root / "lgar_force_last" / folder_name / "pred" / "summary.json"
    probe_path = root / "router_probe" / folder_name / "summary.json"
    force_lgar = json.loads(force_path.read_text(encoding="utf-8"))
    probe = json.loads(probe_path.read_text(encoding="utf-8"))
    summary["lengths"][folder_name] = {
        "ce": ce,
        "lgar": old_lgar,
        "lgar_force_last": force_lgar,
        "router_probe": probe,
        "delta_lgar_vs_ce": old_lgar["mean_score"] - ce["mean_score"],
        "delta_force_last_vs_ce": force_lgar["mean_score"] - ce["mean_score"],
        "delta_force_last_vs_lgar": force_lgar["mean_score"] - old_lgar["mean_score"],
    }

out_path = root / "compare_summary.json"
out_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
print(out_path)
PY
