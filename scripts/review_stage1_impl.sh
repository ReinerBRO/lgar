#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/gemini/space/private/zjc/goals/lgar}"
ENV_PATH="${ENV_PATH:-/gemini/space/private/zjc/envs/zjc_env}"

cd "${PROJECT_ROOT}"
set +u
source "${ENV_PATH}/bin/activate"
set -u
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

python -m py_compile lgar_cpt/*.py
python -m unittest discover -s tests

python - <<'PY'
from pathlib import Path

run_script = Path("scripts/run_stage1.sh").read_text(encoding="utf-8")
train_script = Path("lgar_cpt/train_stage1.py").read_text(encoding="utf-8")
train_core = Path("lgar_cpt/train_stage0.py").read_text(encoding="utf-8")
eval_script = Path("lgar_cpt/evaluate.py").read_text(encoding="utf-8")
review_script = Path("lgar_cpt/review_stage1_results.py").read_text(encoding="utf-8")

checks = {
    "stage1_uses_torchrun": "torchrun --standalone --nproc_per_node" in run_script,
    "stage1_defaults_to_8_ranks": 'NPROC_PER_NODE="${NPROC_PER_NODE:-8}"' in run_script,
    "stage1_sets_all_8_gpus": 'CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"' in run_script,
    "stage1_initializes_process_group": "dist.init_process_group" in train_script,
    "stage1_steps_use_world_size": "args.seq_len * args.batch_size * world_size" in train_script,
    "stage1_passes_rank_world_to_train_one": "world_size=world_size" in train_script and "rank=rank" in train_script,
    "train_one_syncs_model_grads": "_sync_module_grads(model, world_size)" in train_core,
    "train_one_syncs_router_grads": "_sync_module_grads(router, world_size)" in train_core,
    "train_one_sums_global_tokens": "global_step_tokens = int(_ddp_sum" in train_core,
    "train_one_rank0_writes_metrics": "if rank == 0:" in train_core and "metrics_partial.json" in train_core,
    "train_one_saves_checkpoints": "checkpoint.pt" in train_core and '"model": model.state_dict()' in train_core,
    "train_one_runs_final_eval": "final_eval" in train_core and "evaluate_model(" in train_core,
    "final_eval_uses_method_mode": "mode=mode" in train_core and "target_budget=effective_routed_budget" in train_core,
    "eval_uses_lgar_forward": "qwen_lgar_forward" in eval_script and "target_budget=target_budget" in eval_script,
    "stage1_review_checks_mechanism": "lgar_router_signal" in review_script and "actual_budget_last_quarter" in review_script,
}
failed = [name for name, ok in checks.items() if not ok]
if failed:
    raise SystemExit(f"stage1 implementation review failed: {failed}")
print("stage1_impl_review_ok", checks)
PY
