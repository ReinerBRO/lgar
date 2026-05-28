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
python -m unittest discover -s tests -p 'test_lgar_*.py'
python - <<'PY'
from transformers import AutoConfig, AutoTokenizer

path = "/gemini/space/private/zjc/models/Qwen2.5-0.5B"
tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
cfg = AutoConfig.from_pretrained(path, trust_remote_code=True)
assert tok.eos_token_id is not None
assert cfg.model_type == "qwen2"
assert cfg.num_hidden_layers > 0
print("qwen_config_ok", cfg.hidden_size, cfg.num_hidden_layers)
PY
