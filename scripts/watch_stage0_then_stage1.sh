#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/gemini/space/private/zjc/goals/lgar}"
LOG_DIR="${LOG_DIR:-/gemini/space/private/zjc/logs/lgar}"
CHECK_INTERVAL_SECONDS="${CHECK_INTERVAL_SECONDS:-300}"

mkdir -p "${LOG_DIR}"
cd "${PROJECT_ROOT}"

while true; do
  if [[ -f reports/stage0_summary.json ]]; then
    status="$(python - <<'PY'
import json
from pathlib import Path

payload = json.loads(Path("reports/stage0_summary.json").read_text(encoding="utf-8"))
checks = payload.get("decision_checks", {})
failed = [k for k, v in checks.items() if not v]
if failed:
    print("FAIL " + ",".join(failed))
elif payload.get("hard_lgar_available") is not True:
    print("FAIL hard_lgar_available")
else:
    print("PASS")
PY
)"
    echo "[$(date -Is)] stage0_status=${status}" | tee -a "${LOG_DIR}/watch_stage0_then_stage1.log"
    if [[ "${status}" == "PASS" ]]; then
      if ! tmux has-session -t lgar_stage1_yui 2>/dev/null; then
        tmux new-session -d -s lgar_stage1_yui "cd ${PROJECT_ROOT} && bash scripts/run_stage1.sh 2>&1 | tee -a ${LOG_DIR}/stage1_yui.log"
      fi
      exit 0
    fi
    if [[ "${status}" == FAIL* ]]; then
      exit 1
    fi
  else
    echo "[$(date -Is)] waiting_for_stage0_summary" | tee -a "${LOG_DIR}/watch_stage0_then_stage1.log"
  fi
  sleep "${CHECK_INTERVAL_SECONDS}"
done
