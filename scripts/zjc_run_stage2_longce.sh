#!/usr/bin/env bash
set -euo pipefail

export METHOD=longce_cpt
export STAGE2_TAG="${STAGE2_TAG:-$(date +%Y%m%d_%H%M%S)}"
export RUN_NAME="${RUN_NAME:-stage2_longce_cpt_${STAGE2_TAG}}"
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/zjc_run_stage2_train_one.sh"
