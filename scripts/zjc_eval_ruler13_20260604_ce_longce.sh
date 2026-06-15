#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PAIR=ce_longce \
TAG="${TAG:-20260604_16k_0p5b}" \
SEQ_LENS="${SEQ_LENS:-4096 8192 16384}" \
MAX_SAMPLES="${MAX_SAMPLES:-200}" \
CKPT_VARIANTS="${CKPT_VARIANTS:-final}" \
CE_BATCH_SIZE="${CE_BATCH_SIZE:-32}" \
LONGCE_BATCH_SIZE="${LONGCE_BATCH_SIZE:-32}" \
CE_16K_BATCH_SIZE="${CE_16K_BATCH_SIZE:-8}" \
LONGCE_16K_BATCH_SIZE="${LONGCE_16K_BATCH_SIZE:-8}" \
bash "${SCRIPT_DIR}/zjc_run_ruler13_pair_eval.sh"
