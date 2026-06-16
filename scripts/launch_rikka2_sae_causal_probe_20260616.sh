#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=/gfs/space/private/zjc/goals/lgar
cd "${PROJECT_ROOT}"

mkdir -p logs reports

LOG_FILE="${PROJECT_ROOT}/logs/launch_rikka2_sae_causal_probe_20260616.out"
OUTPUT="${PROJECT_ROOT}/reports/sae_causal_probe_stage2_sae_cure_layer21_sparse_from_base_frozen_20260615_16k_0p5b_rikka2_b8.json"
CHECKPOINT_PATH="${PROJECT_ROOT}/runs/stage2_sae_cure_layer21_sparse_from_base_frozen_20260615_16k_0p5b/runs/cure/stage2_sae_cure_layer21_sparse_from_base_frozen_20260615_16k_0p5b/checkpoint.pt"

rm -f "${LOG_FILE}"

nohup bash -lc "
set -euo pipefail
cd '${PROJECT_ROOT}'
CUDA_VISIBLE_DEVICES=0 \
CHECKPOINT_PATH='${CHECKPOINT_PATH}' \
TAG=stage2_sae_cure_layer21_sparse_from_base_frozen_20260615_16k_0p5b_rikka2_b8 \
OUTPUT='${OUTPUT}' \
BATCHES=8 \
BATCH_SIZE=1 \
LOGPROB_CHUNK_SIZE=512 \
DTYPE=bf16 \
ATTN_IMPLEMENTATION=sdpa \
bash scripts/run_sae_causal_probe.sh
" > "${LOG_FILE}" 2>&1 < /dev/null &

echo "probe_pid=$!"
echo "log=${LOG_FILE}"
echo "output=${OUTPUT}"
