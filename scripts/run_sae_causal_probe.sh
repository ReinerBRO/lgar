#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ZJC_ROOT="${ZJC_ROOT:-/gfs/space/private/zjc}"
PYTHON="${PYTHON:-${ZJC_ROOT}/envs/zjc_env/bin/python}"

MODEL_PATH="${MODEL_PATH:-${ZJC_ROOT}/models/Qwen2.5-0.5B}"
CACHE_DIR="${CACHE_DIR:-${ZJC_ROOT}/data/qwen_fineweb_stage2}"
SIGNAL_DIR="${SIGNAL_DIR:-${PROJECT_ROOT}/data/offline_signal_16k_1b_sw2048_lw2560}"

SAE_SIGNAL_TAG="${SAE_SIGNAL_TAG:-20260615_layer21_sae_signal_8shard}"
SAE_LAYER="${SAE_LAYER:-21}"
SAE_ROOT="${SAE_ROOT:-${PROJECT_ROOT}/runs/${SAE_SIGNAL_TAG}}"
SAE_CHECKPOINT="${SAE_CHECKPOINT:-${SAE_ROOT}/sae/sae_layer${SAE_LAYER}.pt}"
FEATURES="${FEATURES:-${SAE_ROOT}/evidence_features_validated.json}"

RUN_NAME="${RUN_NAME:-stage2_sae_cure_layer21_sparse_from_base_frozen_20260615_16k_0p5b}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-${PROJECT_ROOT}/runs/${RUN_NAME}/checkpoint.pt}"
TAG="${TAG:-${RUN_NAME}}"
OUTPUT="${OUTPUT:-${PROJECT_ROOT}/reports/sae_causal_probe_${TAG}.json}"

BATCH_SIZE="${BATCH_SIZE:-1}"
BATCHES="${BATCHES:-8}"
SEED="${SEED:-1337}"
DTYPE="${DTYPE:-bf16}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"
ABLATION_SCALE="${ABLATION_SCALE:-1.0}"
LOGPROB_CHUNK_SIZE="${LOGPROB_CHUNK_SIZE:-512}"

cd "${PROJECT_ROOT}"

if [[ ! -s "${CHECKPOINT_PATH}" ]]; then
  echo "missing checkpoint: ${CHECKPOINT_PATH}" >&2
  echo "set CHECKPOINT_PATH=/path/to/checkpoint_latest.pt if final checkpoint is not ready" >&2
  exit 2
fi

"${PYTHON}" -m curcpt.sae_causal_probe \
  --model-path "${MODEL_PATH}" \
  --checkpoint-path "${CHECKPOINT_PATH}" \
  --sae-checkpoint "${SAE_CHECKPOINT}" \
  --features "${FEATURES}" \
  --cache-dir "${CACHE_DIR}" \
  --signal-dir "${SIGNAL_DIR}" \
  --output "${OUTPUT}" \
  --batch-size "${BATCH_SIZE}" \
  --batches "${BATCHES}" \
  --seed "${SEED}" \
  --dtype "${DTYPE}" \
  --attn-implementation "${ATTN_IMPLEMENTATION}" \
  --ablation-scale "${ABLATION_SCALE}" \
  --logprob-chunk-size "${LOGPROB_CHUNK_SIZE}" \
  "$@"

echo "wrote ${OUTPUT}"
