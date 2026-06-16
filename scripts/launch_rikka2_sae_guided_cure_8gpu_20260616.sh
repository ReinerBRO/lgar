#!/usr/bin/env bash
set -euo pipefail

export ZJC_ROOT=/gfs/space/private/zjc
export PROJECT_ROOT=/gfs/space/private/zjc/goals/lgar
cd "${PROJECT_ROOT}"

mkdir -p logs configs

RUN_NAME=stage2_cure_sae_guided_heads_from_base_frozen_20260616_16k_0p5b
LAUNCH_LOG="${PROJECT_ROOT}/logs/launch_rikka2_${RUN_NAME}.out"

if [[ "${1:-}" != "--worker" ]]; then
  rm -f "${LAUNCH_LOG}"
  nohup bash "$0" --worker > "${LAUNCH_LOG}" 2>&1 < /dev/null &
  echo "pid=$!"
  echo "log=${LAUNCH_LOG}"
  echo "run_name=${RUN_NAME}"
  exit 0
fi

PYTHON=/gfs/space/private/zjc/envs/zjc_env/bin/python
MODEL_PATH="${ZJC_ROOT}/models/Qwen2.5-0.5B"
CACHE_DIR="${ZJC_ROOT}/data/qwen_fineweb_stage2"
SIGNAL_DIR="${PROJECT_ROOT}/data/offline_signal_16k_1b_sw2048_lw2560"

SAE_SIGNAL_TAG=20260615_layer21_sae_signal_8shard
SAE_ROOT="${PROJECT_ROOT}/runs/${SAE_SIGNAL_TAG}"
SAE_LAYER=21
SAE_CKPT="${SAE_ROOT}/sae/sae_layer${SAE_LAYER}.pt"
FEATURES_VALID="${SAE_ROOT}/evidence_features_validated.json"

HEADS_JSON="${PROJECT_ROOT}/configs/cure_sae_guided_top24_heads_layer21_20260616.json"
SELECT_LOG="${PROJECT_ROOT}/logs/zjc_select_sae_guided_heads_layer21_20260616.log"
TRAIN_LOG="${PROJECT_ROOT}/logs/zjc_${RUN_NAME}.log"
OUTPUT_ROOT="${PROJECT_ROOT}/runs/${RUN_NAME}"
TB_DIR="${PROJECT_ROOT}/runs/tensorboard/20260616_sae_guided_heads/${RUN_NAME}"

NUM_GPUS=8
SEQ_LEN=16384
GLOBAL_TOKENS=500000000
TOKENS_PER_RANK=$((GLOBAL_TOKENS / NUM_GPUS))
SAVE_INTERVAL=763
WAIT_FOR_IDLE_GPUS="${WAIT_FOR_IDLE_GPUS:-1}"
GPU_IDLE_MEMORY_MIB="${GPU_IDLE_MEMORY_MIB:-1000}"
GPU_WAIT_SECONDS="${GPU_WAIT_SECONDS:-43200}"
GPU_POLL_SECONDS="${GPU_POLL_SECONDS:-60}"

wait_for_idle_gpus() {
  if [[ "${WAIT_FOR_IDLE_GPUS}" != "1" ]]; then
    return 0
  fi
  local start_ts now_ts used_csv busy
  start_ts=$(date +%s)
  while true; do
    used_csv="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr '\n' ' ')"
    busy=0
    for used in ${used_csv}; do
      if [[ "${used}" -ge "${GPU_IDLE_MEMORY_MIB}" ]]; then
        busy=1
        break
      fi
    done
    if [[ "${busy}" == "0" ]]; then
      echo "all GPUs idle enough: memory.used MiB=${used_csv}"
      return 0
    fi
    now_ts=$(date +%s)
    if [[ $((now_ts - start_ts)) -ge "${GPU_WAIT_SECONDS}" ]]; then
      echo "timed out waiting for idle GPUs; memory.used MiB=${used_csv}" >&2
      nvidia-smi || true
      exit 5
    fi
    echo "waiting for idle GPUs; memory.used MiB=${used_csv}"
    sleep "${GPU_POLL_SECONDS}"
  done
}

echo "=== rikka2 SAE-guided CURE launch ==="
date -Is
echo "project=${PROJECT_ROOT}"
echo "model=${MODEL_PATH}"
echo "cache=${CACHE_DIR}"
echo "signal=${SIGNAL_DIR}"
echo "sae=${SAE_CKPT}"
echo "features=${FEATURES_VALID}"
echo "heads_json=${HEADS_JSON}"
echo "run_name=${RUN_NAME}"

for path in "${MODEL_PATH}" "${CACHE_DIR}" "${SIGNAL_DIR}"; do
  if [[ ! -d "${path}" ]]; then
    echo "missing dir: ${path}" >&2
    exit 1
  fi
done
for path in "${SAE_CKPT}" "${FEATURES_VALID}"; do
  if [[ ! -s "${path}" ]]; then
    echo "missing file: ${path}" >&2
    exit 1
  fi
done

wait_for_idle_gpus

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export HF_HOME="${ZJC_ROOT}/cache/huggingface"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export TRANSFORMERS_CACHE="${HF_HOME}/hub"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_IB_DISABLE=1
export NCCL_NVLS_ENABLE=0
export NCCL_DEBUG=WARN

if [[ ! -s "${HEADS_JSON}" ]]; then
  echo "=== select SAE-guided retrieval heads ==="
  CUDA_VISIBLE_DEVICES=0 "${PYTHON}" scripts/select_sae_guided_heads.py \
    --model-path "${MODEL_PATH}" \
    --cache-dir "${CACHE_DIR}" \
    --offline-signal-dir "${SIGNAL_DIR}" \
    --sae-checkpoint "${SAE_CKPT}" \
    --features "${FEATURES_VALID}" \
    --output "${HEADS_JSON}" \
    --candidate-layers 16-21 \
    --top-k-heads 24 \
    --batches 32 \
    --batch-size 1 \
    --max-tokens 0 \
    --seed 1337 \
    --dtype bf16 \
    --attn-implementation sdpa \
    --attention-mask-mode causal \
    2>&1 | tee -a "${SELECT_LOG}"
else
  echo "reuse existing heads_json=${HEADS_JSON}"
fi

"${PYTHON}" - <<PY
import json
from pathlib import Path
path = Path("${HEADS_JSON}")
payload = json.loads(path.read_text(encoding="utf-8"))
heads = payload.get("retrieval_heads") or []
print(json.dumps({
    "event": "sae_guided_heads_ready",
    "path": str(path),
    "num_heads": len(heads),
    "heads": heads,
    "source": payload.get("source"),
}, sort_keys=True), flush=True)
if len(heads) != 24:
    raise SystemExit(f"{path} has {len(heads)} heads; expected 24")
PY

echo "=== train CURE/A3 from base frozen with SAE-guided heads ==="
METHOD=cure_cpt \
STAGE2_TAG=20260616_sae_guided_heads \
RUN_NAME="${RUN_NAME}" \
OUTPUT_ROOT="${OUTPUT_ROOT}" \
TENSORBOARD_DIR="${TB_DIR}" \
LOG_FILE="${TRAIN_LOG}" \
ZJC_ROOT="${ZJC_ROOT}" \
PROJECT_ROOT="${PROJECT_ROOT}" \
PYTHON="${PYTHON}" \
MODEL_PATH="${MODEL_PATH}" \
CACHE_DIR="${CACHE_DIR}" \
OFFLINE_SIGNAL_DIR="${SIGNAL_DIR}" \
TOP_HEADS_JSON="${HEADS_JSON}" \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE="${NUM_GPUS}" \
SEQ_LEN="${SEQ_LEN}" \
LOCAL_WINDOW=2048 \
BATCH_SIZE=1 \
GLOBAL_TOKENS="${GLOBAL_TOKENS}" \
TOKENS_PER_RANK="${TOKENS_PER_RANK}" \
EVAL_INTERVAL=50 \
SAVE_INTERVAL="${SAVE_INTERVAL}" \
RESUME_IF_AVAILABLE=1 \
CURE_FROM_BASE=1 \
CURE_BASE_LR=0.0 \
FREEZE_BASE_MODEL=1 \
CURE_REQUIRE_FREEZE_BASE=1 \
ADAPTER_LR=1e-4 \
ADAPTER_WEIGHT_DECAY=0.0 \
LAMBDA_RH_CE=0.0 \
LAMBDA_RH_KD=0.0 \
LAMBDA_COV=0.0 \
LAMBDA_FULL_HU_CE=0.5 \
LAMBDA_NONHU_LOGP=0.02 \
CURE_MAIN_LOSS_MASK=valid_remote \
UTILITY_TOP_FRACTION_TRAINING=0.10 \
LORA_RANK=16 \
LORA_ALPHA=32.0 \
DTYPE=bf16 \
ATTN_IMPLEMENTATION=sdpa \
ATTENTION_MASK_MODE=causal \
GRADIENT_CHECKPOINTING=1 \
FSDP=0 \
SHOW_GPU_INFO=1 \
bash scripts/zjc_run_stage2_train_one.sh

echo "=== rikka2 SAE-guided CURE done ==="
date -Is
