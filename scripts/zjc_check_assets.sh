#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/zjc_env.sh
source "${SCRIPT_DIR}/zjc_env.sh"
cd "${PROJECT_ROOT}"

ALLOW_MISSING="${ALLOW_MISSING:-0}"
missing=0

check_path() {
  local kind="$1"
  local label="$2"
  local path="$3"
  if [[ "$kind" == "file" && -f "$path" ]] || [[ "$kind" == "dir" && -d "$path" ]]; then
    echo "[ok] ${label}: ${path}"
    return 0
  fi
  echo "[missing] ${label}: ${path}" >&2
  missing=$((missing + 1))
}

echo "=== ZJC lgar asset check ==="
echo "project_root=${PROJECT_ROOT}"
echo "python=${PYTHON}"
echo "nproc_per_node=${NPROC_PER_NODE}"
echo "raw_data_dir=${RAW_DATA_DIR}"
echo "cache_dir=${CACHE_DIR}"

check_path dir "model" "${MODEL_PATH}"
check_path dir "raw FineWeb data" "${RAW_DATA_DIR}"
check_path dir "pretrain token cache" "${CACHE_DIR}"
check_path file "top heads json" "${TOP_HEADS_JSON}"

check_path file "ce checkpoint" "${CE_CKPT}"
check_path file "adapter checkpoint" "${ADAPTER_CKPT}"
check_path file "longce checkpoint" "${LONGCE_CKPT}"
check_path file "cure checkpoint" "${CURE_CKPT}"

check_path dir "RULER root" "${RULER_DIR}"
check_path file "RULER prepare.py" "${RULER_DIR}/_assets/RULER/scripts/data/prepare.py"
check_path dir "RULER generated ${RULER_FOLDER}" "${RULER_DIR}/generated/${RULER_FOLDER}"

check_path dir "LongBench root" "${LONGBENCH_ROOT}"
check_path dir "LongEval root" "${LONGEVAL_ROOT}"
check_path dir "BABILong root" "${BABILONG_ROOT}"
check_path file "NoLiMA needle set" "${NOLIMA_NEEDLE_SET}"
check_path dir "NoLiMA haystack" "${NOLIMA_HAYSTACK_DIR}"
check_path dir "prefix scripts" "${PREFIX_SCRIPTS_DIR}"

if [[ "${missing}" -gt 0 ]]; then
  echo "missing=${missing}" >&2
  if [[ "${ALLOW_MISSING}" != "1" ]]; then
    exit 1
  fi
fi

echo "=== asset check done ==="
