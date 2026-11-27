#!/usr/bin/env bash
# Repository-relative defaults; export any variable below to override it.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="${DATA_ROOT:-${SCRIPT_DIR}/data}"
OUT_ROOT="${OUT_ROOT:-${SCRIPT_DIR}/output}"
GS_ROOT="${GS_ROOT:-${SCRIPT_DIR}/gaussian-splatting}"
GS_TRAIN="${GS_TRAIN:-${GS_ROOT}/train.py}"
GS_RENDER="${GS_RENDER:-${GS_ROOT}/render.py}"
GS_METRICS="${GS_METRICS:-${GS_ROOT}/metrics.py}"
CONDA_ENV="${CONDA_ENV:-gs}"
