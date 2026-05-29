#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="$(mktemp -d /tmp/esp_pswf_direct_smoke.XXXXXX)"
trap 'rm -rf "${BUILD_DIR}"' EXIT

CXX="${CXX:-c++}"
ENV_PREFIX="${CONDA_PREFIX:-${PIXI_PROJECT_ROOT:-}}"
INCLUDE_FLAGS=(-I"${ROOT_DIR}")
LIBRARY_FLAGS=()
if [[ -n "${ENV_PREFIX}" && -d "${ENV_PREFIX}/include" ]]; then
  INCLUDE_FLAGS+=(-I"${ENV_PREFIX}/include")
  if [[ -d "${ENV_PREFIX}/include/fftw" ]]; then
    INCLUDE_FLAGS+=(-I"${ENV_PREFIX}/include/fftw")
  fi
fi
if [[ ! -f "${ENV_PREFIX:-}/include/fftw/fftw3.h" &&
      -f "${ROOT_DIR}/.pixi/envs/dev-cpu/include/fftw/fftw3.h" ]]; then
  INCLUDE_FLAGS+=(-I"${ROOT_DIR}/.pixi/envs/dev-cpu/include/fftw")
fi
if [[ -n "${ENV_PREFIX}" && -d "${ENV_PREFIX}/lib" ]]; then
  LIBRARY_FLAGS+=(-L"${ENV_PREFIX}/lib")
fi

"${CXX}" -std=c++17 -O2 "${INCLUDE_FLAGS[@]}" \
  "${ROOT_DIR}/scripts/esp_pswf_direct_smoke.cpp" \
  "${ROOT_DIR}/SPONGE/PM_force/esp_pswf.cpp" \
  "${LIBRARY_FLAGS[@]}" \
  -o "${BUILD_DIR}/esp_pswf_direct_smoke"

"${BUILD_DIR}/esp_pswf_direct_smoke"
