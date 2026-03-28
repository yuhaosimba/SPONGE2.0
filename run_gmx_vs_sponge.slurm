#!/bin/bash
#SBATCH --job-name=gmx_vs_sponge
#SBATCH --partition=gaoyiqinlab
#SBATCH --nodelist=gnho039
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=gmx_vs_sponge_%j.out
#SBATCH --error=gmx_vs_sponge_%j.err

set -euo pipefail

REPO_DIR="/storage/gaoyiqinLab/hujinyuan/SPG_GMX_Bench/SPONGE2.0"
PIXI_BIN="${HOME}/.pixi/bin/pixi"
PIXI_ENV="dev-cuda12"
DEFAULT_GMX_BIN="/storage/gaoyiqinLab/hujinyuan/tools/gromacs_cuda_build/install/bin/gmx"
DEFAULT_CASE_DIR="${REPO_DIR}/benchmarks/performance/gmx_vs_sponge/statics/water_160k"
TIMING_LOG="gmx_vs_sponge_wallclock_${SLURM_JOB_ID:-manual}.log"

cd "${REPO_DIR}"

if [[ ! -x "${PIXI_BIN}" ]]; then
  echo "pixi not found at ${PIXI_BIN}" >&2
  exit 1
fi

export PATH="${HOME}/.pixi/bin:${PATH}"
export BENCH_WARMUP=1
export BENCH_REPEATS=1
export BENCH_STEPS=500000
export BENCH_CASE_DIR="${BENCH_CASE_DIR:-${DEFAULT_CASE_DIR}}"
export BENCH_GPU_ID=0
export BENCH_GMX_NTOMP=8
export BENCH_SPONGE_BIN="${BENCH_SPONGE_BIN:-SPONGE}"
export BENCH_GMX_BIN="${BENCH_GMX_BIN:-${DEFAULT_GMX_BIN}}"

if ! "${PIXI_BIN}" run -e "${PIXI_ENV}" which "${BENCH_SPONGE_BIN}" >/dev/null 2>&1; then
  echo "Required executable not available in pixi env ${PIXI_ENV}: ${BENCH_SPONGE_BIN}" >&2
  exit 1
fi

if [[ ! -x "${BENCH_GMX_BIN}" ]] && ! command -v "${BENCH_GMX_BIN}" >/dev/null 2>&1; then
  echo "Required GROMACS executable not found: ${BENCH_GMX_BIN}" >&2
  echo "Set BENCH_GMX_BIN to an existing GROMACS binary before re-submitting." >&2
  exit 1
fi

wallclock_start="$(date '+%F %T %z')"
epoch_start="$(date +%s)"
{
  echo "start=${wallclock_start}"
  echo "job_id=${SLURM_JOB_ID:-manual}"
  echo "node_list=${SLURM_NODELIST:-unknown}"
  echo "bench_gmx_bin=${BENCH_GMX_BIN}"
} > "${TIMING_LOG}"

if command -v /usr/bin/time >/dev/null 2>&1; then
  /usr/bin/time -v -o "${TIMING_LOG}.time" "${PIXI_BIN}" run -e "${PIXI_ENV}" bench-gmx-vs-sponge
else
  "${PIXI_BIN}" run -e "${PIXI_ENV}" bench-gmx-vs-sponge
fi

epoch_end="$(date +%s)"
wallclock_end="$(date '+%F %T %z')"
{
  echo "end=${wallclock_end}"
  echo "elapsed_seconds=$((epoch_end - epoch_start))"
} >> "${TIMING_LOG}"
