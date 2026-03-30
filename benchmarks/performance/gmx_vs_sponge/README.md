# SPONGE vs GROMACS GPU Benchmark (water_160k)

This benchmark compares throughput (`ns/day`, `steps/s`) for NVT and NPT.

## Default static files

Located in `benchmarks/performance/gmx_vs_sponge/statics/water_160k/`:

- `water.top` + `water.gro` (GROMACS)
- `water.prmtop` + `npt1.rst7` (SPONGE)
- `water_nvt_eq.gro` + `water_nvt_eq.rst7` (aligned NVT start)
- `water_npt_eq.gro` + `water_npt_eq.rst7` (aligned NPT start)

## Important precondition

For each mode, GROMACS `gro` and SPONGE `rst7` must represent the same coordinates/box.

Before benchmark, the test performs strict checks:

- atom count equality
- box length consistency
- coordinate RMSD under tolerance (`BENCH_COORD_TOL_ANGSTROM`, default `0.02` A)

If mismatch is detected, benchmark exits with an explicit error.

## Run with pixi

```bash
pixi run -e dev-cuda12 bench-gmx-vs-sponge
```

## Environment variables

- `BENCH_CASE_DIR` (default: repo static dir)
- `BENCH_WARMUP` (default: `3`)
- `BENCH_REPEATS` (default: `5`)
- `BENCH_STEPS` (default: `200000`)
- `BENCH_GPU_ID` (default: `0`)
- `BENCH_GMX_NTMPI` (default: `1`)
- `BENCH_GMX_NTOMP` (default: `8`)
- `BENCH_GMX_BIN` (default: `gmx`)
- `BENCH_SPONGE_PARM7_FILE` (default: auto-detect `water.prmtop`)
- `BENCH_SPONGE_RST7_FILE_NVT` (default: `water_nvt_eq.rst7`)
- `BENCH_SPONGE_RST7_FILE_NPT` (default: `water_npt_eq.rst7`)
- `BENCH_TOP_FILE` (default: `water.top`)
- `BENCH_GRO_FILE_NVT` (default: `water_nvt_eq.gro`)
- `BENCH_GRO_FILE_NPT` (default: `water_npt_eq.gro`)
- `BENCH_CUTOFF_ANGSTROM` (default: `10.0`)
- `BENCH_SKIN_ANGSTROM` (default: `2.0`)
- `BENCH_NSTLIST` (default: `50`)
- `BENCH_PME_GRID_ANGSTROM` (default: `1.0`, used when FFT grid is not explicitly set)
- `BENCH_PME_FFTX`, `BENCH_PME_FFTY`, `BENCH_PME_FFTZ` (default: unset; when all 3 are set, both engines use explicit identical FFT grid)
- `BENCH_GMX_DISABLE_PME_TUNING` (default: `1`; adds `-notunepme` for strict PME-grid comparability)

## Outputs

Generated under `benchmarks/performance/gmx_vs_sponge/outputs/water160k_gpu/`:

- `summary.json`
- `summary.csv`
