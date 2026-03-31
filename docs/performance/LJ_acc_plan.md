# LJ Atom-Pair Performance Optimization Plan

> Scope: keep the current atom-pair neighbor-list structure (`ATOM_GROUP`) unchanged. Optimize the LJ/direct-Coulomb path as far as possible without introducing cluster-pair list semantics.

## Goal

Improve SPONGE GPU nonbonded throughput on the current atom-pair path while preserving the existing neighbor-list interface used by other modules.

## Boundaries

- Do not change `ATOM_GROUP` semantics or the external neighbor-list API.
- Do not convert the codebase to cluster-pair lists in this branch.
- Allow internal kernel/data-path changes inside the LJ module.
- Every optimization patch must be followed by benchmark measurement.

## Profiling Baseline

The current bottleneck is not register spilling. The main issues from Nsight Compute are:

- Low SM throughput: about `15.66%`
- Low memory throughput: about `15.59%`
- Low achieved occupancy: about `42.09%`
- Poor cache locality: `L2 hit ~34.80%`, `L1/TEX hit ~0%`
- Main warp stalls: `wait`, `short_scoreboard`, `branch_resolving`, `drain`

Reference files:

- General LJ kernel: `SPONGE/Lennard_Jones_force/Lennard_Jones_force.cpp`
- LJ math helpers: `SPONGE/Lennard_Jones_force/Lennard_Jones_force.h`
- Water-specialized staging path: `SPONGE/Lennard_Jones_force/solvent_LJ.cpp`
- External benchmark harness: `/home/ylj/SPONGE_GITHUB/SPG_GMX_TEST/runner.py`

Important note:

- The `ncu` per-launch kernel duration is not directly comparable between SPONGE and GROMACS as a total-step cost. Use throughput, cache, occupancy, stall, and end-to-end `ns/day` as the primary evidence.

## Optimization Roadmap

### P0: Arithmetic Cleanup

Target files:

- `SPONGE/Lennard_Jones_force/Lennard_Jones_force.h`
- `SPONGE/Lennard_Jones_force/Lennard_Jones_force.cpp`
- Any duplicated LJ formulas in `solvent_LJ.cpp`, `LJ_soft_core`, or SITS variants if affected

Tasks:

1. Replace `sqrt + powf` heavy code with `dr2 / inv_r / inv_r2 / inv_r6` math.
2. Switch cutoff checks from `dr < cutoff` to `dr2 < cutoff2`.
3. Keep force/energy/virial semantics unchanged for the current template specializations.
4. Re-test numerical stability after the rewrite.

Why first:

- This is the highest expected gain for the lowest architectural risk.

### P0: Launch Configuration Sweep

Target files:

- `SPONGE/Lennard_Jones_force/Lennard_Jones_force.cpp`

Tasks:

1. Stop assuming the current `(32, 32, 1)` launch shape is optimal.
2. Benchmark at least:
   - `(32, 4, 1)`
   - `(32, 8, 1)`
   - `(32, 16, 1)`
   - `(64, 4, 1)` if the mapping remains valid
3. Keep the fastest shape only after end-to-end benchmark confirmation.

Why now:

- The current kernel is memory-irregular. A smaller block may improve active blocks per SM and reduce latency-hiding failure.

### P1: Read-Only Path Cleanup

Target files:

- `SPONGE/Lennard_Jones_force/Lennard_Jones_force.cpp`
- `SPONGE/Lennard_Jones_force/Lennard_Jones_force.h`

Tasks:

1. Add `const __restrict__` where aliasing is impossible.
2. Make read-only arrays eligible for better caching paths.
3. Remove avoidable inner-loop branches in the force-only hot path.

Why:

- Low development cost. Can help cache behavior and compiler scheduling.

### P1: General-Kernel J-Tile Staging

Target files:

- `SPONGE/Lennard_Jones_force/Lennard_Jones_force.cpp`
- Use `solvent_LJ.cpp` as reference, not as a direct template copy

Tasks:

1. Stage a tile of `atom_j` indices into shared memory.
2. Stage corresponding `x/y/z/q/type` into shared memory or registers.
3. Reuse staged data across the block before loading the next tile.

Why:

- The current kernel performs random gathers for each pair. This is consistent with the poor L2/L1 results from `ncu`.

### P1: Atomic Writeback Reduction

Target files:

- `SPONGE/Lennard_Jones_force/Lennard_Jones_force.cpp`

Tasks:

1. Reduce `atomicAdd` pressure on `frc[atom_j]`.
2. Evaluate warp-local or block-local accumulation before global flush.
3. Keep the current physical semantics unchanged.

Why:

- Random atomic writeback is one of the most likely contributors to low issue rate and poor memory efficiency.

### P2: Hot-Buffer Layout Split

Target files:

- LJ setup path and launch path
- Potentially add an internal packed SoA-like buffer for the LJ kernel

Tasks:

1. Keep external API unchanged.
2. Build a kernel-friendly hot buffer for `x/y/z/q/type`.
3. Use that buffer only for the LJ path.

Why:

- `VECTOR_LJ` is AoS-oriented. The current memory pattern is not ideal for the GPU kernel.

### P2: Type-Table Specialization

Target files:

- `SPONGE/Lennard_Jones_force/Lennard_Jones_force.cpp`
- `SPONGE/Lennard_Jones_force/Lennard_Jones_force.h`

Tasks:

1. Review how `Get_LJ_Type` and `LJ_type_A/B` are used in the hot loop.
2. For low-type-count systems such as water, evaluate constant-memory or packed-parameter paths.
3. Keep this optional unless profiling shows it is still material after P0/P1.

## Benchmark Discipline

Every optimization change must be measured. No patch should be considered an improvement without fresh benchmark evidence.

### Environment

```bash
source /home/ylj/SPONGE_GITHUB/SPG_GMX_TEST/env_gpu.sh
export SPONGE_BIN=/home/ylj/SPONGE_GITHUB/SPONGE/build-dev-cuda12/SPONGE
export GMX_BIN=/home/ylj/应用/gromacs-2026.1/install-cuda/bin/gmx
```

If a different SPONGE binary is under test, update `SPONGE_BIN` explicitly before each run.

### Build Step

Use a fresh CUDA build before benchmarking:

```bash
cd /home/ylj/SPONGE_GITHUB/SPONGE
pixi run -e dev-cuda12 compile
```

### Quick Regression Benchmark

Run after every meaningful LJ patch:

```bash
cd /home/ylj/SPONGE_GITHUB/SPG_GMX_TEST
python runner.py bench \
  --mode NVT \
  --engine sponge \
  --sponge-bin "$SPONGE_BIN" \
  --gpu-id 0 \
  --steps 20000 \
  --warmup 1 \
  --repeats 1 \
  --run-id lj_acc_quick_<tag>
```

Record from `outputs/<run_id>/summary.json`:

- `elapsed_s`
- `steps_per_s`
- `ns_per_day`

### Acceptance Benchmark

Run after each completed P0/P1 phase:

```bash
cd /home/ylj/SPONGE_GITHUB/SPG_GMX_TEST
python runner.py bench \
  --mode NVT \
  --engine sponge \
  --sponge-bin "$SPONGE_BIN" \
  --gpu-id 0 \
  --steps 100000 \
  --warmup 1 \
  --repeats 3 \
  --run-id lj_acc_accept_<tag>
```

### Optional Reference Comparison Against GROMACS 1 CPU

Use when checking whether SPONGE closes the kernel-level gap while preserving current fairness settings:

```bash
cd /home/ylj/SPONGE_GITHUB/SPG_GMX_TEST
python runner.py bench \
  --mode NVT \
  --engine both \
  --sponge-bin "$SPONGE_BIN" \
  --gmx-bin "$GMX_BIN" \
  --gmx-ntomp 1 \
  --gpu-id 0 \
  --steps 20000 \
  --warmup 1 \
  --repeats 1 \
  --run-id lj_acc_vs_gmx1_<tag>
```

## Profiler Checkpoints

Use profiler checkpoints only after a patch changes the kernel behavior materially.

### Nsight Systems

Purpose:

- Confirm that kernel launch cadence and end-to-end GPU pipeline do not regress

### Nsight Compute

Purpose:

- Check whether a patch improves:
  - SM throughput
  - memory throughput
  - achieved occupancy
  - L2 hit rate
  - stall breakdown

Minimum comparison set:

- current branch result
- previous baseline result

## Required Evidence Per Patch

For each LJ optimization patch, record:

1. Git commit hash
2. Patch scope
3. Quick regression benchmark `ns/day`
4. Delta vs previous baseline
5. Whether numerical output stayed acceptable
6. Whether profiler evidence was collected

Recommended log format:

```markdown
| Commit | Scope | Steps | ns/day | Delta | Notes |
|--------|-------|-------|--------|-------|-------|
| abc123 | arithmetic rewrite | 20000 | 78.4 | +12.1% | NVT quick regression |
```

## Execution Order

Recommended order for this branch:

1. Arithmetic cleanup
2. Launch-shape sweep
3. Read-only path cleanup
4. J-tile staging
5. Atomic writeback reduction
6. Optional hot-buffer layout split
7. Optional type-table specialization

## Exit Criteria

The branch is considered technically successful only if:

1. SPONGE `ns/day` improves on the same NVT case and same fairness settings
2. Numerical behavior remains acceptable
3. No neighbor-list API changes are introduced
4. Benchmark evidence is stored for each phase

