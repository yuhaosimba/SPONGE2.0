# PM Direct Neutral Refactor Plan

## Goal

Keep the LJ kernels responsible for LJ plus direct electrostatics while making
the PM direct backend interface neutral:

- external call sites use only PM-neutral names
- PME and ESP are peer direct backends
- no performance-sensitive runtime abstraction is added to the pairwise hot path

## Constraints

- `main.cpp` must not encode PME-vs-ESP direct naming
- LJ pair kernels keep ownership of pair traversal and LJ accumulation
- direct Coulomb evaluators remain header-inline helpers
- current GPU launch shapes and validated `order=5` ESP reciprocal kernels stay unchanged

## Refactor Shape

1. Keep LJ kernels as the execution owner for direct-space pair interactions.
2. Move PME direct scalar formulas into PM direct helper headers so PME and ESP
   live at the same abstraction level.
3. Expose unified PM direct helpers:
   - `PM_Get_Direct_Coulomb_Energy`
   - `PM_Get_Direct_Coulomb_Force`
   - `PM_Get_Excluded_Coulomb_Energy`
   - `PM_Get_Excluded_Coulomb_Force`
4. Rename public direct entrypoints from `*PME_Direct*` to `*PM_Direct*`.
5. Keep selective and soft-core ESP support out of scope for this pass; they
   remain guarded by `Validate_Direct_Force_Path(...)`.

## Implementation Steps

### Step 1: PM-neutral direct evaluator layer

- add `SPONGE/PM_force/pm_direct.h`
- make `PM_Direct_Parameters` carry backend-neutral state
- provide PME scalar direct/excluded formulas there
- provide unified dispatch helpers there

### Step 2: Rename public direct entrypoints

- `LJ_PME_Direct_Force_With_Atom_Energy_And_Virial`
  -> `LJ_PM_Direct_Force_With_Atom_Energy_And_Virial`
- `LJ_Soft_Core_PME_Direct_Force_With_Atom_Energy_And_Virial`
  -> `LJ_Soft_Core_PM_Direct_Force_With_Atom_Energy_And_Virial`
- solvent direct entrypoint renamed the same way

### Step 3: Keep hot path zero-cost

- no virtual dispatch
- no function pointers in the pair loop
- evaluator helpers stay `__host__ __device__ __forceinline__`
- backend choice is still a simple inline branch on the direct parameter pack

### Step 4: Verification

- CPU build
- CUDA build
- `scripts/esp_pswf_direct_smoke.sh`
- `scripts/esp_wat_single_step_smoke.sh`
- `scripts/esp_wat_npt_smoke.sh`

## Out of Scope

- selective-interaction ESP direct support
- soft-core ESP direct support
- autotune or launch retuning
- reciprocal backend reorganization
