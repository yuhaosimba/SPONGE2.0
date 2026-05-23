# REST2 + FEP Soft-Core Shared Selective Kernel Migration Plan

This plan describes how to support REST2 on FEP soft-core nonbonded
interactions while reducing duplication between SITS and REST2. The target
design keeps the public SPONGE input interface unchanged and refactors only the
Selective_Interaction implementation.

## Background

SITS and REST2 are both selective-interaction methods:

- SITS selects hot-region interactions, records selected force/energy/virial,
  and later applies ITS bias/enhancement.
- REST2 selects hot-region interactions and directly scales the Hamiltonian
  with `lambda_m` / `sqrt(lambda_m)` for REMD.

The current REST2 implementation covers normal short-range LJ/direct Coulomb
interactions, but the FEP `LJ_soft_core` direct path is effectively unscaled
under REST2. When REST2 is enabled together with FEP soft-core, the current
facade calls the original `LJ_SOFT_CORE` kernel instead of a REST2-aware
soft-core kernel. As a result, `REST2_unscaled`, `REST2_effective`, and
`REST2_bias` can remain zero for FEP-dominated systems.

SITS already has a selective soft-core kernel:

```text
SITS_LJ_Soft_Core_Direct_CF_Force_With_Atom_Energy_And_Virial
  -> Selective_Lennard_Jones_And_Direct_Coulomb_Soft_Core_Device
```

This kernel should be used as the reference for the REST2 soft-core path, but
the final implementation should avoid permanently maintaining two independent
copies of the same soft-core pair loop.

## Goals

- Add scientifically meaningful REST2 scaling for FEP `LJ_soft_core` direct
  short-range LJ and direct Coulomb interactions.
- Share the complex selective soft-core pair-loop logic between SITS and REST2.
- Preserve current mdin and `manager.toml` user-facing syntax.
- Preserve existing SITS behavior and validation results.
- Keep the main MD loop stable: `main.cpp` should continue calling the
  `Selective_Interaction` facade instead of knowing about SITS/REST2 internals.
- Keep the implementation compatible with existing SPONGE device abstraction
  style (`Launch_Device_Kernel`, device API wrappers, current warp-sum helpers).

## Non-Goals

- Do not change the manager/worker communication protocol.
- Do not change REST2 input names such as `REST2_mode`, `REST2_atom_numbers`,
  `REST2_atom_in_file`, or `REST2_lambda_m`.
- Do not claim REST2 coverage for reciprocal PME, bonded terms, 1-4 terms, or
  long-range correction unless those paths are explicitly implemented and
  validated.
- Do not rewrite all SITS logic in the first patch.
- Do not introduce runtime polymorphism inside CUDA/HIP kernels.

## Target Architecture

Use a shared template kernel with small compile-time policy objects.

```text
Selective_Interaction facade
  |
  +-- normal LJ/direct Coulomb selective pair kernel
  |     +-- SITS policy
  |     +-- REST2 policy
  |
  +-- FEP LJ_soft_core/direct Coulomb selective pair kernel
        +-- SITS policy
        +-- REST2 policy
```

The shared kernel owns the expensive and error-prone mechanics:

- neighbor-list traversal
- PBC displacement
- hot/cold pair classification from `atom_sys_mark_local`
- A/B topology LJ type lookup
- hard-core versus soft-core branch selection
- soft-core distance evaluation
- LJ force and energy calculation
- direct Coulomb force and energy calculation
- optional `dU/dlambda`
- warp reduction and atomic updates

The policy owns how each pair contribution is scaled and where it is written:

- SITS policy writes normal MD force/energy/virial plus selected
  force/energy/virial buffers for later ITS enhancement.
- REST2 policy writes scaled force/energy/virial directly into the MD buffers
  and records REST2 unscaled/effective energy.

## Proposed File Layout

Keep the public facade files and add implementation-only shared kernel headers:

```text
SPONGE/Selective_Interaction/
  Selective_Interaction.h
  Selective_Interaction.cpp
  SITS.h
  SITS.cpp
  REST2.h
  REST2.cpp
  Selective_Pair_Kernels.h
  Selective_Policies.h
```

`Selective_Pair_Kernels.h` should contain template device helpers and shared
kernel definitions. `Selective_Policies.h` should contain small policy structs
or policy helper functions. These files are implementation details and should
not expose new user-facing inputs.

## Policy Semantics

### Pair Classification

Use the existing mark convention:

```text
atom_sys_mark = 0  hot / selected region
atom_sys_mark = 1  cold / environment
mark_sum = mark_i + mark_j
```

### SITS Policy

SITS keeps its existing semantics:

```text
hot-hot:  selected factor = 1
hot-cold: selected factor = pwwp_enhance_factor
cold-cold: selected factor = 0
```

The normal physical soft-core contribution is written to the normal MD buffers.
The selected contribution is written to SITS selection buffers:

```text
pw_select.select_force[0]
pw_select.select_atom_energy[0]
pw_select.select_atom_virial_tensor[0]
```

SITS then applies bias/enhancement in `Update_And_Enhance()`.

### REST2 Policy

REST2 should directly scale the active Hamiltonian:

```text
hot-hot:  scale = REST2_lambda_m
hot-cold: scale = sqrt(REST2_lambda_m)
cold-cold: scale = 1
```

For force and virial, REST2 writes `scale * raw_contribution` to the normal MD
buffers. For energy, REST2 writes the scaled contribution to normal atom-energy
buffers and records both:

```text
REST2_unscaled   += raw hot-related contribution
REST2_effective  += scaled hot-related contribution
REST2_bias        = REST2_effective - REST2_unscaled
```

Only hot-related pairs (`mark_sum < 2`) should contribute to REST2 diagnostic
unscaled/effective energy. Cold-cold pairs remain part of the normal total
potential but should not contribute to REST2 bias diagnostics.

## Migration Steps

### Step 1: Add REST2 Soft-Core Entry Point

Add a REST2 method:

```cpp
REST2_INFORMATION::LJ_Soft_Core_Direct_CF_Force_With_Atom_Energy_And_Virial(...)
```

and route the REST2 branch in `Selective_Interaction.cpp` to this method instead
of calling:

```cpp
lj_info->LJ_Soft_Core_PME_Direct_Force_With_Atom_Energy_And_Virial(...)
```

This is the first correctness-critical change.

### Step 2: Introduce Shared Soft-Core Kernel Helpers

Extract the common pair-loop mechanics from the existing SITS soft-core kernel
into `Selective_Pair_Kernels.h`.

The first version may be conservative:

- preserve current loop order
- preserve current hard-core/soft-core branch formulas
- preserve current `Warp_Sum_To` and `atomicAdd` patterns
- keep template bool parameters for `need_force`, `need_energy`,
  `need_virial`, `need_coulomb`, and `need_du_dlambda`

Avoid clever abstraction that makes generated code harder to inspect.

### Step 3: Implement REST2 Soft-Core Policy

Add a REST2 policy that receives:

```text
lambda_m
sqrt_lambda_m
atom_sys_mark_local
d_unscaled_atom_energy
d_effective_atom_energy
normal force/energy/virial buffers
```

The policy should:

- scale force by `lambda_m`, `sqrt(lambda_m)`, or `1`
- scale LJ and direct Coulomb energies consistently with force scaling
- scale virial consistently with force scaling
- accumulate unscaled and effective REST2 diagnostic energies
- leave cold-cold diagnostic energy unchanged

### Step 4: Keep SITS Behavior Stable

Initially, SITS may continue to use its existing kernel if that keeps the first
REST2 fix small. After REST2 soft-core passes validation, migrate SITS to the
shared soft-core kernel with a SITS policy.

This two-stage approach makes failures easier to localize:

```text
first patch: REST2 gains soft-core correctness
second patch: SITS switches to shared implementation without semantic change
```

### Step 5: Share Normal LJ/Direct Coulomb Kernel If Safe

After soft-core is stable, consider applying the same policy-based structure to
the normal LJ/direct Coulomb selective kernel. This is lower priority because
REST2 already has a normal LJ/direct Coulomb implementation.

## Validation Plan

### Build and Format

Run:

```bash
pixi run -e dev-cuda13 cmake --build build-dev-cuda13 --target SPONGE SPONGE_MANAGER -j 8
pixi run -e dev-cuda13 cmake --install build-dev-cuda13
pixi run -e dev-cuda13 format-check
```

### SITS Regression

Run the existing SITS validation/performance tests before and after switching
SITS to the shared kernel:

```bash
pixi run -e dev-cuda13 perf-sits
```

If `perf-sits` is too expensive for every iteration, run the smallest available
SITS smoke/validation case first, then run the full task before committing.

Expected result:

- SITS runs to completion.
- SITS printed energies/bias remain numerically consistent with the old path
  within normal floating-point tolerance.
- No new CUDA memory error or NaN appears.

### REST2 Soft-Core Smoke Test

Use the repository FEP + REST2 NPT manager example or the TMP FEP fixture.
Run a short REST2 exchange/smoke test with `REST2_lambda_m` values that differ
from `1.0`.

Expected result:

- The job runs to completion.
- `REST2_lambda_m` is printed and differs across replicas.
- `REST2_unscaled` and `REST2_effective` are nonzero for FEP soft-core systems.
- `REST2_bias = REST2_effective - REST2_unscaled`.
- Changing `REST2_lambda_m` changes REST2 diagnostics and Hamiltonian probes.

### REST2 Scientific Sanity Checks

For a fixed coordinate snapshot, compare:

- REST2 disabled versus REST2 enabled with `REST2_lambda_m = 1.0`
- REST2 enabled with `REST2_lambda_m < 1.0`
- hot-region all atoms versus a smaller hot-region atom list

Expected result:

- `REST2_lambda_m = 1.0` should reproduce the unscaled soft-core path.
- `REST2_lambda_m < 1.0` should reduce hot-hot soft-core contributions and
  partially reduce hot-cold contributions.
- `REST2_atom_numbers = ALL` should make diagnostics respond strongly.
- A small hot region should only affect pairs touching that region.

### REMD Validation

Run a short REST2-REMD manager test:

```bash
SPONGE_MANAGER --config <rest2-fep-manager.toml>
```

Expected result:

- Exchange attempts use nonzero Hamiltonian differences.
- Accepted/rejected exchanges are logged normally.
- RuntimeState exchange still works.
- No worker restart or memory growth is introduced by the kernel refactor.

## Risks and Guardrails

- Soft-core force and energy formulas are delicate. Keep formula changes minimal
  and copy the existing SITS/LJ_soft_core expressions exactly before applying
  policy scaling.
- Energy scaling and force scaling must be consistent; otherwise REST2 exchange
  probabilities may look plausible while dynamics are wrong.
- `dU/dlambda` is important for FEP/TI outputs. Do not accidentally break the
  existing `LJ_SOFT_CORE` derivative path. If the shared kernel handles
  `need_du_dlambda`, preserve the old result for non-REST2/SITS paths.
- SITS bias logic should remain outside the shared kernel. The shared kernel
  should only compute pair contributions and selected buffers.
- Avoid hidden user-interface changes. REST2 + FEP should work with the same
  mdin and manager TOML syntax already used by current tests.
- Do not introduce temporary build flags or compatibility names such as
  `isoc23_compact`.

## Completion Criteria

The migration is complete when:

- REST2 has a dedicated soft-core entry point.
- `Selective_Interaction.cpp` routes REST2 soft-core calls to the REST2-aware
  implementation.
- REST2 + FEP soft-core produces nonzero and lambda-dependent REST2 diagnostics.
- Existing SITS tests still pass.
- The FEP + REST2 manager example runs through a short NPT/REMD smoke test.
- `format-check` passes.
- The implementation keeps the SPONGE public input interface unchanged.
