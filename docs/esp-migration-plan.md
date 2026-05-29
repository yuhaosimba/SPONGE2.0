# ESP Migration Plan

This document describes how to migrate ESP, Ewald summation with prolates, into
SPONGE as an optional particle-mesh backend. The implementation target is
single-GPU correctness first, with CPU compatibility kept in every code path and
NPT compatibility included from the start. Multi-process PME is out of scope for
the first implementation.

## Goals

- Keep the existing external PME shape: `Particle_Mesh::Initial`,
  `PME_Excluded_Force_With_Atom_Energy`,
  `PME_Reciprocal_Force_With_Energy_And_Virial`, `Update_Box`, and the main
  force loop call sites should remain stable.
- Add ESP as an optional backend selected from `mdin`, without changing default
  PME behavior.
- Port the paper-level ESP optimization points:
  - PSWF-based Ewald splitting kernel.
  - PSWF-based particle-to-grid spreading and grid-to-particle interpolation.
  - PSWF Fourier-space influence function.
  - Smaller FFT grid support through ESP parameter selection.
  - Table or polynomial evaluation for PSWF kernels instead of evaluating PSWF
    functions inside every MD step.
  - Correct self energy, zero-frequency handling, energy, virial, and pressure
    tensor support for NPT.
- Support both GPU and CPU backends through SPONGE's existing device abstraction.
  GPU validation is the first test target, but CPU code must compile and remain
  behaviorally equivalent.

## Non-Goals For The First Pass

- Multi-process PME and distributed FFT.
- GPU-specific communication overlap.
- Replacing LJ-PME or dispersion PME.
- Changing neighbor-list, topology, or external force-loop ownership.
- Making ESP the default backend before force, energy, virial, and NPT checks
  pass against reference PME and direct-sum fixtures.

## References To Port From

- Liang et al., "Accelerating molecular dynamics simulations using fast Ewald
  summation with prolates", ESP main method:
  <https://arxiv.org/abs/2505.09727>
- Liang, Lu, Jiang, "Fast Ewald Summation with Prolates for Charged Systems in
  the NPT Ensemble", pressure tensor and cell derivatives:
  <https://arxiv.org/abs/2601.00161>
- Bostrom, Tornberg, af Klinteberg, "Fast Ewald Summation using Prolate
  Spheroidal Wave Functions", error estimates and parameter selection:
  <https://arxiv.org/abs/2602.16591>
- GROMACS-ESP implementation:
  <https://github.com/lu1and10/Ewald-Splitting-with-Prolates>
- LAMMPS-ESP implementation:
  <https://github.com/LiangJiuyang/Ewald-Splitting-with-Prolates>

## Current SPONGE Touch Points

- `SPONGE/PM_force/PM_force.h`
  - Add backend enum and ESP-owned buffers under `Particle_Mesh`.
  - Preserve existing public method signatures.
- `SPONGE/PM_force/PM_force.cpp`
  - Current B-spline PME initialization, spreading, influence multiplication,
    inverse FFT, gather, energy, self energy, and `Update_Box` all live here.
  - The file already notes PSWF as a future particle-mesh improvement.
- `SPONGE/Lennard_Jones_force/Lennard_Jones_force.h`
  - Current direct Coulomb uses `erfc(beta*r)/r`; ESP needs a PSWF split
    direct-space kernel.
- `SPONGE/Lennard_Jones_force/LJ_soft_core.h`
  - Soft-core Coulomb uses the same Gaussian split assumptions and must either
    stay on PME only initially or get an ESP-compatible path.
- `docs/input-reference/pme.md`
  - Document the new mdin keys after implementation.

## Proposed mdin Interface

Default behavior remains PME:

```toml
[PM]
backend = "pme"          # default: "pme"; allowed: "pme", "esp"
grid_spacing = 1.0
Direct_Tolerance = 1e-5
```

ESP example:

```toml
[PM]
backend = "esp"
Direct_Tolerance = 1e-5
esp_tolerance = 1e-5
esp_order = 8
esp_grid_spacing = 1.5
esp_parameter_mode = "auto"
esp_table_mode = "poly"
esp_table_points = 4096
esp_print_detail = true
```

Parameter meanings:

| Key | Scope | Type | Default | Meaning |
| --- | --- | --- | --- | --- |
| `backend` | `PM` | string | `"pme"` | Selects legacy PME or ESP. |
| `esp_tolerance` | `PM` | float | `Direct_Tolerance` | Target ESP electrostatic tolerance. |
| `esp_order` | `PM` | int | auto | Compact support width `P`; controls `P^3` grid points per atom. |
| `esp_grid_spacing` | `PM` | float | auto | Optional ESP-specific grid spacing. If absent, derive from parameter mode. |
| `esp_parameter_mode` | `PM` | string | `"auto"` | `"auto"` uses error model; `"manual"` trusts order and grid. |
| `esp_table_mode` | `PM` | string | `"poly"` | `"poly"` stores polynomial coefficients; `"table"` stores sampled tables. |
| `esp_table_points` | `PM` | int | `4096` | Lookup-table resolution for table mode and diagnostics. |
| `esp_print_detail` | `PM` | bool | `false` | Print ESP coefficients, grid, estimated errors, and table stats. |

Compatibility rule: existing `[PME]` keys such as `update_interval`,
`calculate_reciprocal_part`, and `calculate_excluded_part` keep their meaning.
`replaced_by_PMC_IZ` remains mutually exclusive with `backend = "esp"`.

## Architecture

### Backend Switch

Add a small backend enum:

```cpp
enum class ParticleMeshBackend
{
    PME,
    ESP
};
```

The public `Particle_Mesh` methods branch internally:

- `Initial`: parse backend, allocate backend-specific buffers, build tables and
  influence function.
- `PME_Reciprocal_Force_With_Energy_And_Virial`: call legacy PME or ESP
  reciprocal implementation.
- `PME_Excluded_Force_With_Atom_Energy`: call legacy exclusion correction or
  ESP exclusion correction.
- `Update_Box`: recompute volume-dependent coefficients for the active backend.

Do not rename the public methods in the first pass; keep the external call graph
unchanged.

### ESP Data Structure

Add an internal POD-style state, for example:

```cpp
struct ESP_Parameters
{
    int order;
    int table_points;
    float tolerance;
    float cutoff;
    float c_spread;
    float c_split;
    float c0_split;
    float psi0_split;
    float lambda_split;
    float self_energy_coeff;
    bool use_polynomial_tables;
};
```

Add buffers for:

- `ESP_atom_near`: `atom_numbers * order^3` grid indices.
- `ESP_window_coeff`: polynomial coefficients or table values for spreading.
- `ESP_window_derivative_coeff`: derivative coefficients or derivative table.
- `ESP_window_fourier`: Fourier-space window values for deconvolution.
- `ESP_split_fourier`: Fourier-space PSWF split values.
- `ESP_BC`: final influence function.
- `ESP_Virial_BC`: NPT and pressure tensor coefficients.
- `ESP_Q`, `ESP_FQ`, `ESP_FBCFQ`: can reuse PME grid buffers when dimensions
  match; keep separate names at first if it reduces risk.

Prefer headers ending in `.h`, consistent with SPONGE style.

## Implementation Phases

### Phase 1: Math And Table Generator

Deliverables:

- Add `SPONGE/PM_force/esp_pswf.h` and `SPONGE/PM_force/esp_pswf.cpp`.
- Port or reimplement the reference PSWF utilities needed at initialization:
  - tolerance to prolate parameter `c`;
  - zero-order PSWF evaluation;
  - integral of PSWF for real-space splitting;
  - Fourier-space splitting function;
  - spreading window and derivative;
  - Fourier transform of spreading window.
- Generate compact polynomial coefficients and optional lookup tables on CPU.
- Add a small deterministic unit/smoke executable or test helper that prints:
  - `c_spread`, `c_split`, `c0_split`, `psi0_split`, `lambda_split`;
  - max table interpolation error on `[0, 1]`;
  - self-energy coefficient.

Notes:

- Expensive PSWF evaluation belongs in initialization only.
- MD-step kernels must only evaluate low-order polynomials or lookup tables.
- The GROMACS-ESP implementation uses Chebyshev and monomial approximations.
  For SPONGE, start with monomial coefficients for device simplicity and keep a
  table mode for verification and future tuning.

### Phase 2: Backend Selection And Buffer Plumbing

Deliverables:

- Parse `[PM] backend = "esp"` and all ESP keys in `Particle_Mesh::Initial`.
- Reject incompatible combinations:
  - `backend = "esp"` with `replaced_by_PMC_IZ = true`;
  - `backend = "esp"` with `PM_MPI_size > 1` for now.
- Allocate `order^3` near-grid arrays and ESP table/coefficient buffers.
- Keep existing PME allocation untouched when `backend = "pme"`.
- Print backend, order, grid, tolerance, table mode, and estimated errors when
  `esp_print_detail = true`.

Acceptance:

- A legacy PME input gives byte-for-byte identical initialization output except
  for any intentionally added backend line.
- An ESP input initializes and exits cleanly before force kernels are enabled.

### Phase 3: PSWF Spreading And Gather

Deliverables:

- Add `ESP_Atom_Near` with dynamic support width `P`.
- Add `ESP_Q_Spread`.
- Add `ESP_Final` for force gather.
- Use separable PSWF windows:
  - spread weight: `W(x) W(y) W(z)`;
  - force gather derivative: `dW/dx`, `dW/dy`, `dW/dz`;
  - coordinate conversion follows the existing PME `rcell` path.

Implementation constraints:

- One source path must compile for CPU and GPU.
- The per-atom grid support is `P^3`, so block layout must not assume 64 points.
- Keep the old B-spline arrays and kernels for PME.

Acceptance:

- For a tiny charged system, spread plus gather runs without invalid memory
  access on GPU.
- CPU build compiles the same kernels through SPONGE's CPU backend macros.

### Phase 4: PSWF Influence Function

Deliverables:

- Build ESP Fourier-space influence coefficients:
  - PSWF split Fourier factor;
  - PSWF spreading-window deconvolution;
  - volume and reciprocal-cell scaling;
  - zero-frequency handling.
- Add `ESP_BCFQ`, or reuse `PME_BCFQ` if `ESP_BC` has the same layout.
- Add `ESP_Sum_Virial` with the NPT-compatible derivative coefficients.
- Keep `PME_BC` and `PME_Virial_BC` unchanged for legacy PME.

Acceptance:

- Reciprocal energy and forces are finite for neutral and non-neutral systems.
- Zero mode matches the chosen neutralizing-background convention.
- For fixed coordinates, ESP reciprocal force is stable under repeated calls.

### Phase 5: Direct-Space Split, Exclusions, And Self Energy

Deliverables:

- Add device functions for ESP direct Coulomb energy and force.
- Update normal LJ+Coulomb direct path to branch on particle-mesh backend.
- Add ESP exclusion correction, replacing the current `erf(beta*r)/r`
  correction with the PSWF long-range correction.
- Replace self energy in ESP mode with the PSWF self coefficient.
- Preserve current Gaussian PME direct path exactly for PME mode.

Important details:

- Direct-space kernel, exclusion correction, reciprocal kernel, and self energy
  must use the same PSWF split definition.
- Soft-core Coulomb should initially be guarded:
  - either implement the ESP-compatible soft-core derivatives in this phase;
  - or reject ESP with soft-core/FEP until the formula is implemented.

Acceptance:

- Direct plus reciprocal plus self energy agrees with direct Ewald/reference
  calculations within `esp_tolerance`.
- Excluded-pair energies do not regress for bonded water/protein fixtures.

### Phase 6: NPT And Pressure Tensor Compatibility

Deliverables:

- Implement ESP `Update_Box`.
- Recompute cell-dependent influence coefficients when box changes.
- Include:
  - volume scaling;
  - reciprocal-cell metric updates;
  - self-energy contribution;
  - zero-frequency contribution;
  - long-range virial and pressure tensor terms.
- Support isotropic and anisotropic box changes at the SPONGE API level already
  exposed through `LTMatrix3 cell`, `LTMatrix3 rcell`, and `LTMatrix3 g`.

Design note:

- The NPT ESP paper separates force evaluation from pressure-tensor evaluation.
  SPONGE's first implementation can keep the existing `PME_Sum_Virial` style:
  compute virial during reciprocal solve and update the active backend's
  `*_Virial_BC` in `Update_Box`.
- A later optimization can add the paper's cheaper pressure-only path, where the
  long-range pressure uses one forward FFT plus diagonal scaling when forces are
  not needed. This is an optimization, not a correctness prerequisite.

Acceptance:

- NPT water does not hit `NaN` pressure, energy, or virial over a short GPU run.
- For a fixed frame, virial from ESP matches finite-difference cell scaling
  within tolerance.

### Phase 7: Parameter Selection

Deliverables:

- Implement `esp_parameter_mode = "manual"` first:
  - user supplies `esp_order` and grid dimensions or spacing.
- Implement `esp_parameter_mode = "auto"` after correctness:
  - estimate Fourier truncation error;
  - estimate aliasing/spreading error;
  - choose `order` and grid dimensions for `esp_tolerance`;
  - prefer FFT-friendly grid sizes through existing `Get_Fft_Patameter`.

Acceptance:

- Auto mode chooses grids no finer than legacy PME for representative systems at
  `1e-3`, `1e-4`, and `1e-5` targets unless forced by box shape.
- Auto mode prints estimated error contributions when `esp_print_detail = true`.

### Phase 8: Verification And Benchmarks

GPU-first validation matrix:

| Fixture | Mode | Checks |
| --- | --- | --- |
| two charges in periodic box | NVE | force direction, energy, self, zero mode |
| small neutral ion/water box | NVT | force RMS vs PME/direct reference |
| water box | NPT | pressure, virial, box update stability |
| non-neutral toy system | NVT | neutralizing-background term |
| bonded molecule with exclusions | NVT | exclusion correction and total energy |

CPU compatibility checks:

- Build CPU target.
- Run at least one tiny ESP single-step force test.
- Compare CPU and GPU ESP force/energy on the same frame within floating-point
  tolerance.

Current smoke diagnostics:

- `scripts/esp_pswf_direct_smoke.sh` builds and runs a tiny host-side
  diagnostic for PSWF table generation, real-space split endpoints, table/poly
  consistency, ESP direct Coulomb closure, ESP exclusion correction closure, and
  self-energy coefficient finiteness. This is not a replacement for the
  single-step MD fixtures, but it guards the shared split definition used by
  direct-space, exclusions, and self energy before enabling full ESP runs.
- `scripts/esp_wat_single_step_smoke.sh` copies the existing
  `WAT_nonortho` fixture, writes a one-step `[PM] backend = "esp"` input, runs
  SPONGE, and rejects missing output or `NaN`/`Inf` energy text. This is the
  first runtime smoke for the single-GPU ESP path; it is not yet an accuracy
  check against PME or direct Ewald.

Performance checks after correctness:

- Compare legacy PME and ESP at equal force RMS tolerance.
- Record grid dimensions, FFT time, spread/gather time, reciprocal time, and
  total MD step time.
- Verify that smaller grid choices are responsible for speedup, not looser
  accuracy.

## Suggested Commit Sequence

1. Add ESP design scaffolding and mdin parsing, no active force change.
2. Add PSWF math generator and table diagnostics.
3. Add ESP spread/gather kernels behind inactive backend tests.
4. Add ESP influence function and reciprocal solve.
5. Add direct-space, exclusion, and self-energy consistency.
6. Add NPT virial and `Update_Box` support.
7. Add auto parameter selection.
8. Add docs, examples, and benchmark fixtures.

Each commit should keep `backend = "pme"` behavior unchanged.

## Completion Criteria For The First Milestone

The first milestone is complete when:

- `mdin.spg.toml` can select `[PM] backend = "esp"`.
- Single-GPU NVT and NPT runs complete without numerical instability.
- ESP forces, energies, self terms, exclusions, and virials pass reference
  checks.
- CPU target compiles and passes a tiny single-step ESP check.
- Existing PME inputs still run through the legacy path.
- Documentation clearly marks ESP as optional and experimental until benchmark
  coverage is complete.
