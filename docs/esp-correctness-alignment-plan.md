# ESP Correctness Alignment Plan

This document defines how to finish correctness validation for the SPONGE ESP
backend by aligning it against legacy PME in force and electrostatic energy.
The immediate scope is single-rank, single-device correctness first, with GPU
as the primary target and CPU parity checked before closeout. Multi-process PME
remains out of scope.

## Objective

Make ESP match PME to the requested electrostatic tolerance under the existing
SPONGE external interface, then prove that match with reproducible tests.

The plan treats force as the primary physical observable and electrostatic
energy as the main decomposition and convention check.

## Definition Of Done

ESP correctness is considered complete only when all of the following are true:

- ESP and PME both pass the same static and short-run validation fixtures.
- ESP force accuracy is within the requested tolerance against a higher-quality
  reference, not only against PME.
- ESP and PME agree on raw electrostatic energy decomposition
  (`PM_direct`, `PM_reciprocal`, `PM_self`, `PM_correction`) to within the
  expected tolerance budget, with no unexplained constant offset left behind.
- NPT runs remain stable and pressure/box updates do not regress while the
  energy and force alignment work is applied.
- CPU and GPU runs produce equivalent conclusions for the same fixtures.

## Comparison Hierarchy

We should not rely on a single comparison mode.

### Tier 1: Analytic Or Host-Reference Checks

Use host-side PSWF reference evaluation to validate:

- direct split scalar
- direct split derivative
- excluded correction scalar
- self-energy coefficient
- Fourier split scalar and derivative

This tier catches normalization and sign mistakes before any MD run.

### Tier 2: Tiny Neutral Static Fixtures

Use very small neutral systems to compare:

- ESP vs PME
- ESP vs an overresolved reference
- PME vs the same overresolved reference

This tier is where we isolate component errors with minimal noise.

### Tier 3: Realistic SPONGE Fixtures

Use existing SPONGE-ready systems that exercise:

- triclinic cell handling
- virtual atoms
- exclusions
- barostat path

This tier proves that the aligned implementation survives real code paths.

## Metrics

### Force Metrics

For every comparison, compute:

```text
abs_max_force_diff = max_i ||F_esp(i) - F_ref(i)||_inf
rms_force_diff     = sqrt(sum_i ||F_esp(i) - F_ref(i)||^2 / (3N))
rel_rms_force_err  = ||F_esp - F_ref||_2 / ||F_ref||_2
rel_rms_pair_err   = ||F_esp - F_pme||_2 / ||F_ref||_2
```

`F_ref` should be the highest-quality available reference, not just PME.

### Energy Metrics

For every frame, record:

- `PM`
- `PM_direct`
- `PM_reciprocal`
- `PM_self`
- `PM_correction`

For energy alignment, compute:

```text
abs_energy_diff(component) = |E_esp(component) - E_ref(component)|
rel_energy_diff(component) = |E_esp(component) - E_ref(component)| /
                             max(1, |E_ref(component)|)
```

Also compare frame-to-frame energy differences:

```text
deltaE_backend(a,b) = E_backend(frame_b) - E_backend(frame_a)
deltaE_diff         = |deltaE_esp(a,b) - deltaE_ref(a,b)|
```

The `deltaE` comparison is important because it catches configuration-dependent
energy errors even when a constant offset is present.

## Acceptance Gates

Let `eps` denote the requested electrostatic target, usually
`esp_tolerance` or `Direct_Tolerance`.

### Gate A: Table And Scalar Sanity

- host PSWF scalar and derivative evaluations agree with their tabulated or
  polynomial forms within the expected interpolation error budget
- no sign mismatch in split, derivative, or excluded correction functions
- self-energy coefficient matches the chosen reference implementation and
  derivation

### Gate B: Tiny-Case Force Correctness

Against the overresolved reference:

- `rel_rms_force_err(ESP) <= 2 * eps`
- `rel_rms_force_err(PME) <= 2 * eps`
- `rel_rms_pair_err(ESP, PME) <= 3 * eps`

If PME itself misses the target on a case, that case is not acceptable as the
final reference until the reference setup is tightened.

### Gate C: Energy Correctness

- `deltaE_esp` must agree with `deltaE_ref` within the same tolerance order as
  the force target
- raw `PM` and its four printed components must not retain any unexplained
  configuration-independent or configuration-dependent bias
- a constant offset is not considered "resolved" until it is either removed in
  code or derived and documented as an intentional convention that still keeps
  SPONGE's external PME semantics unchanged

### Gate D: Realistic Fixture And NPT Stability

- short NVE and NPT runs remain finite and stable
- no regression in box updates or pressure reporting
- ESP vs PME force and energy conclusions remain the same on GPU and CPU

## Validation Matrix

### Case 1: PSWF Math Smoke

Use the existing direct PSWF smoke as the starting point:

- [scripts/esp_pswf_direct_smoke.sh](/mnt/data8t/Software/SPONGE/SPONGE/scripts/esp_pswf_direct_smoke.sh)

Extend it, if needed, to print and compare:

- split real value and derivative at representative radii
- excluded correction value and derivative
- self-energy coefficient

Purpose:

- validate scalar math before involving MD kernels

### Case 2: Tiny Static Orthorhombic Fixture

Create a dedicated neutral electrostatic fixture with:

- no barostat
- no thermostat
- `update_interval = 1`
- `print_detail = true`
- no virtual atoms in the first pass

Purpose:

- isolate `PM_direct`, `PM_reciprocal`, `PM_self`, and `PM_correction`
- make PME and ESP component-by-component comparison easy

### Case 3: Tiny Static Triclinic Fixture

Create a second dedicated neutral fixture with:

- triclinic cell
- same charges and similar size to Case 2

Purpose:

- validate `rcell` handling, reciprocal metric terms, and triclinic force gather

### Case 4: Existing `WAT_nonortho` Single-Step Fixture

Use:

- [scripts/esp_wat_single_step_smoke.sh](/mnt/data8t/Software/SPONGE/SPONGE/scripts/esp_wat_single_step_smoke.sh)

Add a PME companion and comparison harness so the same initial structure can be
run under both backends.

Purpose:

- exercise virtual atoms and real SPONGE topology plumbing

### Case 5: Existing `WAT_nonortho` NPT Fixture

Use:

- [scripts/esp_wat_npt_smoke.sh](/mnt/data8t/Software/SPONGE/SPONGE/scripts/esp_wat_npt_smoke.sh)

Add a PME companion comparison mode and collect:

- `PM*` components
- pressure
- final box frames

Purpose:

- keep NPT compatibility under active validation while energy alignment is fixed

## Implementation Plan

### Phase 1: Build A Reusable Comparison Harness

Add a small comparison tool, preferably Python-based, for example:

- `scripts/esp_compare_metrics.py`

Inputs:

- two case directories or two `mdout.txt` plus two `frc.dat` files
- atom count
- optional reference label

Outputs:

- parsed `PM*` components
- force norms and relative errors
- a small text or JSON summary suitable for CI-like checks

This tool should use the existing extraction helpers in
[benchmarks/utils.py](/mnt/data8t/Software/SPONGE/SPONGE/benchmarks/utils.py)
where possible.

### Phase 2: Freeze A Fair Comparison Configuration

Before debugging correctness, lock the comparison knobs:

- identical coordinates
- identical box
- identical cutoff
- identical `update_interval`
- identical exclusion settings
- identical `print_detail = true`
- no thermostat or barostat for the static alignment phase

For static force/energy alignment, prefer `mode = "nve"` and `step_limit = 0`
or `1`.

### Phase 3: Align Component Energies In This Order

Debug in the following order:

1. `PM_self`
2. `PM_correction`
3. `PM_direct`
4. `PM_reciprocal`
5. total `PM`
6. total force

Reason:

- `PM_self` is a scalar convention term and currently appears to dominate the
  mismatch
- `PM_correction` is usually easier to isolate than the full reciprocal path
- `PM_direct` and `PM_reciprocal` then become cleaner to compare

For each stage, compare against both PME and the overresolved reference before
moving on.

### Phase 4: Align Forces After Energy Components Stop Drifting

Once the scalar energy decomposition is stable:

- compare forces on tiny orthorhombic and triclinic fixtures
- compare against the overresolved reference first
- then ensure ESP and PME differ only within the expected tolerance budget

If force mismatches remain while energies look aligned, inspect:

- PSWF gather derivative
- reciprocal-space scaling by `rcell`
- direct split derivative normalization
- excluded-force derivative sign

### Phase 5: Validate Realistic SPONGE Fixtures

Run the same comparison harness on:

- `WAT_nonortho` single-step
- `WAT_nonortho` short NPT

At this phase, we are checking that the tiny-case fix survives:

- virtual atoms
- exclusions
- triclinic box updates
- NPT virial path

### Phase 6: CPU Parity

Repeat the final passing GPU cases on CPU.

CPU is not the primary tuning target, but it is part of the correctness
contract for this work. The CPU result does not need bitwise identity with GPU;
it must support the same correctness conclusion.

## Planned Deliverables

- `scripts/esp_compare_metrics.py`
- a PME companion mode for the ESP smoke fixtures, or one wrapper that can run
  both backends on the same input
- at least one tiny orthorhombic validation fixture
- at least one tiny triclinic validation fixture
- one test or script that reports force and energy alignment in a machine- and
  human-readable form
- updated closeout notes in the migration plan once acceptance gates pass

## Round Structure

This work should be executed in rounds, with the migration plan and this
alignment plan re-read at the end of each round.

Recommended round order:

1. build comparison harness
2. tiny orthorhombic energy decomposition alignment
3. tiny triclinic force alignment
4. realistic `WAT_nonortho` single-step alignment
5. NPT validation
6. CPU parity and final acceptance

## Practical Stop Rules

Do not declare correctness complete if any of the following remain true:

- ESP only matches PME after manually subtracting an unexplained constant energy
  offset
- force agreement is good on one realistic case but not on tiny gold-reference
  fixtures
- GPU passes but CPU changes the correctness conclusion
- NPT stability is recovered only by loosening tolerances or skipping virial
  reporting

If the plan is followed, the final result should be a defensible statement that
ESP is correct to the requested tolerance, rather than merely stable enough to
run.
