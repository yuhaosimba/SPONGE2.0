# SPONGE NBNXM Frozen Kernel Package

This directory keeps only three components:

- ABI-aligned pairlist/layout data structures (`sci/cjPacked/excl`)
- frozen CUDA LJ_EWALD kernel port (`frozen_port/`)
- golden fixtures (`fixtures/`)
- staged dump I/O / comparators for incremental GROMACS parity checks
- orthorhombic/no-prune builder contract used to migrate `params -> grid -> pairlist` in steps

`nbnxm_frozen_bench` can run either fixture-provided pairlists or SPONGE live-built pairlists.
`nbnxm_stage_compare` compares staged `params/grid/pairlist` dumps bytewise/fieldwise.
