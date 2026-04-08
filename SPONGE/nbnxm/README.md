# SPONGE NBNXM Frozen Kernel Package

This directory keeps only three components:

- ABI-aligned pairlist/layout data structures (`sci/cjPacked/excl`)
- frozen CUDA LJ_EWALD kernel port (`frozen_port/`)
- golden fixtures (`fixtures/`)

`nbnxm_frozen_bench` runs the frozen kernel using fixture-provided pairlists only.
