# PME Electrostatics Parameters

PME is initialized through the `PM` module in the current source tree, so new
TOML examples should prefer the `[PM]` scope:

```toml
[PM]
backend = "pme"
grid_spacing = 1.0
Direct_Tolerance = 1e-5
MPI_size = 1
print_detail = false
```

Some PME control keys are still read from the compatibility scope `[PME]`. This
is a source-level behavior, not a separate algorithm.

## `[PM]` Parameters

| Parameter | Scope | Type | Default | Description |
|-----------|-------|------|---------|-------------|
| `backend` | `PM` | string | `pme` | Particle-mesh backend. Allowed values are `pme` and experimental `esp` |
| `fftx` | `PM` | int | auto | FFT grid size in X |
| `ffty` | `PM` | int | auto | FFT grid size in Y |
| `fftz` | `PM` | int | auto | FFT grid size in Z |
| `grid_spacing` | `PM` | float | `1.0` | Grid spacing in angstrom |
| `Direct_Tolerance` | `PM` | float | `1e-5` | Direct-space Ewald tolerance |
| `MPI_size` | `PM` | int | controller value | PME process count |
| `print_detail` | `PM` | bool | `false` | Print detailed PME energy breakdown |

If `fftx`, `ffty`, and `fftz` are omitted, SPONGE derives them from
`grid_spacing` and the current box dimensions.

`MPI_size > 1` is currently rejected by the source code even if the key is
present.

## Experimental ESP Backend Keys

ESP (Ewald summation with prolates) is being added as an optional backend. The
current implementation is experimental and should be validated against PME or
direct-reference fixtures before production use:

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

| Parameter | Scope | Type | Default | Description |
|-----------|-------|------|---------|-------------|
| `esp_tolerance` | `PM` | float | `Direct_Tolerance` | Target ESP electrostatic tolerance |
| `esp_order` | `PM` | int | auto | PSWF compact support width `P` |
| `esp_grid_spacing` | `PM` | float | auto | Optional ESP-specific grid spacing in angstrom |
| `esp_parameter_mode` | `PM` | string | `auto` | `auto` or `manual` parameter selection |
| `esp_table_mode` | `PM` | string | `poly` | `poly` coefficient mode or sampled `table` mode |
| `esp_table_points` | `PM` | int | `4096` | Table resolution for ESP lookup-table mode and diagnostics |
| `esp_print_detail` | `PM` | bool | `false` | Print ESP parameters and diagnostics during initialization |

When `backend = "esp"` and `esp_grid_spacing` is provided, it overrides
`grid_spacing` for the ESP Fourier grid. If it is omitted, the ordinary
`grid_spacing` value is used.

## Compatibility Keys In `[PME]`

The following keys are read from the `[PME]` scope in the current source:

| Parameter | Scope | Type | Default | Description |
|-----------|-------|------|---------|-------------|
| `update_interval` | `PME` | int | `1` | Reciprocal-space update interval |
| `calculate_reciprocal_part` | `PME` | bool | `true` | Whether to compute reciprocal-space PME |
| `calculate_excluded_part` | `PME` | bool | `true` | Whether to compute excluded-pair PME terms |
| `replaced_by_PMC_IZ` | `PME` | bool | `false` | Replace PME reciprocal evaluation with PMC-IZ |

`replaced_by_PMC_IZ = true` cannot be used in `npt` mode.
