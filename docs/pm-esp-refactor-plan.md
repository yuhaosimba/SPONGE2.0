## PM / ESP refactor plan

### Goals

1. Keep `main.cpp` backend-agnostic.
2. Keep the external particle-mesh call shape stable.
3. Split `PM_force.cpp` by responsibility without introducing a second public PM stack.
4. Preserve current CPU/GPU and PME/ESP behavior.

### Target structure

- `SPONGE/main.cpp`
  - Only calls PM-generic interfaces.
  - Must not reference `ESP_*` names directly.
- `SPONGE/PM_force/PM_force.h`
  - Public `Particle_Mesh` state and PM-generic interfaces.
  - PM-direct parameter type exposed as PM-generic naming.
- `SPONGE/PM_force/PM_force.cpp`
  - Core initialization, cleanup, shared helpers, and public dispatch.
- `SPONGE/PM_force/PM_force_esp.cpp`
  - ESP reciprocal backend kernels and ESP backend helpers.
- `SPONGE/PM_force/esp_pswf.cpp`
  - PSWF math, table generation, and default parameter selection.

### Execution phases

#### Phase 1

- Replace `main.cpp` ESP-specific direct-parameter access with PM-generic access.
- Move ESP-only direct-path validation behind `Particle_Mesh`.
- Move ESP-only final-force sanitize behind `Particle_Mesh`.

#### Phase 2

- Extract ESP reciprocal backend implementation from `PM_force.cpp`.
- Keep PME and shared code in `PM_force.cpp` for this step.
- Update CMake runtime source list.

#### Phase 3

- Revisit whether PME code should be split into a dedicated file.
- Only do this if the remaining `PM_force.cpp` still mixes too many concerns.

### Acceptance

- `main.cpp` does not mention `ESP_Direct_Parameters`, `Get_ESP_Direct_Parameters`, or `ESP_Sanitize_Final_Force`.
- ESP remains selectable through PM input parameters only.
- CPU and CUDA builds succeed.
- Existing ESP smoke tests still pass.
