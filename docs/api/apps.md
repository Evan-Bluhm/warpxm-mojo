# `warpxm.apps`

Physics modules. Each app struct carries the physics parameters (γ,
speed of light, …) and exposes static flux methods that the DG kernels
call.

## Convention

Every app has, at minimum:

* `internal_flux_*_static(q, ...)` — the physical flux along axis `*`,
  computed at a single state.
* `numerical_flux_static(q_l, q_r, n_components, ..., gamma, ...)` —
  Rusanov / Lax-Friedrichs numerical flux at an interface with a
  given unit normal.

MHD modules additionally have:

* `numerical_flux_hlld_static(...)` — the HLLD five-state solver.

All these methods are `@staticmethod` so they're callable without an
instance — DG kernels invoke them as `IMHD2D.numerical_flux_static(...)`.

## Index constants and NC

Each app file exports `*_NC` (the component count) and an `IDX_*`
constant per component. The 3D modules **re-use** the 2D constants —
the state layout is identical, just on a 3D mesh.

## Module list

### `advection`

```mojo
from warpxm.apps.advection import Advection1D, Advection2D, Advection3D
```

Parameters: `velocity`, `cfl_max`. NC is open (multi-component
advection works). The "numerical flux" is exact upwinding, not Rusanov.

### `euler_{1d,2d,3d}`

```mojo
from warpxm.apps.euler_2d import Euler2D, EULER_NC, IDX_RHO, IDX_MX, IDX_MY, IDX_MZ, IDX_E
```

5-component compressible Euler. Parameters: `gas_gamma`, `min_density`,
`min_pressure`, `cfl_max`.

### `maxwell_{2d,3d}`

```mojo
from warpxm.apps.maxwell_2d import (
    Maxwell2D, MAXWELL_NC,
    IDX_EX, IDX_EY, IDX_EZ, IDX_BX, IDX_BY, IDX_BZ,
)
```

8-component (six fields + two reserved). Parameters: `c_light`,
`cfl_max`. No 1D Maxwell (the lowest-rank plane wave is 2D).

### `imhd_{2d,3d}`

```mojo
from warpxm.apps.imhd_2d import (
    IMHD2D, MHD_NC,
    IDX_RHO, IDX_MX, IDX_MY, IDX_MZ, IDX_E, IDX_BX, IDX_BY, IDX_BZ,
)
```

8-component ideal MHD. Parameters: `gas_gamma`, `min_density`,
`min_pressure`, `cfl_max`. Has both `numerical_flux_static` (Rusanov)
and `numerical_flux_hlld_static` (HLLD); selection via the DG solver's
`USE_HLLD` comptime parameter.

### `imhd_glm_{2d,3d}`

```mojo
from warpxm.apps.imhd_glm_2d import ImhdGLM2D, MHD_GLM_NC, IDX_PSI
```

9-component MHD with Dedner GLM cleaning. Adds `c_h`, `c_p_sq`
parameters and the `IDX_PSI` component at index 8.

## Switching MHD flux

```mojo
# Default (Rusanov):
var spatial = DGImhdSolver2D[P](app)

# HLLD:
var spatial = DGImhdSolver2D[P, USE_HLLD=True](app)
```

Same for 3D and for `DGImhdGLMSolver*` once the HLLD wiring is
extended (currently only on plain `DGImhdSolver*`).
