# `warpxm.apps`

Physics modules. Each physics is a **single file** that defines an app
struct (parameters: γ, speed of light, …) and exposes static flux
methods that the DG kernels call. The DG launcher owns dimensionality;
the app operates on local node values via 3-component-normal numerical
fluxes and axis-named (or axis-parametric) internal fluxes.

## Convention

Every app has, at minimum:

* `internal_flux_x_static(q, ...)`, `internal_flux_y_static`,
  `internal_flux_z_static` — the physical flux along that axis at a
  single state. Some apps also expose `internal_flux_axis_static[axis]`.
* `numerical_flux_static(q_l, q_r, nx, ny, nz, ...physics-params...)` —
  Rusanov / central+LF numerical flux on a 3-component unit normal.
  1D callers pass `(±1, 0, 0)`; 2D callers pass `(nx, ny, 0)`.

MHD modules additionally have:

* `numerical_flux_hlld_static(q_l, q_r, nx, ny, nz, ...)` — Miyoshi-
  Kusano HLLD via axis-aligned permutation into a shared
  `_hlld_core_3d`.

All these methods are `@staticmethod` so they're callable without an
instance — DG kernels invoke them as `Euler.numerical_flux_static(...)`.

## Index constants and NC

Fixed-NC apps (`euler`, `maxwell`, `imhd`, `imhd_glm`) each export a
`*_NC` component count and an `IDX_*` constant per component. The
`advection` app is the exception: `Advection[NC: Int]` is NC-parametric
(used at NC=1 for single-component advection, NC>1 for multi-component
tests), so it carries no module-level `ADVECTION_NC` or `IDX_*`
symbols.

## Module list

### `advection`

```mojo
from warpxm.apps.advection import Advection
```

`Advection[NC: Int]` — parameters: `velocity_x`, `velocity_y`,
`velocity_z` (defaults 0), `cfl_max`. NC is open (multi-component
advection works). The "numerical flux" is exact upwinding, not
Rusanov.

### `euler`

```mojo
from warpxm.apps.euler import (
    Euler, EULER_NC, IDX_RHO, IDX_MX, IDX_MY, IDX_MZ, IDX_E,
)
```

5-component compressible Euler. Parameters: `gas_gamma`,
`min_density`, `min_pressure`, `cfl_max`.

### `maxwell`

```mojo
from warpxm.apps.maxwell import (
    Maxwell, MAXWELL_NC,
    IDX_EX, IDX_EY, IDX_EZ, IDX_BX, IDX_BY, IDX_BZ,
)
```

6-component electromagnetic. Parameters: `c_light`, `cfl_max`.

### `imhd`

```mojo
from warpxm.apps.imhd import (
    IMHD, MHD_NC,
    IDX_RHO, IDX_MX, IDX_MY, IDX_MZ, IDX_E, IDX_BX, IDX_BY, IDX_BZ,
)
```

8-component ideal MHD. Parameters: `gas_gamma`, `min_density`,
`min_pressure`, `cfl_max`. Has both `numerical_flux_static` (Rusanov)
and `numerical_flux_hlld_static` (HLLD); selection via the DG
solver's `USE_HLLD` comptime parameter.

### `imhd_glm`

```mojo
from warpxm.apps.imhd_glm import ImhdGLM, MHD_GLM_NC, IDX_PSI
```

9-component MHD with Dedner GLM cleaning. Adds `c_h`, `c_p_sq`
parameters and the `IDX_PSI` component at index 8. HLLD path composes
IMHD HLLD with the exact 2×2 (B·n, ψ) Riemann block.

## Switching MHD flux

```mojo
# Default (Rusanov):
var spatial = DGImhdSolver2D[P](app)

# HLLD:
var spatial = DGImhdSolver2D[P, USE_HLLD=True](app)
```

Same for 3D and for `DGImhdGLMSolver*` (HLLD is wired into GLM-MHD).
