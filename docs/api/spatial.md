# `warpxm.spatial`

DG spatial solvers and source-term kernels. One spatial solver per
(physics, dimension) combination — these are the host-side launchers
that issue the GPU kernels.

## DG spatial solvers

Every spatial solver exposes:

```mojo
def compute_rhs(
    self,
    var ctx: DeviceContext,
    mut mesh: BlockMeshNd[P],
    mut q: DGStateNd[P, NC],
    mut rhs: DGStateNd[P, NC],
) raises:
```

with the per-physics overload of the kernel doing the volume + surface
work.

### Generic 1D

```mojo
from warpxm.spatial.dg_1d import DGSpatialSolver1D
var spatial = DGSpatialSolver1D[P, NC](app)
```

Generic in `NC` and in the app type — works for advection (NC=1 or
multi-component) and 1D Euler (NC=5). Used by the generic
`SSPRK3` stepper and the 1D DP45.

### Per-physics 2D / 3D

```mojo
from warpxm.spatial.dg_euler_2d import DGEulerSolver2D
from warpxm.spatial.dg_euler_3d import DGEulerSolver3D

from warpxm.spatial.dg_maxwell_2d import DGMaxwellSolver2D
from warpxm.spatial.dg_maxwell_3d import DGMaxwellSolver3D

from warpxm.spatial.dg_imhd_2d import DGImhdSolver2D
from warpxm.spatial.dg_imhd_3d import DGImhdSolver3D

from warpxm.spatial.dg_imhd_glm_2d import DGImhdGLMSolver2D
from warpxm.spatial.dg_imhd_glm_3d import DGImhdGLMSolver3D
```

Each is parameterised on the polynomial order `P` and takes the
matching app struct as an argument:

```mojo
var app = IMHD2D(gas_gamma=Real(5.0/3.0), ...)
var spatial = DGImhdSolver2D[P](app)
```

### MHD: USE_HLLD flag

```mojo
var spatial = DGImhdSolver2D[P, USE_HLLD=True](app)
var spatial = DGImhdSolver3D[P, USE_HLLD=True](app)
```

Both default to `USE_HLLD=False` (Rusanov). Setting it to `True`
switches the face flux to HLLD — a comptime-only change with zero
runtime cost. See [HLLD](../numerical-methods/hlld.md).

## Source-term kernels

Source terms aren't bundled into the DG solver — they're free functions
applied between or alongside the RK stages.

### Gravity (Euler)

```mojo
from warpxm.spatial.euler_2d_sources import apply_gravity_source_2d
apply_gravity_source_2d[P](ctx, q, rhs, g)   # adds -ρ g to ρv RHS, ρv·g to E RHS
```

### Lorentz force (charged fluid)

```mojo
from warpxm.spatial.lorentz_source_2d import apply_lorentz_source_2d
apply_lorentz_source_2d[P](ctx, q_fluid, q_maxwell, rhs_fluid, alpha)
```

Adds $\alpha\rho(\mathbf{E} + \mathbf{u}\times\mathbf{B})$ to the fluid
momentum RHS and $\alpha\rho\mathbf{u}\cdot\mathbf{E}$ to the energy RHS.

### Current source (charged fluid → Maxwell)

```mojo
from warpxm.spatial.maxwell_current_source_2d import apply_current_source_2d
apply_current_source_2d[P](ctx, q_fluid, rhs_maxwell, alpha, eps0)
```

Adds $-\alpha\rho\mathbf{u}/\varepsilon_0$ to the Maxwell E-field RHS.

### GLM damping (MHD)

```mojo
from warpxm.spatial.imhd_glm_2d import apply_glm_damping_2d
apply_glm_damping_2d[P](ctx, q, rhs, c_h_sq, c_p_sq)
```

Adds $-(c_h^{2}/c_p^{2})\psi$ to the $\psi$ RHS.

## All sources are NC-parameterised where it matters

The Lorentz / current sources are dimension-specialised because they
need the right number of velocity / B components, but they don't care
which physics module the fluid state came from. The plasma stepper
calls them once per species.

## See also

* [Time integration](../concepts/time-integration.md) — where sources
  get applied within the RK structure.
* [GPU architecture](../concepts/gpu-architecture.md) — how the DG
  kernels are launched.
