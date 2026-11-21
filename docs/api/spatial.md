# `warpxm.spatial`

DG spatial solvers and source-term kernels. One spatial solver per
(physics, dimension) combination — these are the host-side launchers
that issue the GPU kernels.

Each per-dim solver is a thin wrapper over a shared NC-parametric DG
core (`dg_core_{1,2,3}d.mojo`). The wrapper declares four
closures (nested `def`s with an `{imm}` capture list) — `flux_x`, `flux_y` (+ `flux_z` in 3D),
`num_flux`, `apply_bc` — that dispatch to the app's static methods
and calls the core with them. This is the "flux launcher owns
dimensionality, app operates on local node values" pattern from
upstream WARPXM.

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

### Advection launchers (1D / 2D / 3D)

```mojo
from warpxm.spatial.dg_1d import DGSpatialSolver1D
from warpxm.spatial.dg_2d import DGSpatialSolver2D
from warpxm.spatial.dg_3d import DGSpatialSolver3D

var spatial = DGSpatialSolver1D[P, NC](app)   # app: Advection[NC]
```

NC-parametric (multi-component advection works), advection-only —
each wraps an `Advection[NC]` and dispatches to the matching
advection RHS kernel.

### Per-physics 1D / 2D / 3D launchers

```mojo
from warpxm.spatial.dg_euler_1d import DGEulerSolver1D
from warpxm.spatial.dg_euler_2d import DGEulerSolver2D
from warpxm.spatial.dg_euler_3d import DGEulerSolver3D

from warpxm.spatial.dg_maxwell_1d import DGMaxwellSolver1D
from warpxm.spatial.dg_maxwell_2d import DGMaxwellSolver2D
from warpxm.spatial.dg_maxwell_3d import DGMaxwellSolver3D

from warpxm.spatial.dg_imhd_2d import DGImhdSolver2D   # no 1D MHD
from warpxm.spatial.dg_imhd_3d import DGImhdSolver3D

from warpxm.spatial.dg_imhd_glm_2d import DGImhdGLMSolver2D
from warpxm.spatial.dg_imhd_glm_3d import DGImhdGLMSolver3D
```

Each is parameterised on the polynomial order `P` and takes the
matching app struct as an argument:

```mojo
var app = IMHD(gas_gamma=Real(5.0/3.0), ...)
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
apply_gravity_source_2d[P](ctx, q, rhs, gx, gy)
# adds (-ρ gx, -ρ gy) to (ρu, ρv) RHS, ρ(u·gx + v·gy) to E RHS
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
from warpxm.spatial.imhd_glm_source_2d import apply_glm_damping_2d
apply_glm_damping_2d[P](ctx, q, rhs, c_h, c_p_sq)
```

Adds $-(c_h^{2}/c_p^{2})\psi$ to the $\psi$ RHS. The kernel squares
`c_h` internally — pass the hyperbolic cleaning speed itself, not its
square.

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
