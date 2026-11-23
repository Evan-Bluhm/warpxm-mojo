# `warpxm.spatial`

DG spatial solvers, the shared kernel launcher, CFL time-step selection,
and source-term kernels.

```text
spatial/
  dg_core.mojo               shared DG core (volume + surface terms), generic over NDIM
  dg_launch.mojo             host-side launcher shared by every wrapper
  dg_<phys>.mojo             per-physics wrappers: flux closures + BC function
  wave_speed.mojo            CFL: max signal speed reduction, cfl_dt
  sources/                   pointwise source kernels, generic over NDIM
  kinetic/                   Vlasov phase-space DG, moments, BGK, coupling
```

## DG spatial solvers

One solver struct per physics, generic over the dimension. Each holds
its app and exposes:

```mojo
def compute_rhs(
    self,
    var ctx: DeviceContext,
    mesh: BlockMesh[NDIM, P],
    mut q: DGState[NDIM, P, NC],
    mut rhs: DGState[NDIM, P, NC],
) raises

def suggest_dt(...) raises -> Real      # see "CFL time step" below
```

```mojo
from warpxm.spatial.dg_advection import DGAdvectionSolver   # [NDIM, P, NC]
from warpxm.spatial.dg_euler import DGEulerSolver           # [NDIM, P]
from warpxm.spatial.dg_maxwell import DGMaxwellSolver       # [NDIM, P]
from warpxm.spatial.dg_imhd import DGImhdSolver             # [NDIM, P, USE_HLLD=False]
from warpxm.spatial.dg_imhd_glm import DGImhdGLMSolver      # [NDIM, P, USE_HLLD=False]

var spatial = DGEulerSolver[2, P](app)              # app: Euler
var adv = DGAdvectionSolver[1, P, NC](advection)    # app: Advection[NC]
```

The advection solver is NC-parametric (every component advects with
the same velocity); the others have a fixed component count. The MHD
solvers are exercised in 2D and 3D only. The
[`physics`](prelude.md) modules re-export the solvers per physics.

### MHD: `USE_HLLD`

```mojo
var spatial = DGImhdSolver[2, P, USE_HLLD=True](app)
var spatial_glm = DGImhdGLMSolver[3, P, USE_HLLD=True](app)
```

Defaults to `False` (Rusanov). `True` switches the face flux to HLLD at
compile time. See [HLLD](../numerical-methods/hlld.md).

## How a wrapper is built

Each `dg_<phys>.mojo` contains three things: a BC function, a kernel,
and the solver struct. All three are written once for every dimension.

The kernel has the fixed signature the launcher expects, unpacks the
physics scalars from `params`, declares the flux closures, and calls the
shared core:

```mojo
def _euler_rhs_kernel[NDIM: Int, P: Int](
    q: RealTensor,
    rhs: RealTensor,
    face_neg_elem: IndexTensor,
    face_pos_elem: IndexTensor,
    elem_face_dir: IndexTensor,
    face_bc: IndexTensor,
    inv_jacobian: InvJacobian,
    num_elements_i64: Int64,
    params: KernelParams,
):
    var gamma = params[0]
    var rho_min = params[1]
    var press_min = params[2]

    def flux_x(qn: Array[Real, EULER_NC]) {imm} -> Array[Real, EULER_NC]:
        return Euler.internal_flux_x_static(qn, gamma, rho_min, press_min)
    # flux_y, flux_z, num_flux, apply_bc likewise

    _dg_compute_rhs_core[
        NDIM, P, EULER_NC, FlatLayout, FlatLayout, FlatLayout, FlatLayout,
        DG_CORE_ELEMS_PER_BLOCK,
    ](
        q, rhs, face_neg_elem, face_pos_elem, elem_face_dir, face_bc,
        inv_jacobian, Int(num_elements_i64),
        flux_x, flux_y, flux_z, num_flux, apply_bc,
    )
```

The core always takes five closures: `flux_x`, `flux_y`, `flux_z`,
`num_flux(q_l, q_r, nx, ny, nz)` and `apply_bc(q, bc_kind, face_axis)`.
It calls only the first `NDIM` flux closures, so the others are dead
code in a lower-dimensional instantiation.

`compute_rhs` is then one call:

```mojo
launch_dg[_euler_rhs_kernel[Self.NDIM, Self.P]](
    ctx, mesh, q, rhs,
    kernel_params(self.app.gas_gamma, self.app.min_density, self.app.min_pressure),
)
```

### `dg_core`

`_dg_compute_rhs_core[NDIM, P, NC, ...]` evaluates the volume term by
tensor-product GLL differentiation and the surface term through the
numerical-flux closure. Every loop over axes is a `comptime for`, so each
(physics, `NDIM`, `P`) instantiation compiles to straight-line code.

`DG_CORE_ELEMS_PER_BLOCK` (4) elements share one block. A mesh whose
element count is not a multiple of it leaves a partly filled last block;
the threads past the last element still reach the kernel's barrier at
the same call site as the others and stop afterwards
(`tests/test_advection_odd_mesh.mojo`).

In 3D the volume and surface terms accumulate into one array; in 1D and
2D each axis keeps its own derivative array and the terms are combined
at the end. The two orders differ at round-off.

### `dg_launch`

```mojo
from warpxm.spatial.dg_launch import (
    FlatLayout, RealTensor, IndexTensor, KernelParams, kernel_params,
    launch_dg,
)
from warpxm.spatial.dg_core import InvJacobian
```

| Symbol | Meaning |
| ------ | ------- |
| `FlatLayout` | layout type of every flat, runtime-sized device buffer |
| `RealTensor`, `IndexTensor` | `TileTensor` over `WXM_DTYPE` / `INDEX_DTYPE` with `FlatLayout` |
| `KernelParams` | `SIMD[WXM_DTYPE, 8]`: up to eight physics scalars, by position |
| `kernel_params(a, b, ...)` | packs `Real`s into a `KernelParams` |
| `InvJacobian` | `SIMD[WXM_DTYPE, 4]`: `1 / jacobian` per axis |
| `launch_dg[kernel](ctx, mesh, q, rhs, params)` | wraps the buffers, picks the grid, enqueues `kernel` |

Every DG kernel has the same signature, in every dimension:

```text
(q, rhs, face_neg_elem, face_pos_elem, elem_face_dir, face_bc,
 inv_jacobian, num_elements, params)
```

The scalars travel in one vector because Mojo cannot forward a variadic
argument pack into `enqueue_function` next to fixed arguments, and a user
struct is not `DevicePassable`. The order of `kernel_params(...)` on the
host must match the `params[i]` unpacking in the kernel.

The launcher passes `1 / jacobian`, not the Jacobian: the core multiplies
by it (see the kernel performance rules in `CLAUDE.md`).

## CFL time step

```mojo
var dt = spatial.suggest_dt(ctx, mesh, q)     # Euler, ideal MHD, GLM-MHD
var dt = spatial.suggest_dt(mesh)             # Maxwell, advection
```

Returns

$$
\Delta t = \frac{\text{cfl\_max}}{(2P + 1)\,\sum_d \lambda_d / \Delta x_d}
$$

with the app's `cfl_max`. The signal speed is:

| Physics | $\lambda$ |
| ------- | --------- |
| Euler | $\max(\lvert\mathbf{u}\rvert + c)$ over all nodes |
| Ideal MHD | $\max(\lvert\mathbf{u}\rvert + \sqrt{c_s^2 + c_A^2})$, the fast speed perpendicular to $\mathbf{B}$, which bounds it in every direction |
| GLM-MHD | the MHD bound or $c_h$, whichever is larger |
| Maxwell | `c_light` (no state needed) |
| Advection | $\lvert v_d\rvert$ per axis (no state needed) |

For the state-dependent cases the maximum is reduced on the GPU by one
single-block kernel (`_max_speed_kernel` in `wave_speed.mojo`) and one
scalar is read back, so a call costs a kernel launch and a host
synchronisation. `SSPRK3.run_until_cfl` can refresh it every N steps
(see [`temporal`](temporal.md)).

For the nonlinear systems one direction-independent $\lambda$ multiplies
`mesh.inv_dx_sum()`, which is conservative on anisotropic meshes.

With SSPRK3, linear advection is stable up to `cfl_max` of about 1.5 for
$P = 1..4$ and unstable by 2 for $P \ge 3$
(`tests/test_cfl_timestep.mojo`). Use 0.5 for problems with shocks.

The lower-level pieces are importable from `warpxm.spatial.wave_speed`:
`cfl_dt[P](cfl_max, speed_over_dx)`, `max_wave_speed_euler(ctx, q, app)`,
`max_wave_speed_imhd`, `max_wave_speed_imhd_glm`. Each is generic over
`NDIM`. `cfl_dt` returns `WXM_INF` when the speed underflows.

## Source-term kernels (`spatial/sources/`)

Source terms are free functions called from the driver's `compute_rhs`
closure after the spatial solver. Each reads `q` and adds to `rhs` in
place. They are pointwise on the flat state buffer, so one function
serves every dimension; `NDIM` and `P` are inferred from the arguments.

### Gravity (Euler)

```mojo
from warpxm.spatial.sources.euler_gravity_source import apply_gravity_source
apply_gravity_source(ctx, q, rhs, gx, gy)        # gz defaults to 0
apply_gravity_source(ctx, q, rhs, gx, gy, gz)
```

Adds $\rho\,\mathbf{g}$ to the momentum RHS and
$\rho\mathbf{u}\cdot\mathbf{g}$ to the energy RHS.

### Lorentz force (Maxwell → fluid)

```mojo
from warpxm.spatial.sources.lorentz_source import apply_lorentz_source
apply_lorentz_source(ctx, q_fluid, q_maxwell, rhs_fluid, alpha)
```

Adds $\alpha(\rho\mathbf{E} + \rho\mathbf{u}\times\mathbf{B})$ to the
fluid momentum RHS and $\alpha\rho\mathbf{u}\cdot\mathbf{E}$ to the
energy RHS, with $\alpha = q/m$.

### Current source (fluid → Maxwell)

```mojo
from warpxm.spatial.sources.maxwell_current_source import apply_current_source
apply_current_source(ctx, q_fluid, rhs_maxwell, alpha, eps0)
```

Adds $-\alpha\rho\mathbf{u}/\varepsilon_0$ to the Maxwell E-field RHS.
Call it once per species.

### GLM damping (MHD)

```mojo
from warpxm.spatial.sources.imhd_glm_source import apply_glm_damping
apply_glm_damping(ctx, q, rhs, c_h, c_p_sq)
```

Adds $-(c_h^{2}/c_p^{2})\psi$ to the $\psi$ RHS. The function squares
`c_h` itself: pass the cleaning speed, not its square.

## Kinetic solvers (`spatial/kinetic/`)

| File | Contents |
| ---- | -------- |
| `dg_vlasov.mojo` | `DGVlasovSolver[XDIM, VDIM, P_x, P_v]`: generic phase-space DG residual with the full Lorentz force, `compute_rhs(ctx, mesh, f, fields, rhs)`, `suggest_dt(ctx, mesh, fields)` |
| `vlasov_moments.mojo` | `VlasovMoments[XDIM, VDIM, P_x, P_v](ctx, mesh).compute`: density, three flux components and energy into one `DGState[XDIM, P_x, VLASOV_MOMENT_NC]`, in one pass |
| `vlasov_current_source.mojo` | `apply_vlasov_current_source[XDIM, VDIM, P]`: flux moments into the Maxwell RHS |
| `dg_vlasov_1d1v.mojo`, `dg_vlasov_1d2v.mojo` | `DGVlasovSolver1D1V[P_x, P_v]`, `DGVlasovSolver1D2V[P_x, P_v1, P_v2]`: phase-space DG residual, `compute_rhs(ctx, mesh, f, aux, rhs)` |
| `vlasov_moments_1d1v.mojo`, `vlasov_moments_1d2v.mojo` | `VlasovMoments1D1V` / `1D2V`: density, current and kinetic-energy moments into `DGState[1, P_x, 1]` |
| `vlasov_current_source_1d.mojo`, `vlasov_current_source_1d2v.mojo` | `apply_vlasov_current_source_1d` / `_1d2v`: moment current into the Maxwell RHS |
| `vlasov_bgk_source_1d1v.mojo` | `VlasovBGKSource1D1V`: BGK relaxation |
| `vlasov_5moment_to_kinetic_1d1v.mojo` | `apply_5moment_to_kinetic_1d1v`: Maxwellian from an Euler state |

These have their own kernels and launch code; they do not go through
`dg_launch`. See [Vlasov-Maxwell](../physics/vlasov.md).

## See also

* [Time integration](../concepts/time-integration.md): where sources are
  applied within the RK structure.
* [GPU architecture](../concepts/gpu-architecture.md): how the DG
  kernels are laid out on the device.
* [Adding a new physics module](../contributing/adding-a-physics.md).
