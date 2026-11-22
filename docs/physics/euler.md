# 5-moment Euler

Compressible Euler equations of gas dynamics with a γ-law equation of
state. The "5-moment" name refers to the five conserved variables
(ρ, ρu_x, ρu_y, ρu_z, E) — present in every dimension to keep the
state layout dimension-independent.

## Equations

$$
\partial_t
\begin{pmatrix}\rho \\ \rho\mathbf{u} \\ E\end{pmatrix}
+
\nabla\cdot
\begin{pmatrix}\rho\mathbf{u} \\ \rho\mathbf{u}\mathbf{u} + P\,I \\ (E + P)\,\mathbf{u}\end{pmatrix}
= \mathbf{0},
\qquad
P = (\gamma - 1)\!\left(E - \tfrac{1}{2}\rho|\mathbf{u}|^{2}\right).
$$

## Module

```mojo
from warpxm.apps.euler import Euler
```

The constructor parameters:

```mojo
var app = Euler(
    gas_gamma=Real(1.4),
    min_density=Real(1e-12),
    min_pressure=Real(1e-12),
    cfl_max=Real(0.5),
)
```

`min_density` and `min_pressure` are flooring values used inside the
gas-pressure recovery to keep the EOS away from singular states; they
do **not** replace the [Zhang-Shu positivity preserver](../numerical-methods/positivity.md),
which is the correct way to enforce ρ > 0, P > 0.

## State layout

```mojo
from warpxm.apps.euler import EULER_NC, IDX_RHO, IDX_MX, IDX_MY, IDX_MZ, IDX_E
# EULER_NC = 5
```

| Component | Index | Meaning |
| --------- | ----- | ------- |
| `q[IDX_RHO]` | 0 | Density $\rho$ |
| `q[IDX_MX]`  | 1 | $\rho u_x$ |
| `q[IDX_MY]`  | 2 | $\rho u_y$ |
| `q[IDX_MZ]`  | 3 | $\rho u_z$ |
| `q[IDX_E]`   | 4 | Total energy $E = P/(\gamma-1) + \tfrac{1}{2}\rho|\mathbf{u}|^{2}$ |

The same constants serve every dimension: there are no per-dimension
copies.

## Boundary conditions

| BC | Behaviour |
| -- | --------- |
| Periodic | Implicit in the mesh constructor. |
| `BC_COPYOUT` | Zero-gradient outflow. |
| `BC_WALL_FREESLIP` | Reflective wall — normal velocity flipped, tangential preserved. |

## Numerical flux

Rusanov / Lax-Friedrichs only. The fast-wave speed is the gas sound
speed: $\lambda_{\max} = |u_n| + c_s$ with $c_s^2 = \gamma P / \rho$.

For sharper shocks, see the discussion of HLLD in the [MHD page](mhd.md)
— the same Miyoshi-Kusano family of solvers can be specialised to Euler
but isn't shipped here yet.

## Sources

* **Gravity** — `apply_gravity_source(ctx, q, rhs, gx, gy, gz=0)` in
  `src/warpxm/spatial/sources/euler_gravity_source.mojo` adds a uniform
  body force in any dimension. Used in the 2D hydrostatic-balance
  regression test, where the source is balanced to within 4 × 10⁻¹⁶.
* **Lorentz force** — when Euler is coupled to Maxwell as a charged
  fluid; see [Plasma](plasma.md).

## Time step

`spatial.suggest_dt(ctx, mesh, q)` returns
`cfl_max / ((2P + 1) λ Σ_d 1/Δx_d)` with
$\lambda = \max(|\mathbf{u}| + c_s)$ reduced over all nodes on the GPU.
Pass it to `SSPRK3.run_until_cfl` through a closure; see
[`temporal`](../api/temporal.md#run_until_cfl).

## Limiter and positivity

For shock problems, wrap with the Moe-Rossmanith slope limiter:

```mojo
from warpxm.prelude import *
from warpxm.physics.euler import *

var stepper = SSPRK3[2, P, EULER_NC](ctx, mesh.num_elements)
var limiter = MoeRossmanith[2, P, EULER_NC](ctx, mesh.num_elements)

def adjust(mut q_: DGState[2, P, EULER_NC]) raises {mut limiter, imm}:
    limiter.apply(ctx, mesh, q_)
```

For strong shocks where the limiter alone can't keep ρ, P positive,
add the Zhang-Shu positivity preserver into the same `adjust` closure:

```mojo
var positivity = PositivityEuler[2, P](mesh.num_elements, gamma=GAMMA)

def adjust(mut q_: DGState[2, P, EULER_NC]) raises {mut limiter, mut positivity, imm}:
    limiter.apply(ctx, mesh, q_)
    positivity.apply(ctx, mesh, q_)
```

See [Time integration](../concepts/time-integration.md) and
[Positivity preserver](../numerical-methods/positivity.md) for the
algorithms.

## Examples and tests

| File | Description |
| ---- | ----------- |
| [`examples/euler_sod_1d.mojo`](https://github.com/your-fork/warpxm-mojo/blob/main/examples/euler_sod_1d.mojo) | 1D Sod tube. |
| [`examples/euler_entropy_wave_2d.mojo`](https://github.com/your-fork/warpxm-mojo/blob/main/examples/euler_entropy_wave_2d.mojo) | Smooth 2D entropy wave. |
| [`examples/euler_sod_2d_vtu.mojo`](https://github.com/your-fork/warpxm-mojo/blob/main/examples/euler_sod_2d_vtu.mojo) | 2D Sod with VTU output. |
| [`examples/euler_sedov_blast_2d_vtu.mojo`](https://github.com/your-fork/warpxm-mojo/blob/main/examples/euler_sedov_blast_2d_vtu.mojo) | Strong 2D Sedov blast exercising positivity preserver. |
| [`examples/euler_kelvin_helmholtz_2d_vtu.mojo`](https://github.com/your-fork/warpxm-mojo/blob/main/examples/euler_kelvin_helmholtz_2d_vtu.mojo) | 2D Kelvin-Helmholtz shear-layer instability with the slope limiter. |
| [`examples/euler_blast_3d_vtu.mojo`](https://github.com/your-fork/warpxm-mojo/blob/main/examples/euler_blast_3d_vtu.mojo) | 3D blast wave. |

Regression tests cover Sod 1D / 2D / 3D, the 1D and 3D entropy waves,
hydrostatic balance, free-slip walls, the Einfeldt 1-2-3 problem
(1D / 2D / 3D) and the Lax shock tube. The shock-tube tests compare with
the exact Riemann solution from `warpxm.testing.riemann`
([`testing`](../api/testing.md)). See
[Numerical Methods → Positivity preserver](../numerical-methods/positivity.md)
for the 1-2-3 results.
