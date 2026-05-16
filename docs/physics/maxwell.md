# Maxwell's equations

Linear electromagnetism in vacuum:

$$
\partial_t \mathbf{E} = c^{2}\,\nabla\times\mathbf{B} - \mathbf{J}/\varepsilon_{0},
\qquad
\partial_t \mathbf{B} = -\nabla\times\mathbf{E}.
$$

We carry both 3-vectors as 8 scalars (E_x, E_y, E_z, B_x, B_y, B_z;
plus two spares) so the same state layout matches both 2D (TE / TM)
and 3D problems.

## Module

```mojo
from warpxm.apps.maxwell import Maxwell
from warpxm.apps.maxwell import Maxwell
```

Constructor:

```mojo
var app = Maxwell(c_light=Real(1.0), cfl_max=Real(0.5))
```

`c_light` is the speed of light in whatever units you choose. The
CFL limit is on $c\,\Delta t / \Delta x$.

## State layout

```mojo
from warpxm.apps.maxwell import (
    MAXWELL_NC, IDX_EX, IDX_EY, IDX_EZ, IDX_BX, IDX_BY, IDX_BZ,
)
# MAXWELL_NC = 6  (Ex, Ey, Ez, Bx, By, Bz — no padding)
```

The 3D module re-uses these constants.

## Boundary conditions

| BC | Behaviour |
| -- | --------- |
| Periodic | Plane-wave propagation. |
| `BC_COPYOUT` | Outflow. |
| `BC_PEC_WALL` | Perfectly-conducting wall — tangential E and normal B vanish. |
| `BC_PMC_WALL` | Perfectly-magnetic wall — tangential B and normal E vanish. |

PEC walls are how the TM₁₁ cavity mode example is set up:

```mojo
var mesh = BlockMesh2D[P](
    ctx, 0.0, 1.0, 0.0, 1.0, nx=NX, ny=NY,
    periodic_x=False, periodic_y=False,
    bc_left=BC_PEC_WALL, bc_right=BC_PEC_WALL,
    bc_bottom=BC_PEC_WALL, bc_top=BC_PEC_WALL,
)
```

## Numerical flux

Central + Lax-Friedrichs dissipation with $\lambda_{\max} = c$. Maxwell
in vacuum has no shocks, so the flux is essentially a dispersive
upwinding.

## Coupling to fluids

Maxwell receives a current source from each charged fluid via
`apply_current_source_2d` / `_3d`:

```mojo
apply_current_source_2d[P](ctx, fluid_state, maxwell_rhs, alpha, eps0)
```

`alpha` is the species' charge-to-mass ratio. For the symmetric pair-
plasma test we set `alpha_1 = -1, alpha_2 = +1`.

The reverse coupling (Lorentz force on fluid momentum) is in
`apply_lorentz_source_2d` — see [Plasma](plasma.md).

## Examples and tests

| File | Description |
| ---- | ----------- |
| [`examples/maxwell_cavity_2d_vtu.mojo`](https://github.com/your-fork/warpxm-mojo/blob/main/examples/maxwell_cavity_2d_vtu.mojo) | 2D TM₁₁ cavity mode, with VTU frame output. Walkthrough in [Tutorials](../tutorials/cavity-mode.md). |
| [`examples/maxwell_cavity_3d_vtu.mojo`](https://github.com/your-fork/warpxm-mojo/blob/main/examples/maxwell_cavity_3d_vtu.mojo) | Same setup, 3D mesh. |

Tests:

* `tests/test_maxwell_1d_cavity.mojo` — 1D TM cavity (E_z + B_y on $x$),
  one period; L2 E_z ~ 8e-6.
* `tests/test_maxwell_1d_plane_wave.mojo` — 1D transverse plane wave
  propagating along $\hat{x}$ with E_z / B_y; one period returns to IC.
* `tests/test_maxwell_2d_cavity.mojo` — TM₁₁ standing wave; unexcited
  components stay at 0.0 exactly.
* `tests/test_maxwell_2d_plane_wave.mojo` — vacuum plane wave.
* `tests/test_maxwell_3d_cavity.mojo`, `tests/test_maxwell_3d_plane_wave.mojo`.

The 1D Maxwell solver lives in `spatial/dg_maxwell_1d.mojo` and uses
the canonical `Maxwell` app from `apps/maxwell.mojo` — the 1D DG kernel
just passes the normal $(\pm 1, 0, 0)$ into the same numerical flux
the 2D and 3D solvers use. A TM wave with E_z and B_y propagating
along $\hat{x}$ is a perfectly valid 1D Maxwell problem.
