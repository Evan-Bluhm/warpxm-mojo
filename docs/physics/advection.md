# Scalar advection

The simplest hyperbolic conservation law:

$$\partial_t \mathbf{q} + \vec{v}\cdot\nabla \mathbf{q} = 0,$$

with $\vec{v}$ a constant advection velocity. Useful as a non-trivial
smooth-problem benchmark for the DG kernels.

## Module

```mojo
from warpxm.apps.advection import Advection
```

`Advection[NC: Int]` is dimension-agnostic — the same struct serves 1D,
2D, and 3D and is parameterised only by the component count. The DG
core supplies the face normal, and the app dots the
struct's `velocity_x`, `velocity_y`, `velocity_z` against it. For
multi-component advection (e.g., colours) the same constant velocity
is applied to every component.

## State layout

Component count is open: pick `NC = 1` for scalar, larger for
multi-component.

```mojo
var q = DGState[1, P, 1](ctx, mesh.num_elements)        # scalar
var q = DGState[2, P, 4](ctx, mesh.num_elements)        # 4 colours
```

## Boundary conditions

* Periodic — the default for the 1D advection examples.
* `BC_COPYOUT` — zero-gradient outflow. Any other BC kind is treated as
  copy-out.

## Time step

The velocity is constant, so `spatial.suggest_dt(mesh)` needs no state:
it returns `cfl_max / ((2P + 1) Σ_d |v_d| / Δx_d)`.

## Numerical flux

Upwind:

$$\mathbf{F}^{\ast} = \mathbf{q}_L \quad \text{if } v_n > 0, \quad
  \mathbf{q}_R \quad \text{if } v_n < 0.$$

This is exact for advection, so there's no Rusanov / HLLD choice.

## Examples

| File | Dim | Notes |
| ---- | --- | ----- |
| [`examples/advection_1d.mojo`](https://github.com/your-fork/warpxm-mojo/blob/main/examples/advection_1d.mojo) | 1D | Sine IC on `[-0.5, 0.5]`, one full period. The walkthrough in [Getting Started](../getting-started/first-example.md). |
| [`examples/advection_2d.mojo`](https://github.com/your-fork/warpxm-mojo/blob/main/examples/advection_2d.mojo) | 2D | Two-velocity advection on a periodic square. |
| [`examples/convergence_advection_1d.mojo`](https://github.com/your-fork/warpxm-mojo/blob/main/examples/convergence_advection_1d.mojo) | 1D | Convergence study at $P = 1, 2, 3$. Confirms $h^{P+1}$ DG convergence rates. |

## Tests

* `tests/test_advection_convergence_{1,2,3}d.mojo` — single-component returns to IC after one period; checks the P+1 rate and bounds the absolute L2 error.
* `tests/test_advection_copyout_1d.mojo` — verifies the BC_COPYOUT path.
* `tests/test_advection_multi_component_1d.mojo` (1D), `..._2d.mojo`, `..._3d.mojo` — same velocity field, multiple independent colours.
  Measured rates: 1.98 / 3.01 / 3.99 in 1D and 1.90 / 3.03 / 4.00 in 2D for $P = 1, 2, 3$; 3.12 / 3.99 in 3D at $P = 2, 3$.
* `tests/test_cfl_timestep.mojo` — SSPRK3 advection keeps its energy at `cfl_max = 1` and blows up at 4, for $P = 1..4$.
