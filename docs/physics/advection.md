# Scalar advection

The simplest hyperbolic conservation law:

$$\partial_t \mathbf{q} + \vec{v}\cdot\nabla \mathbf{q} = 0,$$

with $\vec{v}$ a constant advection velocity. Useful as a non-trivial
smooth-problem benchmark for the DG kernels.

## Module

```mojo
from warpxm.apps.advection import Advection1D
```

There are dimension-specific structs `Advection1D[NC]`,
`Advection2D[NC]`, `Advection3D[NC]`, each parameterised by the
component count. For multi-component advection (e.g., colours) the same
constant velocity is applied to every component.

## State layout

Component count is open: pick `NC = 1` for scalar, larger for
multi-component.

```mojo
var q = DGState1D[P, 1](ctx, mesh.num_elements)        # scalar
var q = DGState2D[P, 4](ctx, mesh.num_elements)        # 4 colours
```

## Boundary conditions

* Periodic — the default for the 1D advection examples.
* `BC_COPYOUT` — zero-gradient outflow.

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

* `tests/test_advection_1d.mojo`
* `tests/test_advection_2d.mojo`
* `tests/test_advection_3d.mojo`
* `tests/test_advection_copyout_1d.mojo` — verifies the BC_COPYOUT path.
* `tests/test_advection_multi_component.mojo` — same velocity, multiple colours.
* `tests/test_dg_convergence_1d.mojo` — measured $h^{P+1}$ rates (1.98 / 3.01 / 3.99 for $P = 1, 2, 3$).
