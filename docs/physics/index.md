# Physics

warpxm-mojo implements seven fluid and field systems across three
dimensions, plus a kinetic Vlasov-Maxwell solver in 1D1V and 1D2V. Every
(physics, dimension) cell checked here has end-to-end regression tests
passing on the GPU; the full matrix is on the [Features](../features.md)
page.

| Physics                       |  1D | 2D  | 3D  |
| ----------------------------- | :-: | :-: | :-: |
| Scalar advection              | ✅  | ✅  | ✅  |
| 5-moment Euler                | ✅  | ✅  | ✅  |
| Maxwell's equations           | ✅  | ✅  | ✅  |
| Single-fluid plasma           | ✅  | ✅  | ✅  |
| Two-species plasma            | ✅  | ✅  | ✅  |
| Ideal MHD                     | —   | ✅  | ✅  |
| MHD with Dedner GLM cleaning  | —   | ✅  | ✅  |

Each page below covers, in the same order:

* the conservation law,
* the **conserved-variable layout** (`NC` and the `IDX_*` constants),
* what boundary conditions work,
* the available numerical fluxes / limiters / positivity options,
* the example drivers and tests.

## Pages

* [Scalar advection](advection.md)
* [5-moment Euler](euler.md)
* [Maxwell's equations](maxwell.md)
* [Ideal MHD](mhd.md)
* [Two-fluid plasma](plasma.md)
* [Vlasov-Maxwell (kinetic)](vlasov.md)

## Conserved-variable conventions

All multi-component physics store their components in a fixed order at
each node. The component indices are exported as `comptime` constants
from each `apps/*.mojo` file (and re-exported by the matching
`warpxm.physics.*` module); use them with `mesh.integrate(q, IDX_RHO)`,
inside a `mesh.fill` closure, or with `DGState.index`:

```mojo
from warpxm.apps.imhd import (
    IDX_RHO, IDX_MX, IDX_MY, IDX_MZ, IDX_E,
    IDX_BX, IDX_BY, IDX_BZ,
)
```

There is a **single** set of constants per physics, shared by every
dimension — no per-dimension copies. The same goes for
Maxwell. This falls naturally out of the apps/ architecture: each
physics is one canonical file (`apps/imhd.mojo`, `apps/maxwell.mojo`)
exporting the constants once, and the DG wrappers import them from there.
