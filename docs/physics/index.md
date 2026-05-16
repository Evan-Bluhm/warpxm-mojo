# Physics

warpxm-mojo implements seven physics modules across three dimensions.
Every (physics, dimension) cell checked here has end-to-end regression
tests passing on the GPU.

| Physics                       |  1D | 2D  | 3D  |
| ----------------------------- | :-: | :-: | :-: |
| Scalar advection              | ✅  | ✅  | ✅  |
| 5-moment Euler                | ✅  | ✅  | ✅  |
| Maxwell's equations           | —   | ✅  | ✅  |
| Single-fluid plasma           | —   | ✅  | ✅  |
| Two-species plasma            | —   | ✅  | ✅  |
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

## Conserved-variable conventions

All multi-component physics use a fixed-order flat layout, so
`q[node_index * NC + IDX_*]` gives you the component. The component
indices are exported as `comptime` constants from each `apps/*.mojo`
file:

```mojo
from warpxm.apps.imhd import (
    IDX_RHO, IDX_MX, IDX_MY, IDX_MZ, IDX_E,
    IDX_BX, IDX_BY, IDX_BZ,
)
```

The 3D MHD module **re-uses** the 2D constants — there's no
`IDX_RHO_3D`. The same goes for Maxwell. This is intentional: a 3D MHD
state has the same eight components as a 2D MHD state.
