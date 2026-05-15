# `warpxm.core`

Shared low-level types, math compatibility shims, and the boundary-
condition enum.

## `types`

```mojo
from warpxm.core.types import (
    Real, WXM_DTYPE, INDEX_DTYPE,
    TWO_PI, PI, WXM_INF,
)
```

| Symbol | Definition |
| ------ | ---------- |
| `Real` | `Scalar[DType.float64]`. Everything physics-related is double precision. |
| `WXM_DTYPE` | `DType.float64`. Used for `TileTensor[WXM_DTYPE, ...]`. |
| `INDEX_DTYPE` | `DType.int32`. Used for face / element index buffers. |
| `TWO_PI` | $2\pi$ as `Real`. |
| `PI` | $\pi$. |
| `WXM_INF` | Sentinel for "no neighbour" in face-index buffers. |

Switching the framework to single precision would mean editing this one
file. No bare `Float32` / `Float64` literals appear in physics code.

## `math_compat`

```mojo
from warpxm.core.math_compat import sqrt, fabs, fmin, fmax
```

These are thin shims around `std.math.*`:

* `sqrt(x)` — `x ** Real(0.5)`. Works around the missing `float64` GPU
  intrinsic for `std.math.sqrt` on NVIDIA targets.
* `fabs`, `fmin`, `fmax` — wrappers giving `Real`-typed return values
  to avoid promotion surprises.

Any kernel that uses these via this module remains portable; reaching
into `std.math` directly can silently produce single-precision results
on some GPUs.

## `bc` (boundary-condition enum)

```mojo
from warpxm.core.bc import (
    BC_INTERIOR,    # 0 — face is between two real elements (default)
    BC_COPYOUT,     # 1 — zero-gradient outflow
    BC_WALL_FREESLIP,  # 2 — reflective wall (normal momentum flipped)
    BC_DIRICHLET,   # 3 — fixed boundary state (rarely used)
    BC_PEC_WALL,    # 4 — perfectly-conducting wall (Maxwell, MHD)
    BC_PMC_WALL,    # 5 — perfectly-magnetic wall (Maxwell)
)
```

The values are plain `Int` constants — the mesh stores them per-face in
a flat `INDEX_DTYPE` buffer. DG kernels switch on the BC integer at
each ghost-face evaluation; the BC adjustment is then physics-aware
(e.g. for Euler `BC_WALL_FREESLIP` flips $\rho u_n$; for Maxwell
`BC_PEC_WALL` flips tangential $\mathbf{E}$).

Per-axis BC selection is in each mesh's `__init__`:

```mojo
var mesh = BlockMesh2D[P](
    ctx, x_min, x_max, y_min, y_max,
    nx=NX, ny=NY,
    periodic_x=False, periodic_y=True,
    bc_left=BC_WALL_FREESLIP, bc_right=BC_COPYOUT,
)
```

If `periodic_x=True`, the `bc_left` / `bc_right` parameters are
ignored.
