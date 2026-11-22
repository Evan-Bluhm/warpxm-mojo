# `warpxm.core`

Shared low-level types, math compatibility shims, and the boundary-
condition enum.

## `types`

```mojo
from warpxm.core.types import (
    Real, WXM_DTYPE, INDEX_DTYPE,
    TWO_PI, PI, HALF_PI,
    WXM_INF, WXM_TINY,
)
```

| Symbol | Definition |
| ------ | ---------- |
| `Real` | `Scalar[DType.float64]`. Everything physics-related is double precision. |
| `WXM_DTYPE` | `DType.float64`. Used for `TileTensor[WXM_DTYPE, ...]`. |
| `INDEX_DTYPE` | `DType.int32`. Used for face / element index buffers. |
| `TWO_PI`, `PI`, `HALF_PI` | $2\pi$, $\pi$, $\pi/2$ as `Real`. |
| `WXM_INF` | `1e308` as `Real`. Returned by `cfl_dt` (and so by `suggest_dt`) when the signal speed underflows, meaning "no CFL constraint". |
| `WXM_TINY` | `1e-14` as `Real`. The underflow threshold for that check: speeds below it return `WXM_INF` instead of dividing by zero. |

Float64 is the only precision that has been run or tested. Kernels are
written against `Real` / `WXM_DTYPE` throughout, but treat the alias as a
name, not as a supported single-precision switch.

## `math_compat`

```mojo
from warpxm.core.math_compat import sqrt, fabs, sign, fmin, fmax, square
```

These are thin shims around `std.math.*`:

* `sqrt(x)` — the `llvm.sqrt` intrinsic, which lowers to the native
  double-precision square root on NVIDIA targets (`std.math.sqrt` has no
  `float64` GPU overload). Do not use `x ** Real(0.5)` in kernels: it calls
  a software `pow` that is many times more expensive.
* `fabs`, `fmin`, `fmax` — wrappers giving `Real`-typed return values
  to avoid promotion surprises.
* `sign(x)` — returns `+1`, `-1`, or `0` (exact zero at the origin).
* `square(x)` — `x * x`, kept as a named helper so callers can
  write `square(B_x) + square(B_y) + square(B_z)` without
  repeating sub-expressions.

Any kernel that uses these via this module remains portable; reaching
into `std.math` directly can silently produce single-precision results
on some GPUs.

## `bc` (boundary-condition enum)

```mojo
from warpxm.core.bc import (
    BC_INTERIOR,    # 0 — face is between two real elements (default)
    BC_COPYOUT,     # 1 — zero-gradient outflow
    BC_WALL_FREESLIP,  # 2 — reflective wall (normal momentum flipped)
    BC_PEC_WALL,    # 4 — perfectly-conducting wall (Maxwell, ideal MHD)
    BC_PMC_WALL,    # 5 — perfectly-magnetic wall (Maxwell)
    BC_VLASOV_OUTFLOW,  # 6 — zero ghost state on velocity-space boundaries
)
```

The values are plain `Int` constants — the mesh stores them per-face in
a flat `INDEX_DTYPE` buffer. DG kernels switch on the BC integer at
each ghost-face evaluation; the BC adjustment is then physics-aware
(e.g. for Euler `BC_WALL_FREESLIP` flips $\rho u_n$; for Maxwell
`BC_PEC_WALL` flips tangential $\mathbf{E}$). A kind a physics does not
implement falls through to copy-out; GLM-MHD treats every kind as
copy-out. The phase meshes set `BC_VLASOV_OUTFLOW` on the velocity-space
boundaries themselves.

Per-axis BC selection is in each mesh's `__init__`:

```mojo
var mesh = BlockMesh[2, P](
    ctx, x_min, x_max, y_min, y_max,
    nx=NX, ny=NY,
    periodic_x=False, periodic_y=True,
    bc_left=BC_WALL_FREESLIP, bc_right=BC_COPYOUT,
)
```

If `periodic_x=True`, the `bc_left` / `bc_right` parameters are
ignored.
