# `warpxm.adjusters`

State adjusters applied between RK stages. Two families: the
Moe-Rossmanith slope limiter and the Zhang-Shu positivity preserver.

Both adjusters are **cell-mean preserving** — they shrink each nodal
value toward the cell mean with a per-element $\theta \in [0, 1]$. The
cell mean is invariant, so mass / momentum / energy / B-field are
exactly conserved.

Each adjuster has one canonical NDIM-parametric implementation; the
per-dim launcher structs (e.g. `PositivityEuler1D`, `PositivityEuler2D`,
`PositivityEuler3D`) all live in the same canonical file as thin
wrappers over the shared kernel.

## Moe-Rossmanith limiter

```mojo
from warpxm.adjusters.moe_rossmanith import MoeRossmanith2D, MoeRossmanith3D
# 1D variant (Euler-only, different mesh API) lives separately:
from warpxm.adjusters.moe_rossmanith_1d import MoeRossmanith1D

var limiter = MoeRossmanith2D[P, NC](
    ctx, num_elements,
    alpha_coeff=Real(1000.0),   # smoothness fudge
)
limiter.apply(ctx, mesh, q)
```

The 2D and 3D variants share an NDIM-parametric `_bounds_kernel` and
`_apply_kernel` in `adjusters/moe_rossmanith.mojo`. The apply step
gathers all 2·NDIM face neighbors and collapses their averages into
per-element bounds.

The 1D version is **Euler-specific** (`NC = EULER_NC` hard-coded) and
uses the older 1D mesh interface (separate `elem_left_face` /
`elem_right_face` arrays). The 2D and 3D versions are
**NC-parameterised** — same limiter for Euler (NC=5), IMHD (NC=8),
GLM-MHD (NC=9), or any other multi-component conservative system.

See [Moe-Rossmanith limiter](../numerical-methods/moe-rossmanith.md)
for the algorithm.

## Zhang-Shu positivity preservers

### Euler

```mojo
from warpxm.adjusters.positivity_euler import (
    PositivityEuler1D, PositivityEuler2D, PositivityEuler3D,
)

var positivity = PositivityEuler2D[P](
    num_elements=mesh.num_elements,
    gamma=Real(1.4),
    rho_eps=Real(1.0e-10),
    p_eps=Real(1.0e-10),
)
positivity.apply(ctx, mesh, q)
```

All three dims share one NDIM-parametric kernel in
`adjusters/positivity_euler.mojo`. The pressure step uses a closed-form
quadratic in θ to solve $P_{\text{gas}}(\theta) = P_{\varepsilon}$.

### MHD

```mojo
from warpxm.adjusters.positivity_mhd import PositivityMHD2D, PositivityMHD3D

var positivity = PositivityMHD2D[P, MHD_NC](
    num_elements=mesh.num_elements,
    gamma=Real(5.0/3.0),
    rho_eps=Real(1.0e-10),
    p_eps=Real(1.0e-10),
)
positivity.apply(ctx, mesh, q)
```

NC-parameterised on `NC = MHD_NC` (8) or `MHD_GLM_NC` (9 — ψ is
shrunk along with the rest but doesn't enter the positivity check).
The pressure step bisects on $f(\theta) = \rho E - \tfrac{1}{2}|m|^{2}
- \tfrac{1}{2}\rho|B|^{2} - \rho p_{\varepsilon}/(\gamma-1)$, which is
cubic in θ.

See [Zhang-Shu positivity preserver](../numerical-methods/positivity.md)
for the algorithm.

## Usage with steppers

Adjusters are composed at the call site via the SSPRK3 `adjust_fn`
closure:

```mojo
@parameter
def adjust(mut q_: DGState2D[P, NC]) raises:
    limiter.apply(ctx, mesh, q_)
    positivity.apply(ctx, mesh, q_)

stepper.step[compute_rhs, adjust](ctx, q, dt)
```

Order matters: limiter first (shape-preserving on conserved
variables), positivity second (enforces ρ > 0, P > 0 by checking
primitives).
