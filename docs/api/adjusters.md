# `warpxm.adjusters`

State adjusters applied between RK stages. Two families: the
Moe-Rossmanith slope limiter and the Zhang-Shu positivity preserver.

Both adjusters are **cell-mean preserving** — they shrink each nodal
value toward the cell mean with a per-element $\theta \in [0, 1]$. The cell
mean is invariant, so mass / momentum / energy / B-field are exactly
conserved.

## Moe-Rossmanith limiter

```mojo
from warpxm.adjusters.moe_rossmanith_1d import MoeRossmanith1D
from warpxm.adjusters.moe_rossmanith_2d import MoeRossmanith2D
from warpxm.adjusters.moe_rossmanith_3d import MoeRossmanith3D

var limiter = MoeRossmanith2D[P, NC](
    ctx, num_elements,
    alpha_coeff=Real(1000.0),   # smoothness fudge
)
limiter.apply(ctx, mesh, q)
```

The 1D version is **Euler-specific** (`NC = EULER_NC` hard-coded). The
2D and 3D versions are **NC-parameterised** — same limiter works for
Euler (NC=5), MHD (NC=8), GLM-MHD (NC=9), or anything else.

See [Moe-Rossmanith limiter](../numerical-methods/moe-rossmanith.md)
for the algorithm.

## Zhang-Shu positivity preservers

### Euler

```mojo
from warpxm.adjusters.positivity_euler_1d import PositivityEuler1D
from warpxm.adjusters.positivity_euler_2d import PositivityEuler2D
from warpxm.adjusters.positivity_euler_3d import PositivityEuler3D

var positivity = PositivityEuler2D[P](
    num_elements=mesh.num_elements,
    gamma=Real(1.4),
    rho_eps=Real(1.0e-10),
    p_eps=Real(1.0e-10),
)
positivity.apply(ctx, mesh, q)
```

The pressure step uses a closed-form quadratic in θ to solve $P_{\text{gas}}(\theta) = P_{\varepsilon}$.

### MHD

```mojo
from warpxm.adjusters.positivity_mhd_2d import PositivityMHD2D
from warpxm.adjusters.positivity_mhd_3d import PositivityMHD3D

var positivity = PositivityMHD2D[P, MHD_NC](
    num_elements=mesh.num_elements,
    gamma=Real(5.0/3.0),
    rho_eps=Real(1.0e-10),
    p_eps=Real(1.0e-10),
)
positivity.apply(ctx, mesh, q)
```

NC-parameterised on `NC = MHD_NC` (8) or `MHD_GLM_NC` (9). The pressure
step uses bisection because the polynomial $f(\theta) = \rho E -
\tfrac{1}{2}|m|^{2} - \tfrac{1}{2}\rho|B|^{2} - \rho p_{\varepsilon}/(\gamma-1)$
is cubic in θ.

See [Zhang-Shu positivity preserver](../numerical-methods/positivity.md)
for the algorithm.

## Usage with steppers

You usually don't call `.apply(...)` directly — the `SSPRK3_*_Positive_*`
steppers thread it between RK stages automatically:

```mojo
stepper.step(ctx, mesh, spatial, limiter, positivity, q, dt)
```

Each stage does:

```text
stage_value = ... (combine with prior stages)
limiter.apply(stage_value)
positivity.apply(stage_value)
```

Order matters: limiter first (shape-preserving on conserved
variables), positivity second (enforces ρ > 0, P > 0 by checking
primitives).
