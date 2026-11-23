# `warpxm.adjusters`

State adjusters applied between RK stages. Two families: the
Moe-Rossmanith slope limiter and the Zhang-Shu positivity preserver.

Both adjusters are **cell-mean preserving** — they shrink each nodal
value toward the cell mean with a per-element $\theta \in [0, 1]$. The
cell mean is invariant, so mass / momentum / energy / B-field are
exactly conserved.

Each adjuster is one struct and one kernel, generic over the dimension:
`MoeRossmanith[NDIM, P, NC]`, `PositivityEuler[NDIM, P]`,
`PositivityMHD[NDIM, P, NC]`. The cell-mean quadrature weight they share
is in `adjusters/cell_mean.mojo`.

## Moe-Rossmanith limiter

```mojo
from warpxm.adjusters.moe_rossmanith import MoeRossmanith

var limiter = MoeRossmanith[2, P, NC](
    ctx, num_elements,
    alpha_coeff=Real(1000.0),   # default
)
limiter.apply(ctx, mesh, q)
```

On several ranks pass the ghost count after the element count
(`ctx, mesh.num_elements, mesh.num_ghost`): the limiter fetches the cell
means of the elements across a partition face from their owners.

The limiter acts only on deviations larger than
$\alpha = \texttt{alpha\_coeff}\cdot\Delta x^{1.5}$. With the default
(upstream's) that is about 0.35 on a 200-cell unit domain, so on a
problem whose solution is much smaller than that the limiter does
nothing. Set `alpha_coeff` so that $\alpha$ is below the smallest scale
you need limited (the Einfeldt 1-2-3 tests use 10).

`_bounds_kernel` and `_apply_kernel` in `adjusters/moe_rossmanith.mojo`
serve every dimension. The apply step gathers all 2·NDIM face neighbors
(at a domain boundary the element stands in for its missing neighbor)
and collapses their averages into per-element bounds.

The limiter is **NC-parameterised** — the same one for Euler (NC=5),
IMHD (NC=8), GLM-MHD (NC=9), or any other multi-component conservative
system. A 1D Euler limiter is `MoeRossmanith[1, P, EULER_NC]`.

See [Moe-Rossmanith limiter](../numerical-methods/moe-rossmanith.md)
for the algorithm.

## Zhang-Shu positivity preservers

### Euler

```mojo
from warpxm.adjusters.positivity_euler import PositivityEuler

var positivity = PositivityEuler[2, P](
    num_elements=mesh.num_elements,
    gamma=Real(1.4),
    rho_eps=Real(1.0e-10),
    p_eps=Real(1.0e-10),
)
positivity.apply(ctx, mesh, q)
```

One kernel in `adjusters/positivity_euler.mojo` serves 1D, 2D and 3D. The pressure step uses a closed-form
quadratic in θ to solve $P_{\text{gas}}(\theta) = P_{\varepsilon}$.

### MHD

```mojo
from warpxm.adjusters.positivity_mhd import PositivityMHD

var positivity = PositivityMHD[2, P, MHD_NC](
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
cubic in θ. Only nodes that violate the bound are bisected, so an
element that is already admissible costs one evaluation per node.

See [Zhang-Shu positivity preserver](../numerical-methods/positivity.md)
for the algorithm.

## Usage with steppers

Adjusters are composed at the call site via the SSPRK3 `adjust_fn`
closure:

```mojo
def adjust(mut q_: DGState[2, P, NC]) raises {mut limiter, mut positivity, imm}:
    limiter.apply(ctx, mesh, q_)
    positivity.apply(ctx, mesh, q_)

stepper.step(ctx, q, dt, compute_rhs, adjust)
```

Order matters: limiter first (shape-preserving on conserved
variables), positivity second (enforces ρ > 0, P > 0 by checking
primitives).
