# Zhang-Shu positivity preserver

A cell-mean-preserving θ-shrink that enforces ρ > $\rho_{\varepsilon}$
and $P_{\text{gas}}$ > $P_{\varepsilon}$ at every GLL node. Runs *after*
the [Moe-Rossmanith limiter](moe-rossmanith.md) inside the SSPRK3
stepper so the two adjusters compose.

Citation: Zhang, X. & Shu, C.-W., "On maximum-principle-satisfying
high order schemes for scalar conservation laws", *J. Comput. Phys.*
229 (2010) 3091–3120; and the MHD extension in Zhang & Shu, "Positivity-
preserving high-order discontinuous Galerkin schemes for compressible
Euler equations on triangular meshes", *J. Sci. Comput.* 50 (2012)
29–62.

## Two steps per element

For each element with cell mean $\bar{\mathbf{q}}$:

### Step 1 — density

If any GLL node has $\rho_n < \rho_{\varepsilon}$, compute

$$
\theta_{\rho} = \min_{n\,:\, \rho_n < \rho_{\varepsilon}}
\frac{\bar{\rho} - \rho_{\varepsilon}}{\bar{\rho} - \rho_n}
$$

and shrink **all conserved variables** at all nodes:

$$
\mathbf{q}_n^{(1)} = \bar{\mathbf{q}} + \theta_{\rho}\,(\mathbf{q}_n - \bar{\mathbf{q}}).
$$

After step 1 every node has $\rho \geq \rho_{\varepsilon}$.

### Step 2 — pressure

At each node check $P_{\text{gas}}(\mathbf{q}_n^{(1)})$. For nodes where
it falls below $P_{\varepsilon}$, solve for the largest $\theta_P$ such
that the convex-combination state $\bar{\mathbf{q}} + \theta_P(\mathbf{q}_n^{(1)} - \bar{\mathbf{q}})$
satisfies $P_{\text{gas}} \geq P_{\varepsilon}$.

This is a **closed-form quadratic** in $\theta_P$ for Euler:

$$
f(\theta) = \rho(\theta)\,E(\theta) - \tfrac{1}{2}|m(\theta)|^{2} - \frac{P_{\varepsilon}}{\gamma - 1}\,\rho(\theta) = 0,
$$

and a **cubic** in $\theta_P$ for MHD because $\tfrac{1}{2}\rho|\mathbf{B}|^{2}$
in the energy decomposition is itself quadratic in θ. For MHD we use
50 bisection iterations on $[0, 1]$ — gets to $2^{-50} \approx 10^{-15}$
accuracy and avoids the conditioning headaches of a closed-form cubic.

Take the minimum $\theta_P$ across all problematic nodes; shrink again.

## Conservation

The cell mean is invariant under each θ-shrink:

$$
\sum_n w_n (\bar{\mathbf{q}} + \theta(\mathbf{q}_n - \bar{\mathbf{q}})) = \bar{\mathbf{q}} + \theta(\bar{\mathbf{q}} - \bar{\mathbf{q}}) = \bar{\mathbf{q}}.
$$

So mass / momentum / energy / B-field are exactly conserved per
element. We've verified this to machine precision on the strong-blast
regression tests.

## Failure mode

If even the cell mean itself has $\bar{\rho} \leq \rho_{\varepsilon}$ or
$P_{\text{gas}}(\bar{\mathbf{q}}) \leq P_{\varepsilon}$, the preserver
can't help — the simulation has already crashed. In practice that
hasn't happened on any of our test problems, even the Einfeldt 1-2-3
problem (two-rarefaction near-vacuum), because the cell mean is the
*average* and stays well-behaved much longer than individual nodes.

## In code

| File | Dim | Physics |
| ---- | --- | ------- |
| `adjusters/positivity_euler_1d.mojo` | 1D | Euler (NC=5) |
| `adjusters/positivity_euler_2d.mojo` | 2D | Euler |
| `adjusters/positivity_euler_3d.mojo` | 3D | Euler |
| `adjusters/positivity_mhd_2d.mojo` | 2D | MHD (NC=8 or 9 with ψ) |
| `adjusters/positivity_mhd_3d.mojo` | 3D | MHD |

Usage (mirrors the limiter):

```mojo
from warpxm.adjusters.positivity_euler_2d import PositivityEuler2D
from warpxm.temporal.ssprk3_2d import SSPRK3_2D

var positivity = PositivityEuler2D[P](
    num_elements=mesh.num_elements, gamma=GAMMA,
    rho_eps=Real(1.0e-10), p_eps=Real(1.0e-10),
)
var stepper = SSPRK3_2D[P, EULER_NC](ctx, mesh.num_elements)

@parameter
def adjust(mut q_: DGState2D[P, EULER_NC]) raises:
    limiter.apply(ctx, mesh, q_)
    positivity.apply(ctx, mesh, q_)

stepper.step[compute_rhs, adjust](ctx, q, dt)
```

The positive stepper runs `limiter.apply(...)` then `positivity.apply(...)`
between each RK stage.

## Validation

* **Einfeldt 1-2-3 (1D, 2D, 3D)** — Two-rarefaction near-vacuum: both
  preserver passes activate, ρ and P clamp to floor (10⁻¹⁰).
  Cell-mean conservation: exact to 10⁻¹⁴.
* **Strong MHD blast (2D, 3D)** — Δ$P$/$P_{\text{amb}} = 1000$ in
  a strong B₀ = 5 background. `min P_gas` hits `p_eps = 10⁻¹⁰` exactly,
  proving the cubic-bisection branch fires correctly. Mass and
  $\int B_x\,dV$ conserved to 2 × 10⁻¹³.
* **Strong Sedov blast (2D)** — P_inner / P_ambient = 10⁴. `min P_gas`
  pegged at 10⁻¹⁰ throughout the run; mass and total energy conserved
  to 7 × 10⁻¹⁴.
