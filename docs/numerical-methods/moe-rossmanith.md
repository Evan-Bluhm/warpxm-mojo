# Moe-Rossmanith slope limiter

The shock-capturing limiter applied between RK stages in every shock-
prone driver. Operates on conserved variables only — this is the
"conserved-variable cut" of the original Moe-Rossmanith scheme; the
companion [positivity preserver](positivity.md) adds the primitive-
bound step.

Citation: Moe, S., Rossmanith, J. & Seal, D., "A simple and effective
high-order shock capturing limiter for discontinuous Galerkin methods",
arXiv:1507.03024 (2015).

## What it does

For each element $e$ and each conserved-variable component $c$, the
limiter:

1. Computes the **cell mean** $\bar{q}_c(e)$ from GLL quadrature.
2. Finds the per-element min / max nodal values $q_{c,\min}(e),
   q_{c,\max}(e)$.
3. Looks at the cell means of the **neighbours** to derive an upper
   and lower bound on what the polynomial should attain inside the
   element:

   $$
   \text{upper}_c = \max\!\bigl(\bar{q}_c(e) + \alpha,\, \max_{n\in\partial e} \bar{q}_c(n)\bigr),
   $$

   $$
   \text{lower}_c = \min\!\bigl(\bar{q}_c(e) - \alpha,\, \min_{n\in\partial e} \bar{q}_c(n)\bigr),
   $$

   with $\alpha = \alpha_{\text{coeff}}\cdot dx^{3/2}$ a smoothness
   margin that disables the limiter in smooth regions.

4. Computes per-component shrink factors and clips them through a
   non-linear $\varphi$ function:

   $$
   y_M = \frac{\text{upper}_c - \bar{q}_c}{q_{c,\max} - \bar{q}_c + \varepsilon},
   \quad
   y_m = \frac{\text{lower}_c - \bar{q}_c}{q_{c,\min} - \bar{q}_c + \varepsilon},
   $$

   $$
   \varphi(y) = \max\!\bigl(\min(|y|/1.1, 1), 0\bigr),
   \quad
   \theta_c = \min(\varphi(y_M), \varphi(y_m)).
   $$

5. Takes the **minimum θ over all components** in the element and
   shrinks the polynomial toward the cell mean:

   $$
   q_{c}^{\text{new}}(e, i) = \bar{q}_c(e) + \theta(e)\,(q_c^{\text{old}}(e, i) - \bar{q}_c(e)).
   $$

Because $\bar{q}_c$ is the GLL-quadrature mean and θ shrinks each node
linearly toward it, the **cell mean is invariant** — mass / momentum /
energy / B-field are exactly conserved per element.

## Properties

* **Conservative**: cell means untouched.
* **TVD-like**: bounded by neighbour means with the smoothness fudge.
* **Generic in NC**: the 2D and 3D versions are parameterised on `NC`
  so the same kernel handles Euler (NC=5), MHD (NC=8), GLM-MHD (NC=9).
* **GPU-friendly**: one thread per element. Two kernels per call —
  bounds-and-means scan, then θ-and-apply.

## What it doesn't do

It does **not** enforce primitive-variable bounds. With NC=5 Euler the
limiter keeps each conserved variable within neighbour bounds, but the
*derived* primitive $P = (\gamma-1)(E - \tfrac{1}{2}\rho|\mathbf{u}|^{2})$
can still go negative on strong shocks. That's the job of the
[Zhang-Shu positivity preserver](positivity.md), which runs *after*
the Moe-Rossmanith limiter.

## In code

| File | Dims | Struct(s) | Notes |
| ---- | ---- | --------- | ----- |
| `adjusters/moe_rossmanith.mojo`    | 2D, 3D | `MoeRossmanith{2,3}D[P, NC]` | NDIM-parametric, NC-parameterised |
| `adjusters/moe_rossmanith_1d.mojo` | 1D     | `MoeRossmanith1D[P]`         | Euler-only (NC=5 hard-coded); uses the 1D mesh's split left/right face arrays |

Usage:

```mojo
from warpxm.adjusters.moe_rossmanith import MoeRossmanith2D

var limiter = MoeRossmanith2D[P, MHD_NC](ctx, mesh.num_elements)
# Inside the stepper, after each RK stage:
limiter.apply(ctx, mesh, q)
```

The smoothness fudge $\alpha_{\text{coeff}}$ defaults to 1000 (matches
upstream WARPXM) — large enough that the limiter is dormant on smooth
runs and only activates at sharp gradients.
