# Numerical methods

The algorithms behind the physics. Each page is a short, self-contained
explanation of one piece — sufficient to read the code, with citations
to the original papers.

## Spatial discretisation

* [Rusanov / Lax-Friedrichs flux](rusanov.md) — the default numerical
  flux for every physics module.
* [HLLD flux for MHD](hlld.md) — Miyoshi-Kusano (2005) five-state
  solver for ideal MHD.

## Limiters and positivity preservers

* [Moe-Rossmanith slope limiter](moe-rossmanith.md) — conserved-
  variable θ-shrink against neighbour bounds.
* [Zhang-Shu positivity preserver](positivity.md) — cell-mean-
  preserving θ-shrink enforcing ρ > 0 and P > 0.

## Divergence cleaning

* [Dedner GLM cleaning](dedner-glm.md) — generalised Lagrange-
  multiplier transport of $\nabla\cdot\mathbf{B}$ for ideal MHD.

## Time integration

* [SSPRK3](ssprk3.md) — three-stage Shu-Osher strong-stability-
  preserving RK3.
* [Dormand-Prince](dormand-prince.md) — embedded 5(4) RK with adaptive
  $\Delta t$.
