# Concepts

This section is the "why" behind warpxm-mojo. None of it is needed to
*run* an example — but it'll help if you want to read the code, extend
it, or trust the results.

## Pages in this section

* [Design departures from upstream](design-departures.md) — what's
  different from the C++ WARPXM, and what motivated each change.
* [Nodal DG basics](nodal-dg.md) — Gauss-Lobatto-Legendre nodes, the
  strong form, surface fluxes, periodic and non-periodic boundary
  conditions.
* [GPU architecture in Mojo](gpu-architecture.md) — how kernels map to
  the device, what `comptime` buys, why `has_accelerator()` is a
  compile-time precondition.
* [Time integration](time-integration.md) — SSPRK3 vs Dormand-Prince,
  source-coupled steppers for plasma, limiter / positivity-preserver
  insertion between stages.
