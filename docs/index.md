# warpxm-mojo

A GPU-only rewrite of [WARPXM](https://github.com/csumlak/warpxm) in
[Mojo](https://www.modular.com/mojo). WARPXM is a C++ framework developed
at the University of Washington's Computational Plasma Dynamics Laboratory
for solving hyperbolic conservation laws — particularly plasma fluid
equations — using nodal discontinuous Galerkin (DG) methods.

This rewrite targets **GPUs exclusively** and replaces the runtime input
file format with **compiled Mojo programs**: an "input file" is a small
Mojo driver that imports the solver modules and calls into them directly.
There is no JSON / INI / HDF5 input parser, and no Kokkos / non-Kokkos
backend split — there is a single device implementation for every piece
of physics.

## What's in the box

| Physics                       |  1D | 2D  | 3D  |
| ----------------------------- | :-: | :-: | :-: |
| Scalar advection              | ✅  | ✅  | ✅  |
| 5-moment Euler                | ✅  | ✅  | ✅  |
| Maxwell's equations           | ✅  | ✅  | ✅  |
| Single-fluid plasma           | ✅  | ✅  | ✅  |
| Two-species plasma            | ✅  | ✅  | ✅  |
| Ideal MHD                     | —   | ✅  | ✅  |
| MHD with Dedner GLM cleaning  | —   | ✅  | ✅  |

(1D MHD is the only physical gap — $B_x$ along a 1D mesh becomes
constant, leaving no useful 1D MHD problem.)

Plus:

* **Slope limiter:** Moe-Rossmanith conserved-variable limiter,
  NC-parameterised across all physics for 2D + 3D
  (`adjusters/moe_rossmanith.mojo`). 1D variant in
  `moe_rossmanith_1d.mojo` is Euler-only.
* **Positivity preservation:** Zhang-Shu cell-mean-preserving θ-shrink
  for Euler (1D / 2D / 3D) and MHD (2D / 3D). Keeps ρ > 0 and P > 0
  even on near-vacuum problems (Einfeldt 1-2-3, strong Sedov blast).
* **MHD flux:** Rusanov (default) and HLLD (Miyoshi-Kusano 2005,
  five-state) on both `DGImhdSolver{2,3}D` and
  `DGImhdGLMSolver{2,3}D`, selectable at compile time via
  `USE_HLLD=True`.
* **Time integration:** Three-stage SSPRK3 (Shu-Osher) and 5(4)
  Dormand-Prince — both dimension-agnostic and closure-driven
  (`DormandPrince45_{1,2,3}D[P, NC]`). DP45 takes 6–40× fewer steps
  than fixed-Δt SSPRK3 on smooth problems.
* **Output:** ParaView-readable `.vtu` writers for every 2D / 3D
  physics, with a `FrameWriter` that emits time-series + a `.pvd`
  collection.

## Where to go next

* **New to the project?** Start with [Getting Started → Installation](getting-started/installation.md)
  and [Run your first example](getting-started/first-example.md).
* **Want the design story?** See [Concepts → Design departures](concepts/design-departures.md).
* **Adapting a physics module?** [Physics](physics/index.md) has the
  per-system conserved-variable layout and the example drivers that
  exercise it.
* **Contributing?** Start at [Repository tour](contributing/repo-tour.md)
  and the [Adding a new physics module](contributing/adding-a-physics.md)
  recipe.

## Status

The framework is fully feature-complete across the physics × dimension
matrix above (1D MHD is the only intentional gap). 66 regression tests
pass, covering every (physics, dimension) combination plus convergence
verification at P=1/2/3. Ongoing work focuses on scale — see
[Roadmap](contributing/index.md#roadmap) for what's open.
