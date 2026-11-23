# warpxm-mojo

A GPU-only rewrite of [WARPXM](https://github.com/UW-Computational-Plasma-Group/warpxm) in
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

There is also a continuum-kinetic **Vlasov-Maxwell** solver (phase-space
DG), generic over the phase-space dimensions (1D1V up to 3D3V), with
moments, specular walls, a BGK collision operator and self-consistent
coupling to Maxwell.

Plus:

* **Slope limiter:** Moe-Rossmanith conserved-variable limiter,
  generic over the dimension and NC-parameterised across all physics
  (`MoeRossmanith[NDIM, P, NC]` in `adjusters/moe_rossmanith.mojo`).
* **Positivity preservation:** Zhang-Shu cell-mean-preserving θ-shrink
  for Euler (1D / 2D / 3D) and MHD (2D / 3D). Keeps ρ > 0 and P > 0
  even on near-vacuum problems (Einfeldt 1-2-3, strong Sedov blast).
* **MHD flux:** Rusanov (default) and HLLD (Miyoshi-Kusano 2005,
  five-state) on both `DGImhdSolver[NDIM, P]` and
  `DGImhdGLMSolver[NDIM, P]`, selectable at compile time via
  `USE_HLLD=True`.
* **Time integration:** Three-stage SSPRK3 (Shu-Osher) and 5(4)
  Dormand-Prince — both closure-driven and generic over the dimension
  (`SSPRK3[NDIM, P, NC]`, `DormandPrince45[NDIM, P, NC]`), with CFL-driven
  step selection (`solver.suggest_dt`). DP45 takes 6–40× fewer steps
  than fixed-Δt SSPRK3 on smooth problems.
* **Output:** one generic ParaView `.vtu` writer (appended binary) with
  per-physics presets, and a `FrameWriter` that emits time series + a
  `.pvd` collection.
* **Driver helpers:** `mesh.fill`, `integrate`, `l2_error` and
  `visit_nodes` for initial conditions and diagnostics, so a driver never
  indexes the state buffer itself.

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
matrix above (1D MHD is the only intentional gap). The regression suite
covers every (physics, dimension) combination plus convergence
verification at P=1/2/3: see [Features](features.md) for the full matrix
and open work, and [Validation](validation.md) for the measured results.
