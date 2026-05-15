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
| Maxwell's equations           | —   | ✅  | ✅  |
| Single-fluid plasma           | —   | ✅  | ✅  |
| Two-species plasma            | —   | ✅  | ✅  |
| Ideal MHD                     | —   | ✅  | ✅  |
| MHD with Dedner GLM cleaning  | —   | ✅  | ✅  |

Plus:

* **Slope limiter:** Moe-Rossmanith conserved-variable limiter, NC-parameterised
  for all (physics, dimension) combinations.
* **Positivity preservation:** Zhang-Shu cell-mean-preserving θ-shrink that
  keeps ρ > 0 and P > 0 even on near-vacuum problems (Einfeldt 1-2-3,
  strong Sedov blast).
* **MHD flux:** Rusanov (default) and HLLD (Miyoshi-Kusano 2005, five-state)
  selectable at compile time.
* **Time integration:** Three-stage SSPRK3 (Shu-Osher); 5(4) Dormand-Prince
  with embedded error estimate and adaptive Δt.
* **Output:** ParaView-readable `.vtu` writers for every 2D / 3D physics,
  with a `FrameWriter` that emits time-series + a `.pvd` collection.

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
matrix above. Ongoing work focuses on quality and scale — see
[Roadmap](contributing/index.md#roadmap) for what's open.
