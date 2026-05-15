# warpxm-mojo

A GPU-only rewrite of [WARPXM](https://github.com/csumlak/warpxm) in
[Mojo](https://www.modular.com/mojo). WARPXM is a C++ framework for
solving hyperbolic conservation laws — particularly plasma fluid
equations — using nodal discontinuous Galerkin methods, developed at
the University of Washington's Computational Plasma Dynamics Lab.

This rewrite targets GPUs exclusively (NVIDIA / AMD / Apple silicon)
and replaces the runtime input file format with **compiled Mojo
programs**: an "input file" is a Mojo driver that imports the solver
modules and calls into them directly.

## Quick start

```fish
git clone https://github.com/<your-fork>/warpxm-mojo.git
cd warpxm-mojo
pixi install
pixi run mojo run -I src examples/advection_1d.mojo
```

That last command runs a 1D advection convergence problem on the GPU.
If you see a near-machine-zero L2 error and `PASSED`, you're set up.

## Documentation

Full documentation site (concepts, physics, numerical methods,
tutorials, API reference, contributing guide) lives in `docs/` and is
built with [Zensical](https://zensical.org).

Preview locally:

```fish
pixi run --environment docs docs-serve
```

then open <http://localhost:8000>.

A static build (for hosting on GitHub Pages, Netlify, etc.):

```fish
pixi run --environment docs docs-build
# output in ./site/
```

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

* Moe-Rossmanith slope limiter and Zhang-Shu positivity preserver
  (Euler 1D / 2D / 3D; MHD 2D / 3D).
* Rusanov flux everywhere and HLLD flux for MHD (2D and 3D, selectable
  at compile time via `DGImhdSolver{2,3}D[P, USE_HLLD=True]`).
* SSPRK3 time integration plus Dormand-Prince adaptive RK45 (1D
  generic, 2D Euler).
* ParaView VTU writers for every 2D / 3D physics, with a `FrameWriter`
  for time-series + `.pvd` collection.

49 regression tests, all passing. See `docs/contributing/index.md` for
the open roadmap items.

## Status

The framework is feature-complete across the physics × dimension
matrix above. Ongoing work focuses on quality and scale; see the
[Contributing → Roadmap](docs/contributing/index.md) page.

## License

TBD — please open an issue if license clarity blocks adoption.
