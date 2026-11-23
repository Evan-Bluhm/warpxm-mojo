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
pixi run example advection_1d
```

That last command runs a 1D periodic advection problem on the GPU
and prints the L2 error after one period (~1e-5 at default settings).
If the L2 line appears, you're set up. (For an actual convergence
study across $P=1,2,3$, run
`examples/convergence_advection_1d.mojo`.)

Then:

```fish
pixi run build              # type-check every example (~30s)
pixi run test               # full regression suite, in parallel (~1 min)
pixi run example            # list the examples; add a name to run one
```

The first two should report no failures on a working install.

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
| Maxwell's equations           | ✅  | ✅  | ✅  |
| Single-fluid plasma           | ✅  | ✅  | ✅  |
| Two-species plasma            | ✅  | ✅  | ✅  |
| Ideal MHD                     | —   | ✅  | ✅  |
| MHD with Dedner GLM cleaning  | —   | ✅  | ✅  |

(1D MHD is the only physical gap — the lowest-rank MHD problem
requires a perpendicular B field that doesn't fit a 1D mesh.)

There is also a continuum-kinetic **Vlasov-Maxwell** solver (phase-space
DG), generic over the phase-space dimensions (1D1V, 1D2V, 1D3V, 2D2V
tested), with moments, a BGK collision operator (1D1V) and
self-consistent coupling to Maxwell.

Plus:

* Moe-Rossmanith slope limiter (`MoeRossmanith[NDIM, P, NC]`, any
  physics, any dimension).
* Zhang-Shu positivity preserver (Euler 1D / 2D / 3D; MHD 2D / 3D).
* Rusanov flux everywhere; Miyoshi-Kusano HLLD for MHD and GLM-MHD
  (2D and 3D, selectable at compile time via
  `DGImhdSolver[NDIM, P, USE_HLLD=True]` and
  `DGImhdGLMSolver[NDIM, P, USE_HLLD=True]`).
* SSPRK3 time integration plus Dormand-Prince adaptive RK45 — both
  closure-driven and generic over the dimension (`SSPRK3[NDIM, P, NC]`,
  `DormandPrince45[NDIM, P, NC]`), with CFL-driven step selection
  (`solver.suggest_dt`). DP45 takes 6–40× fewer steps
  than fixed-Δt SSPRK3 on smooth problems.
* One generic ParaView `.vtu` writer (appended binary) with per-physics
  presets, and a `FrameWriter` for time series + `.pvd` collection.
* Host-side helpers on the meshes (`fill`, `integrate`, `l2_error`,
  `visit_nodes`) so a driver never indexes the state buffer itself.

The full matrix and capability notes are in [docs/features.md](docs/features.md); the
regression suite's quantitative results are in [docs/validation.md](docs/validation.md).

## Architecture

Each physics is a **single file** in `apps/` plus one thin DG wrapper in
`spatial/`. The shared DG core (`spatial/dg_core.mojo`) owns
dimensionality, and the app operates on local node values via a
3-component-normal numerical flux. Everything else is written once for
every dimension: the mesh (`BlockMesh[NDIM, P]`), the state
(`DGState[NDIM, P, NC]`), the solvers (`DGEulerSolver[NDIM, P]`, ...),
the time steppers, the limiter and positivity preservers, and the
pointwise source kernels. A new physics therefore works in 1D, 2D and 3D
as soon as its app and wrapper exist; see
`docs/contributing/adding-a-physics.md`.

## Status

The framework is feature-complete across the physics × dimension
matrix above (1D MHD is a physical-not-numerical skip). Ongoing work
focuses on quality and scale; see the
[Contributing → Roadmap](docs/contributing/index.md) page.

## License

TBD — please open an issue if license clarity blocks adoption.
