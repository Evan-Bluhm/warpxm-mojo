# Repository tour

The shortest possible introduction to where things live.

```text
warpxm-mojo/
├── CLAUDE.md             # code-editing rules: layout, style, kernel performance
├── README.md             # public landing page
├── pyproject.toml        # pixi config (Mojo + numpy + docs feature, tasks)
├── zensical.toml         # this documentation site's config
├── bench/                # benchmark drivers and BASELINE.md (history, ncu notes)
├── profiling/            # drivers used under ncu / nsys
├── docs/                 # documentation Markdown sources
├── examples/             # driver programs, one per problem
├── scripts/              # run_tests.sh, build_examples.sh, run_example.sh,
│                         # gll_tables.py (GLL table generator)
├── src/warpxm/
│   ├── prelude.mojo      # re-exports everything not tied to one physics
│   ├── physics/          # one re-export module per physics
│   ├── core/             # types, math helpers, BC kinds
│   ├── basis/            # GLL nodal basis tables
│   ├── mesh/             # BlockMesh[NDIM, P], phase meshes, host-side helpers
│   ├── variable/         # DGState[NDIM, P, NC], PhaseState1D{1,2}V
│   ├── apps/             # physics: one file per system (advection, euler,
│   │                     # maxwell, imhd, imhd_glm, vlasovmaxwell)
│   ├── spatial/          # dg_core (shared core), dg_launch (launcher),
│   │   │                 # dg_<phys> wrappers, wave_speed (CFL)
│   │   ├── sources/      # pointwise source kernels, generic over NDIM
│   │   └── kinetic/      # Vlasov phase-space DG, moments, BGK, coupling
│   ├── temporal/         # SSPRK3 and DP45 (generic over NDIM), plasma and
│   │                     # kinetic steppers
│   ├── adjusters/        # Moe-Rossmanith limiter, Zhang-Shu positivity
│   ├── io/               # generic VTU writer, presets, FrameWriter
│   └── testing/          # ExactRiemann, check_convergence
└── tests/                # regression tests, test_<physics>_<problem>_<dim>.mojo
```

`pixi run` tasks: `test` (the suite), `build` (type-check every
example), `example <name>` (run one example; with no name it lists
them).

## How this maps to upstream WARPXM

| Upstream (C++) | This repo (Mojo) |
| -------------- | ---------------- |
| `src/apps/simple/advection.{h,cc}` | `src/warpxm/apps/advection.mojo` |
| `src/apps/5-moment/euler.h` | `src/warpxm/apps/euler.mojo` |
| `src/apps/maxwell/maxwell.{h,cc}` | `src/warpxm/apps/maxwell.mojo` |
| `src/apps/mhd/mhd.{h,cc}` | `src/warpxm/apps/imhd.mojo` |
| `src/apps/mhd/divergence_cleaning/` | `src/warpxm/apps/imhd_glm.mojo` |
| `src/dfem/spatial_solvers/dg/kokkos/ndg_kokkos.{h,cc}` | `src/warpxm/spatial/dg_core.mojo` (shared) + `src/warpxm/spatial/dg_<phys>.mojo` (per-physics wrappers) |
| `src/dfem/temporal_solvers/ssprk3.h` | `src/warpxm/temporal/ssprk3.mojo` + plasma and kinetic variants |
| `src/apps/vlasovmaxwell/` | `src/warpxm/apps/vlasovmaxwell.mojo` + `src/warpxm/spatial/kinetic/` |
| `src/dfem/variable_adjusters/limiter/kokkos/moe_rossmanith_kokkos.{h,cc}` | `src/warpxm/adjusters/moe_rossmanith.mojo` (generic over the dimension) |

If you know upstream, finding the equivalent Mojo file is almost
always a one-step lookup.

## Tests vs examples

* **Tests** in `tests/` are short, assertion-driven, and print just
  enough for a log to be useful. Run the suite with `pixi run test`, or
  one with `pixi run mojo run -I src tests/<file>.mojo`.
* **Examples** in `examples/` are driver programs meant to be read
  top to bottom. Many write VTU frames to `./output/<name>/` for
  ParaView. Run one with `pixi run example <name>`. They are not part of
  the test suite; `pixi run build` type-checks them.

Both are written against the same helpers (`mesh.fill`,
`mesh.integrate`, `stepper.run_until`, …) rather than the state buffer
layout.

Both can serve as "how do I do X" references — pick whichever is
closer to your use case.

## What's *not* here

* **No CMake / Meson / Makefile**. pixi resolves the toolchain;
  `pixi run mojo run -I src <file>.mojo` builds and runs (`-I src` puts
  the `warpxm` package on the module search path).
* **No runtime input parser**. See [Design departures](../concepts/design-departures.md).
* **No CPU implementation**. See [GPU architecture](../concepts/gpu-architecture.md).
* **No `.h` / `.hpp` files**. Mojo unifies declaration and definition.
