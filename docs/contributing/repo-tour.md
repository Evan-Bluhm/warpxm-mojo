# Repository tour

The shortest possible introduction to where things live.

```text
warpxm-mojo/
├── CLAUDE.md             # design + progress notes (developer-facing)
├── README.md             # public landing — points here
├── pyproject.toml        # pixi config (Mojo + numpy + docs feature)
├── zensical.toml         # this documentation site's config
├── docs/                 # documentation Markdown sources
├── examples/             # polished driver programs, one per problem
├── scripts/              # ancillary tooling (GLL table generator, etc.)
├── src/warpxm/
│   ├── core/             # types, math helpers, BC enum
│   ├── basis/            # GLL nodal basis tables
│   ├── mesh/             # BlockMesh{1,2,3}D
│   ├── variable/         # DGState{1,2,3}D[P, NC]
│   ├── apps/             # physics: Advection, Euler, Maxwell, IMHD, IMHD-GLM
│   ├── spatial/          # DG kernels (one per (physics, dimension)) + source kernels
│   ├── temporal/         # SSPRK3 + DP45 steppers (specialised variants)
│   ├── adjusters/        # Moe-Rossmanith limiter, Zhang-Shu positivity
│   └── driver/           # VTU writers, FrameWriter
└── tests/                # regression tests, one per validation problem
```

## How this maps to upstream WARPXM

| Upstream (C++) | This repo (Mojo) |
| -------------- | ---------------- |
| `src/apps/simple/advection.{h,cc}` | `src/warpxm/apps/advection.mojo` |
| `src/apps/5-moment/euler.h` | `src/warpxm/apps/euler_{1d,2d,3d}.mojo` |
| `src/apps/maxwell/maxwell.{h,cc}` | `src/warpxm/apps/maxwell_{2d,3d}.mojo` |
| `src/apps/mhd/mhd.{h,cc}` | `src/warpxm/apps/imhd_{2d,3d}.mojo` |
| `src/apps/mhd/divergence_cleaning/` | `src/warpxm/apps/imhd_glm_{2d,3d}.mojo` |
| `src/dfem/spatial_solvers/dg/kokkos/ndg_kokkos.{h,cc}` | `src/warpxm/spatial/dg_*_{1,2,3}d.mojo` (one per physics) |
| `src/dfem/temporal_solvers/ssprk3.h` | `src/warpxm/temporal/ssprk3.mojo` + variants |
| `src/dfem/variable_adjusters/limiter/kokkos/moe_rossmanith_kokkos.{h,cc}` | `src/warpxm/adjusters/moe_rossmanith_{1,2,3}d.mojo` |

If you know upstream, finding the equivalent Mojo file is almost
always a one-step lookup.

## Tests vs examples

* **Tests** in `tests/` are short, assertion-driven, and dump just
  enough output for a CI log to be useful. They live alongside the
  rest of the source and run with `pixi run mojo run -I src
  tests/<file>.mojo`.
* **Examples** in `examples/` are polished, well-commented driver
  programs intended to be read top-to-bottom by users. Many produce
  VTU output for ParaView. They are not part of the test suite.

Both can serve as "how do I do X" references — pick whichever is
closer to your use case.

## What's *not* here

* **No CMake / Meson / Makefile**. pixi resolves the toolchain;
  `pixi run mojo run -I src <file>.mojo` builds and runs.
* **No runtime input parser**. See [Design departures](../concepts/design-departures.md).
* **No CPU implementation**. See [GPU architecture](../concepts/gpu-architecture.md).
* **No `.h` / `.hpp` files**. Mojo unifies declaration and definition.
