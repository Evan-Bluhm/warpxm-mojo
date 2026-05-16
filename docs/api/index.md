# API reference

Module-by-module summary of the public types and functions in
`src/warpxm/`. Hand-written — Mojo doesn't yet ship docstring
extraction tooling, so this page set evolves manually as the library
changes.

## Module map

The directory layout mirrors upstream WARPXM (see
[Repository tour](../contributing/repo-tour.md)).

```text
src/warpxm/
  core/        # Real-type alias, math helpers, BC enum
  basis/       # GLL nodal basis tables (P = 1..4)
  mesh/        # BlockMesh{1,2,3}D
  variable/    # DGState{1,2,3}D[P, NC]
  apps/        # Physics modules — one canonical file per physics
               # (advection, euler, maxwell, imhd, imhd_glm). The struct
               # is dimension-agnostic; the per-dim DG kernel supplies
               # the right normal triples (e.g. 1D passes (±1, 0, 0)).
  spatial/     # Shared dg_core_{1,2,3}d.mojo templates + thin per-physics
               # dg_<phys>_<N>d.mojo wrappers. Plus source kernels.
  temporal/    # SSPRK3 / Dormand-Prince steppers — all dimension-agnostic
               # via the closure pattern.
  adjusters/   # Moe-Rossmanith slope limiter (1D Euler-only + NC-parametric
               # 2D/3D) and Zhang-Shu positivity preserver
               # (Euler 1D/2D/3D, MHD 2D/3D — there's no 1D MHD).
  driver/      # VTU writers + FrameWriter
```

## Pages

* [`core`](core.md) — types, math compatibility shims, BC constants
* [`basis`](basis.md) — `NodalBasis1D[P]` (GLL nodes + weights, $P = 1..4$)
* [`mesh`](mesh.md) — `BlockMesh{1,2,3}D[P]`
* [`variable`](variable.md) — `DGState{1,2,3}D[P, NC]`
* [`apps`](apps.md) — physics module structs and their static flux methods
* [`spatial`](spatial.md) — DG spatial solvers and source kernels
* [`temporal`](temporal.md) — explicit RK steppers
* [`adjusters`](adjusters.md) — limiters and positivity preservers
* [`driver`](driver.md) — output writers

## Convention summary

* `comptime` parameters: polynomial order `P`, component count `NC`,
  mesh dimension, optional flags like `USE_HLLD`. Drive all
  specialisation.
* Static methods on `apps/*` structs: `internal_flux_*_static`,
  `numerical_flux_static` (Rusanov), `numerical_flux_hlld_static`
  (MHD only). Used by DG kernels.
* Index constants: each physics module exports `IDX_*` constants
  (e.g. `IDX_RHO`, `IDX_MX`, `IDX_BX`) and an `NC` constant. Single
  source of truth per physics — no per-dimension copies.
