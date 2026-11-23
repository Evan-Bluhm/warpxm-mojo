# API reference

Module-by-module summary of the public types and functions in
`src/warpxm/`. The pages are written by hand, so check the source when
a signature matters.

## Module map

The directory layout mirrors upstream WARPXM (see
[Repository tour](../contributing/repo-tour.md)).

```text
src/warpxm/
  prelude.mojo # Re-exports everything not tied to one physics
  physics/     # One re-export module per physics (app, constants,
               # solvers, adjusters, sources, VTU preset)
  core/        # Real-type alias, math helpers, BC kinds
  basis/       # GLL nodal basis tables (P = 1..4)
  mesh/        # BlockMesh[NDIM, P], PhaseMesh[XDIM, VDIM, P_x, P_v] (and 1D1V / 1D2V), the host-side
               # helpers (fill, visit_nodes, integrate, l2_error)
  variable/    # DGState[NDIM, P, NC], PhaseState[XDIM, VDIM, ...] (and 1D1V / 1D2V)
  apps/        # Physics modules, one file per physics (advection, euler,
               # maxwell, imhd, imhd_glm, vlasovmaxwell). The struct is
               # dimension-agnostic; the DG core supplies the normal
               # triples (1D passes (±1, 0, 0)).
  spatial/     # dg_core.mojo (the shared DG core), dg_launch.mojo (the
               # host-side launcher), dg_<phys>.mojo wrappers, all generic
               # over the dimension; wave_speed.mojo (CFL)
    sources/   # Pointwise source kernels, generic over the dimension
    kinetic/   # Vlasov phase-space DG, moments, BGK, current coupling
  temporal/    # SSPRK3 and Dormand-Prince, generic over the dimension;
               # coupled fluid + Maxwell and kinetic steppers
  adjusters/   # Moe-Rossmanith slope limiter (NC-parametric) and Zhang-Shu
               # positivity preservers (Euler, MHD), generic over the
               # dimension
  io/          # Generic VTU writer, per-physics presets, FrameWriter
  testing/     # ExactRiemann, check_convergence (shared by the tests)
```

## Pages

* [`prelude` and `physics`](prelude.md): the re-export modules a driver
  imports
* [`core`](core.md): types, math compatibility shims, BC constants
* [`basis`](basis.md): `NodalBasis1D[P]` (GLL nodes + weights, $P = 1..4$)
* [`mesh`](mesh.md): `BlockMesh[NDIM, P]` and the host-side helpers
* [`variable`](variable.md): `DGState[NDIM, P, NC]`
* [`apps`](apps.md): physics module structs and their static flux methods
* [`spatial`](spatial.md): DG spatial solvers, the launcher, CFL, source
  kernels
* [`temporal`](temporal.md): explicit RK steppers
* [`adjusters`](adjusters.md): limiters and positivity preservers
* [`io`](io.md): output writers
* [`testing`](testing.md): helpers shared by the regression tests

## Convention summary

* `comptime` parameters: polynomial order `P`, component count `NC`,
  mesh dimension, optional flags like `USE_HLLD`. Drive all
  specialisation.
* Static methods on `apps/*` structs: `internal_flux_*_static`,
  `numerical_flux_static` (Rusanov), `numerical_flux_hlld_static`
  (MHD only), `max_signal_speed_static` (Euler, MHD, GLM-MHD; used by
  `suggest_dt`). Used by the kernels.
* `NDIM` is a `comptime` parameter, always first: `BlockMesh[2, P]`,
  `DGState[2, P, NC]`, `DGEulerSolver[2, P]`,
  `MoeRossmanith[2, P, NC]`, `SSPRK3[2, P, NC]`. The kinetic (Vlasov)
  types take `[XDIM, VDIM, P_x, P_v]` (`PhaseMesh[1, 3, P_x, P_v]`); the
  original ones carry a dimension suffix instead (`1D1V`, `1D2V`).
* Index constants: each fixed-NC physics module (`euler`, `maxwell`,
  `imhd`, `imhd_glm`) exports `IDX_*` constants (e.g. `IDX_RHO`,
  `IDX_MX`, `IDX_BX`) and a `*_NC` constant. Single source of truth
  per physics — no per-dimension copies. The `advection` app is the
  exception: `Advection[NC: Int]` is NC-parametric, so it carries no
  module-level `ADVECTION_NC` or `IDX_*` symbols.
