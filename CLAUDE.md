# warpxm-mojo

A GPU-only rewrite of [WARPXM](https://github.com/UW-Computational-Plasma-Group/warpxm) in Mojo. The
reference C++ codebase lives at `~/GitHub/warpxm`.

## Design departures from upstream WARPXM

1. **GPU only.** No Kokkos / non-Kokkos split, no CPU spatial solver, no
   runtime backend dispatch. There is a single device implementation for every
   piece of physics. We always assume a GPU is available — `has_accelerator()`
   is a compile-time precondition, not a runtime branch.
2. **Compiled input files.** Upstream WARPXM ships an executable that loads
   Python-generated input descriptions at runtime. Here, an input file *is* a
   Mojo program that imports the solver modules and calls into them directly.
   No runtime input parser, no JSON/INI/HDF5 input deserialization layer, no
   `WxCryptSet`. The executable for a given physics problem is built by
   compiling its driver file.

## Layout

```
src/warpxm/
  apps/        # One file per physics: advection, euler, maxwell, imhd,
               # imhd_glm. Each exposes axis-parametric fluxes and a
               # 3-component-normal numerical flux; the DG launcher owns
               # dimensionality. See "Apps architecture" below.
  adjusters/   # Moe-Rossmanith slope limiter (1D / 2D / 3D) and Zhang-Shu
               # positivity preservers (Euler 1D/2D/3D, MHD 2D/3D).
  basis/       # GLL nodal basis (P = 1..4)
  core/        # real-type alias, math helpers, BC enum
  driver/      # VTU writers (Euler 2D/3D, MHD 2D/3D, Maxwell 2D/3D) + FrameWriter for time series
  mesh/        # BlockMesh{1,2,3}D with periodic + per-side BC catalog
  spatial/     # dg_core_{1,2,3}d.mojo: shared DG cores. Per-physics dg_<phys>_<N>d.mojo
               # wrappers supply flux/BC closures. Plus source kernels (gravity, Lorentz,
               # current, GLM damping).
  temporal/    # ssprk3_{1,2,3}d.mojo: one generic SSPRK3 stepper per dimension. Driver
               # supplies compute_rhs and adjust closures. dormand_prince_{1,2,3}d.mojo
               # for adaptive RK45. ssprk3_plasma_{1,2,3}d.mojo and
               # ssprk3_twofluid_{1,2,3}d.mojo for two-species and single-fluid plasma
               # (multi-state coupled steppers).
  variable/    # DGState{1,2,3}D[P, NC] device buffers
examples/      # polished driver programs (one .mojo per physics setup)
tests/         # 70 regression tests covering every (physics, dimension) combination
scripts/       # tooling: gll_tables.py (GLL basis table generator),
               # run_tests.sh (pixi `test` task), build_examples.sh
               # (pixi `build` task).
```

The subtree organisation mirrors upstream's `src/apps/`,
`src/dfem/variable_adjusters/`, `src/dfem/spatial_solvers/`, and
`src/dfem/temporal_solvers/` directories.

## Build / run

Mojo + helpers are pinned via pixi:

```fish
pixi install
pixi run mojo run -I src tests/test_advection_1d.mojo
pixi run mojo run -I src examples/imhd_orszag_tang_2d_vtu.mojo
```

`pixi run mojo run -I src <file>` is the canonical incantation: `-I src` adds
the `warpxm` package to the module search path. Every test and example uses
the same form.

## Feature matrix

All cells with a check have an end-to-end regression test passing on the GPU.

|                            |  1D | 2D  | 3D  |
| -------------------------- | :-: | :-: | :-: |
| Advection                  | ✅  | ✅  | ✅  |
| 5-moment Euler             | ✅  | ✅  | ✅  |
| Maxwell                    | ✅  | ✅  | ✅  |
| Single-fluid plasma        | ✅  | ✅  | ✅  |
| Two-species plasma         | ✅  | ✅  | ✅  |
| Ideal MHD                  | —   | ✅  | ✅  |
| MHD + Dedner GLM cleaning  | —   | ✅  | ✅  |
| Moe-Rossmanith limiter     | ✅  | ✅  | ✅  |
| ParaView VTU + .pvd output | —   | ✅  | ✅  |
| Non-periodic BCs           | ✅  | ✅  | ✅  |

Supporting infrastructure:

- **BCs**: `BC_PERIODIC` (implicit), `BC_COPYOUT`, `BC_WALL_FREESLIP`,
  `BC_PEC_WALL` (Maxwell, MHD), `BC_PMC_WALL` (Maxwell). Per-axis selection in
  the mesh constructor.
- **GLL polynomial orders**: P = 1, 2, 3, 4 tables; convergence rates of
  P+1 verified for P=1, 2, 3 on the 1D advection convergence test.
- **Time integration**: 3-stage Shu-Osher SSPRK3, device-resident, with
  optional Moe-Rossmanith limiter inter-stage application.
- **Sources**: gravity (Euler), Lorentz force (Maxwell → fluid), current
  density (fluid → Maxwell), parabolic ψ damping (GLM-MHD). Reusable
  generic-NC infrastructure.

## Notable validations

| Test                            | Quantitative result                                     |
| ------------------------------- | ------------------------------------------------------- |
| 1D advection convergence        | observed rates 1.98 / 3.01 / 3.99 (theoretical P+1)     |
| 2D advection convergence        | observed rates 1.90 / 3.03 / 4.00 (theoretical P+1)     |
| 3D advection convergence        | observed rates 3.12 / 3.99 at P=2,3 (theoretical P+1)   |
| 1D Maxwell plane-wave conv      | observed rates 1.98 / 3.01 / 4.00 (theoretical P+1)     |
| 1D Euler entropy-wave conv      | observed rates 1.99 / 2.95 / 4.00 (theoretical P+1)     |
| 2D MHD Alfvén-wave conv         | observed rates 1.98 / 2.99 / 4.00 (theoretical P+1)     |
| 3D MHD Alfvén-wave conv         | observed rates 2.98 / 4.02 at P=2,3 (theoretical P+1)   |
| 1D1V Vlasov free-streaming conv | observed rates 1.78 / 3.10 / 4.29 (theoretical P+1)     |
| 1D2V Vlasov free-streaming conv | observed rates 1.97 / 3.00 / 3.99 (theoretical P+1)     |
| 2D Euler hydrostatic balance    | \|v\| ~ 4e-16 after 50 steps (well-balanced source)     |
| 2D Sod tube                     | analytic plateaus match to 3e-3                         |
| 1D Langmuir (single-fluid)      | mass drift 2.73e-13, ρ at sample within 7e-6 of IC after period |
| 1D pair-plasma Langmuir         | ρ values match 2D-extruded result to 14 sig figs; mass drift 1.9e-13 per species |
| 2D pair-plasma Langmuir         | mass conserved per species to machine epsilon           |
| Pair-plasma asym δ-scaling      | ratio 100.0 / 100.0 between δ ∈ {0.01, 0.001, 0.0001} — physical O(δ²) |
| 2D Maxwell PEC cavity TM₁₁      | unexcited components stay at 0.0 exactly                |
| 2D MHD Brio-Wu                  | 1D-extruded into 2D, y-drift 2.8e-13                    |
| 2D Orszag-Tang (upstream WARPXM units) | mass drift 6e-14, ψ² bounded — domain [0, 2π]², ρ₀=γ², p₀=γ |
| 2D Orszag-Tang (Athena units)   | ρ ∈ [0.047, 0.547] at t=0.3 (Athena ref), mass drift 8e-14, ψ² 6e-5 |
| 3D Orszag-Tang (Athena, z-extruded) | z-drift 0.0 exactly; ρ + P + ψ² match 2D bit-for-bit, mass drift 1.7e-12 |
| 2D GLM ψ-bump decay             | 45000× decay over 4 time units                          |
| 3D Alfvén wave (SSPRK3)         | B_y / B_z L2 2.4e-5 bit-identical; ρ-drift 2.2e-4       |
| 3D MHD blast                    | ∫B_x dV preserved to 4e-15 (Faraday's law)              |
| 3D cyclotron orbit              | ΔE = 3e-13 (magnetic force does no work)                |
| 1D Einfeldt 1-2-3 problem       | near-vacuum centre survives (min ρ 1e-3, min P 6e-3)   |
| 2D Einfeldt 1-2-3 problem       | y-extruded 1D; y-drift 0 exactly; min ρ 3e-4, min P 9e-4 |
| 3D Einfeldt 1-2-3 problem       | y/z-extruded; both drifts 0; min ρ 2e-3, min P 6e-3    |
| 2D strong MHD blast (ΔP=100)    | P_gas clamped to p_eps; mass / ∫B_x preserved 2e-13    |
| 3D strong MHD blast (ΔP=100)    | P_gas clamped to p_eps; mass / ∫B_x preserved 2e-13    |
| 2D Brio-Wu (HLLD)               | passes with same far-field tolerances as Rusanov path  |
| 3D Brio-Wu (HLLD)               | exact y/z-extrusion (drifts 0), passes Rusanov tolerances |
| Adaptive DP45 (smooth advection)| 187 steps (vs ~2000 SSPRK3), L2 1.2e-7, dt up to 0.008  |
| Adaptive DP45 (2D entropy wave) | 200 steps for t=1, dt 5.3e-4..1e-2, ρ L2 6.6e-7        |
| Adaptive DP45 (2D MHD Alfvén)   | 169 steps for t=1 (vs 1000 SSPRK3), L2 ~4.3e-6 on B/u  |
| 1D Maxwell PEC cavity TM        | L2 E_z 8e-6 after one period, unexcited components 0.0 |
| Adaptive DP45 (1D Maxwell)      | 52 steps for t=1 (vs ~2000 SSPRK3), L2 1e-4, transv 0.0 |
| Adaptive DP45 (2D Maxwell)      | 76 steps for t=1/√2 (vs ~1414 SSPRK3), L2 ~1e-4 on E/B |
| Adaptive DP45 (3D advection)    | 64 steps for t=1, dt 0.008..0.016, L2 1.76e-3          |
| Adaptive DP45 (3D MHD Alfvén)   | 131 steps for t=1 (vs 1000 SSPRK3), L2 ~2.4e-5 on B/u  |
| Adaptive DP45 (3D Maxwell)      | 50 steps for t=1 (vs ~2000 SSPRK3), L2 ~8e-4 on E/B    |

**Two Orszag-Tang setups, on purpose.** `test_imhd_orszag_tang_2d`
uses the upstream WARPXM convention (domain `[0, 2π]²`, `ρ₀ = γ²`,
`p₀ = γ`) so any port-time regression shows up against the C++
reference. `test_imhd_orszag_tang_2d_athena` uses the Athena
convention (domain `[0, 1]²`, `ρ₀ = 25/(36π)`, `p₀ = 5/(12π)`,
`v₀ = 1`, `B₀ = 1/√(4π)`) so results compare directly with the
canonical reference at
<https://www.astro.princeton.edu/~jstone/Athena/tests/orszag-tang/>
and the broader MHD-code literature that adopts Athena's form. The
3D variant only exists in Athena units (z-extrusion of the Athena
2D IC).

Plus 40+ more in `tests/` (94 tests total). Run the suite via:
```fish
pixi run test
# or, for a single test:
pixi run mojo run -I src tests/test_advection_1d.mojo
```

## Open work

Feature-matrix-complete (modulo 1D MHD, an intentional physical gap).
Remaining items are scale and performance:

1. **Kinetic Vlasov-Maxwell solver** — different architecture
   (phase-space DG instead of configuration-space DG). Sizeable port.
2. **MPI halo exchange** — multi-GPU support. Sizeable engineering.

## Capability inventory

What's shipped (so it's easy to see at a glance):

- **Primitive-bound limiting** — Zhang-Shu positivity preservers for
  Euler (1D / 2D / 3D) and ideal MHD (2D / 3D). MHD preserver also
  works with GLM-MHD (NC=9 ψ component shrunk along with the rest).
- **Slope limiter** — Moe-Rossmanith conserved-variable limiter,
  NDIM-parametric (2D + 3D) and NC-parameterised. 1D variant
  (Euler-only) lives in `adjusters/moe_rossmanith_1d.mojo`.
- **Adaptive RK45 (Dormand-Prince)** — NC-parameterised generic
  steppers for all three dimensions
  (`DormandPrince45_{1,2,3}D[P, NC]`). Fixed-dt `step` and adaptive
  `step_adaptive` (embedded 4th-order estimate, dt clipped to
  [0.2·dt, 5·dt], atol/rtol scaled RMS norm). On smooth problems
  takes 6–40× fewer steps than fixed-dt SSPRK3. Per-step error norm
  is reduced fully on the GPU via a single-block tree-reduction
  kernel (`_dp45_sum_sq_kernel` in `temporal/dormand_prince_1d.mojo`,
  reused by 2D and 3D), so only one scalar lands on the host per
  step — no `map_to_host` of the full state buffers.
- **HLLD numerical flux** — shipped on `DGImhdSolver{2,3}D` and
  `DGImhdGLMSolver{2,3}D` (`USE_HLLD=True`, Miyoshi-Kusano 2005
  five-state solver). Permutation-based core
  (`_hlld_core_3d` in `apps/imhd.mojo`) handles all three
  axis-aligned face directions in ~215 lines. Default path stays
  Rusanov (no regression). GLM-MHD path composes IMHD HLLD with the
  exact (B_n, ψ) 2x2 Riemann block (Dedner et al. 2002 §3.2;
  Mignone & Tzeferacos 2010 §4).
- **Pair-plasma diagnostic** — the 1.85e-3 antisymmetry residual on
  the 2D pair-plasma test was originally listed as a Strang-splitting
  motivation. `tests/test_pair_plasma_asym_scaling.mojo` shows the
  residual scales as δ² (ratio 100 between successive δ values),
  confirming it's the *physical* O(δ²) coupling from the ρ-dependent
  Lorentz force `F = α·ρ·E` — species 1 has ρ ≈ 1+δ·sin, species 2
  has ρ ≈ 1−δ·sin, so their force amplitudes differ by 2·δ². No
  operator-splitting fix can eliminate it; closed as "physical, not
  algorithmic."

## Key references in upstream

- `src/dfem/spatial_solvers/dg/kokkos/ndg_kokkos.{h,cc}` — physical-space DG
  solver, kernel-launcher registry pattern.
- `src/apps/simple/advection.{h,cc}` — simplest reference app.
- `src/apps/5-moment/euler.h` — Rusanov flux template I followed.
- `src/apps/mhd/divergence_cleaning/` — Dedner MixedGLM source.
- `src/dfem/variable_adjusters/limiter/kokkos/moe_rossmanith_kokkos.{h,cc}` —
  limiter reference.
- `~/GitHub/warpxm/CLAUDE.md` — long-form architecture notes.

## Style notes

Use the in-repo Mojo skills (`mojo-syntax`, `mojo-gpu-fundamentals`) whenever
writing Mojo. The most common pretraining-era mistakes:

- `alias` → `comptime`
- `fn` → `def` (with explicit `raises` where needed)
- `inout` → `mut`
- `@parameter if/for` → `comptime if/for`
- `List[T](a, b, c)` → bracket literals `[a, b, c]`
- `from collections import …` → `from std.collections import …`
- Struct-parameter access inside a struct → `Self.T`, not bare `T`

Mojo 1.2.0 specifics (toolchain is pinned to `mojo ==1.2.0` +
`max-core ==26.7.0` from the stable `max` channel):

- `InlineArray` → `Array`. It is `Copyable` but not `ImplicitlyCopyable`:
  return locals with `return out^`, copy borrowed arguments with
  `q.copy()`, and bring comptime tables to runtime with
  `materialize[_TABLE]()`.
- `std.gpu` → `max.gpu` (`max.gpu.host`, `max.gpu.memory`, `max.gpu.sync`).
  `layout` and `max` come from the `max-core` conda package, not `mojo`.
- Closures are unified closures: a nested `def` with a capture list,
  passed as an ordinary trailing argument. `@parameter` / `@__parameter`
  and `capturing [_]` parameters are gone.
  - Host side: `def adjust(mut q: ...) raises {mut limiter, imm}:` then
    `stepper.step(ctx, q, dt, compute_rhs, adjust)`. Capture `mut` only what
    the closure mutates; two closures passed to one call may not both
    capture the same object mutably (that is why solver / adjuster methods
    take `mesh` read-only).
  - Kernel side: `def flux_x(qn: Array[Real, NC]) {imm} -> Array[Real, NC]:`
    passed as `_dg_<N>d_compute_rhs_core[P, NC, ...layouts, EB](..., flux_x,
    num_flux, apply_bc)`.
  - Accepting one: a type parameter bounded by the closure type plus an
    argument of that type — `def step[ComputeRhs: def(mut S, mut S) raises
    -> None](..., compute_rhs_fn: ComputeRhs)`.
- `Int` is not `DevicePassable`. Kernel scalar arguments are `Int64`
  (`size_i64: Int64`, then `var size = Int(size_i64)` in the body) and
  launch sites pass `Int64(size)`.
- `row_major(Idx(n))` → `row_major(Int(n))`.
- Prefer `comptime match` over `comptime if / elif` chains on a parameter.
- `NodalBasis1D.nodes()/weights()/diff_matrix_flat()` are `@inline(.never)` to
  dodge a 1.2.0 optimizer bug (an `Array` returned by an inlined function
  reads back as garbage in host code after a `ctx.enqueue_copy`). See the
  note in `basis/nodal_basis_1d.mojo` before removing it.

### Apps architecture

Each of the five core physics packages (advection, euler, maxwell, imhd,
imhd_glm) is a single file with a dimension-agnostic struct. The app
exposes:

- `internal_flux_x_static`, `internal_flux_y_static`, `internal_flux_z_static`
  (or an axis-parametric `internal_flux_axis_static[axis]` where it pays off
  — advection, euler).
- `numerical_flux_static(q_l, q_r, nx, ny, nz, ...physics-params...)` taking a
  3-component unit normal. 1D callers pass `(±1, 0, 0)`; 2D callers pass
  `(nx, ny, 0)`. Unused lanes multiply by zero and are folded away.
- Optionally `numerical_flux_hlld_static(...)` with the same shape (currently
  IMHD and GLM-MHD; uses axis-aligned permutation dispatch into
  `_hlld_core_3d`).

This mirrors upstream WARPXM's pattern: the *flux launcher* owns
dimensionality; the *app* operates on local node values. Per-dim DG kernel
wrappers (`dg_<phys>_<N>d.mojo`) still exist because mesh layouts are
dimension-specific, but they call into the shared app file. Callers import
the canonical struct directly: `from warpxm.apps.euler import Euler`,
`from warpxm.apps.imhd import IMHD`, etc.

### Adding a new physics

1. **App**: write `apps/<phys>.mojo` with `internal_flux_{x,y,z}_static`
   methods on the struct and a `numerical_flux_static(q_l, q_r, nx, ny, nz,
   ...)` taking a 3-component unit normal. Module-level helpers
   (`_rho_floored`, `_gas_pressure_floored`, etc.) for the physics-specific
   conversions.
2. **DG kernel**: write `spatial/dg_<phys>_<N>d.mojo` for each dimension you
   need as a thin wrapper over the shared `_dg_<N>d_compute_rhs_core` (in
   `spatial/dg_core_<N>d.mojo`). The wrapper declares `def ... {imm}`
   closures that dispatch to the app's static methods and pass the right
   normal triples (e.g. 2D passes `(nx, ny, 0)`):
   - 1D: `flux_x`, `num_flux`, `apply_bc` (three closures);
   - 2D: + `flux_y` (four closures);
   - 3D: + `flux_z` (five closures).
3. **Stepper**: nothing new. The generic `SSPRK3_<N>D[P, NC]` and
   `DormandPrince45_<N>D[P, NC]` already handle any single-state problem.
   Compose the limiter / positivity preserver / source kernels at the call
   site via the driver's `adjust` and `compute_rhs` closures.
4. **BCs**: add a `_apply_bc_<phys>` helper in your DG-kernel wrapper. It
   takes interior `q_n`, `bc_kind`, and `face_axis`, returns the ghost-side
   state. The shared DG core invokes it at every boundary face.
5. **Test**: clone an analogous (phys, N±1) test, adjust the IC, run.

Source-term kernels (gravity, Lorentz, GLM damping, etc.) and the slope
limiter are NC-parameterised — a new app reuses them with the right NC
value.
