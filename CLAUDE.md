# warpxm-mojo

A GPU-only rewrite of [WARPXM](https://github.com/csumlak/warpxm) in Mojo. The
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
  apps/        # physics apps (advection, euler, maxwell, mhd, glm-mhd)
  adjusters/   # slope limiters (Moe-Rossmanith for 1D, 2D, 3D)
  basis/       # GLL nodal basis (P = 1..4)
  core/        # real-type alias, math helpers, BC enum
  driver/      # VTU writers (Euler 2D/3D, MHD 2D/3D, Maxwell 2D/3D) + FrameWriter for time series
  mesh/        # BlockMesh{1,2,3}D with periodic + per-side BC catalog
  spatial/     # DG kernels per (app, dimension); source kernels (gravity, Lorentz, current, GLM damping)
  temporal/    # SSPRK3 steppers per (app, dimension), with optional limiter / GLM / source variants
  variable/    # DGState{1,2,3}D[P, NC] device buffers
examples/      # polished driver programs (one .mojo per physics setup)
tests/         # 49 regression tests covering every (physics, dimension) combination
scripts/       # tooling (GLL table generator, etc.)
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
| Maxwell                    | —   | ✅  | ✅  |
| Single-fluid plasma        | —   | ✅  | ✅  |
| Two-species plasma         | —   | ✅  | ✅  |
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
| 2D Euler hydrostatic balance    | \|v\| ~ 4e-16 after 50 steps (well-balanced source)     |
| 2D Sod tube                     | analytic plateaus match to 3e-3                         |
| 2D pair-plasma Langmuir         | mass conserved per species to machine epsilon           |
| 2D Maxwell PEC cavity TM₁₁      | unexcited components stay at 0.0 exactly                |
| 2D MHD Brio-Wu                  | 1D-extruded into 2D, y-drift 6.5e-14                    |
| 2D Orszag-Tang vortex           | mass drift 6e-14, ψ² bounded                            |
| 2D GLM ψ-bump decay             | 45000× decay over 4 time units                          |
| 3D Alfvén wave                  | B_y / B_z L2 4e-6 bit-identical; ρ-drift 3e-5           |
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
| Adaptive DP45 (2D entropy wave) | 199 steps for t=1, dt 4.4e-3..1e-2, ρ L2 7e-7          |

Plus 25 more in `tests/`. Run the suite via:
```fish
for t in tests/*.mojo; do pixi run mojo run -I src "$t"; done
```

## Remaining work

Mostly *quality* and *scale* items, not feature gaps:

1. **Primitive-bound limiting** — Zhang-Shu positivity preservers shipped
   for Euler (1D / 2D / 3D) and ideal MHD (2D / 3D). 1D MHD doesn't exist
   in the framework so there's no analogue there. The MHD preserver also
   works with GLM-MHD (NC=9 ψ component is shrunk along with the rest).
2. **Strang-style symmetric source splitting** for two-species plasma — the
   current first-order splitting introduces O(δ·dt) antisymmetry residual.
3. **Kinetic Vlasov-Maxwell solver** — different architecture (phase-space DG
   instead of configuration-space DG). Sizeable port.
4. **MPI halo exchange** — multi-GPU support. Sizeable engineering effort.
5. **Adaptive RK45 (Dormand-Prince)** — shipped for 1D generic DG fields
   and 2D Euler (`temporal/dormand_prince_1d.mojo` and
   `temporal/dormand_prince_euler_2d.mojo`). Fixed-dt `step` and adaptive
   `step_adaptive` (embedded 4th-order estimate, dt clipped to [0.2·dt,
   5·dt], atol/rtol scaled RMS norm). On smooth problems takes ~10× fewer
   steps than fixed-dt SSPRK3 at the same tolerance. 3D port and other 2D
   physics (MHD, Maxwell, etc.) still to do.
6. **HLLD numerical flux for MHD** — shipped in 2D and 3D
   (`DGImhdSolver{2,3}D[P, USE_HLLD=True]`, Miyoshi-Kusano 2005 five-state
   solver). 3D version uses a permutation-based core
   (`_hlld_core_3d` in `apps/imhd_3d.mojo`) so all three axis-aligned face
   directions share one ~150-line implementation. Default path stays
   Rusanov (no regression).

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

When adding a new (physics, dimension) combination, the proven recipe is:

1. App: write `apps/<phys>_<N>d.mojo` with `internal_flux_{x,y[,z]}_static` and
   a Rusanov / central+LF `numerical_flux_static` taking a unit normal.
2. DG kernel: clone `spatial/dg_<existing>_<N>d.mojo`, swap the flux helpers.
3. Stepper: clone `temporal/ssprk3_<existing>_<N>d.mojo`, swap the spatial
   solver type.
4. BCs: add a `_apply_bc_<phys>_<N>d` helper, thread `face_bc` through the
   kernel, dispatch on the BC kind at each ghost-sentinel branch.
5. Test: clone an analogous (phys, N±1) test, adjust the IC, run.

Source-term kernels and the slope limiter are NC-parameterized — they don't
care about the physics, so a new app reuses them with the right NC value.
