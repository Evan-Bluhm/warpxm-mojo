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
  prelude.mojo # Re-exports everything not tied to one physics (Real, BC kinds,
               # meshes, DGState, steppers, FrameWriter, write_vtu).
  physics/     # One re-export module per physics (advection, euler, maxwell,
               # imhd, imhd_glm, plasma, vlasov): app, index constants,
               # solvers, adjusters, sources, VTU preset. A driver needs
               # `from warpxm.prelude import *` and one of these.
  apps/        # One file per physics: advection, euler, maxwell, imhd,
               # imhd_glm, vlasovmaxwell. Each exposes axis-parametric fluxes,
               # a 3-component-normal numerical flux and a signal-speed bound;
               # the DG launcher owns dimensionality. See "Apps architecture".
  adjusters/   # MoeRossmanith[NDIM, P, NC] slope limiter and Zhang-Shu
               # positivity preservers (PositivityEuler[NDIM, P],
               # PositivityMHD[NDIM, P, NC]).
  basis/       # GLL nodal basis (P = 1..4)
  core/        # real-type alias, math helpers, BC kinds
  io/          # vtu.mojo: one generic appended-binary VTU writer (2D / 3D).
               # vtu_presets.mojo: per-physics field lists. frame_writer.mojo:
               # time series + .pvd.
  mesh/        # block.mojo: BlockMesh[NDIM, P] (periodic + per-side BCs).
               # phase_mesh.mojo: PhaseMesh[XDIM, VDIM, P_x, P_v], the
               # generic phase mesh. phase_mesh_1d{1,2}v.mojo: the original
               # per-dimension ones. Meshes own the host-side helpers:
               # fill, visit_nodes, integrate, integrate_fn, l2_error.
  spatial/     # dg_core.mojo: the DG core, generic over NDIM. dg_launch.mojo:
               # the one host-side launcher. dg_<phys>.mojo: one wrapper per
               # physics (flux closures + BC function + solver struct, e.g.
               # DGEulerSolver[NDIM, P]). wave_speed.mojo: CFL.
    sources/   # Pointwise source kernels, generic over NDIM: gravity,
               # Lorentz, Maxwell current, GLM damping.
    kinetic/   # Vlasov phase-space DG, moments, BGK, current coupling.
               # dg_vlasov.mojo, vlasov_moments.mojo and
               # vlasov_current_source.mojo are generic over [XDIM, VDIM];
               # the *_1d1v / *_1d2v files are the original ones.
  temporal/    # ssprk3.mojo, dormand_prince.mojo: generic over NDIM.
               # ssprk3_plasma.mojo / ssprk3_twofluid.mojo: coupled fluid +
               # Maxwell. ssprk3_phase.mojo, ssprk3_vlasov_maxwell.mojo:
               # kinetic, generic over [XDIM, VDIM]. ssprk3_1d{1,2}v.mojo,
               # ssprk3_vlasov_maxwell_1d{1,2}v.mojo: the original kinetic
               # ones. Every stepper has `step` and `run_until`.
               # pacing.mojo: CflPacer, shared by the run_until_cfl loops.
  testing/     # ExactRiemann, check_convergence (shared by tests).
  variable/    # DGState[NDIM, P, NC]; PhaseState[XDIM, VDIM, P_x, P_v, NC];
               # PhaseState1D{1,2}V.
examples/      # polished driver programs (one .mojo per physics setup)
tests/         # regression tests, named test_<physics>_<problem>_<dim>.mojo
scripts/       # gll_tables.py (GLL table generator), run_tests.sh (pixi
               # `test`), build_examples.sh (`build`), run_example.sh (`example`).
```

The subtree organisation mirrors upstream's `src/apps/`,
`src/dfem/variable_adjusters/`, `src/dfem/spatial_solvers/`, and
`src/dfem/temporal_solvers/` directories.

## Build / run

Mojo + helpers are pinned via pixi:

```fish
pixi install
pixi run test                    # whole suite in parallel, about a minute
pixi run test euler 3d           # only tests whose name contains "euler" or "3d"
pixi run test -t                 # also list the ten slowest tests
pixi run build                   # type-check every example
pixi run example imhd_orszag_tang_2d_vtu   # run one (output in ./output/<name>/)
pixi run mojo run -I src tests/test_advection_convergence_1d.mojo
```

`pixi run mojo run -I src <file>` is the canonical incantation: `-I src` adds
the `warpxm` package to the module search path.

## Where the rest lives

This file holds what you need while editing code: layout, style and
performance rules, and the architecture conventions. Everything descriptive
is in `docs/` and is not repeated here:

- `docs/features.md`: feature matrix (fluid and kinetic), BCs, capability
  notes (limiter tolerance, DP45, HLLD), open work.
- `docs/validation.md`: quantitative results of the regression suite.
- `docs/api/`: per-package reference. `docs/tutorials/`: worked drivers.
- `bench/BASELINE.md`: benchmark history and the `ncu` profiling notes
  behind the kernel performance rules.

## Driver conventions

A driver (test or example) is a Mojo program. Write it against the helpers,
not the buffer layout:

```mojo
from warpxm.prelude import *
from warpxm.physics.euler import *

comptime State = DGState[2, P, EULER_NC]

def initial(x: Real, y: Real) -> Array[Real, EULER_NC]: ...
mesh.fill(q, initial)                       # never hand-roll the node loop
var mass = mesh.integrate(q, IDX_RHO)       # also integrate_fn, l2_error
mesh.visit_nodes(q, visit)                  # visit(x, y, w, q_node) for the rest

def compute_rhs(mut q_in: State, mut q_out: State) raises {imm}: ...
def adjust(mut q_: State) raises {mut limiter, imm}: ...
def suggest_dt(mut q_: State) raises {imm} -> Real:
    return spatial.suggest_dt(ctx, mesh, q_)
t = stepper.run_until_cfl(ctx, q, t, t_next, compute_rhs, adjust, suggest_dt)
# or, with a fixed step: stepper.run_until(ctx, q, t, t_next, dt, compute_rhs, adjust)
```

- Sample a specific node with `State.index(elem, node, comp)` inside
  `with q.buf.map_to_host() as h:`. Do not re-derive the flat index.
- Do not hard-code a time step in examples: use `suggest_dt` with the app's
  `cfl_max`. Tests that pin a step count may keep a fixed `dt`.
- Output goes through `FrameWriter(name)` (default `./output/<name>/`,
  created on demand) and `dump_<phys>_frame` or `write_vtu` with your own
  field closure. No `/tmp` paths.
- Tests print `PASSED` last and name themselves
  `test_<physics>_<problem>_<dim>.mojo`. Shared test code (exact Riemann
  solver, rate check) lives in `src/warpxm/testing/`.

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
    `stepper.step(ctx, q, dt, compute_rhs, adjust)`. A closure that reads
    an outer runtime variable needs a capture list (`{imm}`); one that
    reads only its arguments and comptime constants needs none. Capture `mut` only what
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
- A function cannot forward a variadic pack into `enqueue_function`
  alongside fixed arguments, and user structs are not `DevicePassable`.
  That is why DG kernels take their physics scalars as one `KernelParams`
  vector (`spatial/dg_launch.mojo`) and unpack it by position.
- A `comptime` alias of a `def(...) thin` type does not match as a kernel
  parameter bound; spell the function type inline (see `launch_dg`).
- `row_major(Idx(n))` → `row_major(Int(n))`.
- Prefer `comptime match` over `comptime if / elif` chains on a parameter.
- Inside kernels use `NodalBasis1D[P].diff_entry[col](row)` (a select over
  immediates) and comptime weights, not the `@inline(.never)` table
  accessors, which become real calls through per-thread local memory.
- `NodalBasis1D.nodes()/weights()/diff_matrix_flat()` are `@inline(.never)` to
  dodge a 1.2.0 optimizer bug (an `Array` returned by an inlined function
  reads back as garbage in host code after a `ctx.enqueue_copy`). See the
  note in `basis/nodal_basis_1d.mojo` before removing it.

### Kernel performance rules

Every DG kernel is bound by the FP64 pipe on consumer GPUs (86–88 % busy on
an RTX 3090), so FP64 instruction count is what matters. Rules that came out
of profiling with `ncu` (see `bench/BASELINE.md`):

- **Square roots go through `core.math_compat.sqrt`** (the `llvm.sqrt`
  intrinsic). Never `x ** 0.5`: that is a software `pow`, and was the
  largest single cost in the MHD flux.
- **Divide once.** Take `inv_rho = 1 / ρ` once per state and multiply; a
  double division is about ten FP64 instructions. The DG cores receive
  `1 / jacobian` from the launcher and use comptime reciprocal weights.
- **Numerical fluxes form F·n directly** (`_normal_flux_and_speed` in
  `apps/euler.mojo`, `apps/imhd.mojo`): one flux evaluation per side. Do
  not build f_x, f_y, f_z and dot with n — `0.0 * f_y` cannot be folded.
- **No per-thread local memory**: index scratch arrays with comptime
  indices only, and do not call non-inlined functions that return arrays.
- **Adjuster kernels run one thread per element**, so their latency is the
  serial depth of the worst element. Keep iterative work (the MHD
  positivity bisection) conditional on the node actually violating.

One correctness rule for any kernel that uses `barrier()`: every thread of
a block must reach the same `barrier()` call site. A `barrier(); return`
on a guard path deadlocks the warp when the block is only partly valid
(the DG core's last block, when the element count is not a multiple of
`DG_CORE_ELEMS_PER_BLOCK`). Clamp the index, do the work, and return
after the barrier instead (see `past_end` in `spatial/dg_core.mojo`;
`tests/test_advection_odd_mesh.mojo` covers it).

### Apps architecture

Each of the five fluid physics packages (advection, euler, maxwell, imhd,
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
- `max_signal_speed_static(q, ...)` where the speed depends on the state
  (euler, imhd, imhd_glm): a direction-independent bound used by
  `suggest_dt`.

This mirrors upstream WARPXM's pattern: the *flux launcher* owns
dimensionality; the *app* operates on local node values. The DG core, the
mesh, the per-physics wrappers (`dg_<phys>.mojo`) and the adjusters are all
generic over `NDIM`; nothing in `src/warpxm` is written per dimension
except the original kinetic 1D1V / 1D2V code. New kinetic work goes on
the generic `[XDIM, VDIM]` stack (`PhaseMesh`, `PhaseState`,
`DGVlasovSolver`, `VlasovMoments`, `SSPRK3Phase`, `SSPRK3VlasovMaxwell`),
not on a new per-dimension copy. Callers import
the canonical struct directly: `from warpxm.apps.euler import Euler`,
`from warpxm.apps.imhd import IMHD`, etc.

### Adding a new physics

1. **App**: write `apps/<phys>.mojo` with `internal_flux_{x,y,z}_static`
   methods on the struct and a `numerical_flux_static(q_l, q_r, nx, ny, nz,
   ...)` taking a 3-component unit normal. Module-level helpers
   (`_rho_floored`, `_gas_pressure_floored`, etc.) for the physics-specific
   conversions.
2. **DG kernel**: write one `spatial/dg_<phys>.mojo`, a thin wrapper over
   the shared `_dg_compute_rhs_core` (`spatial/dg_core.mojo`), generic over
   `NDIM`. The kernel `_<phys>_rhs_kernel[NDIM, P]` has the fixed signature
   that `launch_dg` expects (see `spatial/dg_launch.mojo`): unpack the
   physics scalars from `params`, then declare five `def ... {imm}`
   closures that dispatch to the app's static methods: `flux_x`, `flux_y`,
   `flux_z`, `num_flux(q_l, q_r, nx, ny, nz)` and
   `apply_bc(q_interior, bc_kind, face_axis)`. The core calls only the
   first `NDIM` flux closures and passes axis-aligned unit normals.
   The solver struct `DG<Phys>Solver[NDIM, P]` has `compute_rhs`, which is
   one `launch_dg[kernel](ctx, mesh, q, rhs, kernel_params(...))` call, and
   `suggest_dt` (add a `max_signal_speed_static` on the app plus a case in
   `spatial/wave_speed.mojo` if the speed depends on the state).
3. **Stepper**: nothing new. The generic `SSPRK3[NDIM, P, NC]` and
   `DormandPrince45[NDIM, P, NC]` already handle any single-state problem.
   Compose the limiter / positivity preserver / source kernels at the call
   site via the driver's `adjust` and `compute_rhs` closures.
4. **BCs**: add a `_apply_bc_<phys>` helper in your DG-kernel wrapper. It
   takes interior `q_n`, `bc_kind`, and `face_axis`, returns the ghost-side
   state. The shared DG core invokes it at every boundary face.
5. **Re-export**: add `physics/<phys>.mojo` listing the app, its constants,
   solvers and adjusters, and a preset in `io/vtu_presets.mojo`.
6. **Test**: clone an analogous test of another physics, adjust the IC, run.

Source-term kernels (`spatial/sources/`) and the slope limiter are generic
over `NDIM` and `NC`, so a new app reuses them as they are.
