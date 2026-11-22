# Adding a new physics module

A proven recipe for adding a new physics. Source-term-only additions
(gravity, drag, …) are simpler and don't need every step.

The steps (one shared app file across dimensions):

1. **App** — `apps/<phys>.mojo`. Define `<PHYS>_NC`, the `IDX_*`
   constants, and a struct with `internal_flux_{x,y,z}_static` methods
   that compute the per-axis flux at a single state, plus a
   `numerical_flux_static(q_l, q_r, nx, ny, nz, ...)` taking a
   3-component unit normal (a 1D caller passes `(±1, 0, 0)`; 2D
   passes `(nx, ny, 0)`). Give the struct a `cfl_max` field. If the
   signal speed depends on the state, add
   `max_signal_speed_static(q, ...)` returning a direction-independent
   bound. Module-level helpers for the physics-specific conversions
   (`_rho_floored`, `_gas_pressure_floored`, etc.). Follow the kernel
   performance rules in `CLAUDE.md` (one division per state, `sqrt`
   from `core.math_compat`, form $F\cdot n$ directly).

2. **DG kernel** — one `spatial/dg_<phys>.mojo`, generic over `NDIM`.
   The kernel has the fixed signature `launch_dg` expects
   (listed in `spatial/dg_launch.mojo` and on the
   [`spatial`](../api/spatial.md#how-a-wrapper-is-built) page): buffers
   as `RealTensor` / `IndexTensor`, then the inverse Jacobians (one
   `InvJacobian` vector), the
   element count, and one `params: KernelParams`. In the body, unpack
   the physics scalars by position (`var gamma = params[0]`), declare
   nested `def ... {imm}` closures that dispatch to the app's static
   methods, and call the shared `_dg_compute_rhs_core[NDIM, P, NC, ...]`
   with five closures: `flux_x`, `flux_y`, `flux_z`,
   `num_flux(q_l, q_r, nx, ny, nz)` and
   `apply_bc(q, bc_kind, face_axis)`. The core supplies the unit normal
   of each axis and calls only the first `NDIM` flux closures.

   The solver struct's `compute_rhs` is then one call,
   `launch_dg[kernel[Self.NDIM, Self.P]](ctx, mesh, q, rhs, kernel_params(...))`,
   with the scalars in the same order the kernel unpacks them (at most
   eight). `dg_euler.mojo` is a complete example in about 120 lines, and
   serves 1D, 2D and 3D.

3. **CFL** — add `suggest_dt` to the solver struct. Constant-speed
   systems compute it from the mesh alone
   (`cfl_dt[Self.P](cfl_max, speed * mesh.inv_dx_sum())`). For a
   state-dependent speed, add a `SPEED_<PHYS>` case and a
   `max_wave_speed_<phys>` function in `spatial/wave_speed.mojo`, which
   reduces the app's `max_signal_speed_static` over the mesh on the GPU.

4. **BCs** — add a `_apply_bc_<phys>` helper in your DG-kernel
   wrapper. It takes the interior state, the BC kind, and (in 2D / 3D) a
   `face_axis` ∈ {0, 1, 2} (which mesh normal direction this face is
   on), and returns the ghost-side state. The shared DG core invokes it
   at every boundary face. Unhandled kinds should fall through to
   copy-out.

5. **Stepper** — nothing new. `SSPRK3[NDIM, P, NC]` and
   `DormandPrince45[NDIM, P, NC]` handle any single-state problem.
   Compose the limiter / positivity preserver / source kernels at the
   call site via the driver's `adjust` and `compute_rhs` closures. The
   coupled steppers (`ssprk3_plasma.mojo`, `ssprk3_twofluid.mojo`) are
   separate because they advance three (or two) states.

6. **Re-export and output** — add `physics/<phys>.mojo` listing the app,
   its constants, solvers and adjusters, so a driver needs only
   `from warpxm.prelude import *` and `from warpxm.physics.<phys> import *`.
   Add a `dump_<phys>_frame` preset to `io/vtu_presets.mojo`: a field
   closure and a name list on top of `write_vtu`, about ten lines.

7. **Test** — clone an analogous (phys, N±1) test, adjust the IC, run.
   Name it `test_<phys>_<problem>_<dim>.mojo`. A convergence test with
   `check_convergence` and a conservation check are the minimum; see
   [Tests](tests.md).

## What's reusable

* **Slope limiter** — `MoeRossmanith[NDIM, P, NC]` is NC-parameterised,
  so you get shock capturing for free if your conserved variables
  carry the same Rankine-Hugoniot bounds (true for everything except
  the magnetic field, which is divergence-constrained).
* **Source-term kernels** — the functions in `spatial/sources/` are
  pointwise on the flat state buffer and generic over the dimension.
  Reuse, don't clone.
* **VTU output** — `write_vtu(path, mesh, q, names, fields)` handles the
  geometry and the file format for any state; you supply only the field
  closure.
* **Driver helpers** — `mesh.fill`, `integrate`, `l2_error` and
  `visit_nodes` work for any `NC`.

## What needs to be new

* **Numerical flux** beyond Rusanov — HLLC, HLLD, Roe, AUSM all need
  per-physics work because they exploit the eigenstructure of the
  system. See `IMHD.numerical_flux_hlld_static` for the Miyoshi-Kusano
  example (axis-aligned permutation into a shared `_hlld_core_3d`).
* **Positivity preserver** — the algorithm is generic but the
  primitive recovery formula isn't. Each "positive primitive" has its
  own quadratic or cubic in θ.
* **Boundary-condition handling** — the BC kinds are shared but the
  ghost-state construction per BC depends on physics (e.g. PEC for
  Maxwell flips tangential E; free-slip for Euler flips normal
  momentum).

## Worked example: 1D Maxwell

Because the app, the DG wrapper, the mesh and the steppers are all
generic over the dimension, running an existing physics in another
dimension needs no new library code:

```text
src/warpxm/
  apps/maxwell.mojo               # unchanged: covers all 3 dims
  spatial/dg_maxwell.mojo         # unchanged: DGMaxwellSolver[NDIM, P]
tests/test_maxwell_convergence_1d.mojo # transverse plane wave: rate + absolute error
tests/test_maxwell_cavity_1d.mojo      # TM cavity
```

The driver instantiates `BlockMesh[1, P]`, `DGState[1, P, MAXWELL_NC]`
and `DGMaxwellSolver[1, P]`. The `Maxwell` app's `numerical_flux_static`
takes a 3-component normal, and in 1D the DG core passes
$(\pm 1, 0, 0)$. The wrapper itself is about 120 lines: a BC function
that handles all three wall normals, the kernel with its five closures,
and a solver struct whose `compute_rhs` is one launcher call.

## Don't add: 1D MHD

The lowest-rank MHD problem (transverse magnetic-field perturbation)
genuinely needs at least 2D — there is no useful "1D MHD" because the
$B_x$ component along the only spatial axis becomes a constant. No
1D MHD problem is tested or supported.
