# Adding a new physics module

A proven recipe for adding a new (physics, dimension) combination.
Source-term-only additions (gravity, drag, …) are simpler and don't
need every step.

The five-step pattern:

1. **App** — `apps/<phys>_<N>d.mojo`. Define `<PHYS>_NC`, the
   `IDX_*` constants, and a struct with `internal_flux_{x,y[,z]}_static`
   plus a `numerical_flux_static` (Rusanov / Lax-Friedrichs taking a
   unit normal).

2. **DG kernel** — `spatial/dg_<phys>_<N>d.mojo`. Clone the closest
   existing spatial solver (e.g. `dg_euler_2d.mojo`) and swap the
   flux helpers. The volume + surface skeleton is the same for every
   physics.

3. **Stepper** — `temporal/ssprk3_<phys>_<N>d.mojo`. Clone the
   matching limited or non-limited SSPRK3 variant, swap the spatial
   solver type. If your physics needs sources, model after
   `ssprk3_plasma_2d.mojo`.

4. **BCs** — add a `_apply_bc_<phys>_<N>d` helper in the DG kernel
   file, thread `face_bc` through the kernel, dispatch on the BC kind
   at each ghost-sentinel branch. The helper takes the interior
   `q_n` and returns a ghost state.

5. **Test** — clone an analogous (phys, N±1) test, adjust the IC, run.

## What's reusable

* **Slope limiter** — `MoeRossmanith{2,3}D[P, NC]` is NC-parameterised,
  so you get shock capturing for free if your conserved variables
  carry the same Rankine-Hugoniot bounds (true for everything except
  the magnetic field, which is divergence-constrained).
* **Source-term kernels** — `apply_*_source_*` are NC-parameterised
  where dimensional analysis allows. Reuse, don't clone.
* **VTU writers** — the geometry decomposition is order-agnostic.
  Cloning an existing writer (e.g. `vtu_imhd_2d.mojo`) and changing
  the `<DataArray>` blocks is ~20 lines.

## What needs to be new

* **Numerical flux** beyond Rusanov — HLLC, HLLD, Roe, AUSM all need
  per-physics work because they exploit the eigenstructure of the
  system. See `IMHD2D.numerical_flux_hlld_static` for the
  Miyoshi-Kusano example.
* **Positivity preserver** — the algorithm is generic but the
  primitive recovery formula isn't. Each "positive primitive" has its
  own quadratic or cubic in θ.
* **Boundary-condition handling** — the BC enum is shared but the
  ghost-state construction per BC depends on physics (e.g. PEC for
  Maxwell flips tangential E; free-slip for Euler flips normal
  momentum).

## Pattern: a 1D Maxwell example

If you wanted to add 1D Maxwell:

```text
src/warpxm/
  apps/maxwell_1d.mojo           # NC=8 reused, copy of maxwell_2d.mojo, drop ny
  spatial/dg_maxwell_1d.mojo     # clone dg_euler_1d.mojo, swap fluxes
  temporal/ssprk3_maxwell_1d.mojo  # clone ssprk3_maxwell_2d.mojo
tests/test_maxwell_1d_plane_wave.mojo  # smooth plane-wave decay
```

Each file is ~150 lines. The whole port is a few hours, dominated by
debugging the BC adjustment for PEC walls.

## Don't add: 1D MHD

The lowest-rank MHD problem (transverse magnetic-field perturbation)
genuinely needs at least 2D — there is no useful "1D MHD" because the
$B_x$ component along the only spatial axis becomes a constant. We
deliberately don't ship a 1D MHD module.
