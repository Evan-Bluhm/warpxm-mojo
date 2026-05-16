# Adding a new physics module

A proven recipe for adding a new physics. Source-term-only additions
(gravity, drag, …) are simpler and don't need every step.

The five-step pattern (one shared app file across dimensions):

1. **App** — `apps/<phys>.mojo`. Define `<PHYS>_NC`, the `IDX_*`
   constants, and a struct with `internal_flux_{x,y,z}_static` methods
   that compute the per-axis flux at a single state, plus a
   `numerical_flux_static(q_l, q_r, nx, ny, nz, ...)` taking a
   3-component unit normal (a 1D caller passes `(±1, 0, 0)`; 2D
   passes `(nx, ny, 0)`). Module-level helpers for the physics-specific
   conversions (`_rho_floored`, `_gas_pressure_floored`, etc.).

2. **DG kernel** — `spatial/dg_<phys>_<N>d.mojo` for each dimension you
   need. Thin wrapper over the shared
   `_dg_<N>d_compute_rhs_core` template in
   `spatial/dg_core_<N>d.mojo`. The wrapper declares `@parameter def`
   closures that dispatch to the app's static methods with the right
   normal triples (e.g. 2D passes `(nx, ny, 0)`):
   - 1D: `flux_x`, `num_flux`, `apply_bc` (three closures);
   - 2D: + `flux_y` (four closures);
   - 3D: + `flux_z` (five closures).

3. **Stepper** — nothing new. The generic `SSPRK3_<N>D[P, NC]` and
   `DormandPrince45_<N>D[P, NC]` already handle any single-state
   problem. Compose the limiter / positivity preserver / source kernels
   at the call site via the driver's `adjust` and `compute_rhs`
   closures. The plasma steppers
   (`ssprk3_plasma_{1,2,3}d`, `ssprk3_twofluid_{1,2,3}d`) are kept
   specialised because they juggle 3 (or 2) coupled states.

4. **BCs** — add a `_apply_bc_<phys>_<N>d` helper in your DG-kernel
   wrapper. It takes the interior `q_n`, the BC kind, and a `face_axis`
   ∈ {0, 1, 2} (which mesh normal direction this face is on), and
   returns the ghost-side state. The shared DG core invokes it at every
   boundary face.

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
  system. See `IMHD.numerical_flux_hlld_static` for the Miyoshi-Kusano
  example (axis-aligned permutation into a shared `_hlld_core_3d`).
* **Positivity preserver** — the algorithm is generic but the
  primitive recovery formula isn't. Each "positive primitive" has its
  own quadratic or cubic in θ.
* **Boundary-condition handling** — the BC enum is shared but the
  ghost-state construction per BC depends on physics (e.g. PEC for
  Maxwell flips tangential E; free-slip for Euler flips normal
  momentum).

## Worked example: how 1D Maxwell was added

The 1D Maxwell solver is a small worked example of the recipe above.
What was needed:

```text
src/warpxm/
  apps/maxwell.mojo               # already existed — covers all 3 dims
  spatial/dg_maxwell_1d.mojo      # NEW: thin wrapper over dg_core_1d
tests/test_maxwell_1d_plane_wave.mojo  # transverse plane wave
tests/test_maxwell_1d_cavity.mojo      # TM cavity
```

The `Maxwell` app struct was reused unchanged because its
`numerical_flux_static` takes a 3-component normal — the 1D DG kernel
just passes $(\pm 1, 0, 0)$. The 1D DG wrapper itself is ~100 lines of
closure plumbing. Total port effort: under an hour.

## Don't add: 1D MHD

The lowest-rank MHD problem (transverse magnetic-field perturbation)
genuinely needs at least 2D — there is no useful "1D MHD" because the
$B_x$ component along the only spatial axis becomes a constant. We
deliberately don't ship a 1D MHD module.
