## Tasks for improving warpxm-mojo

Ergonomics

1. Add a host-side helper module for initial conditions and diagnostics. This is the biggest win.
   - 102 test and example files hand-roll the same nested loop (elements × nodes, compute x/y, index (e * nodes_per_elem + n) * NC + IDX) to fill the initial state.
   - Integration and convergence helpers are copy-pasted: _integrate_field ×4, compute_total_mass ×4, _check_rate ×9, _exact_riemann ×5.
   - A mesh.fill(q, ic) taking a closure (x, y) -> Array[Real, NC], plus integrate[F](q, mesh), l2_error, and a testing/ module for the Riemann solver and rate check, would cut the Orszag-Tang example from about 200 lines to about 60.
   - It would also hide the buffer layout, which every driver currently hard-codes.
2. Add CFL-driven time stepping. Every example hard-codes DT, from 2.5e-5 to 1e-3, found by trial.
   - cfl_max is stored on the apps but only advection and Vlasov ever read it; for Euler and MHD it is a dead parameter.
   - The cheap version is a GPU max-wave-speed reduction (the DP45 error-norm kernel is already the right shape) feeding a suggest_dt(q, mesh).
   - A run_until(t_target, ...) helper would also replace the frame-loop arithmetic repeated in each VTU example.
3. Write one generic VTU writer. The six vtu_* files (about 1,200 lines) differ only in their field lists; geometry and connectivity are identical.
   - One writer per dimension taking a list of (name, closure) fields, with per-physics presets as thin wrappers, would do.
   - While there, switch from ASCII to appended-binary output; ASCII will hurt at 3D resolutions.
4. Smaller ergonomics.
   - Examples write to hard-coded /tmp/... and require a manual mkdir -p; let FrameWriter create the directory and default to ./output/<name>.
   - Add a pixi run example <name> task, since pixi run mojo run -I src examples/... is a lot to type.
   - Wrap app parameters and index constants in a per-physics re-export so a driver needs 3 imports rather than 12.

Clearer code

1. Collapse the per-dimension duplication where Mojo allows.
   - ssprk3_{1,3}d differ only in the DGState type name, and the DP45 2D and 3D files differ by about 100 diff lines.
   - The state buffer is flat and the RK kernels are already dimension-agnostic, so a single DGState[NDIM, P, NC], or steppers generic over a state trait, would replace 3 SSPRK3 files, 3 DP45 files and 3 no-op adjusters with one each.
   - The six lorentz_source_* and maxwell_current_source_* files are also pointwise and could be folded the same way.
   - The same applies to the six plasma and two-fluid steppers.
2. Shrink the DG wrappers. Each dg_<phys>_<N>d.mojo repeats about 40 lines of TileTensor and layout plumbing in compute_rhs. A shared launch_dg_core_2d(ctx, mesh, q, rhs, kernel, params...) would leave only the closures and the BC function, which are the parts that actually differ.
3. Fix naming inconsistencies.
   - spatial/dg_{1,2,3}d.mojo and DGSpatialSolver2D are the advection solvers; rename to dg_advection_* and DGAdvectionSolver*.
   - driver/vtu_2d.mojo is Euler-only; rename to vtu_euler_2d.
   - euler_2d_sources.mojo breaks the <thing>_source_<N>d pattern used elsewhere.
   - Tests mix test_euler_3d_blast with test_imhd_3d_blast-style and test_euler_sod_3d-style names; pick physics, problem, then dimension.
   - driver/ only contains output code; io/ would be more honest.
4. Split spatial/. It now mixes DG solvers, fluid sources and six Vlasov files. Subfolders spatial/sources/ and spatial/kinetic/ (or a top-level kinetic/) would make the fluid/kinetic boundary visible.
5. Remove stale content.
   - "Open work" in CLAUDE.md still lists the Vlasov-Maxwell solver as unstarted, though it ships with about 20 tests.
   - The feature matrix has no Vlasov row.
   - README.md and docs/index.md never mention Vlasov.
   - examples/imhd_orszag_tang_2d_vtu.mojo and euler_blast_3d_vtu.mojo reference SSPRK3_IMHD_GLM_Limited_2D-style steppers that no longer exist.
   - BC_DIRICHLET is a placeholder with no implementation; implement it or delete it.
   - ACCUM_DTYPE and the "flip WXM_DTYPE" comment imply float32 support that has not been tested.
6. Stop maintaining CLAUDE.md as a second README. The validation table and capability inventory duplicate docs/. Keep the style and performance rules there and link out for the rest.
