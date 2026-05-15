# Contributing

The project is small enough that one person can hold the full picture
in their head. The pages below are aimed at making it easier for a
second person — you — to do the same.

## Where to start

* [Repository tour](repo-tour.md) — what every directory is for and
  how the upstream WARPXM layout maps onto it.
* [Adding a new physics module](adding-a-physics.md) — the standard
  recipe for porting a new conservation law into the framework. Covers
  the (app, spatial, temporal, BC, test) five-step pattern.
* [Mojo style notes](style.md) — the pretraining-era Mojo syntax
  pitfalls and the in-repo conventions that aren't enforced by the
  compiler.
* [Tests](tests.md) — how to run the suite, how to add a new test,
  what counts as a "regression test" here.

## Roadmap

What's still open, roughly in order of impact:

1. **Strang source splitting** for the plasma stepper (#26 in
   `CLAUDE.md`). The current source-coupled integration is
   second-order via RK averaging; Strang would be a cleaner factoring
   for stiff sources.
2. **DP45 to other 2D / 3D physics**. Currently shipped for 1D
   generic and 2D Euler. Each port is ~100 lines of
   `temporal/dormand_prince_<phys>_<dim>.mojo`.
3. **HLLD wiring into the GLM-MHD spatial solver**. Currently only on
   plain `DGImhdSolver{2,3}D`.
4. **Vlasov-Maxwell solver**. Different architecture (phase-space DG
   vs configuration-space DG) — a sizeable port. Lives in upstream
   under `src/apps/kinetic/`.
5. **MPI halo exchange**. Multi-GPU support; sizeable engineering.
6. **GPU-side reduction for the DP45 error norm**. Currently the
   per-step error is reduced on the host via `map_to_host`, which is
   the bottleneck for very large meshes.

If you tackle one of these, see the [Tests](tests.md) page for what a
"complete" regression looks like.

## Code of conduct

Don't be a jerk. That's it.
