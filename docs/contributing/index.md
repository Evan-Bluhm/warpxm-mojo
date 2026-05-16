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

The framework is feature-complete across the (physics, dimension)
matrix that physics permits (only 1D MHD is missing — intentional,
$B_x$ becomes constant in 1D). What's still open, roughly in order of
impact:

1. **Vlasov-Maxwell solver**. Different architecture (phase-space DG
   vs configuration-space DG) — a sizeable port. Lives in upstream
   under `src/apps/kinetic/`.
2. **MPI halo exchange**. Multi-GPU support; sizeable engineering.

Recently closed (see git log for details):

* GPU-side reduction for the DP45 error norm — `_dp45_sum_sq_kernel`
  in `temporal/dormand_prince_1d.mojo` tree-reduces Σ (err/sc)² in a
  single block. Replaces three full-buffer `map_to_host` copies per
  adaptive step with a one-scalar readback. Shared by the 2D and 3D
  DP45 steppers.
* DP45 generalised to 1D / 2D / 3D for any NC-component problem
  (`DormandPrince45_{1,2,3}D[P, NC]`, closure-driven).
* HLLD wired into both `DGImhdSolver{2,3}D` and
  `DGImhdGLMSolver{2,3}D` (axis-aligned permutation dispatch).
* Strang source splitting was originally motivated by a 1.85e-3
  antisymmetry residual on the 2D pair-plasma test. The
  `test_pair_plasma_asym_scaling` diagnostic showed the residual
  scales as $\delta^{2}$ — confirming it's the *physical* Lorentz
  coupling between species with antisymmetric density perturbations,
  not an algorithmic asymmetry. No splitting fix can eliminate it.

If you tackle one of the open items, see the [Tests](tests.md) page
for what a "complete" regression looks like.

## Code of conduct

Don't be a jerk. That's it.
