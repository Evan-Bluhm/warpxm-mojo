# Contributing

The project is small enough that one person can hold the full picture
in their head. The pages below are aimed at making it easier for a
second person — you — to do the same.

## Where to start

* [Repository tour](repo-tour.md) — what every directory is for and
  how the upstream WARPXM layout maps onto it.
* [Adding a new physics module](adding-a-physics.md) — the standard
  recipe for porting a new conservation law into the framework: app,
  DG wrapper, BCs, re-export module, test.
* [Mojo style notes](style.md) — the pretraining-era Mojo syntax
  pitfalls and the in-repo conventions that aren't enforced by the
  compiler.
* [Tests](tests.md) — how to run the suite, how to add a new test,
  what counts as a "regression test" here.

## Roadmap

The framework is feature-complete across the (physics, dimension)
matrix that physics permits (only 1D MHD is missing, on purpose). Open
work and the full feature matrix are on the [Features](../features.md)
page; the measured results of the regression suite are on the
[Validation](../validation.md) page.

If you tackle one of the open items, see the [Tests](tests.md) page
for what a "complete" regression looks like.

## Code of conduct

Don't be a jerk. That's it.
