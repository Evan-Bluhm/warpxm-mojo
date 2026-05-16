# Tests

Every (physics, dimension) combination has at least one regression
test in `tests/`. Each test is a standalone `main()` that:

1. Builds a mesh + state.
2. Runs a known problem to a known time.
3. Asserts something specific — a conservation drift bound, an analytic-
   solution L2 tolerance, a positivity bound, etc.
4. Prints `PASSED` on success.

## Running the suite

```fish
pixi run test               # full suite, tally pass/fail at the end
pixi run test -k            # stop at first failure
pixi run test -v            # verbose: print full output of failing tests
```

The `test` pixi task wraps `scripts/run_tests.sh` — a shell loop with
robust pass detection (grep for `^PASSED$`) so noise like build
warnings or trailing newlines doesn't confuse the tally. Exit code 1
on any failure.

To run a single test directly:

```fish
pixi run mojo run -I src tests/test_advection_1d.mojo
```

Tests run sequentially. Each takes a few seconds on a modern GPU; the
full 66-test sweep is ~5 minutes.

## A "good" test

The canonical pattern, from `tests/test_imhd_briowu_2d.mojo`:

```mojo
def main() raises:
    comptime assert has_accelerator(), "Requires GPU"

    # ... build mesh, IC, run to T_FINAL ...

    var any_nan = False
    var max_y_drift: Real = 0.0
    with q.buf.map_to_host() as h:
        # ... compute diagnostics ...

    assert_true(not any_nan, "Brio-Wu produced NaN")
    print("max y-drift in ρ (1D-extruded uniformity check):", max_y_drift)
    assert_true(max_y_drift < Real(1.0e-10),
                "result should be uniform in y for a 1D-extruded Brio-Wu")
    # ... more asserts on far-field state ...
    print("PASSED")
```

What makes this useful:

* **The assert tells you what went wrong.** "Brio-Wu produced NaN",
  "result should be uniform in y for a 1D-extruded Brio-Wu" — both
  are diagnostic, not generic.
* **Print before assert.** The diagnostic prints land in CI logs even
  on failure.
* **Reasonable, not generous tolerances.** `1e-10` for a 1D-extruded
  problem is a tight enough bound to catch real regressions.

## Don't write a test that just runs

```mojo
# BAD
def main() raises:
    var mesh = BlockMesh2D[P](...)
    var q = DGState2D[P, NC](...)
    var stepper = SSPRK3_*_*(...)
    while t < T_FINAL: stepper.step(...)
    print("PASSED")
```

This passes if the simulation doesn't crash. It doesn't catch
silent regressions (e.g. flipping a sign on a flux term).

## Naming convention

`tests/test_<physics>_<problem>_<dim>.mojo`. Examples:

* `test_euler_sod_1d.mojo` — Euler Sod tube, 1D
* `test_imhd_briowu_2d_hlld.mojo` — MHD Brio-Wu, 2D, HLLD path
* `test_maxwell_2d_cavity.mojo` — Maxwell, 2D, cavity mode

If a physics has multiple problem variants (Sod, blast, entropy wave),
use the problem name as the middle slug.

## What counts as "regression"

A test is a regression check, not a unit test. We're verifying that:

* the conservation laws are still satisfied to machine precision on
  the test problem,
* shocks / contacts land in the right place,
* limiter and positivity preserver behave conservatively,
* energy/momentum/mass drifts stay within established tolerances.

If a code change causes a previously-passing test to fail, that's a
regression. Don't relax tolerances to make tests pass — investigate
why the new code is worse.

## Adding a test for a new physics

See [Adding a new physics module](adding-a-physics.md) for the
five-step recipe. The test step is: clone an analogous (phys, N±1)
test, adjust the IC, run, and tune the tolerances to whatever the new
problem genuinely achieves.
