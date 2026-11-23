# Tests

Every (physics, dimension) combination has at least one regression
test in `tests/`. What each one measures is tabulated on the
[Validation](../validation.md) page. Each test is a standalone `main()`
that:

1. Builds a mesh + state.
2. Runs a known problem to a known time.
3. Asserts something specific — a conservation drift bound, an analytic-
   solution L2 tolerance, a positivity bound, etc.
4. Prints `PASSED` on success.

## Running the suite

```fish
pixi run test               # full suite (16 tests at a time), about a minute
pixi run test euler 3d      # only tests whose name contains "euler" or "3d"
pixi run test -j 4          # at most 4 tests at once (-j 1 = sequential)
pixi run test -k            # stop launching tests after the first failure
pixi run test -v            # verbose: print full output of failing tests
pixi run test -t            # also list the ten slowest tests
```

The `test` pixi task wraps `scripts/run_tests.sh`. Tests are independent
programs, so the runner compiles and runs them in parallel. A test passes
only if it exits with status 0 **and** prints the line `PASSED`. Exit
code 1 on any failure.

Compilation parallelises well; GPU work does not (one GPU serialises the
processes, and each process pays about a second of driver start-up). So
the suite's wall time is set by total GPU work — keep new tests as small
as their assertion allows:

* Use the fewest transverse elements that still exercise the code path
  (2 layers for a problem that is uniform in that direction).
* Choose `dt = T / n_steps` so the run ends exactly on the target time;
  a truncated last step otherwise shows up as a spurious phase error
  and forces a loose tolerance.
* Stop at the earliest time with a known exact solution (a half period
  with the sign-flipped IC, or the true period of a diagonal wave).
* Set tolerances from the measured error (roughly 5–30× above it), not
  from a generic bound — a tolerance orders of magnitude above the
  measured value cannot catch a regression.

To run a single test directly:

```fish
pixi run mojo run -I src tests/test_advection_convergence_1d.mojo
```

A single test takes a few seconds, most of it compilation.

## Multi-rank tests

```fish
pixi run test-mpi           # tests/mpi/, each at several rank counts
pixi run test-mpi euler     # by name
pixi run test-mpi -v        # print the output of failing runs
```

The `test-mpi` task (`scripts/run_mpi_tests.sh`) needs a system MPI
(`mpicc`, `mpirun`). It builds each `tests/mpi/*.mojo` once with
`scripts/mpi_build.sh` and runs it under `mpirun` at the rank counts named
in the test's `# MPI_RANKS:` line (default `1 2 3 4`); the ranks share
whatever GPUs are visible. A test passes if every run prints `PASSED`
**and** its `CHECKSUM` lines are identical at every rank count:

```mojo
var sum = mesh.checksum(q)        # hash of the field, any decomposition
if is_root():
    print("CHECKSUM", sum)
```

So a multi-rank test asserts that the decomposition does not change the
field by a single bit. Give the mesh element counts that the rank counts
do not divide evenly, and make it small: the ranks take turns on one GPU.
A test of anything that sums over ranks in floating point (the adaptive
Dormand-Prince step) should print no checksum for that part and check it
against a tolerance instead.

## Writing a test

Tests use the same helpers as any driver, so they contain the problem
and its assertions and little else. `tests/test_euler_sod_2d.mojo` is a
representative example; its skeleton:

```mojo
from warpxm.prelude import *
from warpxm.physics.euler import *
from warpxm.testing.riemann import ExactRiemann

comptime State = DGState[2, P, EULER_NC]

def main() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = DeviceContext()
    var mesh = BlockMesh[2, P](...)
    var q = State(ctx, mesh.num_elements)

    def initial(x: Real, y: Real) -> Array[Real, EULER_NC]:
        ...
    mesh.fill(q, initial)
    var mass_0 = mesh.integrate(q, IDX_RHO)

    # ... solver, stepper, compute_rhs / adjust closures ...
    var t = stepper.run_until(ctx, q, Real(0.0), T_FINAL, DT, compute_rhs, adjust)

    var err_mass = fabs(mesh.integrate(q, IDX_RHO) - mass_0 - expected_flux)
    print("mass balance error:", err_mass)
    assert_true(err_mass < Real(1.0e-11), "mass balance violated")
    print("PASSED")
```

Rules:

* **Use the helpers, not the buffer layout.** `mesh.fill` for the
  initial condition; `mesh.integrate`, `integrate_fn`, `l2_error` and
  `visit_nodes` for diagnostics ([`mesh`](../api/mesh.md#host-side-helpers)).
  To sample one specific node, use `State.index(elem, node, comp)`
  inside `with q.buf.map_to_host() as h:`. Do not re-derive the flat
  index.
* **Shared test code lives in `src/warpxm/testing/`**: the exact Riemann
  solver (`ExactRiemann`) and the rate check (`check_convergence`). Add
  to it rather than copying a helper into a second test
  ([`testing`](../api/testing.md)).
* **A fixed `dt` is fine in a test.** Tests pin the step count so that
  results are reproducible to round-off; use
  `stepper.run_until(ctx, q, t0, t_end, dt, ...)`. Examples use the
  CFL-driven `run_until_cfl` instead.
* **The assert tells you what went wrong.** "mass balance violated",
  "result not uniform in y" — diagnostic, not generic.
* **Print before assert.** The numbers land in the log even on failure.
* **Reasonable, not generous tolerances** (see above).

Three tests cover the infrastructure itself, independently of any
physics:

* `test_mesh_helpers.mojo` — `fill`, `visit_nodes`, `integrate`,
  `integrate_fn`, `l2_error` on every mesh (including the phase
  meshes), against the mesh's own coordinate buffers and closed-form
  integrals.
* `test_cfl_timestep.mojo` — `suggest_dt` against the analytic bound for
  every physics, the stability limit of SSPRK3, and `run_until` /
  `run_until_cfl` landing on the target time.
* `test_advection_odd_mesh.mojo` — meshes whose element count is not a
  multiple of the DG core's elements-per-block packing, in 1D, 2D and
  3D. A kernel that mishandles the partly filled last block either
  never returns or leaves the last elements stale.

`test_vtu_writers.mojo` does the same for the output path.

## Don't write a test that just runs

```mojo
# BAD
def main() raises:
    var mesh = BlockMesh[2, P](...)
    var q = DGState[2, P, NC](...)
    var stepper = SSPRK3[2, P, NC](...)
    _ = stepper.run_until(...)
    print("PASSED")
```

This passes if the simulation doesn't crash. It doesn't catch
silent regressions (e.g. flipping a sign on a flux term).

## Naming convention

`tests/test_<physics>_<problem>_<dim>.mojo`: physics first, then the
problem, then the dimension, then an optional variant. Examples:

* `test_euler_sod_1d.mojo` — Euler Sod tube, 1D
* `test_advection_convergence_2d.mojo` — advection, convergence study, 2D
* `test_imhd_briowu_2d_hlld.mojo` — MHD Brio-Wu, 2D, HLLD path
* `test_maxwell_cavity_2d.mojo` — Maxwell, 2D, cavity mode
* `test_vlasov_maxwell_langmuir_1d1v.mojo` — kinetic, Langmuir wave, 1D1V

The physics slugs in use are `advection`, `euler`, `maxwell`, `imhd`,
`langmuir` / `pair_plasma` / `cyclotron` (fluid plasma), `vlasov`, and
`dp45` for the adaptive-stepper tests. A test that spans dimensions
(`test_euler_positivity`, `test_mesh_helpers`, `test_cfl_timestep`,
`test_advection_odd_mesh`, `test_vtu_writers`) has no dimension suffix.

The name is what `pixi run test <substring>` filters on, so
`pixi run test imhd 3d` runs every MHD test and every 3D test.

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
recipe. The test step is: clone an analogous (phys, N±1)
test, adjust the IC, run, and tune the tolerances to whatever the new
problem genuinely achieves.
