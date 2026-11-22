# CFL time-step selection and the `run_until` / `run_until_cfl` drivers.
#
#   1. `suggest_dt` equals `cfl / ((2P+1) Σ_d λ_d / Δx_d)` with the right
#      signal speed for each physics: |u| + c (Euler), |u| + the
#      perpendicular fast speed (ideal MHD), that or c_h (GLM-MHD), c
#      (Maxwell), and the per-axis velocity (advection). The state has one
#      fast node in an otherwise slow field, so the GPU reduction has to
#      find a maximum rather than read any node; the fast node is placed
#      last so a reduction that drops the tail fails.
#   2. The bound is meaningful: linear advection at P = 1..4 keeps its
#      energy with `cfl_max = 1` and blows up with `cfl_max = 4`.
#   3. `run_until` lands on the target time exactly when dt does not divide
#      the interval, and takes the expected number of steps.
#   4. `run_until_cfl` lands on the target, refreshes dt as asked, gives
#      the same answer as `run_until` with the same (constant) dt, and
#      raises on a step below `dt_min` or a NaN step.

from std.collections import Array
from std.math import isnan, sin
from std.sys import has_accelerator
from std.testing import assert_true
from max.gpu.host import DeviceContext

from warpxm.prelude import *
from warpxm.physics.advection import *
from warpxm.physics.plasma import *
from warpxm.apps.imhd import IMHD, MHD_NC
from warpxm.apps.imhd_glm import ImhdGLM, MHD_GLM_NC
from warpxm.spatial.dg_imhd import DGImhdSolver
from warpxm.spatial.dg_imhd_glm import DGImhdGLMSolver


comptime GAMMA: Real = 1.4


def _close(a: Real, b: Real, what: String) raises:
    print(what, ": got", a, "expected", b)
    assert_true(fabs(a - b) <= Real(1.0e-13) * fabs(b), what)


def _test_formulas(var ctx: DeviceContext) raises:
    comptime P = 2
    var mesh2 = BlockMesh[2, P](
        ctx, Real(0.0), Real(1.0), Real(0.0), Real(0.5),
        nx=20, ny=4, periodic_x=True, periodic_y=True,
    )
    var x_fast = mesh2.x_max
    var y_fast = mesh2.y_max

    # --- Euler: ρ = 1, p = 1 at rest, one node moving at |u| = 5. ------
    var q = DGState[2, P, EULER_NC](ctx, mesh2.num_elements)
    def euler_ic(x: Real, y: Real) {imm} -> Array[Real, EULER_NC]:
        var fast = x == x_fast and y == y_fast
        var mx: Real = 3.0 if fast else 0.0
        var mz: Real = 4.0 if fast else 0.0
        return [
            Real(1.0), mx, Real(0.0), mz,
            Real(1.0) / (GAMMA - Real(1.0)) + Real(0.5) * (mx * mx + mz * mz),
        ]
    mesh2.fill(q, euler_ic)
    var euler = DGEulerSolver[2, P](
        Euler(gas_gamma=GAMMA, min_density=Real(1e-12),
              min_pressure=Real(1e-12), cfl_max=Real(0.5)),
    )
    _close(
        euler.suggest_dt(ctx, mesh2, q),
        Real(0.5) / (Real(5.0) * (Real(5.0) + sqrt(GAMMA)) * (Real(20.0) + Real(8.0))),
        "Euler 2D suggest_dt",
    )

    # --- GLM-MHD: |u| = 0, c_s² = γ, c_A² = 9; c_h below, then above. --
    var qg = DGState[2, P, MHD_GLM_NC](ctx, mesh2.num_elements)
    def glm_ic(x: Real, y: Real) {imm} -> Array[Real, MHD_GLM_NC]:
        var fast = x == x_fast and y == y_fast
        var bz: Real = 3.0 if fast else 0.0
        var v = Array[Real, MHD_GLM_NC](fill=Real(0.0))
        v[0] = 1.0
        v[4] = Real(1.0) / (GAMMA - Real(1.0)) + Real(0.5) * bz * bz
        v[7] = bz
        return v^
    mesh2.fill(qg, glm_ic)
    var fast_speed = sqrt(GAMMA + Real(9.0))
    var glm_lo = DGImhdGLMSolver[2, P](
        ImhdGLM(gas_gamma=GAMMA, min_density=Real(1e-12), min_pressure=Real(1e-12),
                cfl_max=Real(0.4), c_h=Real(1.0), c_p_sq=Real(0.5)),
    )
    _close(
        glm_lo.suggest_dt(ctx, mesh2, qg),
        Real(0.4) / (Real(5.0) * fast_speed * Real(28.0)),
        "GLM-MHD 2D suggest_dt (fast speed)",
    )
    var glm_hi = DGImhdGLMSolver[2, P](
        ImhdGLM(gas_gamma=GAMMA, min_density=Real(1e-12), min_pressure=Real(1e-12),
                cfl_max=Real(0.4), c_h=Real(10.0), c_p_sq=Real(0.5)),
    )
    _close(
        glm_hi.suggest_dt(ctx, mesh2, qg),
        Real(0.4) / (Real(5.0) * Real(10.0) * Real(28.0)),
        "GLM-MHD 2D suggest_dt (c_h)",
    )

    # --- Ideal MHD 3D, P = 3: moving and magnetised fast node. ---------
    comptime P3 = 3
    var mesh3 = BlockMesh[3, P3](
        ctx, Real(0.0), Real(1.0), Real(0.0), Real(1.0), Real(0.0), Real(2.0),
        nx=4, ny=5, nz=4,
    )
    var z_fast = mesh3.z_max
    var x3_fast = mesh3.x_max
    var y3_fast = mesh3.y_max
    var qm = DGState[3, P3, MHD_NC](ctx, mesh3.num_elements)
    def mhd_ic(x: Real, y: Real, z: Real) {imm} -> Array[Real, MHD_NC]:
        var fast = x == x3_fast and y == y3_fast and z == z_fast
        var my: Real = 4.0 if fast else 0.0      # ρ = 2, so |u| = 2
        var bx: Real = 2.0 if fast else 0.0
        var v = Array[Real, MHD_NC](fill=Real(0.0))
        v[0] = 2.0
        v[2] = my
        v[4] = (
            Real(1.0) / (GAMMA - Real(1.0)) + Real(0.25) * my * my
            + Real(0.5) * bx * bx
        )
        v[5] = bx
        return v^
    mesh3.fill(qm, mhd_ic)
    var mhd = DGImhdSolver[3, P3](
        IMHD(gas_gamma=GAMMA, min_density=Real(1e-12),
             min_pressure=Real(1e-12), cfl_max=Real(0.5)),
    )
    _close(
        mhd.suggest_dt(ctx, mesh3, qm),
        Real(0.5) / (
            Real(7.0) * (Real(2.0) + sqrt((GAMMA + Real(4.0)) / Real(2.0)))
            * (Real(4.0) + Real(5.0) + Real(2.0))
        ),
        "IMHD 3D suggest_dt",
    )

    # --- Constant-speed systems. --------------------------------------
    var maxwell = DGMaxwellSolver[2, P](Maxwell(c_light=Real(3.0), cfl_max=Real(0.5)))
    _close(
        maxwell.suggest_dt(mesh2),
        Real(0.5) / (Real(5.0) * Real(3.0) * Real(28.0)),
        "Maxwell 2D suggest_dt",
    )
    var adv = DGAdvectionSolver[2, P, 1](
        Advection[1](Real(1.0), Real(-2.0), Real(0.0), Real(0.5)),
    )
    _close(
        adv.suggest_dt(mesh2),
        Real(0.5) / (Real(5.0) * (Real(20.0) + Real(2.0) * Real(8.0))),
        "advection 2D suggest_dt",
    )


def _advect[P: Int](var ctx: DeviceContext, cfl: Real) raises -> Real:
    """Energy after 1000 steps of periodic advection at the suggested dt,
    relative to the initial energy. The IC has element-scale content so
    the highest modes are excited."""
    comptime State = DGState[1, P, 1]
    var mesh = BlockMesh[1, P](ctx, Real(0.0), Real(1.0), 32, True)
    var q = State(ctx, mesh.num_elements)
    def ic(x: Real) -> Array[Real, 1]:
        return [sin(TWO_PI * x) + Real(0.01) * sin(TWO_PI * Real(37.0) * x)]
    mesh.fill(q, ic)
    def sq(v: Array[Real, 1]) -> Real:
        return v[0] * v[0]
    var e0 = mesh.integrate_fn(q, sq)
    var spatial = DGAdvectionSolver[1, P, 1](
        Advection[1](Real(1.0), Real(0.0), Real(0.0), cfl),
    )
    var stepper = SSPRK3[1, P, 1](ctx, mesh.num_elements)
    def rhs(mut a: State, mut b: State) raises {imm}:
        spatial.compute_rhs(ctx, mesh, a, b)
    var dt = spatial.suggest_dt(mesh)
    for _ in range(1000):
        stepper.step(ctx, q, dt, rhs, no_op_adjust[1, P, 1])
    return mesh.integrate_fn(q, sq) / e0


def _test_stability[P: Int](var ctx: DeviceContext) raises:
    var stable = _advect[P](ctx, Real(1.0))
    var unstable = _advect[P](ctx, Real(4.0))
    print("P =", P, " energy ratio at cfl 1:", stable, " at cfl 4:", unstable)
    assert_true(stable <= Real(1.0) + Real(1.0e-12) and stable > Real(0.5),
                "advection at cfl_max = 1 is not stable")
    assert_true(isnan(unstable) or unstable > Real(1.0e3),
                "advection at cfl_max = 4 did not blow up: the bound is slack")


def _test_run_until(var ctx: DeviceContext) raises:
    comptime P = 2
    comptime State = DGState[1, P, EULER_NC]
    var mesh = BlockMesh[1, P](ctx, Real(0.0), Real(1.0), 40, True)
    var app = Euler(gas_gamma=GAMMA, min_density=Real(1e-12),
                    min_pressure=Real(1e-12), cfl_max=Real(0.5))
    var spatial = DGEulerSolver[1, P](app)
    var stepper = SSPRK3[1, P, EULER_NC](ctx, mesh.num_elements)

    # Entropy wave: uniform u = 1 and p = 1, so |u| + c stays (nearly)
    # constant and so does the suggested dt.
    def ic(x: Real) -> Array[Real, EULER_NC]:
        var rho = Real(1.0) + Real(0.1) * sin(TWO_PI * x)
        return [rho, rho, Real(0.0), Real(0.0),
                Real(1.0) / (GAMMA - Real(1.0)) + Real(0.5) * rho]
    var rhs_calls = 0
    def rhs(mut a: State, mut b: State) raises {mut rhs_calls, imm}:
        spatial.compute_rhs(ctx, mesh, a, b)
        rhs_calls += 1

    # (3) fixed dt that does not divide the interval: 0.1 / 0.03 → 4 steps.
    var q = State(ctx, mesh.num_elements)
    mesh.fill(q, ic)
    var t = stepper.run_until(
        ctx, q, Real(0.25), Real(0.35), Real(0.03),
        rhs, no_op_adjust[1, P, EULER_NC],
    )
    assert_true(t == Real(0.35), "run_until did not land on the target time")
    assert_true(rhs_calls == 12, "run_until took the wrong number of steps")
    # An exact multiple takes exactly that many steps, with no round-off step.
    rhs_calls = 0
    t = stepper.run_until(
        ctx, q, Real(0.0), Real(0.3), Real(0.001),
        rhs, no_op_adjust[1, P, EULER_NC],
    )
    assert_true(rhs_calls == 900, "run_until added or dropped a step")
    assert_true(fabs(t - Real(0.3)) < Real(1.0e-13), "run_until drifted off the target")
    # Already there: no step.
    rhs_calls = 0
    t = stepper.run_until(
        ctx, q, Real(0.3), Real(0.3), Real(0.001),
        rhs, no_op_adjust[1, P, EULER_NC],
    )
    assert_true(rhs_calls == 0 and t == Real(0.3), "run_until stepped past its target")

    # (4) CFL-driven. The reference run uses the same dt, fixed.
    mesh.fill(q, ic)
    var dt0 = spatial.suggest_dt(ctx, mesh, q)
    var c_max = Real(1.0) + sqrt(GAMMA / Real(0.9))
    _close(dt0, Real(0.5) / (Real(5.0) * c_max * Real(40.0)), "entropy-wave suggest_dt")
    var dt_calls = 0
    def suggest(mut a: State) raises {mut dt_calls, imm} -> Real:
        dt_calls += 1
        return spatial.suggest_dt(ctx, mesh, a)
    rhs_calls = 0
    var t_end: Real = 0.05
    t = stepper.run_until_cfl(
        ctx, q, Real(0.0), t_end, rhs, no_op_adjust[1, P, EULER_NC], suggest,
        dt_interval=5,
    )
    var steps = rhs_calls // 3
    print("run_until_cfl:", steps, "steps,", dt_calls, "dt refreshes, dt0 =", dt0)
    assert_true(t == t_end, "run_until_cfl did not land on the target time")
    var expected_steps = Int(t_end / dt0) + 1
    assert_true(steps == expected_steps or steps == expected_steps + 1,
                "run_until_cfl took an unexpected number of steps")
    assert_true(dt_calls == (steps + 4) // 5, "dt was not refreshed every dt_interval steps")

    # A suggestion at or below `dt_min`, or a NaN one, raises before any
    # step is taken instead of stalling.
    rhs_calls = 0
    var raised = False
    try:
        _ = stepper.run_until_cfl(
            ctx, q, t_end, Real(1.0), rhs, no_op_adjust[1, P, EULER_NC],
            suggest, dt_min=Real(1.0),
        )
    except e:
        raised = True
    assert_true(raised and rhs_calls == 0, "run_until_cfl ignored dt_min")
    def suggest_nan(mut a: State) raises -> Real:
        var zero: Real = 0.0
        return zero / zero
    raised = False
    try:
        _ = stepper.run_until_cfl(
            ctx, q, t_end, Real(1.0), rhs, no_op_adjust[1, P, EULER_NC],
            suggest_nan,
        )
    except e:
        raised = True
    assert_true(raised and rhs_calls == 0, "run_until_cfl accepted a NaN step")

    var q_ref = State(ctx, mesh.num_elements)
    mesh.fill(q_ref, ic)
    _ = stepper.run_until(
        ctx, q_ref, Real(0.0), t_end, dt0, rhs, no_op_adjust[1, P, EULER_NC],
    )
    var diff: Real = 0.0
    with q.buf.map_to_host() as a:
        with q_ref.buf.map_to_host() as b:
            for i in range(q.buf_len):
                diff = fmax(diff, fabs(a[i] - b[i]))
    print("max |CFL run − fixed-dt run| =", diff)
    # The suggested dt moves in the last digits as the wave advects.
    assert_true(diff < Real(1.0e-9), "CFL-driven run differs from the fixed-dt run")
    # And the wave has actually moved: it is not the initial condition.
    def exact_rho(x: Real) {imm} -> Real:
        return Real(1.0) + Real(0.1) * sin(TWO_PI * (x - t_end))
    def initial_rho(x: Real) -> Real:
        return Real(1.0) + Real(0.1) * sin(TWO_PI * x)
    var err = mesh.l2_error(q, IDX_RHO, exact_rho)
    var moved = mesh.l2_error(q, IDX_RHO, initial_rho)
    print("L2(ρ) vs advected wave:", err, " vs initial wave:", moved)
    assert_true(err < Real(2.0e-5), "CFL-driven run has the wrong solution")
    assert_true(moved > Real(100.0) * err, "solution did not advance")


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = DeviceContext()
    _test_formulas(ctx)
    _test_stability[1](ctx)
    _test_stability[2](ctx)
    _test_stability[3](ctx)
    _test_stability[4](ctx)
    _test_run_until(ctx)
    print("PASSED")
