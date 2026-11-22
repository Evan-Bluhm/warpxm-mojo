# Adaptive DP45 controller on a 2D Euler entropy wave.
#
# Smooth solution ρ(x, t) = 1 + 0.01·sin(2π(x − Ut)), uniform p and
# uniform u. Periodic mesh, t_final = 1 → solution returns to IC.
# Starts with a CFL-violating initial dt; controller must shrink it and
# then relax it. Validates the 2D port of `step_adaptive`.

from std.collections import Array
from std.math import sin
from std.sys import has_accelerator
from max.gpu.host import DeviceContext
from std.testing import assert_true

from warpxm.prelude import *
from warpxm.physics.euler import *


comptime P: Int = 2
comptime NX: Int = 32
comptime NY: Int = 4
comptime GAMMA: Real = 5.0 / 3.0
comptime EPS: Real = 0.01
comptime U: Real = 1.0
comptime T_FINAL: Real = 1.0
comptime DT_INIT: Real = 1.0e-2     # CFL-violating; controller will shrink
comptime ATOL: Real = 1.0e-8
comptime RTOL: Real = 1.0e-6
comptime State = DGState[2, P, EULER_NC]


def _exact_rho(x: Real, y: Real) -> Real:
    return Real(1.0) + EPS * sin(TWO_PI * x)


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = DeviceContext()
    var mesh = BlockMesh[2, P](
        ctx, Real(0.0), Real(1.0), Real(0.0), Real(0.125),
        nx=NX, ny=NY, periodic_x=True, periodic_y=True,
    )
    var q = State(ctx, mesh.num_elements)

    def initial(x: Real, y: Real) -> Array[Real, EULER_NC]:
        var rho = _exact_rho(x, y)
        var ke = Real(0.5) * rho * U * U
        return [
            rho, rho * U, Real(0.0), Real(0.0),
            Real(1.0) / (GAMMA - Real(1.0)) + ke,
        ]
    mesh.fill(q, initial)

    var app = Euler(
        gas_gamma=GAMMA, min_density=Real(1e-12),
        min_pressure=Real(1e-12), cfl_max=Real(0.5),
    )
    var spatial = DGEulerSolver[2, P](app)
    var stepper = DormandPrince45[2, P, EULER_NC](ctx, mesh.num_elements)

    def compute_rhs(mut q_in: State, mut q_out: State) raises {imm}:
        spatial.compute_rhs(ctx, mesh, q_in, q_out)

    var t: Real = 0.0
    var dt = DT_INIT
    var num_steps: Int = 0
    var min_dt = dt
    var max_dt: Real = 0.0
    while t < T_FINAL - Real(1.0e-12):
        var dt_try = dt if (t + dt) < T_FINAL else (T_FINAL - t)
        var ret = stepper.step_adaptive(
            ctx,
            q,
            dt_try,
            compute_rhs,
            atol=ATOL,
            rtol=RTOL,
        )
        var dt_used = ret[0]
        var dt_next = ret[1]
        # Controller contract: never step further than asked, and keep the
        # suggested next step inside the [0.2, 5] clip of the step taken.
        assert_true(
            dt_used > Real(0.0)
            and dt_used <= dt_try * (Real(1.0) + Real(1.0e-12)),
            "step taken is longer than the step requested",
        )
        assert_true(
            dt_next >= Real(0.2) * dt_used * (Real(1.0) - Real(1.0e-12))
            and dt_next <= Real(5.0) * dt_used * (Real(1.0) + Real(1.0e-12)),
            "suggested next step outside the [0.2, 5] clip",
        )
        t += dt_used
        num_steps += 1
        if dt_used < min_dt: min_dt = dt_used
        if dt_used > max_dt: max_dt = dt_used
        dt = dt_next
        if num_steps > 100000:
            break

    ctx.synchronize()

    # L2 error in ρ vs analytic IC (it returns to IC at t=1 since U=1, L=1).
    var l2 = mesh.l2_error(q, IDX_RHO, _exact_rho)

    print("Adaptive DP45 2D entropy wave (P=2, NX=32, NY=4, atol=1e-8, rtol=1e-6):")
    print("  steps taken     :", num_steps)
    print("  min dt used     :", min_dt)
    print("  max dt used     :", max_dt)
    print("  ρ L2 error      :", l2)

    assert_true(
        num_steps > 170 and num_steps < 230,
        "step count outside the band around the measured value",
    )
    assert_true(l2 < Real(2.0e-6),
                "Adaptive DP45 ρ error worse than expected")
    print("PASSED")
