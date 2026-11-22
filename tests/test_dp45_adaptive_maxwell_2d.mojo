# Adaptive DP45 on a 2D Maxwell plane wave.
#
# Same TM-mode IC as test_maxwell_plane_wave_2d.mojo (E_z = sin(2π(x+y)),
# B propagates on the diagonal). The system is linear and perfectly
# smooth, so DP45 should pull a much larger dt than the SSPRK3 baseline
# (5e-4 fixed × ~1414 steps for t = 1/√2 ≈ 0.7071) and still return to
# the IC within the SSPRK3 tolerances.

from std.collections import Array
from std.math import sin
from std.sys import has_accelerator
from max.gpu.host import DeviceContext
from std.testing import assert_true

from warpxm.prelude import *
from warpxm.physics.maxwell import *


comptime P: Int = 2
comptime NX: Int = 32
comptime NY: Int = 32
comptime C_LIGHT: Real = 1.0
comptime DT_INIT: Real = 1.0e-2     # CFL-violating; controller must shrink
comptime ATOL: Real = 1.0e-8
comptime RTOL: Real = 1.0e-6
comptime State = DGState[2, P, MAXWELL_NC]


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"

    var ctx = DeviceContext()
    var mesh = BlockMesh[2, P](
        ctx, Real(0.0), Real(1.0), Real(0.0), Real(1.0),
        nx=NX, ny=NY, periodic_x=True, periodic_y=True,
    )
    var q = State(ctx, mesh.num_elements)

    var inv_sqrt2: Real = Real(1.0) / sqrt(Real(2.0))

    def initial(x: Real, y: Real) {imm} -> Array[Real, MAXWELL_NC]:
        var s = sin(TWO_PI * (x + y))
        return [
            Real(0.0), Real(0.0), s, inv_sqrt2 * s, -inv_sqrt2 * s, Real(0.0),
        ]
    mesh.fill(q, initial)

    var spatial = DGMaxwellSolver[2, P](Maxwell(c_light=C_LIGHT, cfl_max=Real(0.5)))
    var stepper = DormandPrince45[2, P, MAXWELL_NC](ctx, mesh.num_elements)

    var t_final: Real = inv_sqrt2

    def compute_rhs(mut q_in: State, mut q_out: State) raises {imm}:
        spatial.compute_rhs(ctx, mesh, q_in, q_out)

    var t: Real = 0.0
    var dt = DT_INIT
    var num_steps: Int = 0
    var min_dt = dt
    var max_dt: Real = 0.0
    while t < t_final - Real(1.0e-12):
        var dt_try = dt if (t + dt) < t_final else (t_final - t)
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

    def exact_ez(x: Real, y: Real) -> Real:
        return sin(TWO_PI * (x + y))
    def exact_bx(x: Real, y: Real) {imm} -> Real:
        return inv_sqrt2 * sin(TWO_PI * (x + y))
    def exact_by(x: Real, y: Real) {imm} -> Real:
        return -inv_sqrt2 * sin(TWO_PI * (x + y))
    var l2_ez = mesh.l2_error(q, IDX_EZ, exact_ez)
    var l2_bx = mesh.l2_error(q, IDX_BX, exact_bx)
    var l2_by = mesh.l2_error(q, IDX_BY, exact_by)

    print("Adaptive DP45 2D Maxwell (P=2, 32x32, atol=1e-8, rtol=1e-6):")
    print("  steps taken     :", num_steps)
    print("  min dt used     :", min_dt)
    print("  max dt used     :", max_dt)
    print("  L2 E_z          :", l2_ez)
    print("  L2 B_x          :", l2_bx)
    print("  L2 B_y          :", l2_by)

    # SSPRK3 baseline at dt = 5e-4 is ~1414 steps for t = 1/√2; adaptive
    # DP45 should be a sizeable fraction of that.
    assert_true(
        num_steps > 64 and num_steps < 88,
        "step count outside the band around the measured value",
    )
    assert_true(l2_ez < Real(3.7e-4), "E_z error worse than SSPRK3 baseline")
    assert_true(l2_bx < Real(2.8e-4), "B_x error worse than SSPRK3 baseline")
    assert_true(l2_by < Real(2.8e-4), "B_y error worse than SSPRK3 baseline")
    print("PASSED")
