# Adaptive DP45 controller test on smooth 3D advection.
#
# Initial condition q(x,y,z) = sin(2π x)·cos(2π y)·sin(2π z) advected with
# v = (1, 1, 1) on a triply-periodic [-0.5, 0.5]^3 mesh, returns exactly to
# the IC at t = 1. Starts from a CFL-violating dt so the controller must
# shrink it; then relax it once the smooth solution permits larger steps.
# Validates the 3D port of DormandPrince45.

from std.collections import Array
from std.math import cos, sin
from std.sys import has_accelerator
from max.gpu.host import DeviceContext
from std.testing import assert_true

from warpxm.prelude import *
from warpxm.physics.advection import *


comptime P: Int = 2
comptime NX: Int = 12
comptime NY: Int = 12
comptime NZ: Int = 12
comptime NC: Int = 1
comptime T_FINAL: Real = 1.0
comptime DT_INIT: Real = 5.0e-2     # CFL-violating; controller must shrink
comptime ATOL: Real = 1.0e-7
comptime RTOL: Real = 1.0e-5
comptime State = DGState[3, P, NC]


def _wave(x: Real, y: Real, z: Real) -> Real:
    return sin(TWO_PI * x) * cos(TWO_PI * y) * sin(TWO_PI * z)


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = DeviceContext()
    var mesh = BlockMesh[3, P](
        ctx, Real(-0.5), Real(0.5), Real(-0.5), Real(0.5), Real(-0.5), Real(0.5),
        nx=NX, ny=NY, nz=NZ,
    )
    var q = State(ctx, mesh.num_elements)

    def initial(x: Real, y: Real, z: Real) -> Array[Real, NC]:
        return [_wave(x, y, z)]
    mesh.fill(q, initial)

    var spatial = DGAdvectionSolver[3, P, NC](
        Advection[NC](
            velocity_x=Real(1.0), velocity_y=Real(1.0), velocity_z=Real(1.0),
            cfl_max=Real(1.0),
        )
    )
    var stepper = DormandPrince45[3, P, NC](ctx, mesh.num_elements)

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

    var l2 = mesh.l2_error(q, 0, _wave)

    print("Adaptive DP45 3D advection (P=2, N=12^3, atol=1e-7, rtol=1e-5):")
    print("  steps taken     :", num_steps)
    print("  min dt used     :", min_dt)
    print("  max dt used     :", max_dt)
    print("  L2 error        :", l2)

    assert_true(
        num_steps > 54 and num_steps < 74,
        "step count outside the band around the measured value",
    )
    assert_true(l2 < Real(2.0e-3),
                "Adaptive DP45 3D L2 error worse than fixed-step SSPRK3 baseline")
    print("PASSED")
