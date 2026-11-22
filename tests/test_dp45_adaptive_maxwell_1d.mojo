# Adaptive DP45 on 1D Maxwell — completes DP45 + Maxwell coverage
# across all three dimensions. Same plane-wave IC as
# test_maxwell_convergence_1d.mojo.

from std.collections import Array
from std.math import sin
from std.sys import has_accelerator
from max.gpu.host import DeviceContext
from std.testing import assert_true

from warpxm.prelude import *
from warpxm.physics.maxwell import *


comptime P: Int = 2
comptime NX: Int = 32
comptime C_LIGHT: Real = 1.0
comptime T_FINAL: Real = 1.0
comptime DT_INIT: Real = 1.0e-2     # CFL-violating
comptime ATOL: Real = 1.0e-8
comptime RTOL: Real = 1.0e-6
comptime State = DGState[1, P, MAXWELL_NC]


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"

    var ctx = DeviceContext()
    var mesh = BlockMesh[1, P](ctx, Real(0.0), Real(1.0), NX, periodic=True)
    var q = State(ctx, mesh.num_elements)

    # +x-propagating plane wave: E_z = sin(2πx), B_y = −E_z / c.
    def initial(x: Real) -> Array[Real, MAXWELL_NC]:
        var s = sin(TWO_PI * x)
        return [Real(0.0), Real(0.0), s, Real(0.0), -s / C_LIGHT, Real(0.0)]
    mesh.fill(q, initial)

    var spatial = DGMaxwellSolver[1, P](Maxwell(c_light=C_LIGHT, cfl_max=Real(0.5)))
    var stepper = DormandPrince45[1, P, MAXWELL_NC](ctx, mesh.num_elements)

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

    var err_Ez: Real = 0.0
    var err_By: Real = 0.0
    var max_other: Real = 0.0
    def accumulate(
        x: Real, w: Real, v: Array[Real, MAXWELL_NC],
    ) {mut err_Ez, mut err_By, mut max_other}:
        var s = sin(TWO_PI * x)
        var d_Ez = v[IDX_EZ] - s
        var d_By = v[IDX_BY] - (-s / C_LIGHT)
        err_Ez += w * d_Ez * d_Ez
        err_By += w * d_By * d_By
        for idx in [IDX_EX, IDX_EY, IDX_BX, IDX_BZ]:
            max_other = fmax(max_other, fabs(v[idx]))
    mesh.visit_nodes(q, accumulate)
    var l2_Ez = sqrt(err_Ez)
    var l2_By = sqrt(err_By)

    print("Adaptive DP45 1D Maxwell (P=2, NX=32, atol=1e-8, rtol=1e-6):")
    print("  steps taken     :", num_steps)
    print("  min dt used     :", min_dt)
    print("  max dt used     :", max_dt)
    print("  L2 E_z          :", l2_Ez)
    print("  L2 B_y          :", l2_By)
    print("  max |E_x,E_y,B_x,B_z| :", max_other)

    # Fixed-dt SSPRK3 baseline is ~2000 steps at dt=5e-4.
    assert_true(
        num_steps > 44 and num_steps < 60,
        "step count outside the band around the measured value",
    )
    assert_true(l2_Ez < Real(3.1e-4), "E_z error worse than baseline")
    assert_true(l2_By < Real(3.1e-4), "B_y error worse than baseline")
    assert_true(max_other < Real(1.0e-12),
                "Transverse components grew — coupling spurious")
    print("PASSED")
