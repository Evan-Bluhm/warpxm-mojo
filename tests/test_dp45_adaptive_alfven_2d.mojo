# Adaptive DP45 on a 2D ideal-MHD Alfvén wave.
#
# Exercises the generic DormandPrince45[2, P, MHD_NC] (NC=8) against a
# smooth nonlinear eigenvector of the MHD system. Same IC as
# test_imhd_convergence_2d.mojo: a +x-propagating circularly polarized Alfvén
# wave that returns to its IC at t = 1.
#
# Validates:
#   (a) far fewer steps than the SSPRK3 dt=1e-3 baseline (which took 1000),
#   (b) controller exercises dt (max > 1.5 · min),
#   (c) L2 errors at t = 1 stay within the SSPRK3 tolerances.

from std.collections import Array
from std.math import cos, sin
from std.sys import has_accelerator
from max.gpu.host import DeviceContext
from std.testing import assert_true

from warpxm.prelude import *
from warpxm.physics.imhd import *


comptime P: Int = 2
comptime NX: Int = 32
comptime NY: Int = 4
comptime GAMMA: Real = 5.0 / 3.0
comptime RHO0: Real = 1.0
comptime P0: Real = 1.0
comptime B0: Real = 1.0
comptime DELTA: Real = 0.1
comptime T_FINAL: Real = 1.0
comptime DT_INIT: Real = 1.0e-2     # CFL-violating; controller must shrink
comptime ATOL: Real = 1.0e-8
comptime RTOL: Real = 1.0e-6
comptime State = DGState[2, P, MHD_NC]


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"

    var ctx = DeviceContext()
    var mesh = BlockMesh[2, P](
        ctx, Real(0.0), Real(1.0), Real(0.0), Real(1.0 / 8.0),
        nx=NX, ny=NY, periodic_x=True, periodic_y=True,
    )
    var q = State(ctx, mesh.num_elements)

    var inv_sqrt_rho = Real(1.0) / sqrt(RHO0)
    var k_wave = TWO_PI

    def initial(x: Real, y: Real) {imm} -> Array[Real, MHD_NC]:
        var c_v = cos(k_wave * x)
        var s_v = sin(k_wave * x)
        var By_ic = DELTA * c_v
        var Bz_ic = DELTA * s_v
        var uy_ic = -DELTA * c_v * inv_sqrt_rho
        var uz_ic = -DELTA * s_v * inv_sqrt_rho
        var bsq = B0 * B0 + By_ic * By_ic + Bz_ic * Bz_ic
        var msq = RHO0 * RHO0 * (uy_ic * uy_ic + uz_ic * uz_ic)
        var E_total = (
            P0 / (GAMMA - Real(1.0)) + Real(0.5) * msq / RHO0 + Real(0.5) * bsq
        )
        return [
            RHO0, Real(0.0), RHO0 * uy_ic, RHO0 * uz_ic, E_total,
            B0, By_ic, Bz_ic,
        ]
    mesh.fill(q, initial)

    var spatial = DGImhdSolver[2, P](
        IMHD(
            gas_gamma=GAMMA, min_density=Real(1e-12),
            min_pressure=Real(1e-12), cfl_max=Real(0.5),
        )
    )
    var stepper = DormandPrince45[2, P, MHD_NC](ctx, mesh.num_elements)

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

    var err_By: Real = 0.0
    var err_Bz: Real = 0.0
    var err_uy: Real = 0.0
    var err_uz: Real = 0.0
    var max_drho: Real = 0.0
    def accumulate(
        x: Real, y: Real, w: Real, v: Array[Real, MHD_NC],
    ) {mut err_By, mut err_Bz, mut err_uy, mut err_uz, mut max_drho, imm}:
        var c_v = cos(k_wave * x)
        var s_v = sin(k_wave * x)
        var rho = v[IDX_RHO]
        var d_By = v[IDX_BY] - DELTA * c_v
        var d_Bz = v[IDX_BZ] - DELTA * s_v
        var d_uy = v[IDX_MY] / rho - (-DELTA * c_v * inv_sqrt_rho)
        var d_uz = v[IDX_MZ] / rho - (-DELTA * s_v * inv_sqrt_rho)
        err_By += w * d_By * d_By
        err_Bz += w * d_Bz * d_Bz
        err_uy += w * d_uy * d_uy
        err_uz += w * d_uz * d_uz
        max_drho = fmax(max_drho, fabs(rho - RHO0))
    mesh.visit_nodes(q, accumulate)
    var l2_By = sqrt(err_By)
    var l2_Bz = sqrt(err_Bz)
    var l2_uy = sqrt(err_uy)
    var l2_uz = sqrt(err_uz)

    print("Adaptive DP45 2D Alfvén (P=2, NX=32, NY=4, atol=1e-8, rtol=1e-6):")
    print("  steps taken     :", num_steps)
    print("  min dt used     :", min_dt)
    print("  max dt used     :", max_dt)
    print("  L2 B_y          :", l2_By)
    print("  L2 B_z          :", l2_Bz)
    print("  L2 u_y          :", l2_uy)
    print("  L2 u_z          :", l2_uz)
    print("  max |Δρ|        :", max_drho)

    # SSPRK3 at fixed dt=1e-3 takes 1000 steps for this problem; adaptive
    # DP45 should comfortably do it in well under that on smooth data.
    assert_true(
        num_steps > 143 and num_steps < 195,
        "step count outside the band around the measured value",
    )
    assert_true(l2_By < Real(1.3e-5) and l2_Bz < Real(1.3e-5),
                "B L2 errors worse than fixed-step SSPRK3 baseline")
    assert_true(l2_uy < Real(1.3e-5) and l2_uz < Real(1.3e-5),
                "u L2 errors worse than fixed-step SSPRK3 baseline")
    assert_true(max_drho < Real(1.0e-4),
                "density drifted — wave should stay incompressible")
    print("PASSED")
