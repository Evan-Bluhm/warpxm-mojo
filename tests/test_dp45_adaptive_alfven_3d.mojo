# Adaptive DP45 on a 3D ideal-MHD Alfvén wave.
#
# Same physics as test_imhd_convergence_3d.mojo (a +x-propagating circularly
# polarized Alfvén wave, uniform in y / z, period 1). Exercises the
# generic DormandPrince45[3, P, MHD_NC] on the most expensive
# (8-component, 3D, MHD) spatial RHS in the framework.

from std.collections import Array
from std.math import cos, sin
from std.sys import has_accelerator
from max.gpu.host import DeviceContext
from std.testing import assert_true

from warpxm.prelude import *
from warpxm.physics.imhd import *


comptime P: Int = 2
comptime NX: Int = 16
comptime NY: Int = 4
comptime NZ: Int = 4
comptime GAMMA: Real = 5.0 / 3.0
comptime RHO0: Real = 1.0
comptime P0: Real = 1.0
comptime B0: Real = 1.0
comptime DELTA: Real = 0.1
comptime T_FINAL: Real = 1.0
comptime DT_INIT: Real = 1.0e-2  # CFL-violating
comptime ATOL: Real = 1.0e-8
comptime RTOL: Real = 1.0e-6
comptime State = DGState[3, P, MHD_NC]


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"

    var ctx = DeviceContext()
    var mesh = BlockMesh[3, P](
        ctx,
        Real(0.0),
        Real(1.0),
        Real(0.0),
        Real(1.0 / 4.0),
        Real(0.0),
        Real(1.0 / 4.0),
        nx=NX,
        ny=NY,
        nz=NZ,
    )
    var q = State(ctx, mesh.num_elements)

    var inv_sqrt_rho = Real(1.0) / sqrt(RHO0)
    var k_wave = TWO_PI

    def initial(x: Real, y: Real, z: Real) {imm} -> Array[Real, MHD_NC]:
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

    var spatial = DGImhdSolver[3, P](
        IMHD(
            gas_gamma=GAMMA,
            min_density=Real(1e-12),
            min_pressure=Real(1e-12),
            cfl_max=Real(0.5),
        )
    )
    var stepper = DormandPrince45[3, P, MHD_NC](ctx, mesh.num_elements)

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
        if dt_used < min_dt:
            min_dt = dt_used
        if dt_used > max_dt:
            max_dt = dt_used
        dt = dt_next
        if num_steps > 100000:
            break
    ctx.synchronize()

    var err_By: Real = 0.0
    var err_Bz: Real = 0.0
    var max_drho: Real = 0.0
    def accumulate(
        x: Real, y: Real, z: Real, w: Real, v: Array[Real, MHD_NC],
    ) {mut err_By, mut err_Bz, mut max_drho, imm}:
        var d_By = v[IDX_BY] - DELTA * cos(k_wave * x)
        var d_Bz = v[IDX_BZ] - DELTA * sin(k_wave * x)
        err_By += w * d_By * d_By
        err_Bz += w * d_Bz * d_Bz
        max_drho = fmax(max_drho, fabs(v[IDX_RHO] - RHO0))
    mesh.visit_nodes(q, accumulate)
    var l2_By = sqrt(err_By)
    var l2_Bz = sqrt(err_Bz)

    print("Adaptive DP45 3D MHD Alfvén (P=2, 16x4x4, atol=1e-8, rtol=1e-6):")
    print("  steps taken     :", num_steps)
    print("  min dt used     :", min_dt)
    print("  max dt used     :", max_dt)
    print("  L2 B_y          :", l2_By)
    print("  L2 B_z          :", l2_Bz)
    print("  max |Δρ|        :", max_drho)

    # SSPRK3 baseline at dt = 1e-3 is 1000 steps for t = 1.
    assert_true(
        num_steps > 111 and num_steps < 151,
        "step count outside the band around the measured value",
    )
    assert_true(l2_By < Real(7.5e-5), "B_y error worse than SSPRK3 baseline")
    assert_true(l2_Bz < Real(7.5e-5), "B_z error worse than SSPRK3 baseline")
    assert_true(
        max_drho < Real(7.0e-4),
        "density drifted — wave should stay incompressible",
    )
    print("PASSED")
