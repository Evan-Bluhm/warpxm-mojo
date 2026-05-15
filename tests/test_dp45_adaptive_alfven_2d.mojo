# Adaptive DP45 on a 2D ideal-MHD Alfvén wave.
#
# Exercises the generic DormandPrince45_2D[P, MHD_NC] (NC=8) against a
# smooth nonlinear eigenvector of the MHD system. Same IC as
# test_imhd_alfven_2d.mojo: a +x-propagating circularly polarized Alfvén
# wave that returns to its IC at t = 1.
#
# Validates:
#   (a) far fewer steps than the SSPRK3 dt=1e-3 baseline (which took 1000),
#   (b) controller exercises dt (max > 1.5 · min),
#   (c) L2 errors at t = 1 stay within the SSPRK3 tolerances.

from std.math import cos, sin
from std.sys import has_accelerator
from std.gpu.host import DeviceContext
from std.testing import assert_true

from warpxm.apps.imhd_2d import (
    IMHD2D, MHD_NC, IDX_RHO, IDX_MX, IDX_MY, IDX_MZ, IDX_E,
    IDX_BX, IDX_BY, IDX_BZ,
)
from warpxm.basis.nodal_basis_1d import NodalBasis1D
from warpxm.core.math_compat import sqrt
from warpxm.core.types import Real, TWO_PI
from warpxm.mesh.block_2d import BlockMesh2D
from warpxm.spatial.dg_imhd_2d import DGImhdSolver2D
from warpxm.temporal.dormand_prince_2d import DormandPrince45_2D
from warpxm.variable.state_2d import DGState2D


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


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"

    var ctx = DeviceContext()
    var mesh = BlockMesh2D[P](
        ctx, Real(0.0), Real(1.0), Real(0.0), Real(1.0 / 8.0),
        nx=NX, ny=NY, periodic_x=True, periodic_y=True,
    )
    var q = DGState2D[P, MHD_NC](ctx, mesh.num_elements)

    var P1 = NodalBasis1D[P].num_nodes
    var ref_nodes = NodalBasis1D[P].nodes()
    var weights = NodalBasis1D[P].weights()
    var nodes_per_elem = P1 * P1
    var inv_sqrt_rho = Real(1.0) / sqrt(RHO0)
    var k_wave = TWO_PI

    var host = ctx.enqueue_create_host_buffer[Real.dtype](q.buf_len)
    for j in range(NY):
        for i in range(NX):
            var xl = mesh.x_min + Real(i) * mesh.dx
            var e = j * NX + i
            for jn in range(P1):
                for in_ in range(P1):
                    var x = xl + (Real(1.0) + ref_nodes[in_]) * mesh.jacobian_x
                    var c_v = cos(k_wave * x)
                    var s_v = sin(k_wave * x)
                    var By_ic = DELTA * c_v
                    var Bz_ic = DELTA * s_v
                    var uy_ic = -DELTA * c_v * inv_sqrt_rho
                    var uz_ic = -DELTA * s_v * inv_sqrt_rho

                    var bsq = B0 * B0 + By_ic * By_ic + Bz_ic * Bz_ic
                    var msq = RHO0 * RHO0 * (uy_ic * uy_ic + uz_ic * uz_ic)
                    var E_total = P0 / (GAMMA - Real(1.0)) + Real(0.5) * msq / RHO0 + Real(0.5) * bsq

                    var n = jn * P1 + in_
                    var b = (e * nodes_per_elem + n) * MHD_NC
                    host[b + IDX_RHO] = RHO0
                    host[b + IDX_MX] = Real(0.0)
                    host[b + IDX_MY] = RHO0 * uy_ic
                    host[b + IDX_MZ] = RHO0 * uz_ic
                    host[b + IDX_E] = E_total
                    host[b + IDX_BX] = B0
                    host[b + IDX_BY] = By_ic
                    host[b + IDX_BZ] = Bz_ic
    ctx.enqueue_copy(dst_buf=q.buf, src_buf=host)

    var spatial = DGImhdSolver2D[P](
        IMHD2D(
            gas_gamma=GAMMA, min_density=Real(1e-12),
            min_pressure=Real(1e-12), cfl_max=Real(0.5),
        )
    )
    var stepper = DormandPrince45_2D[P, MHD_NC](ctx, mesh.num_elements)

    @parameter
    def compute_rhs(
        mut q_in: DGState2D[P, MHD_NC],
        mut q_out: DGState2D[P, MHD_NC],
    ) raises:
        spatial.compute_rhs(ctx, mesh, q_in, q_out)

    var t: Real = 0.0
    var dt = DT_INIT
    var num_steps: Int = 0
    var min_dt = dt
    var max_dt: Real = 0.0
    while t < T_FINAL - Real(1.0e-12):
        var dt_try = dt if (t + dt) < T_FINAL else (T_FINAL - t)
        var ret = stepper.step_adaptive[compute_rhs](
            ctx, q, dt_try, atol=ATOL, rtol=RTOL,
        )
        var dt_used = ret[0]
        var dt_next = ret[1]
        t += dt_used
        num_steps += 1
        if dt_used < min_dt: min_dt = dt_used
        if dt_used > max_dt: max_dt = dt_used
        dt = dt_next
        if num_steps > 100000:
            break
    ctx.synchronize()

    var ja = mesh.jacobian_x * mesh.jacobian_y
    var err_By: Real = 0.0
    var err_Bz: Real = 0.0
    var err_uy: Real = 0.0
    var err_uz: Real = 0.0
    var max_drho: Real = 0.0
    with q.buf.map_to_host() as h:
        for j in range(NY):
            for i in range(NX):
                var xl = mesh.x_min + Real(i) * mesh.dx
                var e = j * NX + i
                for jn in range(P1):
                    for in_ in range(P1):
                        var x = xl + (Real(1.0) + ref_nodes[in_]) * mesh.jacobian_x
                        var c_v = cos(k_wave * x)
                        var s_v = sin(k_wave * x)
                        var By_exact = DELTA * c_v
                        var Bz_exact = DELTA * s_v
                        var uy_exact = -DELTA * c_v * inv_sqrt_rho
                        var uz_exact = -DELTA * s_v * inv_sqrt_rho

                        var n = jn * P1 + in_
                        var b = (e * nodes_per_elem + n) * MHD_NC
                        var rho = h[b + IDX_RHO]
                        var uy = h[b + IDX_MY] / rho
                        var uz = h[b + IDX_MZ] / rho

                        var d_By = h[b + IDX_BY] - By_exact
                        var d_Bz = h[b + IDX_BZ] - Bz_exact
                        var d_uy = uy - uy_exact
                        var d_uz = uz - uz_exact
                        var d_rho = rho - RHO0
                        var jw = ja * weights[in_] * weights[jn]
                        err_By += jw * d_By * d_By
                        err_Bz += jw * d_Bz * d_Bz
                        err_uy += jw * d_uy * d_uy
                        err_uz += jw * d_uz * d_uz
                        if d_rho > max_drho: max_drho = d_rho
                        if -d_rho > max_drho: max_drho = -d_rho
    var l2_By = err_By ** Real(0.5)
    var l2_Bz = err_Bz ** Real(0.5)
    var l2_uy = err_uy ** Real(0.5)
    var l2_uz = err_uz ** Real(0.5)

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
    assert_true(num_steps < 500,
                "Adaptive DP45 took too many steps for smooth Alfvén wave")
    assert_true(l2_By < Real(2.0e-3) and l2_Bz < Real(2.0e-3),
                "B L2 errors worse than fixed-step SSPRK3 baseline")
    assert_true(l2_uy < Real(2.0e-3) and l2_uz < Real(2.0e-3),
                "u L2 errors worse than fixed-step SSPRK3 baseline")
    assert_true(max_drho < Real(1.0e-3),
                "density drifted — wave should stay incompressible")
    assert_true(max_dt > min_dt * Real(1.5),
                "Controller did not vary dt — likely a bug")
    print("PASSED")
