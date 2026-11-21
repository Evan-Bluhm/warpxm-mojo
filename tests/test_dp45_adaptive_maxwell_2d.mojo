# Adaptive DP45 on a 2D Maxwell plane wave.
#
# Same TM-mode IC as test_maxwell_2d_plane_wave.mojo (E_z = sin(2π(x+y)),
# B propagates on the diagonal). The system is linear and perfectly
# smooth, so DP45 should pull a much larger dt than the SSPRK3 baseline
# (5e-4 fixed × ~1414 steps for t = 1/√2 ≈ 0.7071) and still return to
# the IC within the SSPRK3 tolerances.

from std.math import sin
from std.sys import has_accelerator
from max.gpu.host import DeviceContext
from std.testing import assert_true

from warpxm.apps.maxwell import (
    Maxwell, MAXWELL_NC, IDX_EX, IDX_EY, IDX_EZ, IDX_BX, IDX_BY, IDX_BZ,
)
from warpxm.basis.nodal_basis_1d import NodalBasis1D
from warpxm.core.math_compat import sqrt
from warpxm.core.types import Real, TWO_PI
from warpxm.mesh.block_2d import BlockMesh2D
from warpxm.spatial.dg_maxwell_2d import DGMaxwellSolver2D
from warpxm.temporal.dormand_prince_2d import DormandPrince45_2D
from warpxm.variable.state_2d import DGState2D


comptime P: Int = 2
comptime NX: Int = 32
comptime NY: Int = 32
comptime C_LIGHT: Real = 1.0
comptime DT_INIT: Real = 1.0e-2     # CFL-violating; controller must shrink
comptime ATOL: Real = 1.0e-8
comptime RTOL: Real = 1.0e-6


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"

    var ctx = DeviceContext()
    var mesh = BlockMesh2D[P](
        ctx, Real(0.0), Real(1.0), Real(0.0), Real(1.0),
        nx=NX, ny=NY, periodic_x=True, periodic_y=True,
    )
    var q = DGState2D[P, MAXWELL_NC](ctx, mesh.num_elements)

    var P1 = NodalBasis1D[P].num_nodes
    var ref_nodes = NodalBasis1D[P].nodes()
    var weights = NodalBasis1D[P].weights()
    var nodes_per_elem = P1 * P1

    var inv_sqrt2: Real = Real(1.0) / sqrt(Real(2.0))

    var host = ctx.enqueue_create_host_buffer[Real.dtype](q.buf_len)
    for j in range(NY):
        var yl = mesh.y_min + Real(j) * mesh.dy
        for i in range(NX):
            var xl = mesh.x_min + Real(i) * mesh.dx
            var e = j * NX + i
            for jn in range(P1):
                var y = yl + (Real(1.0) + ref_nodes[jn]) * mesh.jacobian_y
                for in_ in range(P1):
                    var x = xl + (Real(1.0) + ref_nodes[in_]) * mesh.jacobian_x
                    var s = sin(TWO_PI * (x + y))
                    var n = jn * P1 + in_
                    var b = (e * nodes_per_elem + n) * MAXWELL_NC
                    host[b + IDX_EX] = Real(0.0)
                    host[b + IDX_EY] = Real(0.0)
                    host[b + IDX_EZ] = s
                    host[b + IDX_BX] = inv_sqrt2 * s
                    host[b + IDX_BY] = -inv_sqrt2 * s
                    host[b + IDX_BZ] = Real(0.0)
    ctx.enqueue_copy(dst_buf=q.buf, src_buf=host)

    var spatial = DGMaxwellSolver2D[P](Maxwell(c_light=C_LIGHT, cfl_max=Real(0.5)))
    var stepper = DormandPrince45_2D[P, MAXWELL_NC](ctx, mesh.num_elements)

    var t_final: Real = inv_sqrt2

    def compute_rhs(
        mut q_in: DGState2D[P, MAXWELL_NC],
        mut q_out: DGState2D[P, MAXWELL_NC],
    ) raises {imm}:
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
        t += dt_used
        num_steps += 1
        if dt_used < min_dt: min_dt = dt_used
        if dt_used > max_dt: max_dt = dt_used
        dt = dt_next
        if num_steps > 100000:
            break
    ctx.synchronize()

    var err_ez: Real = 0.0
    var err_bx: Real = 0.0
    var err_by: Real = 0.0
    var ja = mesh.jacobian_x * mesh.jacobian_y
    with q.buf.map_to_host() as h:
        for j in range(NY):
            var yl = mesh.y_min + Real(j) * mesh.dy
            for i in range(NX):
                var xl = mesh.x_min + Real(i) * mesh.dx
                var e = j * NX + i
                for jn in range(P1):
                    var y = yl + (Real(1.0) + ref_nodes[jn]) * mesh.jacobian_y
                    for in_ in range(P1):
                        var x = xl + (Real(1.0) + ref_nodes[in_]) * mesh.jacobian_x
                        var s = sin(TWO_PI * (x + y))
                        var n = jn * P1 + in_
                        var b = (e * nodes_per_elem + n) * MAXWELL_NC
                        var d_ez = h[b + IDX_EZ] - s
                        var d_bx = h[b + IDX_BX] - inv_sqrt2 * s
                        var d_by = h[b + IDX_BY] - (-inv_sqrt2 * s)
                        var jw = ja * weights[in_] * weights[jn]
                        err_ez += jw * d_ez * d_ez
                        err_bx += jw * d_bx * d_bx
                        err_by += jw * d_by * d_by
    var l2_ez = err_ez ** Real(0.5)
    var l2_bx = err_bx ** Real(0.5)
    var l2_by = err_by ** Real(0.5)

    print("Adaptive DP45 2D Maxwell (P=2, 32x32, atol=1e-8, rtol=1e-6):")
    print("  steps taken     :", num_steps)
    print("  min dt used     :", min_dt)
    print("  max dt used     :", max_dt)
    print("  L2 E_z          :", l2_ez)
    print("  L2 B_x          :", l2_bx)
    print("  L2 B_y          :", l2_by)

    # SSPRK3 baseline at dt = 5e-4 is ~1414 steps for t = 1/√2; adaptive
    # DP45 should be a sizeable fraction of that.
    assert_true(num_steps < 700,
                "Adaptive DP45 took too many steps for smooth Maxwell wave")
    assert_true(l2_ez < Real(5.0e-3), "E_z error worse than SSPRK3 baseline")
    assert_true(l2_bx < Real(5.0e-3), "B_x error worse than SSPRK3 baseline")
    assert_true(l2_by < Real(5.0e-3), "B_y error worse than SSPRK3 baseline")
    assert_true(max_dt > min_dt * Real(1.5),
                "Controller did not vary dt — likely a bug")
    print("PASSED")
