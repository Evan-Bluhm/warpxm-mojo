# 2D Sod shock tube — extrudes the 1D Sod problem along y.
#
# Initial discontinuity at x = 0.5; uniform in y. Top/bottom walls are
# free-slip (= periodic in this case since no y-flux ever develops); left
# and right are copyout outflows.
#
# Expected outcome at t = 0.2:
#   * Result is uniform in y at every x (within machine precision).
#   * Profile in x matches the 1D Sod analytic solution to the same
#     tolerance as the 1D test.

from std.math import isnan
from std.sys import has_accelerator
from std.gpu.host import DeviceContext
from std.testing import assert_true

from warpxm.adjusters.moe_rossmanith import MoeRossmanith2D
from warpxm.apps.euler import (
    Euler, EULER_NC, IDX_RHO, IDX_MX, IDX_MY, IDX_MZ, IDX_E,
)
from warpxm.basis.nodal_basis_1d import NodalBasis1D
from warpxm.core.bc import BC_COPYOUT, BC_WALL_FREESLIP
from warpxm.core.math_compat import fabs
from warpxm.core.types import Real
from warpxm.mesh.block_2d import BlockMesh2D
from warpxm.spatial.dg_euler_2d import DGEulerSolver2D
from warpxm.temporal.ssprk3_2d import SSPRK3_2D
from warpxm.variable.state_2d import DGState2D


comptime P: Int = 2
comptime NX: Int = 200
comptime NY: Int = 8
comptime GAMMA: Real = 1.4
comptime T_FINAL: Real = 0.2
comptime DT: Real = 2.0e-4


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"

    var ctx = DeviceContext()
    var mesh = BlockMesh2D[P](
        ctx, Real(0.0), Real(1.0), Real(0.0), Real(0.1),
        nx=NX, ny=NY, periodic_x=False, periodic_y=False,
        bc_left=BC_COPYOUT, bc_right=BC_COPYOUT,
        bc_bottom=BC_WALL_FREESLIP, bc_top=BC_WALL_FREESLIP,
    )
    var q = DGState2D[P, EULER_NC](ctx, mesh.num_elements)

    var P1 = NodalBasis1D[P].num_nodes
    var ref_nodes = NodalBasis1D[P].nodes()
    var nodes_per_elem = P1 * P1

    # IC: classic Sod step in x, uniform in y.
    var host = ctx.enqueue_create_host_buffer[Real.dtype](q.buf_len)
    for j in range(NY):
        for i in range(NX):
            var xl = mesh.x_min + Real(i) * mesh.dx
            var e = j * NX + i
            for jn in range(P1):
                for in_ in range(P1):
                    var x = xl + (Real(1.0) + ref_nodes[in_]) * mesh.jacobian_x
                    var rho: Real
                    var pp: Real
                    if x < Real(0.5):
                        rho = 1.0
                        pp = 1.0
                    else:
                        rho = 0.125
                        pp = 0.1
                    var n = jn * P1 + in_
                    var b = (e * nodes_per_elem + n) * EULER_NC
                    host[b + IDX_RHO] = rho
                    host[b + IDX_MX] = Real(0.0)
                    host[b + IDX_MY] = Real(0.0)
                    host[b + IDX_MZ] = Real(0.0)
                    host[b + IDX_E] = pp / (GAMMA - Real(1.0))
    ctx.enqueue_copy(dst_buf=q.buf, src_buf=host)

    var app = Euler(
        gas_gamma=GAMMA,
        min_density=Real(1e-12),
        min_pressure=Real(1e-12),
        cfl_max=Real(0.5),
    )
    var spatial = DGEulerSolver2D[P](app)
    var stepper = SSPRK3_2D[P, EULER_NC](ctx, mesh.num_elements)
    var limiter = MoeRossmanith2D[P, EULER_NC](ctx, mesh.num_elements)

    @parameter
    def compute_rhs(
        mut q_in: DGState2D[P, EULER_NC],
        mut q_out: DGState2D[P, EULER_NC],
    ) raises:
        spatial.compute_rhs(ctx, mesh, q_in, q_out)

    @parameter
    def adjust(mut q: DGState2D[P, EULER_NC]) raises:
        limiter.apply(ctx, mesh, q)

    var t: Real = 0.0
    while t < T_FINAL - Real(0.5) * DT:
        var dt_use = DT if (t + DT) < T_FINAL else (T_FINAL - t)
        stepper.step[compute_rhs, adjust](ctx, q, dt_use)
        t += dt_use
    ctx.synchronize()

    # Pull the entire field back and validate.
    var mid_node = nodes_per_elem // 2
    var any_nan = False
    var rho_at = InlineArray[Real, NX](fill=Real(0.0))   # row 0 sample
    var max_y_drift: Real = 0.0
    with q.buf.map_to_host() as h:
        for j in range(NY):
            for i in range(NX):
                var e = j * NX + i
                var b = (e * nodes_per_elem + mid_node) * EULER_NC
                var rho_ji = h[b + IDX_RHO]
                if isnan(rho_ji):
                    any_nan = True
                if j == 0:
                    rho_at[i] = rho_ji
                else:
                    # Compare every other row to row 0 at the same x.
                    var d = fabs(rho_ji - rho_at[i])
                    if d > max_y_drift: max_y_drift = d

    assert_true(not any_nan, "2D Sod result contains NaN")
    print("max y-drift in ρ (should be ~0):", max_y_drift)
    assert_true(max_y_drift < Real(1.0e-10),
                "result not uniform in y — 1D-extruded structure broken")

    # Sample x-profile values: plateaus from 1D Sod analytic.
    var ix01 = Int(0.1 / mesh.dx)
    var ix06 = Int(0.6 / mesh.dx)
    var ix075 = Int(0.75 / mesh.dx)
    var ix095 = Int(0.95 / mesh.dx)
    print("rho at x=0.1   :", rho_at[ix01], "  (analytic 1.0)")
    print("rho at x=0.6   :", rho_at[ix06], "  (analytic 0.4263)")
    print("rho at x=0.75  :", rho_at[ix075], "  (analytic 0.2655)")
    print("rho at x=0.95  :", rho_at[ix095], "  (analytic 0.125)")
    assert_true(fabs(rho_at[ix01] - Real(1.0)) < Real(0.02), "left state off")
    assert_true(fabs(rho_at[ix06] - Real(0.4263)) < Real(0.03), "post-rarefaction off")
    assert_true(fabs(rho_at[ix075] - Real(0.2655)) < Real(0.03), "post-contact off")
    assert_true(fabs(rho_at[ix095] - Real(0.125)) < Real(0.02), "right state off")
    print("PASSED")
