# 2D Euler hydrostatic-balance regression with uniform gravity.
#
# IC (a column under downward gravity):
#
#     ρ(x, y) = 1
#     u(x, y) = v(x, y) = 0
#     P(x, y) = P_0 - y                   (linear in y; P_0 chosen so P > 0)
#     E      = P / (γ - 1)
#     g      = (0, -1)
#
# Hydrostatic balance: ∂P/∂y = ρ · g_y, i.e. -1 = 1 · (-1). The continuous
# RHS for ρv is `-∂P/∂y + ρ·g_y = 1 + (-1) = 0`. With the strong-form DG
# discretization at P ≥ 1, the discrete ∂P/∂y is exact for linear P, so the
# gravity source kernel cancels the pressure-gradient flux divergence
# exactly. The state should sit still to machine precision.
#
# We integrate for ~50 timesteps and assert max|v|, max|u|, and the maximum
# deviation of ρ and E from their initial values all stay below 1e-12.

from std.sys import has_accelerator
from std.gpu.host import DeviceContext
from std.testing import assert_true

from warpxm.apps.euler import (
    Euler, EULER_NC, IDX_RHO, IDX_MX, IDX_MY, IDX_MZ, IDX_E,
)
from warpxm.basis.nodal_basis_1d import NodalBasis1D
from warpxm.core.bc import BC_WALL_FREESLIP
from warpxm.core.math_compat import fabs
from warpxm.core.types import Real
from warpxm.mesh.block_2d import BlockMesh2D
from warpxm.spatial.dg_euler_2d import DGEulerSolver2D
from warpxm.spatial.euler_2d_sources import apply_gravity_source_2d
from warpxm.temporal.ssprk3_2d import SSPRK3_2D
from warpxm.variable.state_2d import DGState2D


comptime P: Int = 2
comptime NX: Int = 8
comptime NY: Int = 16
comptime GAMMA: Real = 5.0 / 3.0
comptime P0: Real = 2.0
comptime GX: Real = 0.0
comptime GY: Real = -1.0


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"

    var ctx = DeviceContext()
    var mesh = BlockMesh2D[P](
        ctx, Real(0.0), Real(1.0), Real(0.0), Real(1.0),
        nx=NX, ny=NY, periodic_x=True, periodic_y=False,
        bc_bottom=BC_WALL_FREESLIP, bc_top=BC_WALL_FREESLIP,
    )
    var q = DGState2D[P, EULER_NC](ctx, mesh.num_elements)

    var P1 = NodalBasis1D[P].num_nodes
    var ref_nodes = NodalBasis1D[P].nodes()
    var nodes_per_elem = P1 * P1

    # Pre-compute the per-node initial state so we can diff against it later.
    var host = ctx.enqueue_create_host_buffer[Real.dtype](q.buf_len)
    var rho0 = Real(1.0)
    for j in range(NY):
        var yl = mesh.y_min + Real(j) * mesh.dy
        for i in range(NX):
            var e = j * NX + i
            for jn in range(P1):
                var y = yl + (Real(1.0) + ref_nodes[jn]) * mesh.jacobian_y
                var p_local = P0 - y
                var energy_local = p_local / (GAMMA - Real(1.0))
                for in_ in range(P1):
                    var n = jn * P1 + in_
                    var b = (e * nodes_per_elem + n) * EULER_NC
                    host[b + IDX_RHO] = rho0
                    host[b + IDX_MX] = Real(0.0)
                    host[b + IDX_MY] = Real(0.0)
                    host[b + IDX_MZ] = Real(0.0)
                    host[b + IDX_E] = energy_local
    ctx.enqueue_copy(dst_buf=q.buf, src_buf=host)

    # Snapshot the IC for later diff.
    var q0_host = ctx.enqueue_create_host_buffer[Real.dtype](q.buf_len)
    ctx.enqueue_copy(dst_buf=q0_host, src_buf=q.buf)
    ctx.synchronize()

    var app = Euler(
        gas_gamma=GAMMA,
        min_density=Real(1e-12),
        min_pressure=Real(1e-12),
        cfl_max=Real(0.5),
    )
    var spatial = DGEulerSolver2D[P](app)
    var stepper = SSPRK3_2D[P, EULER_NC](ctx, mesh.num_elements)

    @parameter
    def compute_rhs(
        mut q_in: DGState2D[P, EULER_NC],
        mut q_out: DGState2D[P, EULER_NC],
    ) raises:
        spatial.compute_rhs(ctx, mesh, q_in, q_out)
        apply_gravity_source_2d[P](ctx, q_in, q_out, GX, GY)

    @parameter
    def no_adjust(mut q_: DGState2D[P, EULER_NC]) raises:
        pass

    var dt: Real = 1.0e-3
    for _ in range(50):
        stepper.step[compute_rhs, no_adjust](ctx, q, dt)
    ctx.synchronize()

    var max_v_abs: Real = 0.0
    var max_u_abs: Real = 0.0
    var max_rho_drift: Real = 0.0
    var max_e_drift: Real = 0.0
    with q.buf.map_to_host() as h:
        for k in range(mesh.num_elements * nodes_per_elem):
            var b = k * EULER_NC
            var u_abs = fabs(h[b + IDX_MX])
            var v_abs = fabs(h[b + IDX_MY])
            var rho_d = fabs(h[b + IDX_RHO] - q0_host[b + IDX_RHO])
            var e_d = fabs(h[b + IDX_E] - q0_host[b + IDX_E])
            if u_abs > max_u_abs: max_u_abs = u_abs
            if v_abs > max_v_abs: max_v_abs = v_abs
            if rho_d > max_rho_drift: max_rho_drift = rho_d
            if e_d > max_e_drift: max_e_drift = e_d

    print("After 50 timesteps under hydrostatic balance:")
    print("  max |u|             =", max_u_abs)
    print("  max |v|             =", max_v_abs)
    print("  max |Δρ| vs IC      =", max_rho_drift)
    print("  max |ΔE| vs IC      =", max_e_drift)
    assert_true(max_u_abs < Real(1.0e-12), "horizontal velocity grew")
    assert_true(max_v_abs < Real(1.0e-12), "vertical velocity grew — hydrostatic balance broken")
    assert_true(max_rho_drift < Real(1.0e-12), "density drifted")
    assert_true(max_e_drift < Real(1.0e-12), "energy drifted")
    print("PASSED")
