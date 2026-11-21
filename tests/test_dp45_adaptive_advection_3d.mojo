# Adaptive DP45 controller test on smooth 3D advection.
#
# Initial condition q(x,y,z) = sin(2π x)·cos(2π y)·sin(2π z) advected with
# v = (1, 1, 1) on a triply-periodic [-0.5, 0.5]^3 mesh, returns exactly to
# the IC at t = 1. Starts from a CFL-violating dt so the controller must
# shrink it; then relax it once the smooth solution permits larger steps.
# Validates the 3D port of DormandPrince45_3D.

from std.math import cos, sin
from std.sys import has_accelerator
from max.gpu.host import DeviceContext
from std.testing import assert_true

from warpxm.apps.advection import Advection
from warpxm.basis.nodal_basis_1d import NodalBasis1D
from warpxm.core.types import Real, TWO_PI
from warpxm.mesh.block_3d import BlockMesh3D
from warpxm.spatial.dg_3d import DGSpatialSolver3D
from warpxm.temporal.dormand_prince_3d import DormandPrince45_3D
from warpxm.variable.state_3d import DGState3D


comptime P: Int = 2
comptime NX: Int = 12
comptime NY: Int = 12
comptime NZ: Int = 12
comptime NC: Int = 1
comptime T_FINAL: Real = 1.0
comptime DT_INIT: Real = 5.0e-2     # CFL-violating; controller must shrink
comptime ATOL: Real = 1.0e-7
comptime RTOL: Real = 1.0e-5


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = DeviceContext()
    var mesh = BlockMesh3D[P](
        ctx, Real(-0.5), Real(0.5), Real(-0.5), Real(0.5), Real(-0.5), Real(0.5),
        nx=NX, ny=NY, nz=NZ,
    )
    var q = DGState3D[P, NC](ctx, mesh.num_elements)

    var P1 = NodalBasis1D[P].num_nodes
    var ref_nodes = NodalBasis1D[P].nodes()
    var weights = NodalBasis1D[P].weights()
    var nodes_per_elem = P1 * P1 * P1

    var host = ctx.enqueue_create_host_buffer[Real.dtype](q.buf_len)
    for k in range(NZ):
        var zl = mesh.z_min + Real(k) * mesh.dz
        for j in range(NY):
            var yl = mesh.y_min + Real(j) * mesh.dy
            for i in range(NX):
                var xl = mesh.x_min + Real(i) * mesh.dx
                var e = k * NX * NY + j * NX + i
                for kn in range(P1):
                    var z = zl + (Real(1.0) + ref_nodes[kn]) * mesh.jacobian_z
                    for jn in range(P1):
                        var y = yl + (Real(1.0) + ref_nodes[jn]) * mesh.jacobian_y
                        for in_ in range(P1):
                            var x = xl + (Real(1.0) + ref_nodes[in_]) * mesh.jacobian_x
                            var n = kn * P1 * P1 + jn * P1 + in_
                            host[(e * nodes_per_elem + n) * NC + 0] = (
                                sin(TWO_PI * x) * cos(TWO_PI * y) * sin(TWO_PI * z)
                            )
    ctx.enqueue_copy(dst_buf=q.buf, src_buf=host)

    var spatial = DGSpatialSolver3D[P, NC](
        Advection[NC](
            velocity_x=Real(1.0), velocity_y=Real(1.0), velocity_z=Real(1.0),
            cfl_max=Real(1.0),
        )
    )
    var stepper = DormandPrince45_3D[P, NC](ctx, mesh.num_elements)

    def compute_rhs(
        mut q_in: DGState3D[P, NC],
        mut q_out: DGState3D[P, NC],
    ) raises {imm}:
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
        t += dt_used
        num_steps += 1
        if dt_used < min_dt: min_dt = dt_used
        if dt_used > max_dt: max_dt = dt_used
        dt = dt_next
        if num_steps > 100000:
            break

    ctx.synchronize()

    var jv = mesh.jacobian_x * mesh.jacobian_y * mesh.jacobian_z
    var acc: Real = 0.0
    with q.buf.map_to_host() as h:
        for k in range(NZ):
            var zl = mesh.z_min + Real(k) * mesh.dz
            for j in range(NY):
                var yl = mesh.y_min + Real(j) * mesh.dy
                for i in range(NX):
                    var xl = mesh.x_min + Real(i) * mesh.dx
                    var e = k * NX * NY + j * NX + i
                    for kn in range(P1):
                        var z = zl + (Real(1.0) + ref_nodes[kn]) * mesh.jacobian_z
                        for jn in range(P1):
                            var y = yl + (Real(1.0) + ref_nodes[jn]) * mesh.jacobian_y
                            for in_ in range(P1):
                                var x = xl + (Real(1.0) + ref_nodes[in_]) * mesh.jacobian_x
                                var n = kn * P1 * P1 + jn * P1 + in_
                                var exact = sin(TWO_PI * x) * cos(TWO_PI * y) * sin(TWO_PI * z)
                                var d = h[(e * nodes_per_elem + n) * NC + 0] - exact
                                var jw = jv * weights[in_] * weights[jn] * weights[kn]
                                acc += jw * d * d
    var l2 = acc ** Real(0.5)

    print("Adaptive DP45 3D advection (P=2, N=12^3, atol=1e-7, rtol=1e-5):")
    print("  steps taken     :", num_steps)
    print("  min dt used     :", min_dt)
    print("  max dt used     :", max_dt)
    print("  L2 error        :", l2)

    assert_true(num_steps < 600,
                "Adaptive DP45 3D took too many steps for smooth advection")
    assert_true(l2 < Real(2.0e-3),
                "Adaptive DP45 3D L2 error worse than fixed-step SSPRK3 baseline")
    assert_true(max_dt > min_dt * Real(1.5),
                "Controller did not vary dt — likely a bug")
    print("PASSED")
