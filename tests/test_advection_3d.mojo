# 3D scalar advection on a triply-periodic block mesh.
#
# Initial condition:
#     q(x, y, z, 0) = sin(2π x) · cos(2π y) · sin(2π z)
# Velocity: (1, 1, 1). Period along the (1,1,1) diagonal is 1 in each axis
# (after one full period in each direction = same as one full period in t
# because every direction wraps), so at t = 1 the IC returns exactly.
#
# Validates the whole 3D stack: BlockMesh3D, DGState3D, Advection3D,
# DGSpatialSolver3D, SSPRK3_3D. Mass conservation falls out of
# periodic + flux-conservative DG; L2 closure against the IC is the main
# accuracy check.

from std.math import cos, sin
from std.sys import has_accelerator
from std.gpu.host import DeviceContext
from std.testing import assert_true

from warpxm.apps.advection_3d import Advection3D
from warpxm.basis.nodal_basis_1d import NodalBasis1D
from warpxm.core.math_compat import fabs
from warpxm.core.types import Real, TWO_PI
from warpxm.mesh.block_3d import BlockMesh3D
from warpxm.spatial.dg_3d import DGSpatialSolver3D
from warpxm.temporal.ssprk3_3d import SSPRK3_3D
from warpxm.variable.state_3d import DGState3D


comptime P: Int = 2
comptime NX: Int = 16
comptime NY: Int = 16
comptime NZ: Int = 16
comptime NC: Int = 1
comptime DT: Real = 1.0e-3


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

    var app = Advection3D[NC](
        velocity_x=Real(1.0), velocity_y=Real(1.0), velocity_z=Real(1.0),
        cfl_max=Real(1.0),
    )
    var spatial = DGSpatialSolver3D[P, NC](app)
    var stepper = SSPRK3_3D[P, NC](ctx, mesh.num_elements)

    var t_final: Real = 1.0

    @parameter
    def compute_rhs(
        mut q_in: DGState3D[P, NC],
        mut q_out: DGState3D[P, NC],
    ) raises:
        spatial.compute_rhs(ctx, mesh, q_in, q_out)

    @parameter
    def no_adjust(mut q_: DGState3D[P, NC]) raises:
        pass

    var t: Real = 0.0
    var step: Int = 0
    while t < t_final - Real(0.5) * DT:
        var dt_use = DT if (t + DT) < t_final else (t_final - t)
        stepper.step[compute_rhs, no_adjust](ctx, q, dt_use)
        t += dt_use
        step += 1
    ctx.synchronize()

    # L2 error vs IC.
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
    print("3D advection — t =", t, " (", step, "steps)")
    print("  L2 error after one period:", l2)
    assert_true(l2 < Real(2.0e-3), "3D advection L2 error too large")
    print("PASSED")
