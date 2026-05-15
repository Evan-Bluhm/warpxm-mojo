# 2D scalar advection on a doubly-periodic block mesh.
#
# Initial condition: q(x, y, 0) = sin(2π x) · cos(2π y) on [-0.5, 0.5]².
# Velocity (v_x, v_y) = (1, 1). Exact solution at t = 1 (one period in each
# axis) returns to the IC.
#
# Build / run:
#     pixi run mojo run -I src examples/advection_2d.mojo

from std.math import sin, cos
from std.sys import has_accelerator
from std.gpu.host import DeviceContext

from warpxm.apps.advection_2d import Advection2D
from warpxm.basis.nodal_basis_1d import NodalBasis1D
from warpxm.core.types import Real, TWO_PI
from warpxm.mesh.block_2d import BlockMesh2D
from warpxm.spatial.dg_2d import DGSpatialSolver2D
from warpxm.temporal.ssprk3_2d import SSPRK3_2D
from warpxm.variable.state_2d import DGState2D


comptime P: Int = 2
comptime NC: Int = 1
comptime NX: Int = 32
comptime NY: Int = 32
comptime DT: Real = 5.0e-4
comptime T_FINAL: Real = 1.0


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"

    var ctx = DeviceContext()
    var mesh = BlockMesh2D[P](
        ctx, Real(-0.5), Real(0.5), Real(-0.5), Real(0.5),
        nx=NX, ny=NY, periodic_x=True, periodic_y=True,
    )
    var q = DGState2D[P, NC](ctx, mesh.num_elements)

    # IC: q = sin(2π x) · cos(2π y)
    var P1 = NodalBasis1D[P].num_nodes
    var ref_nodes = NodalBasis1D[P].nodes()
    var weights = NodalBasis1D[P].weights()
    var nodes_per_elem = P1 * P1
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
                    var n = jn * P1 + in_
                    host[(e * nodes_per_elem + n) * NC + 0] = sin(TWO_PI * x) * cos(TWO_PI * y)
    ctx.enqueue_copy(dst_buf=q.buf, src_buf=host)

    var app = Advection2D[NC](velocity_x=Real(1.0), velocity_y=Real(1.0), cfl_max=Real(1.0))
    var spatial = DGSpatialSolver2D[P, NC](app)
    var stepper = SSPRK3_2D[P, NC](ctx, mesh.num_elements)

    var t: Real = 0.0
    var step: Int = 0
    while t < T_FINAL - Real(0.5) * DT:
        var dt_use = DT if (t + DT) < T_FINAL else (T_FINAL - t)
        stepper.step(ctx, mesh, spatial, q, dt_use)
        t += dt_use
        step += 1
    ctx.synchronize()

    # L2 error vs IC
    var acc: Real = 0.0
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
                        var n = jn * P1 + in_
                        var exact = sin(TWO_PI * x) * cos(TWO_PI * y)
                        var diff = h[(e * nodes_per_elem + n) * NC + 0] - exact
                        acc += ja * weights[in_] * weights[jn] * diff * diff
    var l2 = acc ** Real(0.5)
    print("Integrated to t =", t, "in", step, "steps")
    print("L2 error after one period:", l2)
