# 2D multi-component advection — direct analogue of
# test_advection_multi_component.mojo. Catches NC-indexing bugs in the
# 2D DG kernel that NC=1 would silently miss.
#
# IC: two independent tracers
#   q_0(x, y, 0) = sin(2π x) · cos(2π y)
#   q_1(x, y, 0) = cos(2π x) · sin(2π y)
# advected with velocity (1, 1) on [0, 1]² periodic. After t=1 both
# tracers return to their ICs.

from std.math import sin, cos
from std.sys import has_accelerator
from max.gpu.host import DeviceContext
from std.testing import assert_true

from warpxm.apps.advection import Advection
from warpxm.basis.nodal_basis_1d import NodalBasis1D
from warpxm.core.types import Real, TWO_PI
from warpxm.mesh.block_2d import BlockMesh2D
from warpxm.spatial.dg_2d import DGSpatialSolver2D
from warpxm.temporal.ssprk3_2d import SSPRK3_2D
from warpxm.variable.state_2d import DGState2D


comptime P: Int = 2
comptime NC: Int = 2
comptime NX: Int = 32
comptime NY: Int = 32


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"

    var ctx = DeviceContext()
    var mesh = BlockMesh2D[P](
        ctx, Real(0.0), Real(1.0), Real(0.0), Real(1.0),
        nx=NX, ny=NY, periodic_x=True, periodic_y=True,
    )
    var q = DGState2D[P, NC](ctx, mesh.num_elements)

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
                    var base = (e * nodes_per_elem + n) * NC
                    host[base + 0] = sin(TWO_PI * x) * cos(TWO_PI * y)
                    host[base + 1] = cos(TWO_PI * x) * sin(TWO_PI * y)
    ctx.enqueue_copy(dst_buf=q.buf, src_buf=host)

    var spatial = DGSpatialSolver2D[P, NC](
        Advection[NC](velocity_x=Real(1.0), velocity_y=Real(1.0), cfl_max=Real(1.0))
    )
    var stepper = SSPRK3_2D[P, NC](ctx, mesh.num_elements)

    var dt: Real = 1.0e-3
    var t_final: Real = 1.0

    def compute_rhs(
        mut q_in: DGState2D[P, NC],
        mut q_out: DGState2D[P, NC],
    ) raises {imm}:
        spatial.compute_rhs(ctx, mesh, q_in, q_out)

    def no_adjust(mut q_: DGState2D[P, NC]) raises {imm}:
        pass

    var t: Real = 0.0
    while t < t_final - Real(0.5) * dt:
        var dt_use = dt if (t + dt) < t_final else (t_final - t)
        stepper.step(ctx, q, dt_use, compute_rhs, no_adjust)
        t += dt_use
    ctx.synchronize()

    var ja = mesh.jacobian_x * mesh.jacobian_y
    var err0: Real = 0.0
    var err1: Real = 0.0
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
                        var base = (e * nodes_per_elem + n) * NC
                        var d0 = h[base + 0] - sin(TWO_PI * x) * cos(TWO_PI * y)
                        var d1 = h[base + 1] - cos(TWO_PI * x) * sin(TWO_PI * y)
                        var jw = ja * weights[in_] * weights[jn]
                        err0 += jw * d0 * d0
                        err1 += jw * d1 * d1
    var l2_0 = err0 ** Real(0.5)
    var l2_1 = err1 ** Real(0.5)

    print("Component 0 (sin·cos) L2 error:", l2_0)
    print("Component 1 (cos·sin) L2 error:", l2_1)
    assert_true(l2_0 < Real(2.0e-4), "component 0 error too large")
    assert_true(l2_1 < Real(2.0e-4), "component 1 error too large")
    print("PASSED")
