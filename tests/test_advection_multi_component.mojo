# Multi-component regression: advect two independent tracer fields (sin and
# cos) together at NC = 2 and verify each lands back on itself after one
# period. Catches indexing / per-component bugs that NC = 1 would silently
# miss.

from std.math import sin, cos
from std.sys import has_accelerator
from max.gpu.host import DeviceContext
from std.testing import assert_true

from warpxm.apps.advection import Advection
from warpxm.basis.nodal_basis_1d import NodalBasis1D
from warpxm.core.types import Real, TWO_PI
from warpxm.mesh.block_1d import BlockMesh1D
from warpxm.spatial.dg_1d import DGSpatialSolver1D
from warpxm.temporal.ssprk3_1d import SSPRK3_1D
from warpxm.variable.state_1d import DGState1D


comptime P: Int = 2
comptime NC: Int = 2
comptime NUM_ELEMENTS: Int = 64


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"

    var ctx = DeviceContext()
    var mesh = BlockMesh1D[P](ctx, Real(-0.5), Real(0.5), NUM_ELEMENTS, periodic=True)
    var q = DGState1D[P, NC](ctx, mesh.num_elements)

    # Component 0: sin(2π x). Component 1: cos(2π x).
    var nodes_per_elem = NodalBasis1D[P].num_nodes
    var ref_nodes = NodalBasis1D[P].nodes()
    var weights = NodalBasis1D[P].weights()
    var host = ctx.enqueue_create_host_buffer[Real.dtype](q.buf_len)
    for e in range(mesh.num_elements):
        var xl = mesh.x_min + Real(e) * mesh.dx
        for i in range(nodes_per_elem):
            var x = xl + (Real(1.0) + ref_nodes[i]) * mesh.jacobian
            var base = (e * nodes_per_elem + i) * NC
            host[base + 0] = sin(TWO_PI * x)
            host[base + 1] = cos(TWO_PI * x)
    ctx.enqueue_copy(dst_buf=q.buf, src_buf=host)

    var app = Advection[NC](velocity_x=Real(1.0), cfl_max=Real(1.0))
    var spatial = DGSpatialSolver1D[P, NC](app)
    var stepper = SSPRK3_1D[P, NC](ctx, mesh.num_elements)

    var dt: Real = 1.0e-3
    var t_final: Real = 1.0

    def compute_rhs(
        mut q_in: DGState1D[P, NC],
        mut q_out: DGState1D[P, NC],
    ) raises {imm}:
        spatial.compute_rhs(ctx, mesh, q_in, q_out)

    def no_adjust(mut q_: DGState1D[P, NC]) raises {imm}:
        pass

    var t: Real = 0.0
    while t < t_final - Real(0.5) * dt:
        var dt_use = dt if (t + dt) < t_final else (t_final - t)
        stepper.step(ctx, q, dt_use, compute_rhs, no_adjust)
        t += dt_use
    ctx.synchronize()

    var err0: Real = 0.0
    var err1: Real = 0.0
    with q.buf.map_to_host() as h:
        for e in range(mesh.num_elements):
            var xl = mesh.x_min + Real(e) * mesh.dx
            for i in range(nodes_per_elem):
                var x = xl + (Real(1.0) + ref_nodes[i]) * mesh.jacobian
                var base = (e * nodes_per_elem + i) * NC
                var d0 = h[base + 0] - sin(TWO_PI * x)
                var d1 = h[base + 1] - cos(TWO_PI * x)
                var jw = weights[i] * mesh.jacobian
                err0 += jw * d0 * d0
                err1 += jw * d1 * d1
    var l2_0 = err0 ** Real(0.5)
    var l2_1 = err1 ** Real(0.5)

    print("Component 0 (sin) L2 error:", l2_0)
    print("Component 1 (cos) L2 error:", l2_1)
    assert_true(l2_0 < Real(5.0e-5), "sin component error too large")
    assert_true(l2_1 < Real(5.0e-5), "cos component error too large")
    print("PASSED")
