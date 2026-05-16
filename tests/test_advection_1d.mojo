# Regression test for the 1D advection vertical slice.
#
# Build / run:
#     pixi run mojo run -I src tests/test_advection_1d.mojo
#
# Asserts the L2 error after one period on a 64-element periodic mesh stays
# below a generous tolerance. With P=2, dt=1e-3 we expect ≲ 5e-5.

from std.math import ceildiv, sin
from std.sys import has_accelerator
from std.gpu.host import DeviceContext
from std.testing import assert_true

from warpxm.apps.advection import Advection
from warpxm.basis.nodal_basis_1d import NodalBasis1D
from warpxm.core.types import Real, TWO_PI
from warpxm.mesh.block_1d import BlockMesh1D
from warpxm.spatial.dg_1d import DGSpatialSolver1D
from warpxm.temporal.ssprk3_1d import SSPRK3_1D
from warpxm.variable.state_1d import DGState1D


comptime P: Int = 2
comptime NUM_ELEMENTS: Int = 64


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"

    var ctx = DeviceContext()
    var mesh = BlockMesh1D[P](ctx, Real(-0.5), Real(0.5), NUM_ELEMENTS, periodic=True)
    var q = DGState1D[P, 1](ctx, mesh.num_elements)

    # Initial condition: sin(2π x).
    var nodes_per_elem = NodalBasis1D[P].num_nodes
    var ref_nodes = NodalBasis1D[P].nodes()
    var host = ctx.enqueue_create_host_buffer[Real.dtype](q.total_nodes)
    for e in range(mesh.num_elements):
        var xl = mesh.x_min + Real(e) * mesh.dx
        for i in range(nodes_per_elem):
            var x = xl + (Real(1.0) + ref_nodes[i]) * mesh.jacobian
            host[e * nodes_per_elem + i] = sin(TWO_PI * x)
    ctx.enqueue_copy(dst_buf=q.buf, src_buf=host)

    # Integrate one period.
    var app = Advection[1](velocity_x=Real(1.0), cfl_max=Real(1.0))
    var spatial = DGSpatialSolver1D[P, 1](app)
    var stepper = SSPRK3_1D[P, 1](ctx, mesh.num_elements)
    var dt: Real = 1.0e-3
    var t_final: Real = 1.0

    @parameter
    def compute_rhs(
        mut q_in: DGState1D[P, 1],
        mut q_out: DGState1D[P, 1],
    ) raises:
        spatial.compute_rhs(ctx, mesh, q_in, q_out)

    @parameter
    def no_adjust(mut q_: DGState1D[P, 1]) raises:
        pass

    var t: Real = 0.0
    while t < t_final - Real(0.5) * dt:
        var dt_use = dt if (t + dt) < t_final else (t_final - t)
        stepper.step[compute_rhs, no_adjust](ctx, q, dt_use)
        t += dt_use

    ctx.synchronize()

    # L2 error vs IC.
    var weights = NodalBasis1D[P].weights()
    var acc: Real = 0.0
    with q.buf.map_to_host() as h:
        for e in range(mesh.num_elements):
            var xl = mesh.x_min + Real(e) * mesh.dx
            for i in range(nodes_per_elem):
                var x = xl + (Real(1.0) + ref_nodes[i]) * mesh.jacobian
                var exact = sin(TWO_PI * x)
                var d = h[e * nodes_per_elem + i] - exact
                acc += weights[i] * mesh.jacobian * d * d
    var l2 = acc ** Real(0.5)

    print("L2 error after one period:", l2)
    assert_true(l2 < Real(5.0e-5), "1D advection L2 error exceeded tolerance")
    print("PASSED")
