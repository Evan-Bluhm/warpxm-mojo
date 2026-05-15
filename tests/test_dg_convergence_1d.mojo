# Convergence regression test: confirm that each polynomial order P attains the
# expected L2 rate of approximately (P + 1) on smooth advection.
#
# We test P = 1, 2, 3. P = 4 is excluded because at N ≥ 32 the L2 error already
# bottoms out at the SSPRK3 (third-order) time-integration floor and the
# observed convergence rate is dominated by that floor rather than the spatial
# discretization.

from std.math import log2, sin
from std.sys import has_accelerator
from std.gpu.host import DeviceContext
from std.testing import assert_true

from warpxm.apps.advection import Advection1D
from warpxm.basis.nodal_basis_1d import NodalBasis1D
from warpxm.core.types import Real, TWO_PI
from warpxm.mesh.block_1d import BlockMesh1D
from warpxm.spatial.dg_1d import DGSpatialSolver1D
from warpxm.temporal.ssprk3_1d import SSPRK3_1D
from warpxm.variable.state_1d import DGState1D


def _one_run[P: Int](
    var ctx: DeviceContext, num_elements: Int, dt: Real, t_final: Real,
) raises -> Real:
    var mesh = BlockMesh1D[P](ctx, Real(-0.5), Real(0.5), num_elements, periodic=True)
    var q = DGState1D[P, 1](ctx, mesh.num_elements)

    var nodes_per_elem = NodalBasis1D[P].num_nodes
    var ref_nodes = NodalBasis1D[P].nodes()
    var weights = NodalBasis1D[P].weights()

    var host = ctx.enqueue_create_host_buffer[Real.dtype](q.total_nodes)
    for e in range(mesh.num_elements):
        var xl = mesh.x_min + Real(e) * mesh.dx
        for i in range(nodes_per_elem):
            var x = xl + (Real(1.0) + ref_nodes[i]) * mesh.jacobian
            host[e * nodes_per_elem + i] = sin(TWO_PI * x)
    ctx.enqueue_copy(dst_buf=q.buf, src_buf=host)

    var spatial = DGSpatialSolver1D[P, 1](
        Advection1D[1](velocity=Real(1.0), cfl_max=Real(1.0))
    )
    var stepper = SSPRK3_1D[P, 1](ctx, mesh.num_elements)

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

    var acc: Real = 0.0
    with q.buf.map_to_host() as h:
        for e in range(mesh.num_elements):
            var xl = mesh.x_min + Real(e) * mesh.dx
            for i in range(nodes_per_elem):
                var x = xl + (Real(1.0) + ref_nodes[i]) * mesh.jacobian
                var exact = sin(TWO_PI * x)
                var d = h[e * nodes_per_elem + i] - exact
                acc += weights[i] * mesh.jacobian * d * d
    return acc ** Real(0.5)


def _check_rate[P: Int](var ctx: DeviceContext, expected: Real) raises:
    var dt: Real = 5.0e-4
    var t_final: Real = 1.0
    var err_lo = _one_run[P](ctx, 32, dt, t_final)
    var err_hi = _one_run[P](ctx, 64, dt, t_final)
    var rate = log2(err_lo / err_hi)
    print("P =", P, " err32 =", err_lo, " err64 =", err_hi, " rate =", rate)
    # Loose tolerance — the asymptotic rate isn't perfectly hit at moderate N.
    assert_true(rate > expected - Real(0.25), "Convergence rate below tolerance")


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = DeviceContext()
    _check_rate[1](ctx, Real(2.0))
    _check_rate[2](ctx, Real(3.0))
    _check_rate[3](ctx, Real(4.0))
    print("PASSED")
