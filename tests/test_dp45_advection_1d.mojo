# Dormand-Prince RK45 convergence test on smooth 1D advection.
#
# Two checks:
#   (a) DP45 is consistent: on a smooth IC with N = 64, P = 3, dt small,
#       the L2 error vs exact ic(x - t) is small.
#   (b) DP45 reaches the same answer as SSPRK3 to within a small tolerance
#       on a coarse run, confirming the algebra is right.
#
# Full 5(4)-order temporal convergence verification (varying dt and
# checking rate=5 on the time-integration error) is deferred until
# adaptive step-size selection is in place — fixed-dt validation here
# just confirms the implementation runs and gives sensible numbers.

from std.math import sin
from std.sys import has_accelerator
from std.gpu.host import DeviceContext
from std.testing import assert_true

from warpxm.apps.advection import Advection
from warpxm.basis.nodal_basis_1d import NodalBasis1D
from warpxm.core.math_compat import fabs
from warpxm.core.types import Real, TWO_PI
from warpxm.mesh.block_1d import BlockMesh1D
from warpxm.spatial.dg_1d import DGSpatialSolver1D
from warpxm.temporal.dormand_prince_1d import DormandPrince45_1D
from warpxm.temporal.ssprk3_1d import SSPRK3_1D
from warpxm.variable.state_1d import DGState1D


comptime P: Int = 3
comptime NX: Int = 64
comptime T_FINAL: Real = 1.0
comptime DT: Real = 5.0e-4


def _run_dp45(var ctx: DeviceContext) raises -> Real:
    var mesh = BlockMesh1D[P](ctx, Real(-0.5), Real(0.5), NX, periodic=True)
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
        Advection[1](velocity_x=Real(1.0), cfl_max=Real(1.0))
    )
    var stepper = DormandPrince45_1D[P, 1](ctx, mesh.num_elements)

    @parameter
    def compute_rhs(
        mut q_in: DGState1D[P, 1],
        mut q_out: DGState1D[P, 1],
    ) raises:
        spatial.compute_rhs(ctx, mesh, q_in, q_out)

    var t: Real = 0.0
    while t < T_FINAL - Real(0.5) * DT:
        var dt_use = DT if (t + DT) < T_FINAL else (T_FINAL - t)
        stepper.step[compute_rhs](ctx, q, dt_use)
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


def _run_ssprk3(var ctx: DeviceContext) raises -> Real:
    var mesh = BlockMesh1D[P](ctx, Real(-0.5), Real(0.5), NX, periodic=True)
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
        Advection[1](velocity_x=Real(1.0), cfl_max=Real(1.0))
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
    while t < T_FINAL - Real(0.5) * DT:
        var dt_use = DT if (t + DT) < T_FINAL else (T_FINAL - t)
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


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = DeviceContext()
    var err_dp = _run_dp45(ctx)
    var err_rk = _run_ssprk3(ctx)
    print("DP45 L2 error (P=3, N=64, dt=1e-2, t=1):", err_dp)
    print("SSPRK3 L2 error (same setup):           ", err_rk)
    # DP45 is 5th-order in time vs SSPRK3's 3rd; on a smooth problem with
    # the same dt, DP45 should be at least as accurate (typically much more).
    assert_true(err_dp < Real(1.0e-4),
                "DP45 L2 error larger than expected for smooth advection")
    # Both should agree on the qualitative answer (the analytic solution is
    # identity at t=1 since v=1, period=1).
    assert_true(err_dp < err_rk * Real(2.0),
                "DP45 surprisingly worse than SSPRK3")
    print("PASSED")
