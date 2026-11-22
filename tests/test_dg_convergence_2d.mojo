# 2D DG convergence regression — confirm rate of approximately (P + 1)
# on smooth 2D advection. Direct analogue of `test_dg_convergence_1d`.
#
# IC: q(x, y, 0) = sin(2π x) · cos(2π y) on [0, 1]² periodic.
# Velocity (1, 1) — diagonal advection; returns to IC at t = 1.
#
# Tests P = 1, 2, 3 by halving the element count and checking
# log2(err_lo / err_hi) ≈ P + 1.

from std.math import cos, log2, sin
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


def _one_run[P: Int](
    var ctx: DeviceContext, num_per_side: Int, dt: Real, t_final: Real,
) raises -> Real:
    var mesh = BlockMesh2D[P](
        ctx, Real(0.0), Real(1.0), Real(0.0), Real(1.0),
        nx=num_per_side, ny=num_per_side,
        periodic_x=True, periodic_y=True,
    )
    var q = DGState2D[P, 1](ctx, mesh.num_elements)

    var P1 = NodalBasis1D[P].num_nodes
    var ref_nodes = NodalBasis1D[P].nodes()
    var weights = NodalBasis1D[P].weights()
    var nodes_per_elem = P1 * P1

    var host = ctx.enqueue_create_host_buffer[Real.dtype](q.buf_len)
    for j in range(num_per_side):
        var yl = mesh.y_min + Real(j) * mesh.dy
        for i in range(num_per_side):
            var xl = mesh.x_min + Real(i) * mesh.dx
            var e = j * num_per_side + i
            for jn in range(P1):
                var y = yl + (Real(1.0) + ref_nodes[jn]) * mesh.jacobian_y
                for in_ in range(P1):
                    var x = xl + (Real(1.0) + ref_nodes[in_]) * mesh.jacobian_x
                    var n = jn * P1 + in_
                    host[(e * nodes_per_elem + n) * 1 + 0] = sin(TWO_PI * x) * cos(TWO_PI * y)
    ctx.enqueue_copy(dst_buf=q.buf, src_buf=host)

    var spatial = DGSpatialSolver2D[P, 1](
        Advection[1](velocity_x=Real(1.0), velocity_y=Real(1.0), cfl_max=Real(1.0))
    )
    var stepper = SSPRK3_2D[P, 1](ctx, mesh.num_elements)

    def compute_rhs(
        mut q_in: DGState2D[P, 1],
        mut q_out: DGState2D[P, 1],
    ) raises {imm}:
        spatial.compute_rhs(ctx, mesh, q_in, q_out)

    def no_adjust(mut q_: DGState2D[P, 1]) raises {imm}:
        pass

    var t: Real = 0.0
    while t < t_final - Real(0.5) * dt:
        var dt_use = dt if (t + dt) < t_final else (t_final - t)
        stepper.step(ctx, q, dt_use, compute_rhs, no_adjust)
        t += dt_use
    ctx.synchronize()

    var ja = mesh.jacobian_x * mesh.jacobian_y
    var acc: Real = 0.0
    with q.buf.map_to_host() as h:
        for j in range(num_per_side):
            var yl = mesh.y_min + Real(j) * mesh.dy
            for i in range(num_per_side):
                var xl = mesh.x_min + Real(i) * mesh.dx
                var e = j * num_per_side + i
                for jn in range(P1):
                    var y = yl + (Real(1.0) + ref_nodes[jn]) * mesh.jacobian_y
                    for in_ in range(P1):
                        var x = xl + (Real(1.0) + ref_nodes[in_]) * mesh.jacobian_x
                        var exact = sin(TWO_PI * x) * cos(TWO_PI * y)
                        var n = jn * P1 + in_
                        var d = h[(e * nodes_per_elem + n) * 1 + 0] - exact
                        var jw = ja * weights[in_] * weights[jn]
                        acc += jw * d * d
    return acc ** Real(0.5)


def _check_rate[P: Int](
    var ctx: DeviceContext, expected: Real, tol_lo: Real, tol_hi: Real,
) raises:
    """Check the observed rate and bound both errors absolutely.

    `tol_lo` / `tol_hi` are about 3x the measured coarse / fine errors, so
    a defect that scales the error uniformly (and leaves the rate intact)
    still fails.
    """
    var dt: Real = 2.5e-4
    var t_final: Real = 1.0
    var err_lo = _one_run[P](ctx, 16, dt, t_final)
    var err_hi = _one_run[P](ctx, 32, dt, t_final)
    var rate = log2(err_lo / err_hi)
    assert_true(err_lo < tol_lo, "coarse-mesh L2 error above tolerance")
    assert_true(err_hi < tol_hi, "fine-mesh L2 error above tolerance")
    print("P =", P, " err16 =", err_lo, " err32 =", err_hi, " rate =", rate)
    assert_true(rate > expected - Real(0.25), "Convergence rate below tolerance")


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = DeviceContext()
    _check_rate[1](ctx, Real(2.0), Real(3.3e-1), Real(9.0e-2))
    _check_rate[2](ctx, Real(3.0), Real(2.5e-3), Real(3.0e-4))
    _check_rate[3](ctx, Real(4.0), Real(5.0e-5), Real(3.2e-6))
    print("PASSED")
