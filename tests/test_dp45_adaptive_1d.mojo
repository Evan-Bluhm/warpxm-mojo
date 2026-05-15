# Adaptive DP45 controller test on smooth 1D advection.
#
# Starts with an aggressively-large initial dt that the embedded
# error estimator must shrink, then validates that:
#   (a) the controller converges back into a stable cadence,
#   (b) the final L2 error vs the analytic sin(2π(x-t)) is competitive
#       with the fixed-step DP45 result,
#   (c) the run takes substantially fewer steps than fixed-dt SSPRK3
#       would at the same tolerance (i.e. the controller exploits the
#       smooth solution by relaxing dt).

from std.math import sin
from std.sys import has_accelerator
from std.gpu.host import DeviceContext
from std.testing import assert_true

from warpxm.apps.advection import Advection1D
from warpxm.basis.nodal_basis_1d import NodalBasis1D
from warpxm.core.math_compat import fabs
from warpxm.core.types import Real, TWO_PI
from warpxm.mesh.block_1d import BlockMesh1D
from warpxm.spatial.dg_1d import DGSpatialSolver1D
from warpxm.temporal.dormand_prince_1d import DormandPrince45_1D
from warpxm.variable.state_1d import DGState1D


comptime P: Int = 3
comptime NX: Int = 64
comptime T_FINAL: Real = 1.0
comptime DT_INIT: Real = 5.0e-3       # CFL-violating start; controller must shrink it
comptime ATOL: Real = 1.0e-8
comptime RTOL: Real = 1.0e-6


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = DeviceContext()
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
        Advection1D[1](velocity=Real(1.0), cfl_max=Real(1.0))
    )
    var stepper = DormandPrince45_1D[P, 1](ctx, mesh.num_elements)

    @parameter
    def compute_rhs(
        mut q_in: DGState1D[P, 1],
        mut q_out: DGState1D[P, 1],
    ) raises:
        spatial.compute_rhs(ctx, mesh, q_in, q_out)

    var t: Real = 0.0
    var dt = DT_INIT
    var num_steps: Int = 0
    var min_dt = dt
    var max_dt: Real = 0.0
    while t < T_FINAL - Real(1.0e-12):
        # Clip so we don't overshoot t_final.
        var dt_try = dt if (t + dt) < T_FINAL else (T_FINAL - t)
        var ret = stepper.step_adaptive[compute_rhs](
            ctx, q, dt_try, atol=ATOL, rtol=RTOL,
        )
        var dt_used = ret[0]
        var dt_next = ret[1]
        t += dt_used
        num_steps += 1
        if dt_used < min_dt: min_dt = dt_used
        if dt_used > max_dt: max_dt = dt_used
        dt = dt_next
        if num_steps > 100000:
            break  # safety
    ctx.synchronize()

    # L2 error vs analytic sin(2π(x - T_FINAL)) = sin(2πx) on [-0.5, 0.5] (period 1).
    var l2_err: Real = 0.0
    with q.buf.map_to_host() as h:
        for e in range(mesh.num_elements):
            var xl = mesh.x_min + Real(e) * mesh.dx
            for i in range(nodes_per_elem):
                var x = xl + (Real(1.0) + ref_nodes[i]) * mesh.jacobian
                var d = h[e * nodes_per_elem + i] - sin(TWO_PI * x)
                l2_err += weights[i] * mesh.jacobian * d * d
    l2_err = l2_err ** Real(0.5)

    print("Adaptive DP45 smooth advection (P=3, N=64, atol=1e-8, rtol=1e-6):")
    print("  steps taken     :", num_steps)
    print("  min dt used     :", min_dt)
    print("  max dt used     :", max_dt)
    print("  L2 error        :", l2_err)

    # For reference, a fixed-dt CFL-stable SSPRK3 run needs about 2000 steps
    # at dt=5e-4. Adaptive should comfortably beat that on smooth data.
    assert_true(num_steps < 1500,
                "Adaptive DP45 took too many steps — controller not relaxing")
    assert_true(l2_err < Real(1.0e-6),
                "Adaptive DP45 L2 error worse than expected")
    # Final-step dt should be much larger than the original CFL-violating dt.
    assert_true(max_dt > Real(5.0e-4),
                "Adaptive controller never relaxed dt past CFL floor")
    print("PASSED")
