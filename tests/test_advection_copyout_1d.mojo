# Non-periodic 1D advection with copyout BCs on both ends.
#
# Set-up:
#   * Domain x ∈ [0, 1], 64 elements at P = 2.
#   * Initial condition: a smooth Gaussian pulse centred at x = 0.25.
#   * Velocity v = +1, so the pulse drifts toward the right (outflow) boundary.
#   * Both ends use `BC_COPYOUT` (zero-gradient).
#
# Expected behaviour at t = 1:
#   * Pulse has fully exited through the right boundary.
#   * Domain is approximately flat at the IC's leftmost value (≈ 0) since
#     copyout at the left boundary keeps soft-feeding the original baseline.
#   * No spurious oscillations / reflections.
#
# The pulse has a small support (≈ 0.05 wide); at t = 1 it sits centred at
# x = 1.25, i.e. well outside the domain, so the post-test L2 norm of `q`
# should be ≪ the initial peak height.

from std.math import exp
from std.sys import has_accelerator
from std.gpu.host import DeviceContext
from std.testing import assert_true

from warpxm.apps.advection import Advection1D
from warpxm.basis.nodal_basis_1d import NodalBasis1D
from warpxm.core.bc import BC_COPYOUT
from warpxm.core.math_compat import fabs
from warpxm.core.types import Real
from warpxm.mesh.block_1d import BlockMesh1D
from warpxm.spatial.dg_1d import DGSpatialSolver1D
from warpxm.temporal.ssprk3_1d import SSPRK3_1D
from warpxm.variable.state_1d import DGState1D


comptime P: Int = 2
comptime NUM_ELEMENTS: Int = 64
comptime SIGMA: Real = 0.025
comptime X0: Real = 0.25


@always_inline
def _gauss(x: Real) -> Real:
    var z = (x - X0) / SIGMA
    return exp(-Real(0.5) * z * z)


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"

    var ctx = DeviceContext()
    var mesh = BlockMesh1D[P](
        ctx, Real(0.0), Real(1.0), NUM_ELEMENTS, periodic=False,
        bc_left=BC_COPYOUT, bc_right=BC_COPYOUT,
    )
    var q = DGState1D[P, 1](ctx, mesh.num_elements)

    var nodes_per_elem = NodalBasis1D[P].num_nodes
    var ref_nodes = NodalBasis1D[P].nodes()

    var host = ctx.enqueue_create_host_buffer[Real.dtype](q.buf_len)
    var initial_peak: Real = 0.0
    for e in range(mesh.num_elements):
        var xl = mesh.x_min + Real(e) * mesh.dx
        for i in range(nodes_per_elem):
            var x = xl + (Real(1.0) + ref_nodes[i]) * mesh.jacobian
            var v = _gauss(x)
            host[e * nodes_per_elem + i] = v
            if v > initial_peak: initial_peak = v
    ctx.enqueue_copy(dst_buf=q.buf, src_buf=host)

    var app = Advection1D[1](velocity=Real(1.0), cfl_max=Real(1.0))
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

    # Check max |q|. The pulse should have exited; remaining "echo" should be
    # tiny (well below 1% of the initial peak).
    var max_abs: Real = 0.0
    with q.buf.map_to_host() as h:
        for k in range(q.buf_len):
            var a = fabs(h[k])
            if a > max_abs: max_abs = a
    print("initial peak =", initial_peak)
    print("final max|q| =", max_abs)
    assert_true(
        max_abs < Real(1.0e-3) * initial_peak,
        "pulse did not exit cleanly through copyout BC",
    )
    print("PASSED")
