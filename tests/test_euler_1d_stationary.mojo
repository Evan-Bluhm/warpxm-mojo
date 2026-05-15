# Sanity test for the 1D Euler DG solver.
#
# Initial condition: constant uniform state with zero velocity. Volume flux
# `f_x = (0, P, 0, 0, 0)` is constant in space → zero volume contribution.
# Numerical-flux jumps are zero across every face. RHS must be exactly zero.

from std.sys import has_accelerator
from std.gpu.host import DeviceContext
from std.testing import assert_true

from warpxm.apps.euler_1d import Euler1D
from warpxm.apps.euler_2d import EULER_NC, IDX_RHO, IDX_MX, IDX_MY, IDX_MZ, IDX_E
from warpxm.core.math_compat import fabs
from warpxm.core.types import Real
from warpxm.mesh.block_1d import BlockMesh1D
from warpxm.spatial.dg_euler_1d import DGEulerSolver1D
from warpxm.temporal.ssprk3_1d import SSPRK3_1D
from warpxm.variable.state_1d import DGState1D


comptime P: Int = 2
comptime NUM_ELEMENTS: Int = 32
comptime GAMMA: Real = 5.0 / 3.0


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"

    var ctx = DeviceContext()
    var mesh = BlockMesh1D[P](
        ctx, Real(0.0), Real(1.0), NUM_ELEMENTS, periodic=True,
    )
    var q = DGState1D[P, EULER_NC](ctx, mesh.num_elements)

    var rho: Real = 1.0
    var p: Real = 1.0
    var energy: Real = p / (GAMMA - Real(1.0))

    var host = ctx.enqueue_create_host_buffer[Real.dtype](q.buf_len)
    for k in range(mesh.num_elements * (P + 1)):
        var b = k * EULER_NC
        host[b + IDX_RHO] = rho
        host[b + IDX_MX] = Real(0.0)
        host[b + IDX_MY] = Real(0.0)
        host[b + IDX_MZ] = Real(0.0)
        host[b + IDX_E] = energy
    ctx.enqueue_copy(dst_buf=q.buf, src_buf=host)

    var app = Euler1D(
        gas_gamma=GAMMA,
        min_density=Real(1e-12),
        min_pressure=Real(1e-12),
        cfl_max=Real(0.5),
    )
    var spatial = DGEulerSolver1D[P](app)
    var rhs = DGState1D[P, EULER_NC](ctx, mesh.num_elements)

    spatial.compute_rhs(ctx, mesh, q, rhs)
    ctx.synchronize()

    var max_abs: Real = 0.0
    with rhs.buf.map_to_host() as h:
        for k in range(rhs.buf_len):
            var a = fabs(h[k])
            if a > max_abs: max_abs = a
    print("max |rhs| on stationary state:", max_abs)
    assert_true(max_abs < Real(1.0e-12), "stationary 1D Euler RHS not zero")

    # Also evolve a few steps and confirm drift is exactly zero.
    var stepper = SSPRK3_1D[P, EULER_NC](ctx, mesh.num_elements)

    @parameter
    def compute_rhs(
        mut q_in: DGState1D[P, EULER_NC],
        mut q_out: DGState1D[P, EULER_NC],
    ) raises:
        spatial.compute_rhs(ctx, mesh, q_in, q_out)

    @parameter
    def no_adjust(mut q_: DGState1D[P, EULER_NC]) raises:
        pass

    var dt: Real = 1.0e-3
    for _ in range(50):
        stepper.step[compute_rhs, no_adjust](ctx, q, dt)
    ctx.synchronize()

    var max_dev: Real = 0.0
    with q.buf.map_to_host() as h:
        for k in range(mesh.num_elements * (P + 1)):
            var b = k * EULER_NC
            var d_rho = fabs(h[b + IDX_RHO] - rho)
            var d_e = fabs(h[b + IDX_E] - energy)
            if d_rho > max_dev: max_dev = d_rho
            if d_e > max_dev: max_dev = d_e
    print("max deviation after 50 steps:", max_dev)
    assert_true(max_dev < Real(1.0e-12), "stationary 1D Euler drifted")
    print("PASSED")
