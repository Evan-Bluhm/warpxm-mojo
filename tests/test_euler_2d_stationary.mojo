# Sanity test for the 2D Euler DG solver.
#
# Initial condition: a smooth, **stationary** state with constant density and
# pressure, zero velocity. Since `f_x(q) = (0, P, 0, 0, 0)` is constant in
# space, the volume term integrates to zero exactly; numerical-flux jumps are
# also zero across every face. RHS should be machine-zero everywhere.
#
# This is the equivalent of the "constant solution → zero RHS" smoke test for
# advection, and it catches sign errors / index-mapping bugs in the new Euler
# kernel cheaply.

from std.sys import has_accelerator
from std.gpu.host import DeviceContext
from std.testing import assert_true

from warpxm.apps.euler_2d import Euler2D, EULER_NC, IDX_RHO, IDX_MX, IDX_MY, IDX_MZ, IDX_E
from warpxm.basis.nodal_basis_1d import NodalBasis1D
from warpxm.core.math_compat import fabs
from warpxm.core.types import Real
from warpxm.mesh.block_2d import BlockMesh2D
from warpxm.spatial.dg_euler_2d import DGEulerSolver2D
from warpxm.temporal.ssprk3_2d import SSPRK3_2D
from warpxm.variable.state_2d import DGState2D


comptime P: Int = 2
comptime NX: Int = 16
comptime NY: Int = 16
comptime GAMMA: Real = 5.0 / 3.0


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"

    var ctx = DeviceContext()
    var mesh = BlockMesh2D[P](
        ctx, Real(0.0), Real(1.0), Real(0.0), Real(1.0),
        nx=NX, ny=NY, periodic_x=True, periodic_y=True,
    )
    var q = DGState2D[P, EULER_NC](ctx, mesh.num_elements)

    # Constant state: ρ = 1, u = 0, P = 1 ⇒ E = P/(γ-1).
    var P1 = NodalBasis1D[P].num_nodes
    var nodes_per_elem = P1 * P1
    var rho: Real = 1.0
    var p: Real = 1.0
    var energy: Real = p / (GAMMA - Real(1.0))

    var host = ctx.enqueue_create_host_buffer[Real.dtype](q.buf_len)
    for e in range(mesh.num_elements):
        for n in range(nodes_per_elem):
            var b = (e * nodes_per_elem + n) * EULER_NC
            host[b + IDX_RHO] = rho
            host[b + IDX_MX] = Real(0.0)
            host[b + IDX_MY] = Real(0.0)
            host[b + IDX_MZ] = Real(0.0)
            host[b + IDX_E] = energy
    ctx.enqueue_copy(dst_buf=q.buf, src_buf=host)

    var app = Euler2D(
        gas_gamma=GAMMA,
        min_density=Real(1e-12),
        min_pressure=Real(1e-12),
        cfl_max=Real(0.5),
    )
    var spatial = DGEulerSolver2D[P](app)
    var rhs = DGState2D[P, EULER_NC](ctx, mesh.num_elements)

    spatial.compute_rhs(ctx, mesh, q, rhs)
    ctx.synchronize()

    var max_abs: Real = 0.0
    with rhs.buf.map_to_host() as h:
        for k in range(rhs.buf_len):
            var a = fabs(h[k])
            if a > max_abs:
                max_abs = a
    print("max |rhs| on stationary state:", max_abs)
    assert_true(max_abs < Real(1.0e-12), "stationary Euler RHS not zero")

    # Also evolve a few steps and confirm nothing drifts.
    var stepper = SSPRK3_2D[P, EULER_NC](ctx, mesh.num_elements)

    @parameter
    def compute_rhs(
        mut q_in: DGState2D[P, EULER_NC],
        mut q_out: DGState2D[P, EULER_NC],
    ) raises:
        spatial.compute_rhs(ctx, mesh, q_in, q_out)

    @parameter
    def no_adjust(mut q_: DGState2D[P, EULER_NC]) raises:
        pass

    var dt: Real = 1.0e-3
    for _ in range(50):
        stepper.step[compute_rhs, no_adjust](ctx, q, dt)
    ctx.synchronize()

    var max_dev: Real = 0.0
    with q.buf.map_to_host() as h:
        for e in range(mesh.num_elements):
            for n in range(nodes_per_elem):
                var b = (e * nodes_per_elem + n) * EULER_NC
                var d_rho = fabs(h[b + IDX_RHO] - rho)
                var d_mx = fabs(h[b + IDX_MX])
                var d_my = fabs(h[b + IDX_MY])
                var d_mz = fabs(h[b + IDX_MZ])
                var d_e = fabs(h[b + IDX_E] - energy)
                if d_rho > max_dev: max_dev = d_rho
                if d_mx > max_dev: max_dev = d_mx
                if d_my > max_dev: max_dev = d_my
                if d_mz > max_dev: max_dev = d_mz
                if d_e > max_dev: max_dev = d_e
    print("max deviation after 50 steps:", max_dev)
    assert_true(max_dev < Real(1.0e-12), "stationary Euler drifted")
    print("PASSED")
