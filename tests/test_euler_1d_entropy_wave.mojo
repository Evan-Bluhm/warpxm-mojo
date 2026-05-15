# 1D 5-moment Euler entropy-wave regression.
#
# A smooth density perturbation `ρ(x, 0) = 1 + ε sin(2π x)` riding on a
# uniform flow `u = 1`, with constant pressure `P = 1`. Pure entropy waves
# are linearly degenerate: density advects at speed u while pressure and
# velocity stay constant. After one period the solution returns to the IC.

from std.math import sin
from std.sys import has_accelerator
from std.gpu.host import DeviceContext
from std.testing import assert_true

from warpxm.apps.euler_1d import Euler1D
from warpxm.apps.euler_2d import EULER_NC, IDX_RHO, IDX_MX, IDX_MY, IDX_MZ, IDX_E
from warpxm.basis.nodal_basis_1d import NodalBasis1D
from warpxm.core.types import Real, TWO_PI
from warpxm.mesh.block_1d import BlockMesh1D
from warpxm.spatial.dg_euler_1d import DGEulerSolver1D
from warpxm.temporal.ssprk3_1d import SSPRK3_1D
from warpxm.variable.state_1d import DGState1D


comptime P: Int = 2
comptime NUM_ELEMENTS: Int = 64
comptime GAMMA: Real = 5.0 / 3.0
comptime EPS: Real = 0.01
comptime U: Real = 1.0


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"

    var ctx = DeviceContext()
    var mesh = BlockMesh1D[P](
        ctx, Real(0.0), Real(1.0), NUM_ELEMENTS, periodic=True,
    )
    var q = DGState1D[P, EULER_NC](ctx, mesh.num_elements)

    var nodes_per_elem = NodalBasis1D[P].num_nodes
    var ref_nodes = NodalBasis1D[P].nodes()
    var weights = NodalBasis1D[P].weights()

    var pp: Real = 1.0

    var host = ctx.enqueue_create_host_buffer[Real.dtype](q.buf_len)
    for e in range(mesh.num_elements):
        var xl = mesh.x_min + Real(e) * mesh.dx
        for i in range(nodes_per_elem):
            var x = xl + (Real(1.0) + ref_nodes[i]) * mesh.jacobian
            var rho = Real(1.0) + EPS * sin(TWO_PI * x)
            var ke = Real(0.5) * rho * U * U
            var b = (e * nodes_per_elem + i) * EULER_NC
            host[b + IDX_RHO] = rho
            host[b + IDX_MX] = rho * U
            host[b + IDX_MY] = Real(0.0)
            host[b + IDX_MZ] = Real(0.0)
            host[b + IDX_E] = pp / (GAMMA - Real(1.0)) + ke
    ctx.enqueue_copy(dst_buf=q.buf, src_buf=host)

    var app = Euler1D(
        gas_gamma=GAMMA,
        min_density=Real(1e-12),
        min_pressure=Real(1e-12),
        cfl_max=Real(0.5),
    )
    var spatial = DGEulerSolver1D[P](app)
    var stepper = SSPRK3_1D[P, EULER_NC](ctx, mesh.num_elements)

    var dt: Real = 1.0e-3
    var t_final: Real = 1.0

    @parameter
    def compute_rhs(
        mut q_in: DGState1D[P, EULER_NC],
        mut q_out: DGState1D[P, EULER_NC],
    ) raises:
        spatial.compute_rhs(ctx, mesh, q_in, q_out)

    @parameter
    def no_adjust(mut q_: DGState1D[P, EULER_NC]) raises:
        pass

    var t: Real = 0.0
    while t < t_final - Real(0.5) * dt:
        var dt_use = dt if (t + dt) < t_final else (t_final - t)
        stepper.step[compute_rhs, no_adjust](ctx, q, dt_use)
        t += dt_use
    ctx.synchronize()

    var rho_err: Real = 0.0
    var mass_drift: Real = 0.0
    with q.buf.map_to_host() as h:
        for e in range(mesh.num_elements):
            var xl = mesh.x_min + Real(e) * mesh.dx
            for i in range(nodes_per_elem):
                var x = xl + (Real(1.0) + ref_nodes[i]) * mesh.jacobian
                var exact = Real(1.0) + EPS * sin(TWO_PI * x)
                var b = (e * nodes_per_elem + i) * EULER_NC
                var d = h[b + IDX_RHO] - exact
                var jw = mesh.jacobian * weights[i]
                rho_err += jw * d * d
                mass_drift += jw * (h[b + IDX_RHO] - Real(1.0))
    var l2 = rho_err ** Real(0.5)
    print("ρ L2 error vs exact:", l2)
    print("net mass drift:", mass_drift)
    assert_true(l2 < Real(1.0e-4), "Euler 1D entropy-wave error too large")
    assert_true(mass_drift < Real(1.0e-12) and mass_drift > -Real(1.0e-12),
                "mass not conserved")
    print("PASSED")
