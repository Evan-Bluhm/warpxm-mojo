# 2D 5-moment Euler entropy wave on a doubly-periodic domain.
#
# Initial state:
#
#     ρ(x, y, 0) = 1 + ε · sin(2π (x + y))
#     u_x = u_y = 1,  u_z = 0
#     P = 1        (γ = 5/3)
#     E = P/(γ-1) + ρ |u|² / 2
#
# Because the velocity is uniform and the pressure is constant, the
# perturbation is a contact / entropy wave — it is linearly degenerate and
# advects with the flow without dispersion. The exact solution at time `t`
# is the IC translated by `(u_x t, u_y t)`. After t = 1 (one period in each
# axis) the solution returns to the IC.
#
# This is a non-trivial test that exercises every component of the Euler
# kernel: continuity, momentum, energy, and the Rusanov dissipation.

from std.math import sin
from std.sys import has_accelerator
from max.gpu.host import DeviceContext

from warpxm.apps.euler import Euler, EULER_NC, IDX_RHO, IDX_MX, IDX_MY, IDX_MZ, IDX_E
from warpxm.basis.nodal_basis_1d import NodalBasis1D
from warpxm.core.types import Real, TWO_PI
from warpxm.mesh.block_2d import BlockMesh2D
from warpxm.spatial.dg_euler_2d import DGEulerSolver2D
from warpxm.temporal.ssprk3_2d import SSPRK3_2D
from warpxm.variable.state_2d import DGState2D


comptime P: Int = 2
comptime NX: Int = 32
comptime NY: Int = 32
comptime GAMMA: Real = 5.0 / 3.0
comptime EPS: Real = 0.01
comptime UX: Real = 1.0
comptime UY: Real = 1.0


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"

    var ctx = DeviceContext()
    var mesh = BlockMesh2D[P](
        ctx, Real(0.0), Real(1.0), Real(0.0), Real(1.0),
        nx=NX, ny=NY, periodic_x=True, periodic_y=True,
    )
    var q = DGState2D[P, EULER_NC](ctx, mesh.num_elements)

    var P1 = NodalBasis1D[P].num_nodes
    var ref_nodes = NodalBasis1D[P].nodes()
    var weights = NodalBasis1D[P].weights()
    var nodes_per_elem = P1 * P1

    # IC
    var host = ctx.enqueue_create_host_buffer[Real.dtype](q.buf_len)
    for j in range(NY):
        var yl = mesh.y_min + Real(j) * mesh.dy
        for i in range(NX):
            var xl = mesh.x_min + Real(i) * mesh.dx
            var e = j * NX + i
            for jn in range(P1):
                var y = yl + (Real(1.0) + ref_nodes[jn]) * mesh.jacobian_y
                for in_ in range(P1):
                    var x = xl + (Real(1.0) + ref_nodes[in_]) * mesh.jacobian_x
                    var rho = Real(1.0) + EPS * sin(TWO_PI * (x + y))
                    var pp: Real = 1.0
                    var ke = Real(0.5) * rho * (UX * UX + UY * UY)
                    var b = (e * nodes_per_elem + (jn * P1 + in_)) * EULER_NC
                    host[b + IDX_RHO] = rho
                    host[b + IDX_MX] = rho * UX
                    host[b + IDX_MY] = rho * UY
                    host[b + IDX_MZ] = Real(0.0)
                    host[b + IDX_E] = pp / (GAMMA - Real(1.0)) + ke
    ctx.enqueue_copy(dst_buf=q.buf, src_buf=host)

    var app = Euler(
        gas_gamma=GAMMA,
        min_density=Real(1e-12),
        min_pressure=Real(1e-12),
        cfl_max=Real(0.5),
    )
    var spatial = DGEulerSolver2D[P](app)
    var stepper = SSPRK3_2D[P, EULER_NC](ctx, mesh.num_elements)

    def compute_rhs(
        mut q_in: DGState2D[P, EULER_NC],
        mut q_out: DGState2D[P, EULER_NC],
    ) raises {imm}:
        spatial.compute_rhs(ctx, mesh, q_in, q_out)

    def no_adjust(mut q_: DGState2D[P, EULER_NC]) raises {imm}:
        pass

    var dt: Real = 1.0e-3
    var t_final: Real = 1.0
    var t: Real = 0.0
    var step: Int = 0
    while t < t_final - Real(0.5) * dt:
        var dt_use = dt if (t + dt) < t_final else (t_final - t)
        stepper.step(ctx, q, dt_use, compute_rhs, no_adjust)
        t += dt_use
        step += 1
    ctx.synchronize()

    # Density L2 error vs IC.
    var ja = mesh.jacobian_x * mesh.jacobian_y
    var rho_err: Real = 0.0
    var mass_drift: Real = 0.0
    with q.buf.map_to_host() as h:
        for j in range(NY):
            var yl = mesh.y_min + Real(j) * mesh.dy
            for i in range(NX):
                var xl = mesh.x_min + Real(i) * mesh.dx
                var e = j * NX + i
                for jn in range(P1):
                    var y = yl + (Real(1.0) + ref_nodes[jn]) * mesh.jacobian_y
                    for in_ in range(P1):
                        var x = xl + (Real(1.0) + ref_nodes[in_]) * mesh.jacobian_x
                        var b = (e * nodes_per_elem + (jn * P1 + in_)) * EULER_NC
                        var exact = Real(1.0) + EPS * sin(TWO_PI * (x + y))
                        var d = h[b + IDX_RHO] - exact
                        var jw = ja * weights[in_] * weights[jn]
                        rho_err += jw * d * d
                        mass_drift += jw * (h[b + IDX_RHO] - Real(1.0))
    print("Integrated to t =", t, "in", step, "steps")
    print("ρ L2 error vs exact:", rho_err ** Real(0.5))
    print("Net mass drift (should be ~0 by conservation):", mass_drift)
