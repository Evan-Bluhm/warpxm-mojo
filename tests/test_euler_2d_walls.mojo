# 2D 5-moment Euler in a closed box with free-slip walls on all four sides.
#
# Initial condition: a smooth Gaussian pressure perturbation in a quiescent
# uniform-density background.
#
#     ρ ≡ 1,  u ≡ 0,  P(x, y) = 1 + ε · exp(-((x-0.5)² + (y-0.5)²)/σ²)
#
# Once integrated forward, mass and total energy in the box must be conserved
# exactly (to round-off), because the walls reflect all wave traffic and the
# domain is closed. Asserting `|∑ρ - ∑ρ₀| ≪ 1` and `|∑E - ∑E₀| ≪ 1` is a
# strong check on the BC + DG kernel; an incorrect ghost-state synthesis
# would leak mass or energy at the boundary.

from std.math import exp
from std.sys import has_accelerator
from std.gpu.host import DeviceContext
from std.testing import assert_true

from warpxm.apps.euler_2d import Euler2D, EULER_NC, IDX_RHO, IDX_MX, IDX_MY, IDX_MZ, IDX_E
from warpxm.basis.nodal_basis_1d import NodalBasis1D
from warpxm.core.bc import BC_WALL_FREESLIP
from warpxm.core.math_compat import fabs
from warpxm.core.types import Real
from warpxm.mesh.block_2d import BlockMesh2D
from warpxm.spatial.dg_euler_2d import DGEulerSolver2D
from warpxm.temporal.ssprk3_2d import SSPRK3_2D
from warpxm.variable.state_2d import DGState2D


comptime P: Int = 2
comptime NX: Int = 32
comptime NY: Int = 32
comptime GAMMA: Real = 5.0 / 3.0
comptime EPS: Real = 0.5
comptime SIGMA: Real = 0.1


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"

    var ctx = DeviceContext()
    var mesh = BlockMesh2D[P](
        ctx, Real(0.0), Real(1.0), Real(0.0), Real(1.0),
        nx=NX, ny=NY, periodic_x=False, periodic_y=False,
        bc_left=BC_WALL_FREESLIP, bc_right=BC_WALL_FREESLIP,
        bc_bottom=BC_WALL_FREESLIP, bc_top=BC_WALL_FREESLIP,
    )
    var q = DGState2D[P, EULER_NC](ctx, mesh.num_elements)

    var P1 = NodalBasis1D[P].num_nodes
    var ref_nodes = NodalBasis1D[P].nodes()
    var weights = NodalBasis1D[P].weights()
    var nodes_per_elem = P1 * P1

    # Initial pressure blob in a uniform-density, quiescent background.
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
                    var r2 = (x - Real(0.5)) ** Real(2.0) + (y - Real(0.5)) ** Real(2.0)
                    var pp = Real(1.0) + EPS * exp(-r2 / (SIGMA * SIGMA))
                    var b = (e * nodes_per_elem + (jn * P1 + in_)) * EULER_NC
                    host[b + IDX_RHO] = Real(1.0)
                    host[b + IDX_MX] = Real(0.0)
                    host[b + IDX_MY] = Real(0.0)
                    host[b + IDX_MZ] = Real(0.0)
                    host[b + IDX_E] = pp / (GAMMA - Real(1.0))
    ctx.enqueue_copy(dst_buf=q.buf, src_buf=host)

    var ja = mesh.jacobian_x * mesh.jacobian_y

    var mass_0: Real = 0.0
    var energy_0: Real = 0.0
    var momx_0: Real = 0.0
    var momy_0: Real = 0.0
    with q.buf.map_to_host() as h:
        for j in range(NY):
            for i in range(NX):
                var e = j * NX + i
                for jn in range(P1):
                    for in_ in range(P1):
                        var n = jn * P1 + in_
                        var jw = ja * weights[in_] * weights[jn]
                        var b = (e * nodes_per_elem + n) * EULER_NC
                        mass_0 += jw * h[b + IDX_RHO]
                        energy_0 += jw * h[b + IDX_E]
                        momx_0 += jw * h[b + IDX_MX]
                        momy_0 += jw * h[b + IDX_MY]

    print("initial: mass =", mass_0, " energy =", energy_0,
          " momx =", momx_0, " momy =", momy_0)

    var app = Euler2D(
        gas_gamma=GAMMA,
        min_density=Real(1e-12),
        min_pressure=Real(1e-12),
        cfl_max=Real(0.5),
    )
    var spatial = DGEulerSolver2D[P](app)
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

    var dt: Real = 5.0e-4
    var t_final: Real = 0.4
    var t: Real = 0.0
    while t < t_final - Real(0.5) * dt:
        var dt_use = dt if (t + dt) < t_final else (t_final - t)
        stepper.step[compute_rhs, no_adjust](ctx, q, dt_use)
        t += dt_use
    ctx.synchronize()

    var mass_1: Real = 0.0
    var energy_1: Real = 0.0
    var momx_1: Real = 0.0
    var momy_1: Real = 0.0
    with q.buf.map_to_host() as h:
        for j in range(NY):
            for i in range(NX):
                var e = j * NX + i
                for jn in range(P1):
                    for in_ in range(P1):
                        var n = jn * P1 + in_
                        var jw = ja * weights[in_] * weights[jn]
                        var b = (e * nodes_per_elem + n) * EULER_NC
                        mass_1 += jw * h[b + IDX_RHO]
                        energy_1 += jw * h[b + IDX_E]
                        momx_1 += jw * h[b + IDX_MX]
                        momy_1 += jw * h[b + IDX_MY]

    print("final  : mass =", mass_1, " energy =", energy_1,
          " momx =", momx_1, " momy =", momy_1)

    var d_mass = fabs(mass_1 - mass_0) / mass_0
    var d_energy = fabs(energy_1 - energy_0) / energy_0
    var d_momx = fabs(momx_1)
    var d_momy = fabs(momy_1)
    print(
        "rel Δmass =", d_mass,
        " rel Δenergy =", d_energy,
        " |momx_final| =", d_momx,
        " |momy_final| =", d_momy,
    )
    assert_true(d_mass < Real(1.0e-12), "mass not conserved in closed box")
    assert_true(d_energy < Real(1.0e-12), "energy not conserved in closed box")
    assert_true(d_momx < Real(1.0e-12), "x-momentum should remain zero by symmetry")
    assert_true(d_momy < Real(1.0e-12), "y-momentum should remain zero by symmetry")
    print("PASSED")
