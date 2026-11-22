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

from std.collections import Array
from std.math import exp
from std.sys import has_accelerator
from max.gpu.host import DeviceContext
from std.testing import assert_true

from warpxm.prelude import *
from warpxm.physics.euler import *


comptime P: Int = 2
comptime NX: Int = 32
comptime NY: Int = 32
comptime GAMMA: Real = 5.0 / 3.0
comptime EPS: Real = 0.5
comptime SIGMA: Real = 0.1
comptime State = DGState[2, P, EULER_NC]


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"

    var ctx = DeviceContext()
    var mesh = BlockMesh[2, P](
        ctx, Real(0.0), Real(1.0), Real(0.0), Real(1.0),
        nx=NX, ny=NY, periodic_x=False, periodic_y=False,
        bc_left=BC_WALL_FREESLIP, bc_right=BC_WALL_FREESLIP,
        bc_bottom=BC_WALL_FREESLIP, bc_top=BC_WALL_FREESLIP,
    )
    var q = State(ctx, mesh.num_elements)

    # Initial pressure blob in a uniform-density, quiescent background.
    def initial(x: Real, y: Real) -> Array[Real, EULER_NC]:
        var r2 = (x - Real(0.5)) ** Real(2.0) + (y - Real(0.5)) ** Real(2.0)
        var pp = Real(1.0) + EPS * exp(-r2 / (SIGMA * SIGMA))
        return [Real(1.0), Real(0.0), Real(0.0), Real(0.0), pp / (GAMMA - Real(1.0))]
    mesh.fill(q, initial)

    var mass_0 = mesh.integrate(q, IDX_RHO)
    var energy_0 = mesh.integrate(q, IDX_E)
    var momx_0 = mesh.integrate(q, IDX_MX)
    var momy_0 = mesh.integrate(q, IDX_MY)
    print("initial: mass =", mass_0, " energy =", energy_0,
          " momx =", momx_0, " momy =", momy_0)

    var app = Euler(
        gas_gamma=GAMMA,
        min_density=Real(1e-12),
        min_pressure=Real(1e-12),
        cfl_max=Real(0.5),
    )
    var spatial = DGEulerSolver[2, P](app)
    var stepper = SSPRK3[2, P, EULER_NC](ctx, mesh.num_elements)

    def compute_rhs(mut q_in: State, mut q_out: State) raises {imm}:
        spatial.compute_rhs(ctx, mesh, q_in, q_out)

    var dt: Real = 5.0e-4
    var t_final: Real = 0.4
    _ = stepper.run_until(
        ctx, q, Real(0.0), t_final, dt, compute_rhs,
        no_op_adjust[2, P, EULER_NC],
    )
    ctx.synchronize()

    var mass_1 = mesh.integrate(q, IDX_RHO)
    var energy_1 = mesh.integrate(q, IDX_E)
    var momx_1 = mesh.integrate(q, IDX_MX)
    var momy_1 = mesh.integrate(q, IDX_MY)
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
