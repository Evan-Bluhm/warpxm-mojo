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
#
#     pixi run example euler_entropy_wave_2d

from std.collections import Array
from std.math import sin
from std.sys import has_accelerator
from max.gpu.host import DeviceContext

from warpxm.prelude import *
from warpxm.physics.euler import *


comptime P: Int = 2
comptime NX: Int = 32
comptime NY: Int = 32
comptime GAMMA: Real = 5.0 / 3.0
comptime EPS: Real = 0.01
comptime UX: Real = 1.0
comptime UY: Real = 1.0
comptime T_FINAL: Real = 1.0
comptime State = DGState[2, P, EULER_NC]


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"

    var ctx = DeviceContext()
    var mesh = BlockMesh[2, P](
        ctx, Real(0.0), Real(1.0), Real(0.0), Real(1.0),
        nx=NX, ny=NY, periodic_x=True, periodic_y=True,
    )
    var q = State(ctx, mesh.num_elements)

    def exact_rho(x: Real, y: Real) -> Real:
        return Real(1.0) + EPS * sin(TWO_PI * (x + y))

    def initial(x: Real, y: Real) -> Array[Real, EULER_NC]:
        var rho = Real(1.0) + EPS * sin(TWO_PI * (x + y))
        var ke = Real(0.5) * rho * (UX * UX + UY * UY)
        return [
            rho, rho * UX, rho * UY, Real(0.0),
            Real(1.0) / (GAMMA - Real(1.0)) + ke,
        ]
    mesh.fill(q, initial)
    var mass_0 = mesh.integrate(q, IDX_RHO)

    # Smooth flow with no limiter, so the step can sit closer to the
    # stability limit than the 0.5 used for shock problems.
    var app = Euler(
        gas_gamma=GAMMA,
        min_density=Real(1e-12),
        min_pressure=Real(1e-12),
        cfl_max=Real(0.8),
    )
    var spatial = DGEulerSolver[2, P](app)
    var stepper = SSPRK3[2, P, EULER_NC](ctx, mesh.num_elements)

    def compute_rhs(mut q_in: State, mut q_out: State) raises {imm}:
        spatial.compute_rhs(ctx, mesh, q_in, q_out)

    def suggest_dt(mut q_: State) raises {imm} -> Real:
        return spatial.suggest_dt(ctx, mesh, q_)

    # |u| + c changes only at the 1e-2 level here: refresh dt every 20 steps.
    var t = stepper.run_until_cfl(
        ctx, q, Real(0.0), T_FINAL, compute_rhs,
        no_op_adjust[2, P, EULER_NC], suggest_dt, dt_interval=20,
    )

    print("Integrated to t =", t, " CFL dt =", suggest_dt(q))
    print("ρ L2 error vs exact:", mesh.l2_error(q, IDX_RHO, exact_rho))
    print("Net mass drift (should be ~0 by conservation):",
          mesh.integrate(q, IDX_RHO) - mass_0)
