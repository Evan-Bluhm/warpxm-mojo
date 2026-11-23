# Multi-rank regression: free-streaming Vlasov in a 2D2V channel, periodic
# along x and with specular walls along y.
#
# The configuration mesh is cut along both axes (4 ranks), so phase-space
# ghost elements are exchanged across a periodic axis and across a walled
# one, where the ranks at the ends apply the specular wall and the rest
# read ghosts. The distribution must be bitwise independent of the number
# of ranks, and the walls must conserve mass.

from std.collections import Array
from std.math import exp, sin, cos
from std.sys import has_accelerator
from max.gpu.host import DeviceContext

from warpxm.prelude import *
from warpxm.physics.vlasov import *
from warpxm.physics.maxwell import *

comptime P: Int = 2
comptime V_MAX: Real = 4.0
comptime State = PhaseState[2, 2, P, P, VLASOV_NC]


def f_initial(x: Real, y: Real, v1: Real, v2: Real) -> Real:
    var w1 = v1 - Real(0.5)
    var w2 = v2 - Real(0.8) * sin(PI * y)
    return (
        (Real(1.0) + Real(0.2) * sin(TWO_PI * x))
        * (Real(1.0) + Real(0.3) * cos(PI * y))
        * exp(-(w1 * w1 + w2 * w2))
    )


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = rank_device_context()
    var mesh = PhaseMesh[2, 2, P, P](
        ctx,
        lo=[0.0, 0.0, -V_MAX, -V_MAX], hi=[1.0, 1.0, V_MAX, V_MAX],
        n=[7, 6, 6, 8],
        periodic=[True, False],
        bc=[BC_COPYOUT, BC_COPYOUT, BC_VLASOV_REFLECT, BC_VLASOV_REFLECT],
    )
    var q = State(ctx, mesh.num_phase_elements, mesh.num_ghost)
    mesh.fill(q, f_initial)
    var mass_0 = mesh.integrate(q)

    var spatial = DGVlasovSolver[2, 2, P, P](VlasovMaxwell(cfl_max=Real(0.5)))
    var stepper = SSPRK3Phase[2, 2, P, P, VLASOV_NC](
        ctx, mesh.num_phase_elements, mesh.num_ghost,
    )
    var em = DGState[2, P, MAXWELL_NC](ctx, mesh.num_x_elements)   # zero

    def compute_rhs(mut q_in: State, mut q_out: State) raises {imm}:
        spatial.compute_rhs(ctx, mesh, q_in, em, q_out)
    def no_adjust(mut q_: State) raises:
        pass
    def suggest_dt(mut q_: State) raises {imm} -> Real:
        return spatial.suggest_dt(ctx, mesh, em)
    _ = stepper.run_until_cfl(
        ctx, q, Real(0.0), Real(0.3), compute_rhs, no_adjust, suggest_dt,
        dt_interval=1000000,
    )
    var mass_1 = mesh.integrate(q)
    var sum = mesh.checksum(q)
    var drift = (mass_1 - mass_0) / mass_0
    if is_root():
        print("ranks =", num_ranks(), " relative mass change =", drift)
        print("CHECKSUM", sum)
    if not (fabs(drift) < Real(1.0e-11)):
        raise Error("specular walls did not conserve mass")
    if is_root():
        print("PASSED")
