# Multi-rank regression: Vlasov-Maxwell Langmuir oscillation in 1D1V.
#
# The full kinetic coupling on several ranks: the Vlasov right-hand side
# reads the fields, the velocity moments give the current that drives
# Maxwell, and the coupled stepper advances both. Moments and sources are
# local to a configuration element, so only the two DG operators exchange
# ghosts. Both fields must be bitwise independent of the number of ranks.

from std.collections import Array
from std.math import exp, sin
from std.sys import has_accelerator
from max.gpu.host import DeviceContext

from warpxm.prelude import *
from warpxm.physics.vlasov import *
from warpxm.physics.maxwell import *

comptime P: Int = 2
comptime V_MAX: Real = 6.0
comptime CHARGE_MASS_RATIO: Real = -1.0
comptime EPS0: Real = 1.0
comptime K: Real = 0.5
comptime LX: Real = TWO_PI / K
comptime Phase = PhaseState[1, 1, P, P, VLASOV_NC]
comptime Fields = DGState[1, P, MAXWELL_NC]
comptime Moments = DGState[1, P, VLASOV_MOMENT_NC]


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = rank_device_context()
    var mesh = PhaseMesh[1, 1, P, P](
        ctx, lo=[0.0, -V_MAX], hi=[LX, V_MAX], n=[19, 24], periodic=[True],
    )
    var x_mesh = BlockMesh[1, P](ctx, Real(0.0), LX, 19, True)
    mesh.check_companion(x_mesh)
    var q_phase = Phase(ctx, mesh.num_phase_elements, mesh.num_ghost)
    var q_maxwell = Fields(ctx, x_mesh.num_elements, x_mesh.num_ghost)

    def f_initial(x: Real, v: Real) -> Real:
        return (
            (Real(1.0) + Real(0.01) * sin(K * x))
            * exp(-Real(0.5) * v * v) / sqrt(TWO_PI)
        )
    mesh.fill(q_phase, f_initial)

    # Gauss: dE/dx = -(n - 1) with the electron charge -1.
    def e_initial(x: Real) -> Array[Real, MAXWELL_NC]:
        var v = Array[Real, MAXWELL_NC](fill=Real(0.0))
        v[IDX_EX] = Real(0.01) * cos_k(x) / K
        return v^
    x_mesh.fill(q_maxwell, e_initial)

    var spatial_vlasov = DGVlasovSolver[1, 1, P, P](
        VlasovMaxwell(
            charge_mass_ratio=CHARGE_MASS_RATIO, skin_depth_norm=Real(1.0),
            cfl_max=Real(0.5),
        )
    )
    var spatial_maxwell = DGMaxwellSolver[1, P](
        Maxwell(c_light=Real(1.0), cfl_max=Real(0.5))
    )
    var moments = VlasovMoments[1, 1, P, P](ctx, mesh)
    var mom = Moments(ctx, x_mesh.num_elements)
    var stepper = SSPRK3VlasovMaxwell[1, 1, P, P](
        ctx, mesh.num_phase_elements, mesh.num_x_elements,
        mesh.num_ghost, x_mesh.num_ghost,
    )

    def compute_joint_rhs(
        mut qp_in: Phase, mut qm_in: Fields,
        mut qp_out: Phase, mut qm_out: Fields,
    ) raises {mut mom, imm}:
        spatial_vlasov.compute_rhs(ctx, mesh, qp_in, qm_in, qp_out)
        spatial_maxwell.compute_rhs(ctx, x_mesh, qm_in, qm_out)
        moments.compute(ctx, mesh, qp_in, mom)
        apply_vlasov_current_source[1, 1, P](
            ctx, mom, qm_out, CHARGE_MASS_RATIO, EPS0,
        )

    def suggest_dt(mut qp: Phase, mut qm: Fields) raises {imm} -> Real:
        return fmin(
            spatial_vlasov.suggest_dt(ctx, mesh, qm),
            spatial_maxwell.suggest_dt(x_mesh),
        )

    def field_energy(v: Array[Real, MAXWELL_NC]) -> Real:
        return Real(0.5) * v[IDX_EX] * v[IDX_EX]

    var mass_0 = mesh.integrate(q_phase)
    var w_0 = x_mesh.integrate_fn(q_maxwell, field_energy)
    # A quarter of a plasma period (omega ~ 1.16 at k = 0.5): the field
    # energy has gone through its first minimum.
    _ = stepper.run_until_cfl(
        ctx, q_phase, q_maxwell, Real(0.0), Real(1.35),
        compute_joint_rhs, suggest_dt, dt_interval=10,
    )
    var mass_1 = mesh.integrate(q_phase)
    var w_1 = x_mesh.integrate_fn(q_maxwell, field_energy)
    var sum_p = mesh.checksum(q_phase)
    var sum_m = x_mesh.checksum(q_maxwell)
    if is_root():
        print("ranks =", num_ranks(), " field energy", w_0, "->", w_1,
              " relative mass change =", (mass_1 - mass_0) / mass_0)
        print("CHECKSUM phase", sum_p)
        print("CHECKSUM maxwell", sum_m)
    if not (fabs(mass_1 - mass_0) < Real(1.0e-9) * mass_0):
        raise Error("mass not conserved")
    if not (w_1 < Real(0.05) * w_0):
        raise Error("field energy did not pass through its minimum")
    if is_root():
        print("PASSED")


def cos_k(x: Real) -> Real:
    return sin(K * x + HALF_PI)
