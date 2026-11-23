# Coupled 1D3V Vlasov-Maxwell in a closed box: specular walls
# (`BC_VLASOV_REFLECT`) on the particles, perfectly conducting walls
# (`BC_PEC_WALL`) on the fields.
#
# This is the boundary pairing of a plasma between conducting plates.
# Neither wall lets anything through: the specular wall passes no
# particles and does no work on them, and the conductor passes no
# Poynting flux. So in the closed box
#
#     M = ∫∫ f dv dx        and        W = ½ ∫ (|E|² + |B|²) dx + ½ ∫∫ |v|² f dv dx
#
# are both constant (with `λ = ε₀ = 1`), whatever the fields do in
# between. A Maxwellian starts with a standing electrostatic field
# `E_x = ε sin πx` and a cavity mode `B_z = δ cos πx`, both compatible
# with the conductor (`E_y = 0` and `B_x = 0` at the walls), so particles
# are driven into the walls along x while the transverse wave reflects
# off them.
#
# Checks:
#   (a) mass conserved to the level of the velocity-boundary loss;
#   (b) total energy conserved to truncation error, measured against the
#       initial field energy, while field and particles exchange a
#       finite fraction of it;
#   (c) the same run with copy-out walls on `f` fails both, so the checks
#       do see the wall.
#
# Build / run:
#     pixi run mojo run -I src tests/test_vlasov_maxwell_walls_1d3v.mojo

from std.collections import Array
from std.math import cos, exp, sin
from std.sys import has_accelerator
from max.gpu.host import DeviceContext
from std.testing import assert_true

from warpxm.prelude import *
from warpxm.physics.vlasov import *
from warpxm.physics.maxwell import *


comptime P: Int = 2
comptime NX: Int = 16
comptime NV: Int = 12
comptime V_MAX: Real = 3.5
comptime SIGMA_SQ: Real = 0.5
comptime CHARGE_MASS_RATIO: Real = -1.0
comptime EPS0: Real = 1.0
comptime EPS_EX: Real = 0.05
comptime DELTA_BZ: Real = 0.05
comptime T_FINAL: Real = 1.5

comptime Phase = PhaseState[1, 3, P, P, VLASOV_NC]
comptime Fields = DGState[1, P, MAXWELL_NC]
comptime Moments = DGState[1, P, VLASOV_MOMENT_NC]


def field_energy(x_mesh: BlockMesh[1, P], mut q_maxwell: Fields) raises -> Real:
    def density(v: Array[Real, MAXWELL_NC]) -> Real:
        var s = Real(0.0)
        for c in range(MAXWELL_NC):
            s += v[c] * v[c]
        return Real(0.5) * s
    return x_mesh.integrate_fn(q_maxwell, density)


def closed_box(var ctx: DeviceContext, bc_kind: Int) raises -> Array[Real, 3]:
    """Run to `T_FINAL` with walls of kind `bc_kind` on `f`. Returns
    `[ΔM / M, ΔW / W_field(0), ΔW_kinetic / W_field(0)]`."""
    var mesh = PhaseMesh[1, 3, P, P](
        ctx,
        lo=[0.0, -V_MAX, -V_MAX, -V_MAX], hi=[1.0, V_MAX, V_MAX, V_MAX],
        n=[NX, NV, NV, NV], periodic=[False], bc=[bc_kind, bc_kind],
    )
    var x_mesh = BlockMesh[1, P](
        ctx, 0.0, 1.0, NX, periodic=False,
        bc_left=BC_PEC_WALL, bc_right=BC_PEC_WALL,
    )
    mesh.check_companion(x_mesh)
    var q_phase = Phase(ctx, mesh.num_phase_elements)
    var q_maxwell = Fields(ctx, NX)

    var norm = Real(1.0) / (PI * SIGMA_SQ * sqrt(PI * SIGMA_SQ))
    def maxwellian(x: Real, v1: Real, v2: Real, v3: Real) {imm} -> Real:
        return norm * exp(-(v1 * v1 + v2 * v2 + v3 * v3) / SIGMA_SQ)
    mesh.fill(q_phase, maxwellian)

    def seed(x: Real) -> Array[Real, MAXWELL_NC]:
        var v = Array[Real, MAXWELL_NC](fill=Real(0.0))
        v[IDX_EX] = EPS_EX * sin(PI * x)
        v[IDX_BZ] = DELTA_BZ * cos(PI * x)
        return v^
    x_mesh.fill(q_maxwell, seed)

    var spatial_vlasov = DGVlasovSolver[1, 3, P, P](
        VlasovMaxwell(
            charge_mass_ratio=CHARGE_MASS_RATIO, skin_depth_norm=Real(1.0),
            cfl_max=Real(0.5),
        )
    )
    var spatial_maxwell = DGMaxwellSolver[1, P](
        Maxwell(c_light=Real(1.0), cfl_max=Real(0.5))
    )
    var moments = VlasovMoments[1, 3, P, P](ctx, mesh)
    var mom = Moments(ctx, NX)
    var stepper = SSPRK3VlasovMaxwell[1, 3, P, P](
        ctx, mesh.num_phase_elements, mesh.num_x_elements,
    )

    def compute_joint_rhs(
        mut qp_in: Phase, mut qm_in: Fields,
        mut qp_out: Phase, mut qm_out: Fields,
    ) raises {mut mom, imm}:
        spatial_vlasov.compute_rhs(ctx, mesh, qp_in, qm_in, qp_out)
        spatial_maxwell.compute_rhs(ctx, x_mesh, qm_in, qm_out)
        moments.compute(ctx, mesh, qp_in, mom)
        apply_vlasov_current_source[1, 3, P](
            ctx, mom, qm_out, CHARGE_MASS_RATIO, EPS0,
        )

    def suggest_dt(mut qp: Phase, mut qm: Fields) raises {imm} -> Real:
        return min(
            spatial_vlasov.suggest_dt(ctx, mesh, qm),
            spatial_maxwell.suggest_dt(x_mesh),
        )

    var mass0 = mesh.integrate(q_phase)
    var field0 = field_energy(x_mesh, q_maxwell)
    moments.compute(ctx, mesh, q_phase, mom)
    var kinetic0 = x_mesh.integrate(mom, MOM_ENERGY)

    _ = stepper.run_until_cfl(
        ctx, q_phase, q_maxwell, Real(0.0), T_FINAL,
        compute_joint_rhs, suggest_dt, dt_interval=20,
    )

    var mass1 = mesh.integrate(q_phase)
    var field1 = field_energy(x_mesh, q_maxwell)
    moments.compute(ctx, mesh, q_phase, mom)
    var kinetic1 = x_mesh.integrate(mom, MOM_ENERGY)
    var out = Array[Real, 3](fill=Real(0.0))
    out[0] = (mass1 - mass0) / mass0
    out[1] = ((field1 + kinetic1) - (field0 + kinetic0)) / field0
    out[2] = (kinetic1 - kinetic0) / field0
    return out^


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = DeviceContext()

    var closed = closed_box(ctx, BC_VLASOV_REFLECT)
    print(
        "specular + PEC: ΔM/M =", closed[0], " ΔW/W_field(0) =", closed[1],
        " ΔW_kinetic/W_field(0) =", closed[2],
    )
    var leaky = closed_box(ctx, BC_COPYOUT)
    print(
        "copy-out + PEC: ΔM/M =", leaky[0], " ΔW/W_field(0) =", leaky[1],
        " ΔW_kinetic/W_field(0) =", leaky[2],
    )

    # Measured in the closed box: ΔM/M = 2.4e-12, ΔW = −1.1e-6 of the
    # initial field energy, with 0.31 of it handed to the particles.
    # With copy-out walls: ΔM/M = 6.9e-3 and ΔW = 6.7 field energies.
    assert_true(abs(closed[0]) < Real(1.0e-10), "mass not conserved")
    assert_true(abs(closed[1]) < Real(1.0e-5), "energy not conserved")
    assert_true(abs(closed[2]) > Real(0.1), "no energy was exchanged")
    assert_true(
        abs(leaky[0]) > Real(1.0e-4) and abs(leaky[1]) > Real(1.0),
        "copy-out walls passed the conservation checks",
    )
    print("PASSED")
