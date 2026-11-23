# Coupled 1D3V Vlasov-Maxwell: electromagnetic waves in both transverse
# polarisations.
#
# The 1D2V test `test_vlasov_maxwell_weibel_1d2v` follows a transverse
# wave `(E_y, B_z)` on a bi-Maxwellian, driven by the `v_2` current. In
# 1D3V the plasma also carries the other polarisation, `(E_z, B_y)`,
# driven by the `v_3` current — the piece of the physics a 2V model does
# not have. With
#
#     f₀ ∝ exp(−v1²/σ_par² − (v2² + v3²)/σ_perp²)
#
# and no background field the two polarisations are decoupled in linear
# theory and obey the same dispersion relation (Weibel 1959; Krall &
# Trivelpiece §9.10)
#
#     ω² − k²c² − ω_p² + ω_p² (σ_perp²/σ_par²) [1 + ζ Z(ζ)] = 0,
#     ζ = ω / (k σ_par).
#
# For `σ_par² = 0.5`, `σ_perp² = 1`, `c = ω_p = 1`, `k = 2π` (above the
# Weibel cutoff `k_max = 1`) the root is `ω = ±6.40797 − 0.049951 i`: a
# standing wave at 2 % above the vacuum frequency, Landau-damped on the
# `v_1 = ω/k` particles.
#
# Both polarisations are seeded, with different amplitudes and spatial
# phases: `B_z = δ_z sin(kx)` and `B_y = δ_y cos(kx)`. Over exactly two
# periods the test checks, for each polarisation separately,
#   (a) the period from the zero crossings of the mode amplitude against
#       `2π/ω_r` (vacuum would be 2 % off),
#   (b) the amplitude after two periods against `e^{−2γT}`,
# and (c) mass conservation. The `(E_z, B_y)` checks exercise
# `E_z + v_1 B_y − v_2 B_x` in the Lorentz force, the `v_3` flux moment
# and its coupling into `∂E_z/∂t`; a sign error in any of them turns the
# 2 % frequency upshift into a downshift or the damping into growth.
#
# Build / run:
#     pixi run mojo run -I src tests/test_vlasov_maxwell_em_wave_1d3v.mojo

from std.collections import Array, List
from std.math import cos, exp, sin
from std.sys import has_accelerator
from max.gpu.host import DeviceContext
from std.testing import assert_true

from warpxm.prelude import *
from warpxm.physics.vlasov import *
from warpxm.physics.maxwell import *


comptime P_X: Int = 2
comptime P_V: Int = 2
comptime NX: Int = 8
comptime NV: Int = 8
comptime X_MIN: Real = 0.0
comptime X_MAX: Real = 1.0
comptime V_MAX: Real = 4.0
comptime SIGMA_PAR_SQ: Real = 0.5    # v1
comptime SIGMA_PERP_SQ: Real = 1.0   # v2 and v3
comptime CHARGE_MASS_RATIO: Real = -1.0
comptime EPS0: Real = 1.0
comptime C_LIGHT: Real = 1.0
comptime OMEGA_R: Real = 6.40797
comptime GAMMA_DAMP: Real = 0.049951
comptime T_FINAL: Real = 2.0 * TWO_PI / OMEGA_R
comptime N_FRAMES: Int = 100
comptime DELTA_BZ: Real = 1.0e-3
comptime DELTA_BY: Real = 0.6e-3

comptime Phase = PhaseState[1, 3, P_X, P_V, VLASOV_NC]
comptime Fields = DGState[1, P_X, MAXWELL_NC]


def mode_amplitudes(
    x_mesh: BlockMesh[1, P_X], mut q_maxwell: Fields,
) raises -> Array[Real, 2]:
    """`[sin(kx) amplitude of B_z, cos(kx) amplitude of B_y]`."""
    var acc = Array[Real, 2](fill=Real(0.0))
    def add(x: Real, w: Real, v: Array[Real, MAXWELL_NC]) {mut acc}:
        acc[0] += w * v[IDX_BZ] * sin(TWO_PI * x)
        acc[1] += w * v[IDX_BY] * cos(TWO_PI * x)
    x_mesh.visit_nodes(q_maxwell, add)
    acc[0] = Real(2.0) * acc[0] / (X_MAX - X_MIN)
    acc[1] = Real(2.0) * acc[1] / (X_MAX - X_MIN)
    return acc^


def check_wave(
    label: String, times: List[Real], amps: List[Real],
) raises:
    """Period from the zero crossings and decay over the whole record."""
    var crossings = List[Real]()
    for k in range(1, len(amps)):
        if amps[k - 1] * amps[k] < 0.0:
            crossings.append(
                times[k - 1] + (times[k] - times[k - 1])
                * amps[k - 1] / (amps[k - 1] - amps[k])
            )
    assert_true(len(crossings) == 4, label + ": expected four zero crossings")
    var period_obs = Real(2.0) * (crossings[3] - crossings[0]) / Real(3.0)
    var period_th = TWO_PI / OMEGA_R
    var period_err = (period_obs - period_th) / period_th
    var ratio_obs = amps[len(amps) - 1] / amps[0]
    var ratio_th = exp(-GAMMA_DAMP * T_FINAL)
    print(
        label, ": period =", period_obs, " theory =", period_th,
        " rel err =", period_err, " b(T)/b(0) =", ratio_obs,
        " theory =", ratio_th, " diff =", ratio_obs - ratio_th,
    )
    # Measured for both polarisations: period error 4.9e-4 (vacuum is
    # 2.0e-2 off) and damping difference 4.9e-3 (undamped is 9.3e-2 off),
    # the same as the 1D2V test on the same 8-element axes.
    assert_true(
        abs(period_err) < Real(2.0e-3), label + ": period off the dispersion relation",
    )
    assert_true(
        abs(ratio_obs - ratio_th) < Real(1.5e-2),
        label + ": damping off the dispersion relation",
    )


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = DeviceContext()

    var mesh = PhaseMesh[1, 3, P_X, P_V](
        ctx,
        lo=[X_MIN, -V_MAX, -V_MAX, -V_MAX],
        hi=[X_MAX, V_MAX, V_MAX, V_MAX],
        n=[NX, NV, NV, NV],
        periodic=[True],
    )
    var x_mesh = BlockMesh[1, P_X](ctx, X_MIN, X_MAX, NX, periodic=True)
    var q_phase = Phase(ctx, mesh.num_phase_elements)
    var q_maxwell = Fields(ctx, NX)

    # ∫ f₀ dv = 1, so ω_p² = n α² / ε₀ = 1.
    var norm = Real(1.0) / (
        PI * sqrt(PI) * sqrt(SIGMA_PAR_SQ) * SIGMA_PERP_SQ
    )
    def bi_maxwellian(x: Real, v1: Real, v2: Real, v3: Real) {imm} -> Real:
        return norm * exp(
            -v1 * v1 / SIGMA_PAR_SQ - (v2 * v2 + v3 * v3) / SIGMA_PERP_SQ
        )
    mesh.fill(q_phase, bi_maxwellian)

    def seed(x: Real) -> Array[Real, MAXWELL_NC]:
        var v = Array[Real, MAXWELL_NC](fill=Real(0.0))
        v[IDX_BZ] = DELTA_BZ * sin(TWO_PI * x)
        v[IDX_BY] = DELTA_BY * cos(TWO_PI * x)
        return v^
    x_mesh.fill(q_maxwell, seed)

    var spatial_vlasov = DGVlasovSolver[1, 3, P_X, P_V](
        VlasovMaxwell(
            charge_mass_ratio=CHARGE_MASS_RATIO, skin_depth_norm=Real(1.0),
            cfl_max=Real(0.5),
        )
    )
    var spatial_maxwell = DGMaxwellSolver[1, P_X](
        Maxwell(c_light=C_LIGHT, cfl_max=Real(0.5))
    )
    var moments = VlasovMoments[1, 3, P_X, P_V](ctx, mesh)
    var mom = DGState[1, P_X, VLASOV_MOMENT_NC](ctx, NX)
    var stepper = SSPRK3VlasovMaxwell[1, 3, P_X, P_V](
        ctx, mesh.num_phase_elements, mesh.num_x_elements,
    )

    def compute_joint_rhs(
        mut qp_in: Phase, mut qm_in: Fields,
        mut qp_out: Phase, mut qm_out: Fields,
    ) raises {mut mom, imm}:
        spatial_vlasov.compute_rhs(ctx, mesh, qp_in, qm_in, qp_out)
        spatial_maxwell.compute_rhs(ctx, x_mesh, qm_in, qm_out)
        moments.compute(ctx, mesh, qp_in, mom)
        apply_vlasov_current_source[1, 3, P_X](
            ctx, mom, qm_out, CHARGE_MASS_RATIO, EPS0,
        )

    def suggest_dt(mut qp: Phase, mut qm: Fields) raises {imm} -> Real:
        return min(
            spatial_vlasov.suggest_dt(ctx, mesh, qm),
            spatial_maxwell.suggest_dt(x_mesh),
        )

    var mass0 = mesh.integrate(q_phase)
    var times = List[Real]()
    var amps_z = List[Real]()
    var amps_y = List[Real]()
    var a0 = mode_amplitudes(x_mesh, q_maxwell)
    times.append(Real(0.0))
    amps_z.append(a0[0])
    amps_y.append(a0[1])

    var t: Real = 0.0
    for frame in range(N_FRAMES):
        var t_next = T_FINAL * Real(frame + 1) / Real(N_FRAMES)
        t = stepper.run_until_cfl(
            ctx, q_phase, q_maxwell, t, t_next, compute_joint_rhs, suggest_dt,
            dt_interval=20,
        )
        var a = mode_amplitudes(x_mesh, q_maxwell)
        times.append(t)
        amps_z.append(a[0])
        amps_y.append(a[1])
    var mass1 = mesh.integrate(q_phase)

    print("1D3V EM waves: t =", t, " ΔM =", mass1 - mass0)
    check_wave("(E_y, B_z) wave", times, amps_z)
    check_wave("(E_z, B_y) wave", times, amps_y)
    assert_true(abs(mass1 - mass0) < Real(1.0e-9), "mass not conserved")
    print("PASSED")
