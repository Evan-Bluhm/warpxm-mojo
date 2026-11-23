# Coupled 2D2V Vlasov-Maxwell: an oblique electrostatic mode.
#
# The 2D counterpart of `test_vlasov_maxwell_langmuir_1d1v`. A uniform
# Maxwellian `f₀ = exp(−|v|²/σ²)/(π σ²)` (`ω_p = 1`, `q/m = −1`) starts
# with the longitudinal field
#
#     E(x, 0) = ε k̂ sin(k · x),      k = 2π (1, 1/2)
#
# on the periodic box `[0, 1] × [0, 2]`. The mode is neither along x nor
# along y nor on the diagonal, so both current components, both
# configuration streaming terms and both velocity accelerations take
# part, each with a different weight. Linear theory is the 1D result in
# the coordinate along `k`: the amplitude `E(t)` of the mode obeys
#
#     dE/dt = −ω_p² ∫₀ᵗ E(τ) G(t − τ) dτ,
#     G(s)  = exp(−a s²) (1 − 2 a s²),      a = |k|² σ² / 4,
#
# the response of the velocity distribution projected on `k̂` (the
# perpendicular velocity integrates out). The field stays longitudinal:
# `E_x / E_y = k_x / k_y` and `B = 0`.
#
# Checks:
#   (a) `√(W_E(t)/W_E(0))` against the Volterra solution at two times;
#   (b) the direction of E: `W_Ex / W_Ey = (k_x/k_y)² = 4` throughout, and
#       no magnetic field or `E_z` appears;
#   (c) total energy `½ ∫ |E|² + ½ ∫∫ |v|² f` conserved, measured against
#       the initial field energy;
#   (d) mass conserved.
#
# Build / run:
#     pixi run mojo run -I src tests/test_vlasov_maxwell_langmuir_2d2v.mojo

from std.collections import Array, List
from std.math import exp, sin
from std.sys import has_accelerator
from max.gpu.host import DeviceContext
from std.testing import assert_true

from warpxm.prelude import *
from warpxm.physics.vlasov import *
from warpxm.physics.maxwell import *


comptime P_X: Int = 2
comptime P_V: Int = 2
comptime NX: Int = 8
comptime NY: Int = 8
comptime NV: Int = 16
comptime LX: Real = 1.0
comptime LY: Real = 2.0
comptime V_MAX: Real = 3.0
comptime SIGMA_SQ: Real = 0.5
comptime CHARGE_MASS_RATIO: Real = -1.0
comptime EPS0: Real = 1.0
comptime KX: Real = TWO_PI / LX
comptime KY: Real = TWO_PI / LY
comptime K_SQ: Real = KX * KX + KY * KY
comptime EPS: Real = 0.01
comptime T_FINAL: Real = 1.0

comptime Phase = PhaseState[2, 2, P_X, P_V, VLASOV_NC]
comptime Fields = DGState[2, P_X, MAXWELL_NC]
comptime Moments = DGState[2, P_X, VLASOV_MOMENT_NC]


def volterra_amplitude(steps: Int) -> List[Real]:
    """`E(t_j)/E(0)` at `t_j = j · T_FINAL / steps` (Heun's method with a
    trapezoidal convolution, as in the 1D1V test)."""
    var a = K_SQ * SIGMA_SQ / Real(4.0)
    var h = T_FINAL / Real(steps)
    var g = List[Real]()
    for j in range(steps + 1):
        var s = Real(j) * h
        g.append(exp(-a * s * s) * (Real(1.0) - Real(2.0) * a * s * s))
    var e = List[Real]()
    e.append(Real(1.0))
    for n in range(steps):
        var i0: Real = 0.0
        for j in range(n + 1):
            var wq = Real(0.5) if (j == 0 or j == n) else Real(1.0)
            i0 += wq * e[j] * g[n - j]
        i0 = i0 * h if n > 0 else Real(0.0)
        var e_pred = e[n] - h * i0
        var i1: Real = Real(0.5) * e_pred * g[0]
        for j in range(n + 1):
            var wq = Real(0.5) if j == 0 else Real(1.0)
            i1 += wq * e[j] * g[n + 1 - j]
        i1 *= h
        e.append(e[n] - Real(0.5) * h * (i0 + i1))
    return e^


def field_energies(
    x_mesh: BlockMesh[2, P_X], mut q_maxwell: Fields,
) raises -> Array[Real, 3]:
    """`[½∫E_x², ½∫E_y², ½∫(E_z² + |B|²)]`."""
    var out = Array[Real, 3](fill=Real(0.0))
    def ex_sq(v: Array[Real, MAXWELL_NC]) -> Real:
        return Real(0.5) * v[IDX_EX] * v[IDX_EX]
    def ey_sq(v: Array[Real, MAXWELL_NC]) -> Real:
        return Real(0.5) * v[IDX_EY] * v[IDX_EY]
    def rest_sq(v: Array[Real, MAXWELL_NC]) -> Real:
        return Real(0.5) * (
            v[IDX_EZ] * v[IDX_EZ] + v[IDX_BX] * v[IDX_BX]
            + v[IDX_BY] * v[IDX_BY] + v[IDX_BZ] * v[IDX_BZ]
        )
    out[0] = x_mesh.integrate_fn(q_maxwell, ex_sq)
    out[1] = x_mesh.integrate_fn(q_maxwell, ey_sq)
    out[2] = x_mesh.integrate_fn(q_maxwell, rest_sq)
    return out^


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = DeviceContext()

    var mesh = PhaseMesh[2, 2, P_X, P_V](
        ctx,
        lo=[0.0, 0.0, -V_MAX, -V_MAX], hi=[LX, LY, V_MAX, V_MAX],
        n=[NX, NY, NV, NV], periodic=[True, True],
    )
    var x_mesh = BlockMesh[2, P_X](
        ctx, 0.0, LX, 0.0, LY, NX, NY, periodic_x=True, periodic_y=True,
    )
    mesh.check_companion(x_mesh)
    var q_phase = Phase(ctx, mesh.num_phase_elements)
    var q_maxwell = Fields(ctx, x_mesh.num_elements)

    def maxwellian(x: Real, y: Real, v1: Real, v2: Real) -> Real:
        return exp(-(v1 * v1 + v2 * v2) / SIGMA_SQ) / (PI * SIGMA_SQ)
    mesh.fill(q_phase, maxwellian)

    def seed(x: Real, y: Real) -> Array[Real, MAXWELL_NC]:
        var v = Array[Real, MAXWELL_NC](fill=Real(0.0))
        var amp = EPS * sin(KX * x + KY * y) / sqrt(K_SQ)
        v[IDX_EX] = amp * KX
        v[IDX_EY] = amp * KY
        return v^
    x_mesh.fill(q_maxwell, seed)

    var spatial_vlasov = DGVlasovSolver[2, 2, P_X, P_V](
        VlasovMaxwell(
            charge_mass_ratio=CHARGE_MASS_RATIO, skin_depth_norm=Real(1.0),
            cfl_max=Real(0.5),
        )
    )
    var spatial_maxwell = DGMaxwellSolver[2, P_X](
        Maxwell(c_light=Real(1.0), cfl_max=Real(0.5))
    )
    var moments = VlasovMoments[2, 2, P_X, P_V](ctx, mesh)
    var mom = Moments(ctx, x_mesh.num_elements)
    var stepper = SSPRK3VlasovMaxwell[2, 2, P_X, P_V](
        ctx, mesh.num_phase_elements, mesh.num_x_elements,
    )

    def compute_joint_rhs(
        mut qp_in: Phase, mut qm_in: Fields,
        mut qp_out: Phase, mut qm_out: Fields,
    ) raises {mut mom, imm}:
        spatial_vlasov.compute_rhs(ctx, mesh, qp_in, qm_in, qp_out)
        spatial_maxwell.compute_rhs(ctx, x_mesh, qm_in, qm_out)
        moments.compute(ctx, mesh, qp_in, mom)
        apply_vlasov_current_source[2, 2, P_X](
            ctx, mom, qm_out, CHARGE_MASS_RATIO, EPS0,
        )

    def suggest_dt(mut qp: Phase, mut qm: Fields) raises {imm} -> Real:
        return min(
            spatial_vlasov.suggest_dt(ctx, mesh, qm),
            spatial_maxwell.suggest_dt(x_mesh),
        )

    comptime N_LIN = 500
    var e_lin = volterra_amplitude(N_LIN)

    var w0 = field_energies(x_mesh, q_maxwell)
    var field0 = w0[0] + w0[1]
    moments.compute(ctx, mesh, q_phase, mom)
    var kinetic0 = x_mesh.integrate(mom, MOM_ENERGY)
    var mass0 = mesh.integrate(q_phase)

    var t: Real = 0.0
    var max_amp_err: Real = 0.0
    var max_ratio_err: Real = 0.0
    var max_other: Real = 0.0
    var w1 = w0.copy()
    for half in range(2):
        t = stepper.run_until_cfl(
            ctx, q_phase, q_maxwell, t, T_FINAL * Real(half + 1) / Real(2.0),
            compute_joint_rhs, suggest_dt, dt_interval=20,
        )
        w1 = field_energies(x_mesh, q_maxwell)
        var amp_obs = sqrt((w1[0] + w1[1]) / field0)
        var amp_lin = e_lin[(half + 1) * (N_LIN // 2)]
        max_amp_err = max(max_amp_err, abs(amp_obs - amp_lin))
        max_ratio_err = max(
            max_ratio_err, abs(w1[0] / w1[1] - KX * KX / (KY * KY)),
        )
        max_other = max(max_other, w1[2] / field0)
        print(
            "  t =", t, " E/E0 observed =", amp_obs, " linear theory =",
            amp_lin, " diff =", amp_obs - amp_lin,
            " W_Ex/W_Ey =", w1[0] / w1[1],
            " (W_Ez + W_B)/W_E(0) =", w1[2] / field0,
        )
    moments.compute(ctx, mesh, q_phase, mom)
    var kinetic1 = x_mesh.integrate(mom, MOM_ENERGY)
    var mass1 = mesh.integrate(q_phase)
    var d_energy = (w1[0] + w1[1] + w1[2] + kinetic1) - (field0 + kinetic0)
    print(
        "2D2V Langmuir: ΔW / W_E(0) =", d_energy / field0,
        " ΔW_K / W_E(0) =", (kinetic1 - kinetic0) / field0,
        " ΔM =", mass1 - mass0,
    )

    # The measured values below are truncation error: on a 16 × 16
    # configuration mesh they drop to 1.6e-4, 3.7e-4, 2.8e-7 and 4.7e-5.
    #
    # (a) Measured 2.5e-4 and 2.7e-4. Without the coupling E/E0 stays at
    # 1, which is 0.06 and 0.08 off.
    assert_true(
        max_amp_err < Real(1.0e-3),
        "field amplitude departs from the linear Vlasov response",
    )
    # (b) Measured: W_Ex/W_Ey within 3.3e-3 of 4 (a current component
    # with the wrong weight would move it by order one); E_z and B hold
    # 1.0e-5 of the initial field energy, from the discrete curl of the
    # longitudinal field.
    assert_true(max_ratio_err < Real(1.0e-2), "E is not along k")
    assert_true(max_other < Real(5.0e-5), "E_z or B appeared")
    # (c) Measured -7.0e-4 of the initial field energy, while the
    # particles gained 0.145 of it.
    assert_true(
        abs(d_energy) / field0 < Real(3.0e-3),
        "energy not conserved on the field-energy scale",
    )
    # (d) Measured 1.9e-10 (the Maxwellian tail at |v| = 3).
    assert_true(abs(mass1 - mass0) < Real(1.0e-9), "mass not conserved")
    print("PASSED")
