# Coupled 1D2V Vlasov-Maxwell — smoke test for the full coupled stack.
#
# Exercises every piece of a coupled 1D2V run:
#   * the `DGVlasovSolver[1, 2, …]` RHS reading `(E_x, E_y, B_z)` from the
#     Maxwell state — including the `v × B_z` Lorentz coupling that is
#     identically zero in 1D1V,
#   * `VlasovMoments[1, 2, …].compute` for the flux moments `(J_x, J_y)`,
#   * `apply_vlasov_current_source[1, 2, …]` feeding both `E_x` and `E_y`,
#   * the `SSPRK3VlasovMaxwell[1, 2, …]` joint stepper composing both
#     sides.
#
# IC is a bi-Maxwellian `f₀ ∝ exp(-v1²/σ_par² − v2²/σ_perp²)` with
# `σ_perp > σ_par` plus a small `B_z(x, 0) = δ · sin(kx)` seed, `k = 2π`.
#
# Linear theory. For transverse perturbations `(E_y, B_z, δf) ∝
# exp(i(kx − ωt))` on this bi-Maxwellian the kinetic dispersion relation
# is (Weibel 1959; Krall & Trivelpiece §9.10)
#
#     ω² − k²c² − ω_p² + ω_p² · (σ_perp²/σ_par²) · [1 + ζ Z(ζ)] = 0,
#     ζ = ω / (k σ_par),
#
# with `Z` the plasma dispersion function. Setting `ω → 0` gives the
# Weibel cutoff `k_max = (ω_p/c) √(σ_perp²/σ_par² − 1)`: modes with
# `k < k_max` grow, modes above it do not. Here `σ_par² = 0.5`,
# `σ_perp² = 1`, `c = ω_p = 1` ⇒ `k_max = 1`, and `k = 2π` is far above
# the cutoff. (An earlier version of this file quoted a "growth rate"
# of 0.70 from the cold-plasma formula; that formula does not apply
# here.) The seed is therefore a standing electromagnetic wave whose
# two counter-propagating branches are the roots
#
#     ω = ±6.40797 − 0.049951 i
#
# (solved with `scipy.special.wofz`; the plasma shifts the vacuum
# frequency `kc = 6.2832` up by 2 % and Landau-damps the wave on the
# `v1 = ω/k` particles). With `B_z = δ sin(kx)`, `E_y = 0` at t = 0 the
# `sin(kx)` amplitude follows
#
#     b(t) ≈ δ · e^{−γt} · [cos(ω_r t) + (γ/ω_r) sin(ω_r t)].
#
# Test checks, over exactly two wave periods:
#   (a) the period of `b(t)` from its zero crossings matches
#       `2π/ω_r` — this needs `E_y ↔ B_z` (Maxwell), `J_y → E_y`
#       (current source) and the transverse Lorentz force together;
#       with the plasma response removed the period is the vacuum
#       `2π/kc`, 2 % off,
#   (b) `b` after two periods matches `δ · e^{−2γT}` — the Landau
#       damping, absent without the `v × B` / `E_y` force on `f`,
#   (c) mass conservation.
#
# Build / run:
#     pixi run mojo run -I src tests/test_vlasov_maxwell_weibel_1d2v.mojo

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
comptime NV1: Int = 8
comptime NV2: Int = 8
comptime X_MIN: Real = 0.0
comptime X_MAX: Real = 1.0
comptime V_MAX: Real = 4.0   # σ_perp = 1 ⇒ erfc(4) ≈ 1.5e-8 (1D), tail squared ≈ 2e-16
comptime SIGMA_PAR_SQ: Real = 0.5    # parallel (v1)
comptime SIGMA_PERP_SQ: Real = 1.0   # perpendicular (v2)
comptime CHARGE_MASS_RATIO: Real = -1.0
comptime SKIN_DEPTH_NORM: Real = 1.0
comptime EPS0: Real = 1.0
comptime C_LIGHT: Real = 1.0
# Complex root of the dispersion relation in the header for these
# parameters: ω = OMEGA_R − i·GAMMA_DAMP.
comptime OMEGA_R: Real = 6.40797
comptime GAMMA_DAMP: Real = 0.049951
# Exactly two wave periods.
comptime T_FINAL: Real = 2.0 * TWO_PI / OMEGA_R
comptime N_STEPS: Int = 500
comptime DT: Real = T_FINAL / Real(N_STEPS)
comptime SAMPLE_EVERY: Int = 5
comptime DELTA_BZ: Real = 1.0e-3   # B_z seed amplitude

comptime Phase = PhaseState[1, 2, P_X, P_V, VLASOV_NC]
comptime MaxwellState = DGState[1, P_X, MAXWELL_NC]


def compute_total_mass(
    phase_mesh: PhaseMesh[1, 2, P_X, P_V], mut q_phase: Phase,
) raises -> Real:
    """M = ∫∫∫ f dv1 dv2 dx via GLL quadrature over the full phase mesh."""
    var acc: Real = 0.0
    def add(x: Real, v1: Real, v2: Real, w: Real, f: Real) {mut acc}:
        acc += w * f
    phase_mesh.visit_nodes(q_phase, add)
    return acc


def bz_sin_amplitude(
    x_mesh: BlockMesh[1, P_X], mut q_maxwell: MaxwellState,
) raises -> Real:
    """Amplitude of the sin(kx) Fourier mode of B_z: (2/L) ∫ B_z sin(kx) dx."""
    var k_wave = TWO_PI / (X_MAX - X_MIN)
    var acc: Real = 0.0
    def add(x: Real, w: Real, v: Array[Real, MAXWELL_NC]) {mut acc, imm}:
        acc += w * v[IDX_BZ] * sin(k_wave * x)
    x_mesh.visit_nodes(q_maxwell, add)
    return Real(2.0) * acc / (X_MAX - X_MIN)


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = DeviceContext()

    var phase_mesh = PhaseMesh[1, 2, P_X, P_V](
        ctx,
        lo=[X_MIN, -V_MAX, -V_MAX],
        hi=[X_MAX, V_MAX, V_MAX],
        n=[NX, NV1, NV2],
        periodic=[True],
    )
    var x_mesh = BlockMesh[1, P_X](
        ctx, X_MIN, X_MAX, NX, periodic=True,
    )

    var q_phase = Phase(ctx, phase_mesh.num_phase_elements)
    var q_maxwell = MaxwellState(ctx, NX)

    # IC: f₀(v1, v2) = norm · exp(-v1²/σ_par² - v2²/σ_perp²), uniform in x.
    # Normalisation chosen so ∫∫ f dv1 dv2 = 1 ⇒ ω_p² = n α² / ε₀ = 1.
    var sigma_par = sqrt(SIGMA_PAR_SQ)
    var sigma_perp = sqrt(SIGMA_PERP_SQ)
    var norm = Real(1.0) / (PI * sigma_par * sigma_perp)

    def maxwellian(x: Real, v1: Real, v2: Real) {imm} -> Real:
        return norm * exp(
            -v1 * v1 / SIGMA_PAR_SQ
            - v2 * v2 / SIGMA_PERP_SQ
        )
    phase_mesh.fill(q_phase, maxwellian)

    # IC: B_z(x, 0) = δ sin(kx), all other field components zero.
    def seed(x: Real) -> Array[Real, MAXWELL_NC]:
        var v = Array[Real, MAXWELL_NC](fill=Real(0.0))
        v[IDX_BZ] = DELTA_BZ * sin(TWO_PI * x)
        return v^
    x_mesh.fill(q_maxwell, seed)
    ctx.synchronize()

    # Initial diagnostics.
    var m0 = compute_total_mass(phase_mesh, q_phase)
    var bz0 = bz_sin_amplitude(x_mesh, q_maxwell)
    print("Weibel IC: M =", m0, " B_z sin(kx) amplitude =", bz0)

    # --- solvers + stepper ---
    var vlasov_app = VlasovMaxwell(
        charge_mass_ratio=CHARGE_MASS_RATIO,
        skin_depth_norm=SKIN_DEPTH_NORM,
        cfl_max=Real(0.5),
    )
    var maxwell_app = Maxwell(c_light=C_LIGHT, cfl_max=Real(0.5))
    var spatial_vlasov = DGVlasovSolver[1, 2, P_X, P_V](vlasov_app)
    var spatial_maxwell = DGMaxwellSolver[1, P_X](maxwell_app)
    var moments = VlasovMoments[1, 2, P_X, P_V](ctx, phase_mesh)
    var stepper = SSPRK3VlasovMaxwell[1, 2, P_X, P_V](
        ctx, phase_mesh.num_phase_elements, phase_mesh.num_x_elements,
    )
    var mom = DGState[1, P_X, VLASOV_MOMENT_NC](ctx, NX)

    def compute_joint_rhs(
        mut qp_in: Phase,
        mut qm_in: MaxwellState,
        mut qp_out: Phase,
        mut qm_out: MaxwellState,
    ) raises {mut mom, imm}:
        spatial_vlasov.compute_rhs(ctx, phase_mesh, qp_in, qm_in, qp_out)
        spatial_maxwell.compute_rhs(ctx, x_mesh, qm_in, qm_out)
        moments.compute(ctx, phase_mesh, qp_in, mom)
        apply_vlasov_current_source[1, 2, P_X](
            ctx, mom, qm_out, CHARGE_MASS_RATIO, EPS0,
        )

    # --- time loop: two wave periods, sampling b(t) every 5 steps ---
    var t: Real = 0.0
    var b_prev = bz0
    var t_prev: Real = 0.0
    var crossings = List[Real]()
    var b_final = bz0
    for step in range(N_STEPS):
        stepper.step(ctx, q_phase, q_maxwell, DT, compute_joint_rhs)
        t += DT
        if (step + 1) % SAMPLE_EVERY == 0:
            ctx.synchronize()
            var b_now = bz_sin_amplitude(x_mesh, q_maxwell)
            if b_prev * b_now < 0.0:
                # Linear interpolation of the zero crossing.
                crossings.append(
                    t_prev + (t - t_prev) * b_prev / (b_prev - b_now)
                )
            b_prev = b_now
            t_prev = t
            b_final = b_now
    var m1 = compute_total_mass(phase_mesh, q_phase)

    # Successive zero crossings are half a period apart.
    assert_true(len(crossings) == 4, "expected four zero crossings of B_z")
    var period_obs = (
        Real(2.0) * (crossings[3] - crossings[0]) / Real(3.0)
    )
    var period_th = TWO_PI / OMEGA_R
    var period_err = (period_obs - period_th) / period_th
    var ratio_obs = b_final / bz0
    var ratio_th = exp(-GAMMA_DAMP * T_FINAL)
    var dM = m1 - m0

    print(
        "Weibel-stable EM wave: steps =", N_STEPS, " t =", t,
        " period =", period_obs, " theory =", period_th,
        " rel err =", period_err,
    )
    print(
        "  b(T)/b(0) =", ratio_obs, " theory e^{-γT} =", ratio_th,
        " diff =", ratio_obs - ratio_th, " ΔM =", dM,
    )

    # (a) Frequency. Measured relative error 4.9e-4; the vacuum
    # period 2π/kc differs by 2.0e-2.
    assert_true(
        abs(period_err) < Real(2.0e-3),
        "EM-wave period off the kinetic dispersion relation",
    )
    # (b) Landau damping over two periods. Undamped would be 1.0
    # (0.093 off); measured difference 4.9e-3, from the coarse
    # (8 × 8, dv = 1) velocity mesh, whose quadrature also gives n = 0.995.
    assert_true(
        abs(ratio_obs - ratio_th) < Real(1.5e-2),
        "EM-wave damping off the kinetic dispersion relation",
    )
    # (c) Mass conservation.
    assert_true(abs(dM) < Real(1.0e-9), "mass not conserved")

    print("PASSED")
