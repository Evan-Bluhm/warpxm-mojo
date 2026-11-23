# Weibel instability growth rate (1D2V Vlasov-Maxwell) vs linear theory.
#
# A bi-Maxwellian `f₀ ∝ exp(-v1²/σ_par² − v2²/σ_perp²)` that is hotter
# across `k` (v2) than along it (v1) is unstable to a purely growing
# transverse mode `(E_y, B_z, δf) ∝ exp(ikx + γt)`.
#
# Linear theory. The kinetic dispersion relation for transverse modes
# on this distribution is (Weibel 1959; Krall & Trivelpiece §9.10)
#
#     ω² − k²c² − ω_p² + ω_p² · (σ_perp²/σ_par²) · [1 + ζ Z(ζ)] = 0,
#     ζ = ω / (k σ_par),
#
# with `Z` the plasma dispersion function. On the imaginary axis,
# `ω = iγ`, `ζ = is` with `s = γ/(k σ_par)`, and `Z(is) = i√π e^{s²}
# erfc(s)`, so the growth rate is the real root of
#
#     ω_p² · [(σ_perp²/σ_par²) · (1 − √π s e^{s²} erfc(s)) − 1]
#         = γ² + k²c².
#
# `γ → 0` gives the cutoff `k_max = (ω_p/c) √(σ_perp²/σ_par² − 1)`.
#
# Parameters: `L = 2π` ⇒ `k = 1`; `σ_par² = 0.09`, `σ_perp² = 1`
# (`T_perp/T_par = 11.1`, `k_max = 3.18`); `c = ω_p = 1` (n = 1,
# α = −1, ε₀ = 1). Bisection on the relation above gives
#
#     γ_W = 0.364360.
#
# (The cold-parallel Davidson formula `γ² = k²c²(A−1)/(1 + k²c²/ω_p²)`
# is not usable here or in the earlier version of this test: it needs
# `γ ≫ k σ_par`, and here `γ/(k σ_par) = 1.2`. For the earlier
# parameters — `k = π/2`, `σ_par² = 0.25` — it gave 0.69 against a true
# kinetic root of 0.064, which is why that version saw no clean growth.)
#
# IC: unperturbed `f₀` plus a `B_z = δ sin(kx)` seed with `δ = 1e-6`.
# The seed projects onto the growing mode, its decaying partner and the
# stable electromagnetic pair (`ω = ±1.496`); by `t = 10` the growing
# mode dominates and the amplitude has grown 12×, 456× by `t = 20` —
# still linear (`B_z ≈ 4.6e-4`).
#
# Checks:
#   (a) least-squares slope of `ln |b(t)|` over `t ∈ [10, 20]`, where
#       `b` is the `sin(kx)` Fourier amplitude of `B_z`, matches `γ_W`.
#       Measured 0.364446 — 2.4e-4 relative. This needs every piece of
#       the 1D2V coupling with the right sign and magnitude: Faraday
#       and Ampère, the `J_y` current source, and both the `E_y` and
#       `v × B_z` terms of the Lorentz force.
#   (b) mass conservation < 1e-9.
#
# Build / run:
#     pixi run mojo run -I src tests/test_vlasov_maxwell_weibel_growth_1d2v.mojo

from std.collections import Array, List
from std.math import exp, sin, log
from std.sys import has_accelerator
from max.gpu.host import DeviceContext
from std.testing import assert_true

from warpxm.prelude import *
from warpxm.physics.vlasov import *
from warpxm.physics.maxwell import *


comptime P_X: Int = 2
comptime P_V: Int = 2
comptime NX: Int = 8
comptime NV1: Int = 12
comptime NV2: Int = 10
comptime X_MIN: Real = 0.0
comptime X_MAX: Real = TWO_PI
comptime V1_MAX: Real = 1.5     # 5 σ_par
comptime V2_MAX: Real = 4.0     # 4 σ_perp
comptime SIGMA_PAR_SQ: Real = 0.09
comptime SIGMA_PERP_SQ: Real = 1.0
comptime CHARGE_MASS_RATIO: Real = -1.0
comptime SKIN_DEPTH_NORM: Real = 1.0
comptime EPS0: Real = 1.0
comptime C_LIGHT: Real = 1.0
# Growing root of the kinetic dispersion relation (see header).
comptime GAMMA_W: Real = 0.364360
comptime T_FINAL: Real = 20.0
comptime N_STEPS: Int = 400
comptime DT: Real = T_FINAL / Real(N_STEPS)
comptime DELTA: Real = 1.0e-6       # B_z seed amplitude
comptime SAMPLE_EVERY: Int = 4      # sample stride for the B_z amplitude
comptime FIT_T_MIN: Real = 10.0     # linear-fit window
comptime FIT_T_MAX: Real = 20.0

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
        lo=[X_MIN, -V1_MAX, -V2_MAX],
        hi=[X_MAX, V1_MAX, V2_MAX],
        n=[NX, NV1, NV2],
        periodic=[True],
    )
    var x_mesh = BlockMesh[1, P_X](
        ctx, X_MIN, X_MAX, NX, periodic=True,
    )

    var q_phase = Phase(ctx, phase_mesh.num_phase_elements)
    var q_maxwell = MaxwellState(ctx, NX)

    var k_wave = TWO_PI / (X_MAX - X_MIN)
    # IC: f = f₀ = norm · exp(-v1²/σ_par² - v2²/σ_perp²), n = 1 ⇒
    # norm = 1/(π σ_par σ_perp). No δf: the B_z seed alone projects onto
    # the growing mode.
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
    def seed(x: Real) {imm} -> Array[Real, MAXWELL_NC]:
        var v = Array[Real, MAXWELL_NC](fill=Real(0.0))
        v[IDX_BZ] = DELTA * sin(k_wave * x)
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

    # --- time loop, sampling the B_z sin(kx) amplitude ---
    var t_samples = List[Real]()
    var bz_samples = List[Real]()
    var t: Real = 0.0
    for step in range(N_STEPS):
        stepper.step(ctx, q_phase, q_maxwell, DT, compute_joint_rhs)
        t += DT
        if (step + 1) % SAMPLE_EVERY == 0:
            ctx.synchronize()
            t_samples.append(t)
            bz_samples.append(bz_sin_amplitude(x_mesh, q_maxwell))
    var bz_final = bz_samples[len(bz_samples) - 1]
    var m1 = compute_total_mass(phase_mesh, q_phase)

    # --- linear least-squares fit of ln|B_z| vs t over [FIT_T_MIN, FIT_T_MAX] ---
    var sum_t: Real = 0.0
    var sum_y: Real = 0.0
    var sum_tt: Real = 0.0
    var sum_ty: Real = 0.0
    var n_fit: Int = 0
    for i in range(len(t_samples)):
        var ti = t_samples[i]
        if ti < FIT_T_MIN or ti > FIT_T_MAX:
            continue
        var yi = log(abs(bz_samples[i]))
        sum_t += ti
        sum_y += yi
        sum_tt += ti * ti
        sum_ty += ti * yi
        n_fit += 1

    var nf = Real(n_fit)
    var gamma_obs = (
        (nf * sum_ty - sum_t * sum_y) / (nf * sum_tt - sum_t * sum_t)
    )
    var rel_err = (gamma_obs - GAMMA_W) / GAMMA_W
    var dM = m1 - m0
    print(
        "Weibel growth: steps =", N_STEPS, " t =", t,
        " samples_in_fit =", n_fit,
        " B_z(T)/B_z(0) =", bz_final / bz0,
    )
    print(
        "Weibel fit: γ_obs =", gamma_obs,
        " γ_theory =", GAMMA_W,
        " rel_err =", rel_err, " ΔM =", dM,
    )

    # Measured 2.4e-4; the residual is the decaying `ω = ±1.496` wave
    # still riding on the fit window.
    assert_true(
        abs(rel_err) < Real(3.0e-3), "Weibel growth rate off linear theory",
    )
    assert_true(abs(dM) < Real(1.0e-9), "mass not conserved")

    print("PASSED")
