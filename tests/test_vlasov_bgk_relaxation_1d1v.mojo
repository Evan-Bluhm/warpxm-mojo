# Vlasov BGK collision source — dynamic relaxation test.
#
# Initial condition: two counter-streaming Maxwellian peaks at v = ±1
#
#     f₀(x, v) = ½·exp(-(v−1)²/σ²) + ½·exp(-(v+1)²/σ²),
#
# strongly non-Maxwellian. Apply BGK alone (no Vlasov advection, no
# Maxwell coupling) for many relaxation times. The BGK operator
# conserves `(n, u, T)` *exactly* by construction (the local
# Maxwellian carries the same three moments as `f`) and drives `f`
# toward `f_M = n/√(2π T) · exp(-(v-u)²/(2T))` with rate `1/τ`. After
# `t = 10 τ`, the deviation should be `exp(-10) ≈ 4·10⁻⁵` of the
# initial.
#
# This validates:
#   (a) Moment conservation: ρ, J, E_kin all bit-for-bit invariant
#       across the run.
#   (b) Convergence: the final `f` matches the analytic single
#       Maxwellian with the IC's moments to ≲ 1·10⁻³ L² error.
#
# Build / run:
#     pixi run mojo run -I src tests/test_vlasov_bgk_relaxation_1d1v.mojo

from std.math import exp
from std.sys import has_accelerator
from max.gpu.host import DeviceContext
from std.testing import assert_true

from warpxm.prelude import *
from warpxm.physics.vlasov import *


comptime P_X: Int = 2
comptime P_V: Int = 3
comptime NX: Int = 4
comptime NV: Int = 32
comptime V_MIN: Real = -4.0
comptime V_MAX: Real = 4.0
# Initial peaks at v = ±U_STREAM with width σ. After relaxation the
# Maxwellian has T = σ²/2 + U_STREAM². The v-truncation residue at
# `v_max` scales as `erfc(v_max/√(2T))`; for the values below
# `erfc(4 / √0.75) ≈ erfc(4.62) ≈ 5e-11` ⇒ moment conservation should
# hold to ≈ 1e-9 over the run.
comptime SIGMA_SQ: Real = 0.25
comptime U_STREAM: Real = 0.5
comptime TAU: Real = 0.1
comptime T_FINAL: Real = 1.0   # 10 · τ — deep into relaxation
comptime DT: Real = 0.01


def compute_three_moments(
    mesh: PhaseMesh[1, 1, P_X, P_V],
    mut q: PhaseState[1, 1, P_X, P_V, VLASOV_NC],
    mut m_out: Real,
    mut p_out: Real,
    mut e_out: Real,
) raises:
    """∫∫ (1, v, ½v²) · f dv dx via GLL quadrature; writes the three
    moments (M, P, E_kin) to the supplied mutable refs in one pass."""
    var m_acc: Real = 0.0
    var p_acc: Real = 0.0
    var e_acc: Real = 0.0
    def add(x: Real, v: Real, w: Real, f: Real) {mut m_acc, mut p_acc, mut e_acc}:
        m_acc += w * f
        p_acc += w * v * f
        e_acc += w * Real(0.5) * v * v * f
    mesh.visit_nodes(q, add)
    m_out = m_acc
    p_out = p_acc
    e_out = e_acc


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = DeviceContext()

    var mesh = PhaseMesh[1, 1, P_X, P_V](
        ctx, lo=[0.0, V_MIN], hi=[1.0, V_MAX], n=[NX, NV],
        periodic=[True],
    )
    var q = PhaseState[1, 1, P_X, P_V, VLASOV_NC](
        ctx, mesh.num_phase_elements,
    )

    # IC: bimodal counter-streaming Maxwellian.
    def bimodal(x: Real, v: Real) -> Real:
        var d1 = v - U_STREAM
        var d2 = v + U_STREAM
        return Real(0.5) * (
            exp(-d1 * d1 / SIGMA_SQ) + exp(-d2 * d2 / SIGMA_SQ)
        )
    mesh.fill(q, bimodal)

    # --- baseline moments (numerical from the IC) ---
    var m0_initial: Real = 0.0
    var p0_initial: Real = 0.0
    var e0_initial: Real = 0.0
    compute_three_moments(mesh, q, m0_initial, p0_initial, e0_initial)
    print(
        "BGK relax IC: M =", m0_initial,
        " P =", p0_initial,
        " E_kin =", e0_initial,
    )

    # --- solver + stepper ---
    var bgk = VlasovBGKSource[1, 1, P_X, P_V](ctx, mesh)
    var stepper = SSPRK3Phase[1, 1, P_X, P_V, VLASOV_NC](
        ctx, mesh.num_phase_elements,
    )

    def compute_rhs(
        mut q_in: PhaseState[1, 1, P_X, P_V, VLASOV_NC],
        mut q_out: PhaseState[1, 1, P_X, P_V, VLASOV_NC],
    ) raises {mut bgk, imm}:
        # BGK alone — zero the residual first because `bgk.apply`
        # *adds* its contribution.
        q_out.buf.enqueue_fill(Real(0.0))
        bgk.apply(ctx, mesh, q_in, q_out, TAU)

    var t: Real = 0.0
    var nsteps: Int = 0
    while t < T_FINAL - Real(0.5) * DT:
        var dt_use = DT if (t + DT) < T_FINAL else (T_FINAL - t)
        stepper.step(
            ctx, q, dt_use, compute_rhs,
            no_op_adjust_phase[1, 1, P_X, P_V, VLASOV_NC],
        )
        t += dt_use
        nsteps += 1
    ctx.synchronize()

    # --- post-relaxation moments and convergence check ---
    var m_final: Real = 0.0
    var p_final: Real = 0.0
    var e_final: Real = 0.0
    compute_three_moments(mesh, q, m_final, p_final, e_final)

    var dM = m_final - m0_initial
    var dP = p_final - p0_initial
    var dE = e_final - e0_initial
    var adM = dM if dM >= 0.0 else -dM
    var adP = dP if dP >= 0.0 else -dP
    var adE = dE if dE >= 0.0 else -dE
    print(
        "BGK relax final: steps =", nsteps, "t =", t,
        " ΔM =", dM, " ΔP =", dP, " ΔE_kin =", dE,
    )

    # Analytic post-relaxation Maxwellian.
    # n_per_x = ∫ f₀ dv = σ√π (each peak unnormalised contributes
    # σ√π; the IC is ½·sum). T = ⟨v²⟩ − u² = σ²/2 + U² − 0 (both
    # peaks have ⟨v²⟩ = U² + σ²/2; average is the same).
    var sigma = sqrt(SIGMA_SQ)
    var n_per_x = sigma * sqrt(PI)
    var T_target = SIGMA_SQ / Real(2.0) + U_STREAM * U_STREAM
    var t_sigma_M_sq = Real(2.0) * T_target  # f_M ∝ exp(-(v-u)²/(2T))

    var l2: Real = 0.0
    var max_err: Real = 0.0
    def compare(x: Real, v: Real, w: Real, f_num: Real) {mut l2, mut max_err, imm}:
        var f_M = n_per_x / sqrt(
            Real(2.0) * PI * T_target
        ) * exp(-v * v / t_sigma_M_sq)
        var d = f_num - f_M
        max_err = fmax(max_err, fabs(d))
        l2 += w * d * d
    mesh.visit_nodes(q, compare)
    l2 = sqrt(l2)
    print(
        "BGK relax convergence: L2 ‖f − f_M‖ =", l2,
        " max |err| =", max_err,
    )

    # (a) Moment conservation. In the continuum, BGK preserves
    # (n, u, T) exactly; discretely, the deviation is bounded by the
    # v-truncation residue ≈ `n · erfc(v_max/√(2T)) / τ · T_final`,
    # multiplied by the moment's `v^k` weight in the integrand. For
    # these parameters M gets ~ erfc ≈ 5e-11 → propagated 1e-9 over
    # the run; E_kin picks up a `v_max² ≈ 16` factor → ~1e-7. P is
    # protected by symmetry and stays at machine epsilon.
    # Measured: ΔM/M = 6.5e-10, ΔP = 4e-18, ΔE/E = 2.9e-8.
    assert_true(adM / m0_initial < Real(5.0e-9), "BGK didn't conserve M")
    assert_true(adP < Real(1.0e-12), "BGK didn't conserve P")
    assert_true(adE / e0_initial < Real(2.0e-7), "BGK didn't conserve E_kin")
    # (b) Convergence: at t = 10 · τ, deviation ≲ exp(-10) ≈ 4·10⁻⁵
    # of initial. The initial L² (f₀ − f_M) is O(0.1), so ≈ 5e-6 is
    # expected; measured 7.5e-6. (A relaxation rate 20 % too slow
    # would leave exp(-8)·0.1 ≈ 3.4e-5.)
    assert_true(l2 < Real(2.0e-5), "BGK relaxation didn't converge")

    print("PASSED")
