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

from std.math import exp, sqrt
from std.sys import has_accelerator
from std.gpu.host import DeviceContext
from std.testing import assert_true

from warpxm.apps.vlasovmaxwell import VLASOV_NC
from warpxm.basis.nodal_basis_1d import NodalBasis1D
from warpxm.core.types import Real, PI
from warpxm.mesh.phase_mesh_1d1v import PhaseMesh1D1V
from warpxm.spatial.vlasov_bgk_source_1d1v import VlasovBGKSource1D1V
from warpxm.temporal.ssprk3_1d1v import SSPRK3_1D1V
from warpxm.variable.phase_state_1d1v import PhaseState1D1V


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
    var ctx: DeviceContext,
    mut mesh: PhaseMesh1D1V[P_X, P_V],
    mut q: PhaseState1D1V[P_X, P_V, VLASOV_NC],
    mut m_out: Real,
    mut p_out: Real,
    mut e_out: Real,
) raises:
    """∫∫ (1, v, ½v²) · f dv dx via GLL quadrature; writes the three
    moments (M, P, E_kin) to the supplied mutable refs in one pass."""
    var x_w = NodalBasis1D[P_X].weights()
    var v_w = NodalBasis1D[P_V].weights()
    var ref_v = NodalBasis1D[P_V].nodes()
    var m_acc: Real = 0.0
    var p_acc: Real = 0.0
    var e_acc: Real = 0.0
    with q.buf.map_to_host() as h:
        for ex in range(NX):
            for ev in range(NV):
                var vl = V_MIN + Real(ev) * mesh.dv
                for ix in range(P_X + 1):
                    for iv in range(P_V + 1):
                        var v = vl + (Real(1.0) + ref_v[iv]) * mesh.jacobian_v
                        var phase_e = ex * NV + ev
                        var idx = (
                            (phase_e * (P_V + 1) + iv) * (P_X + 1) + ix
                        ) * VLASOV_NC
                        var f = h[idx]
                        var w = (
                            x_w[ix] * mesh.jacobian_x
                            * v_w[iv] * mesh.jacobian_v
                        )
                        m_acc += w * f
                        p_acc += w * v * f
                        e_acc += w * Real(0.5) * v * v * f
    m_out = m_acc
    p_out = p_acc
    e_out = e_acc


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = DeviceContext()

    var mesh = PhaseMesh1D1V[P_X, P_V](
        ctx,
        Real(0.0), Real(1.0), V_MIN, V_MAX,
        num_x=NX, num_v=NV,
        periodic_x=True,
    )
    var q = PhaseState1D1V[P_X, P_V, VLASOV_NC](ctx, NX, NV)

    # IC: bimodal counter-streaming Maxwellian.
    var ref_v = NodalBasis1D[P_V].nodes()
    var host = ctx.enqueue_create_host_buffer[Real.dtype](q.buf_len)
    for ex in range(NX):
        for ev in range(NV):
            var vl = V_MIN + Real(ev) * mesh.dv
            for ix in range(P_X + 1):
                for iv in range(P_V + 1):
                    var v = vl + (Real(1.0) + ref_v[iv]) * mesh.jacobian_v
                    var phase_e = ex * NV + ev
                    var idx = (
                        (phase_e * (P_V + 1) + iv) * (P_X + 1) + ix
                    ) * VLASOV_NC
                    var d1 = v - U_STREAM
                    var d2 = v + U_STREAM
                    host[idx] = Real(0.5) * (
                        exp(-d1 * d1 / SIGMA_SQ) + exp(-d2 * d2 / SIGMA_SQ)
                    )
    ctx.enqueue_copy(dst_buf=q.buf, src_buf=host)
    ctx.synchronize()

    # --- baseline moments (numerical from the IC) ---
    var m0_initial: Real = 0.0
    var p0_initial: Real = 0.0
    var e0_initial: Real = 0.0
    compute_three_moments(
        ctx, mesh, q, m0_initial, p0_initial, e0_initial,
    )
    print(
        "BGK relax IC: M =", m0_initial,
        " P =", p0_initial,
        " E_kin =", e0_initial,
    )

    # --- solver + stepper ---
    var bgk = VlasovBGKSource1D1V[P_X, P_V](ctx, NX)
    var stepper = SSPRK3_1D1V[P_X, P_V, VLASOV_NC](ctx, NX, NV)

    @parameter
    def compute_rhs(
        mut q_in: PhaseState1D1V[P_X, P_V, VLASOV_NC],
        mut q_out: PhaseState1D1V[P_X, P_V, VLASOV_NC],
    ) raises:
        # BGK alone — zero the residual first because `bgk.apply`
        # *adds* its contribution.
        q_out.buf.enqueue_fill(Real(0.0))
        bgk.apply(ctx, mesh, q_in, q_out, TAU)

    @parameter
    def no_adjust(mut _q: PhaseState1D1V[P_X, P_V, VLASOV_NC]) raises:
        pass

    var t: Real = 0.0
    var nsteps: Int = 0
    while t < T_FINAL - Real(0.5) * DT:
        var dt_use = DT if (t + DT) < T_FINAL else (T_FINAL - t)
        stepper.step[compute_rhs, no_adjust](ctx, q, dt_use)
        t += dt_use
        nsteps += 1
    ctx.synchronize()

    # --- post-relaxation moments and convergence check ---
    var m_final: Real = 0.0
    var p_final: Real = 0.0
    var e_final: Real = 0.0
    compute_three_moments(ctx, mesh, q, m_final, p_final, e_final)

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

    var x_w = NodalBasis1D[P_X].weights()
    var v_w = NodalBasis1D[P_V].weights()
    var l2: Real = 0.0
    var max_err: Real = 0.0
    with q.buf.map_to_host() as h:
        for ex in range(NX):
            for ev in range(NV):
                var vl = V_MIN + Real(ev) * mesh.dv
                for ix in range(P_X + 1):
                    for iv in range(P_V + 1):
                        var v = vl + (Real(1.0) + ref_v[iv]) * mesh.jacobian_v
                        var phase_e = ex * NV + ev
                        var idx = (
                            (phase_e * (P_V + 1) + iv) * (P_X + 1) + ix
                        ) * VLASOV_NC
                        var f_num = h[idx]
                        var f_M = n_per_x / sqrt(
                            Real(2.0) * PI * T_target
                        ) * exp(-v * v / t_sigma_M_sq)
                        var d = f_num - f_M
                        var ad = d if d >= 0.0 else -d
                        if ad > max_err:
                            max_err = ad
                        var w = (
                            x_w[ix] * mesh.jacobian_x
                            * v_w[iv] * mesh.jacobian_v
                        )
                        l2 += w * d * d
    l2 = l2 ** Real(0.5)
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
    assert_true(adM / m0_initial < Real(1.0e-8), "BGK didn't conserve M")
    assert_true(adP < Real(1.0e-12), "BGK didn't conserve P")
    assert_true(adE / e0_initial < Real(1.0e-6), "BGK didn't conserve E_kin")
    # (b) Convergence: at t = 10 · τ, deviation ≲ exp(-10) ≈ 4·10⁻⁵
    # of initial. The initial L² (f₀ − f_M) is O(0.1); generous 1e-3
    # absorbs any tail residue.
    assert_true(l2 < Real(1.0e-3), "BGK relaxation didn't converge")

    print("PASSED")
