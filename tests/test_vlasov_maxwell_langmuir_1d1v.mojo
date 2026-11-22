# Coupled Vlasov-Maxwell 1D1V — spatial Langmuir mode smoke test.
#
# Builds on `test_vlasov_maxwell_uniform_oscillator_1d1v.mojo`: same
# normalised Maxwellian background, same `α = q/m = -1` electron-like
# setup, but the initial `E_x` carries a sinusoidal spatial
# perturbation `E_x(x, 0) = ε · sin(2π x / L)`. This activates the
# full coupling — Vlasov x-advection, Maxwell evolution of E_x via
# the current source, and the position-dependent v-acceleration that
# develops as `f` becomes x-dependent.
#
# With these parameters (ω_p = 1, k = 2π, σ² = 0.5) the mode is far
# into the kinetic regime, `k·λ_D = k·σ/√2 ≈ 3.1`: there is no weakly
# damped Langmuir branch (see `test_vlasov_maxwell_bohm_gross_1d1v`
# for the dispersion check at small `k·λ_D`). Instead the response is
# the ballistic / phase-mixing transient, which linear theory gives in
# closed form. Linearising about `f₀(v) = exp(-v²/σ²)/(σ√π)` and
# integrating the perturbed Vlasov equation along free-streaming
# characteristics, the amplitude `E(t)` of the `sin(kx)` mode obeys the
# Volterra equation
#
#     dE/dt = -ω_p² ∫₀ᵗ E(τ) · G(t − τ) dτ,
#     G(s)  = exp(-a s²) · (1 − 2 a s²),      a = k² σ² / 4.
#
# (`G` is the current response to a unit field impulse:
# `∫ v ∂_v f₀ e^{-ikvs} dv = -G(s)`.) `∫₀^∞ G ds = 0`, so the current
# phase-mixes away and `E` settles at a non-zero plateau — here
# `E(1)/E(0) = 0.90692`, i.e. the field gives up 17.7 % of its energy
# to the particles.
#
# Checks:
#
#   (a) Field amplitude vs linear theory. `√(W_E(t)/W_E(0))` is
#       compared with the Volterra solution (integrated on the host)
#       at t = 0.5 and t = 1. This fails if either direction of the
#       coupling, or the x-advection that sets `k`, is wrong.
#
#   (b) Total energy `W = ½ ∫ E_x² dx + ½ ∫∫ v² f dv dx` is conserved
#       by the continuous system when `λ_skin = ε₀`. The drift is
#       measured against the initial FIELD energy (2.5e-5), not the
#       total (0.125, almost all thermal) — relative to the total even
#       losing the whole field would be a 2e-4 change.
#
#   (c) Total particle mass `M = ∫∫ f dv dx` is conserved (periodic
#       x; `f ≈ 0` at the v boundaries).
#
# Build / run:
#     pixi run mojo run -I src tests/test_vlasov_maxwell_langmuir_1d1v.mojo

from std.collections import List
from std.math import exp, sin, sqrt
from std.sys import has_accelerator
from max.gpu.host import DeviceContext
from std.testing import assert_true

from warpxm.apps.maxwell import Maxwell, MAXWELL_NC, IDX_EX
from warpxm.apps.vlasovmaxwell import VlasovMaxwell, VLASOV_NC
from warpxm.basis.nodal_basis_1d import NodalBasis1D
from warpxm.core.types import Real, PI, TWO_PI
from warpxm.mesh.block_1d import BlockMesh1D
from warpxm.mesh.phase_mesh_1d1v import PhaseMesh1D1V
from warpxm.spatial.dg_maxwell_1d import DGMaxwellSolver1D
from warpxm.spatial.dg_vlasov_1d1v import DGVlasovSolver1D1V
from warpxm.spatial.vlasov_current_source_1d import (
    apply_vlasov_current_source_1d,
)
from warpxm.spatial.vlasov_moments_1d1v import VlasovMoments1D1V
from warpxm.temporal.ssprk3_vlasov_maxwell_1d1v import (
    SSPRK3_VlasovMaxwell_1D1V,
)
from warpxm.variable.phase_state_1d1v import PhaseState1D1V
from warpxm.variable.state_1d import DGState1D


comptime P_X: Int = 2
comptime P_V: Int = 2
comptime NX: Int = 8
comptime NV: Int = 16
comptime SIGMA_SQ: Real = 0.5
comptime CHARGE_MASS_RATIO: Real = -1.0
comptime SKIN_DEPTH_NORM: Real = 1.0
comptime EPS0: Real = 1.0
comptime C_LIGHT: Real = 1.0
comptime T_FINAL: Real = 1.0
comptime EPS: Real = 0.01  # amplitude of the E_x perturbation


def compute_total_energy(
    var ctx: DeviceContext,
    mut phase_mesh: PhaseMesh1D1V[P_X, P_V],
    mut x_mesh: BlockMesh1D[P_X],
    mut q_phase: PhaseState1D1V[P_X, P_V, VLASOV_NC],
    mut q_maxwell: DGState1D[P_X, MAXWELL_NC],
) raises -> Tuple[Real, Real]:
    """(W_E, W_K) = (½ ∫ E_x² dx, ½ ∫∫ v² f dv dx) via GLL quadrature."""
    var x_weights = NodalBasis1D[P_X].weights()
    var v_weights = NodalBasis1D[P_V].weights()
    var ref_v_nodes = NodalBasis1D[P_V].nodes()

    # Electric energy.
    var w_E: Real = 0.0
    with q_maxwell.buf.map_to_host() as h_m:
        for ex in range(NX):
            for ix in range(P_X + 1):
                var node_x = ex * (P_X + 1) + ix
                var ex_val: Real = h_m[node_x * MAXWELL_NC + IDX_EX]
                var w = x_weights[ix] * x_mesh.jacobian
                w_E += w * Real(0.5) * ex_val * ex_val

    # Kinetic energy: ½ ∫∫ v² f dv dx — full phase-space integral.
    var w_K: Real = 0.0
    with q_phase.buf.map_to_host() as h_p:
        for ex in range(NX):
            for ev in range(NV):
                var vl = Real(-3.0) + Real(ev) * phase_mesh.dv
                for ix in range(P_X + 1):
                    for iv in range(P_V + 1):
                        var v = vl + (Real(1.0) + ref_v_nodes[iv]) * phase_mesh.jacobian_v
                        var phase_e = ex * NV + ev
                        var idx = (
                            (phase_e * (P_V + 1) + iv) * (P_X + 1) + ix
                        ) * VLASOV_NC
                        var f: Real = h_p[idx]
                        var w = (
                            x_weights[ix] * phase_mesh.jacobian_x
                            * v_weights[iv] * phase_mesh.jacobian_v
                        )
                        w_K += w * Real(0.5) * v * v * f
    return Tuple[Real, Real](w_E, w_K)


def compute_total_mass(
    var ctx: DeviceContext,
    mut phase_mesh: PhaseMesh1D1V[P_X, P_V],
    mut q_phase: PhaseState1D1V[P_X, P_V, VLASOV_NC],
) raises -> Real:
    """M = ∫∫ f dv dx."""
    var x_weights = NodalBasis1D[P_X].weights()
    var v_weights = NodalBasis1D[P_V].weights()
    var acc: Real = 0.0
    with q_phase.buf.map_to_host() as h:
        for ex in range(NX):
            for ev in range(NV):
                for ix in range(P_X + 1):
                    for iv in range(P_V + 1):
                        var phase_e = ex * NV + ev
                        var idx = (
                            (phase_e * (P_V + 1) + iv) * (P_X + 1) + ix
                        ) * VLASOV_NC
                        var f: Real = h[idx]
                        var w = (
                            x_weights[ix] * phase_mesh.jacobian_x
                            * v_weights[iv] * phase_mesh.jacobian_v
                        )
                        acc += w * f
    return acc


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = DeviceContext()

    var phase_mesh = PhaseMesh1D1V[P_X, P_V](
        ctx, Real(0.0), Real(1.0), Real(-3.0), Real(3.0),
        num_x=NX, num_v=NV, periodic_x=True,
    )
    var x_mesh = BlockMesh1D[P_X](
        ctx, Real(0.0), Real(1.0), NX, periodic=True,
    )

    var q_phase = PhaseState1D1V[P_X, P_V, VLASOV_NC](ctx, NX, NV)
    var q_maxwell = DGState1D[P_X, MAXWELL_NC](ctx, NX)

    # IC: f₀(v) = exp(-v²/σ²) / (σ√π) — x-independent, ∫f dv ≈ 1.
    var sigma = sqrt(SIGMA_SQ)
    var norm = Real(1.0) / (sigma * sqrt(PI))
    var ref_x_nodes = NodalBasis1D[P_X].nodes()
    var ref_v_nodes = NodalBasis1D[P_V].nodes()
    var p_host = ctx.enqueue_create_host_buffer[Real.dtype](q_phase.buf_len)
    for ex in range(NX):
        for ev in range(NV):
            var vl = Real(-3.0) + Real(ev) * phase_mesh.dv
            for ix in range(P_X + 1):
                for iv in range(P_V + 1):
                    var v = vl + (Real(1.0) + ref_v_nodes[iv]) * phase_mesh.jacobian_v
                    var phase_e = ex * NV + ev
                    var idx = (
                        (phase_e * (P_V + 1) + iv) * (P_X + 1) + ix
                    ) * VLASOV_NC
                    p_host[idx] = norm * exp(-v * v / SIGMA_SQ)
    ctx.enqueue_copy(dst_buf=q_phase.buf, src_buf=p_host)

    # IC: E_x(x) = ε · sin(2π x / L), other components zero.
    var m_host = ctx.enqueue_create_host_buffer[Real.dtype](q_maxwell.buf_len)
    for k in range(q_maxwell.buf_len):
        m_host[k] = Real(0.0)
    for ex in range(NX):
        var xl = Real(0.0) + Real(ex) * x_mesh.dx
        for ix in range(P_X + 1):
            var x = xl + (Real(1.0) + ref_x_nodes[ix]) * x_mesh.jacobian
            var node_x = ex * (P_X + 1) + ix
            m_host[node_x * MAXWELL_NC + IDX_EX] = EPS * sin(TWO_PI * x)
    ctx.enqueue_copy(dst_buf=q_maxwell.buf, src_buf=m_host)
    ctx.synchronize()

    # Initial diagnostics.
    var w0 = compute_total_energy(
        ctx, phase_mesh, x_mesh, q_phase, q_maxwell,
    )
    var m0 = compute_total_mass(ctx, phase_mesh, q_phase)
    print("Vlasov-Maxwell Langmuir IC: W_E =", w0[0], " W_K =", w0[1], " M =", m0)

    # --- solvers + stepper ---
    var vlasov_app = VlasovMaxwell(
        charge_mass_ratio=CHARGE_MASS_RATIO,
        skin_depth_norm=SKIN_DEPTH_NORM,
        cfl_max=Real(0.5),
    )
    var maxwell_app = Maxwell(c_light=C_LIGHT, cfl_max=Real(0.5))
    var spatial_vlasov = DGVlasovSolver1D1V[P_X, P_V](vlasov_app)
    var spatial_maxwell = DGMaxwellSolver1D[P_X](maxwell_app)
    var moments = VlasovMoments1D1V[P_X, P_V]()
    var stepper = SSPRK3_VlasovMaxwell_1D1V[P_X, P_V](ctx, NX, NV)
    var j_current = DGState1D[P_X, 1](ctx, NX)

    def compute_joint_rhs(
        mut qp_in: PhaseState1D1V[P_X, P_V, VLASOV_NC],
        mut qm_in: DGState1D[P_X, MAXWELL_NC],
        mut qp_out: PhaseState1D1V[P_X, P_V, VLASOV_NC],
        mut qm_out: DGState1D[P_X, MAXWELL_NC],
    ) raises {mut j_current, imm}:
        spatial_vlasov.compute_rhs(ctx, phase_mesh, qp_in, qm_in, qp_out)
        spatial_maxwell.compute_rhs(ctx, x_mesh, qm_in, qm_out)
        moments.first_moment(ctx, phase_mesh, qp_in, j_current)
        apply_vlasov_current_source_1d[P_X](
            ctx, j_current, qm_out, CHARGE_MASS_RATIO, EPS0,
        )

    # --- linear theory: Volterra solution for E(t)/E(0) ---
    # Heun's method with trapezoidal convolution; N = 500 is converged
    # to 1e-6 (N = 4000 changes the result by 7e-7).
    comptime N_LIN = 500
    var k_wave = TWO_PI
    var a_lin = k_wave * k_wave * SIGMA_SQ / Real(4.0)
    var h_lin = T_FINAL / Real(N_LIN)
    var g_lin = List[Real]()
    for j in range(N_LIN + 1):
        var s_lag = Real(j) * h_lin
        g_lin.append(
            exp(-a_lin * s_lag * s_lag)
            * (Real(1.0) - Real(2.0) * a_lin * s_lag * s_lag)
        )
    var e_lin = List[Real]()
    e_lin.append(Real(1.0))
    for n in range(N_LIN):
        # ∫₀^{t_n} E(τ) G(t_n − τ) dτ
        var i0: Real = 0.0
        for j in range(n + 1):
            var wq = Real(0.5) if (j == 0 or j == n) else Real(1.0)
            i0 += wq * e_lin[j] * g_lin[n - j]
        i0 = i0 * h_lin if n > 0 else Real(0.0)
        var e_pred = e_lin[n] - h_lin * i0
        # ∫₀^{t_{n+1}} with the predictor as the last sample.
        var i1: Real = Real(0.5) * e_pred * g_lin[0]
        for j in range(n + 1):
            var wq = Real(0.5) if j == 0 else Real(1.0)
            i1 += wq * e_lin[j] * g_lin[n + 1 - j]
        i1 *= h_lin
        e_lin.append(e_lin[n] - Real(0.5) * h_lin * (i0 + i1))

    # --- time loop: two halves, sampled at t = 0.5 and t = 1 ---
    comptime STEPS_PER_HALF = 100
    var dt = T_FINAL / Real(2 * STEPS_PER_HALF)
    var t: Real = 0.0
    var max_amp_err: Real = 0.0
    var w1 = w0
    for half in range(2):
        for _ in range(STEPS_PER_HALF):
            stepper.step(ctx, q_phase, q_maxwell, dt, compute_joint_rhs)
            t += dt
        ctx.synchronize()
        w1 = compute_total_energy(
            ctx, phase_mesh, x_mesh, q_phase, q_maxwell,
        )
        var amp_obs = sqrt(w1[0] / w0[0])
        var amp_lin = e_lin[(half + 1) * (N_LIN // 2)]
        var amp_err = abs(amp_obs - amp_lin)
        if amp_err > max_amp_err:
            max_amp_err = amp_err
        print(
            "  t =", t, " E/E0 observed =", amp_obs,
            " linear theory =", amp_lin, " diff =", amp_obs - amp_lin,
        )
    var m1 = compute_total_mass(ctx, phase_mesh, q_phase)

    var dW = (w1[0] + w1[1]) - (w0[0] + w0[1])
    var dW_field_rel = abs(dW) / w0[0]
    var adM = abs(m1 - m0)
    print(
        "Vlasov-Maxwell Langmuir: t =", t,
        " W_E =", w1[0], " ΔW_K =", w1[1] - w0[1],
        " ΔW / W_E(0) =", dW / w0[0],
        " ΔM =", m1 - m0,
    )

    # (a) Field amplitude follows linear theory. Measured differences:
    # 1.6e-5 at t = 0.5, 2.3e-4 at t = 1 (velocity-grid resolution of
    # the phase-mixed perturbation). With the Lorentz force or the
    # current source removed E/E0 stays at 1 (0.09 off).
    assert_true(
        max_amp_err < Real(1.0e-3),
        "E_x amplitude departs from the linear Vlasov response",
    )
    # (b) Energy conservation, relative to the field energy. Measured
    # 1.3e-5: the 4.4e-6 lost by the field reappears as kinetic energy
    # to within 3e-10.
    assert_true(
        dW_field_rel < Real(1.0e-4),
        "energy not conserved on the field-energy scale",
    )
    # (c) Mass conservation. Measured 7e-11 (copyout flux of the
    # Maxwellian tail at |v| = 3).
    assert_true(adM < Real(5.0e-10), "mass not conserved")

    print("PASSED")
