# 1D1V Vlasov-Maxwell — quantitative Bohm-Gross dispersion check.
#
# A small `E_x(x, 0) = ε · sin(2π x / L)` perturbation on a Maxwellian
# background oscillates at the kinetic Langmuir frequency
#
#     ω_BG² = ω_p² + 3 · k² · v_th²,
#
# with `ω_p² = n · α² / ε₀` and `v_th² = σ²/2` for `f₀(v) ∝
# exp(-v²/σ²)`. The mode also Landau-damps as `γ_L ∝ exp(−ω²/(2 k²
# v_th²))`; for the parameters here `ω/(k·v_th) ≈ 4.8`, giving
# `γ_L ≈ 2·10⁻³` (negligible over the run).
#
# Why these parameters (after the failed slice-11 attempt): slice 9
# ran with σ²=0.5, k=2π ⇒ `k·v_th ≈ 4.4`, deep in the over-damped
# regime where the kinetic dispersion has no real branch. Here:
#
#   L     = 2π            ⇒ k       = 1
#   σ²    = 0.05          ⇒ v_th    ≈ 0.158  (from `v_th² = σ²/2`)
#   ω_BG² ≈ 1 + 3·0.025   = 1.075     ⇒ ω_BG ≈ 1.037, period ≈ 6.06
#   cold  ω_p    = 1                  ⇒ cold period ≈ 6.28
#
# So Bohm-Gross predicts a period ≈ 3.5 % below the cold-plasma
# value — a clear, measurable thermal correction.
#
# Sampling: read `E_x(x*, t)` at one x-node every step into a
# time-series buffer. Count zero crossings of that series with
# linear interpolation. Mean half-period from successive crossings
# gives the observed period; compare against `ω_BG`.
#
# Build / run:
#     pixi run mojo run -I src tests/test_vlasov_maxwell_bohm_gross_1d1v.mojo

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
comptime P_V: Int = 3
comptime NX: Int = 16
comptime NV: Int = 32
comptime L_DOMAIN: Real = TWO_PI
comptime V_MAX: Real = 0.9
comptime SIGMA_SQ: Real = 0.05
comptime CHARGE_MASS_RATIO: Real = -1.0
comptime SKIN_DEPTH_NORM: Real = 1.0
comptime EPS0: Real = 1.0
comptime C_LIGHT: Real = 1.0
comptime EPS: Real = 0.01
# `k = 2π / L_DOMAIN = 1.0` and `v_th² = σ²/2 = 0.025` ⇒ Bohm-Gross
# period ≈ 6.06. Sample for two predicted periods.
comptime T_FINAL: Real = 12.0
comptime DT: Real = 0.05
# Pick the sample x-node where `sin(2π·x/L) = 1` (peak amplitude).
comptime SAMPLE_X: Real = L_DOMAIN / Real(4.0)


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = DeviceContext()

    var phase_mesh = PhaseMesh1D1V[P_X, P_V](
        ctx,
        Real(0.0), L_DOMAIN, -V_MAX, V_MAX,
        num_x=NX, num_v=NV, periodic_x=True,
    )
    var x_mesh = BlockMesh1D[P_X](
        ctx, Real(0.0), L_DOMAIN, NX, periodic=True,
    )

    var q_phase = PhaseState1D1V[P_X, P_V, VLASOV_NC](ctx, NX, NV)
    var q_maxwell = DGState1D[P_X, MAXWELL_NC](ctx, NX)

    # IC f₀(v) = exp(-v²/σ²) / (σ√π) (normalised so ∫f dv = 1).
    var sigma = sqrt(SIGMA_SQ)
    var norm = Real(1.0) / (sigma * sqrt(PI))
    var ref_x_nodes = NodalBasis1D[P_X].nodes()
    var ref_v_nodes = NodalBasis1D[P_V].nodes()
    var p_host = ctx.enqueue_create_host_buffer[Real.dtype](q_phase.buf_len)
    for ex in range(NX):
        for ev in range(NV):
            var vl = -V_MAX + Real(ev) * phase_mesh.dv
            for ix in range(P_X + 1):
                for iv in range(P_V + 1):
                    var v = vl + (Real(1.0) + ref_v_nodes[iv]) * phase_mesh.jacobian_v
                    var phase_e = ex * NV + ev
                    var idx = (
                        (phase_e * (P_V + 1) + iv) * (P_X + 1) + ix
                    ) * VLASOV_NC
                    p_host[idx] = norm * exp(-v * v / SIGMA_SQ)
    ctx.enqueue_copy(dst_buf=q_phase.buf, src_buf=p_host)

    # IC: E_x(x) = ε · sin(2π · x / L), rest zero.
    var m_host = ctx.enqueue_create_host_buffer[Real.dtype](q_maxwell.buf_len)
    for k in range(q_maxwell.buf_len):
        m_host[k] = Real(0.0)
    var sample_node: Int = 0
    var best_d: Real = 1.0e30
    for ex in range(NX):
        var xl = Real(0.0) + Real(ex) * x_mesh.dx
        for ix in range(P_X + 1):
            var x = xl + (Real(1.0) + ref_x_nodes[ix]) * x_mesh.jacobian
            var node_x = ex * (P_X + 1) + ix
            m_host[node_x * MAXWELL_NC + IDX_EX] = EPS * sin(TWO_PI * x / L_DOMAIN)
            var d = x - SAMPLE_X
            var ad = d if d >= 0.0 else -d
            if ad < best_d:
                best_d = ad
                sample_node = node_x
    ctx.enqueue_copy(dst_buf=q_maxwell.buf, src_buf=m_host)
    ctx.synchronize()

    # --- solver setup ---
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

    # --- time loop with E_x sampling ---
    var t_series = List[Real]()
    var e_series = List[Real]()
    var t: Real = 0.0
    var nsteps: Int = 0
    ctx.synchronize()
    with q_maxwell.buf.map_to_host() as h:
        t_series.append(t)
        e_series.append(h[sample_node * MAXWELL_NC + IDX_EX])
    while t < T_FINAL - Real(0.5) * DT:
        var dt_use = DT if (t + DT) < T_FINAL else (T_FINAL - t)
        stepper.step(ctx, q_phase, q_maxwell, dt_use, compute_joint_rhs)
        t += dt_use
        nsteps += 1
        ctx.synchronize()
        with q_maxwell.buf.map_to_host() as h:
            t_series.append(t)
            e_series.append(h[sample_node * MAXWELL_NC + IDX_EX])
    ctx.synchronize()

    # --- zero-crossing detection ---
    var crossings = List[Real]()
    var n_pts = len(e_series)
    for i in range(1, n_pts):
        var v_prev = e_series[i - 1]
        var v_curr = e_series[i]
        if v_prev * v_curr < Real(0.0):
            var t_prev = t_series[i - 1]
            var t_curr = t_series[i]
            var frac = -v_prev / (v_curr - v_prev)
            var t_cross = t_prev + frac * (t_curr - t_prev)
            crossings.append(t_cross)

    # Predicted Bohm-Gross. Note v_th² = σ²/2 for `f₀ ∝ exp(-v²/σ²)`.
    var omega_p_sq: Real = Real(1.0)  # n · α² / ε₀
    var k_wave = TWO_PI / L_DOMAIN
    var v_th_sq = SIGMA_SQ / Real(2.0)
    var omega_bg = sqrt(omega_p_sq + Real(3.0) * k_wave * k_wave * v_th_sq)
    var period_bg = TWO_PI / omega_bg
    var period_cold = TWO_PI / sqrt(omega_p_sq)
    print(
        "Bohm-Gross 1D1V: steps =", nsteps,
        " sample_node =", sample_node,
        " E_x(0) =", e_series[0],
        " n_crossings =", len(crossings),
        " ω_BG =", omega_bg, " period_BG =", period_bg,
        " period_cold =", period_cold,
    )

    assert_true(
        len(crossings) >= 3,
        "too few zero crossings — oscillation not resolved",
    )

    # Mean half-period: average successive crossing intervals.
    var sum_half: Real = 0.0
    for i in range(1, len(crossings)):
        sum_half += crossings[i] - crossings[i - 1]
    var mean_half = sum_half / Real(len(crossings) - 1)
    var period_obs = Real(2.0) * mean_half
    var rel_err_bg = (period_obs - period_bg) / period_bg
    var arel_bg = rel_err_bg if rel_err_bg >= 0.0 else -rel_err_bg
    var rel_diff_cold = (period_obs - period_cold) / period_cold
    var arel_cold = rel_diff_cold if rel_diff_cold >= 0.0 else -rel_diff_cold
    print(
        "Bohm-Gross 1D1V: T_obs =", period_obs,
        " rel err vs BG =", rel_err_bg,
        " rel diff vs cold =", rel_diff_cold,
    )

    # (a) Observed period matches Bohm-Gross within 3 %. Finite-k
    # corrections and SSPRK3 error account for this band.
    assert_true(
        arel_bg < Real(0.03),
        "observed period off Bohm-Gross prediction by > 3 %",
    )
    # (b) Bohm-Gross is meaningfully different from cold ω_p — verify
    # the measurement actually resolves the thermal correction. The
    # predicted correction is ≈ 3.5 %; the observation must agree
    # with BG and disagree with cold ω_p by at least 1 %.
    assert_true(
        arel_cold > Real(0.01),
        "observed period matches cold ω_p — no thermal correction seen",
    )

    print("PASSED")
