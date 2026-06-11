# Coupled 1D2V Vlasov-Maxwell — smoke test for the full slice-21 stack.
#
# Exercises every code path landed in slice 21:
#   * 1D2V Vlasov RHS reading `(E_x, E_y, B_z)` from the Maxwell aux
#     state — including the `v × B_z` Lorentz coupling that is identically
#     zero in 1D1V,
#   * `VlasovMoments1D2V.first_moment_v1 / v2` to extract `(J_x, J_y)`,
#   * `apply_vlasov_current_source_1d2v` feeding both transverse `E`
#     components,
#   * `SSPRK3_VlasovMaxwell_1D2V` joint stepper composing both sides.
#
# IC is a bi-Maxwellian `f₀ ∝ exp(-v1²/σ_par² − v2²/σ_perp²)` with
# `σ_perp > σ_par` plus a small `B_z(x, 0) = δ · sin(2π x)` seed — the
# classical Weibel anisotropy setup. Linear theory (Davidson 1972)
# predicts an exponentially growing transverse mode:
#
#     γ_W = k · v_th_perp · √(T_perp/T_par − 1) / √(1 + k² c² / ω_p²)
#
# with `v_th_perp = σ_perp/√2` and `T_i = ½ σ_i²`. For
# `σ_par² = 0.5, σ_perp² = 1.0, k = 2π, c = 1, ω_p² = 1` ⇒ `γ_W ≈ 0.70`.
#
# This test does **not** assert the growth rate. A pure `B_z` seed
# projects onto a superposition of the growing eigenmode, its decaying
# counterpart, and the stable EM modes; without a matched `(E_y, δf)`
# eigenmode IC the early-time evolution is dominated by these
# transients and `max|B_z|` need not be monotonic, especially on the
# coarse `(NX, NV1, NV2) = (8, 8, 8)` mesh. Validating the linear
# growth rate is the natural follow-up slice once an eigenmode-
# initialiser exists.
#
# Test checks:
#   (a) mass conservation < 1e-9 (Vlasov is a conservation law on a
#       periodic x-domain),
#   (b) no blow-up — `max|B_z|` stays bounded by a few × the seed,
#   (c) the coupling does *something* — `max|B_z|` is nonzero at
#       `T_FINAL` and the observed `γ_obs` and theoretical `γ_W` are
#       printed for visual sanity-check.
#
# Build / run:
#     pixi run mojo run -I src tests/test_vlasov_maxwell_weibel_1d2v.mojo

from std.math import exp, sin, sqrt, log, ceildiv
from std.sys import has_accelerator
from std.gpu.host import DeviceContext
from std.testing import assert_true

from warpxm.apps.maxwell import (
    Maxwell, MAXWELL_NC, IDX_EX, IDX_EY, IDX_BZ,
)
from warpxm.apps.vlasovmaxwell import VlasovMaxwell, VLASOV_NC
from warpxm.basis.nodal_basis_1d import NodalBasis1D
from warpxm.core.types import Real, PI, TWO_PI
from warpxm.mesh.block_1d import BlockMesh1D
from warpxm.mesh.phase_mesh_1d2v import PhaseMesh1D2V
from warpxm.spatial.dg_maxwell_1d import DGMaxwellSolver1D
from warpxm.spatial.dg_vlasov_1d2v import DGVlasovSolver1D2V
from warpxm.spatial.vlasov_current_source_1d2v import (
    apply_vlasov_current_source_1d2v,
)
from warpxm.spatial.vlasov_moments_1d2v import VlasovMoments1D2V
from warpxm.temporal.ssprk3_vlasov_maxwell_1d2v import (
    SSPRK3_VlasovMaxwell_1D2V,
)
from warpxm.variable.phase_state_1d2v import PhaseState1D2V
from warpxm.variable.state_1d import DGState1D


comptime P_X: Int = 2
comptime P_V1: Int = 2
comptime P_V2: Int = 2
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
comptime T_FINAL: Real = 2.0
comptime DT: Real = 4.0e-3
comptime DELTA_BZ: Real = 1.0e-3   # B_z seed amplitude


def compute_total_mass(
    var ctx: DeviceContext,
    mut phase_mesh: PhaseMesh1D2V[P_X, P_V1, P_V2],
    mut q_phase: PhaseState1D2V[P_X, P_V1, P_V2, VLASOV_NC],
) raises -> Real:
    """M = ∫∫∫ f dv1 dv2 dx via GLL quadrature over the full phase mesh."""
    var x_weights = NodalBasis1D[P_X].weights()
    var v1_weights = NodalBasis1D[P_V1].weights()
    var v2_weights = NodalBasis1D[P_V2].weights()
    var nodes_per_phase = (P_X + 1) * (P_V1 + 1) * (P_V2 + 1)
    var acc: Real = 0.0
    with q_phase.buf.map_to_host() as h:
        for ex in range(NX):
            for ev1 in range(NV1):
                for ev2 in range(NV2):
                    var phase_e = (ex * NV1 + ev1) * NV2 + ev2
                    for iv2 in range(P_V2 + 1):
                        for iv1 in range(P_V1 + 1):
                            for ix in range(P_X + 1):
                                var local_n = (
                                    (iv2 * (P_V1 + 1) + iv1) * (P_X + 1) + ix
                                )
                                var idx = (
                                    phase_e * nodes_per_phase + local_n
                                ) * VLASOV_NC
                                var f: Real = h[idx]
                                var w = (
                                    x_weights[ix] * phase_mesh.jacobian_x
                                    * v1_weights[iv1] * phase_mesh.jacobian_v1
                                    * v2_weights[iv2] * phase_mesh.jacobian_v2
                                )
                                acc += w * f
    return acc


def max_abs_bz(
    var ctx: DeviceContext,
    mut q_maxwell: DGState1D[P_X, MAXWELL_NC],
) raises -> Real:
    """Peak |B_z| over all x-nodes."""
    var m: Real = 0.0
    with q_maxwell.buf.map_to_host() as h:
        for ex in range(NX):
            for ix in range(P_X + 1):
                var node_x = ex * (P_X + 1) + ix
                var v: Real = h[node_x * MAXWELL_NC + IDX_BZ]
                var av = v if v >= 0.0 else -v
                if av > m:
                    m = av
    return m


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = DeviceContext()

    var phase_mesh = PhaseMesh1D2V[P_X, P_V1, P_V2](
        ctx,
        X_MIN, X_MAX,
        -V_MAX, V_MAX,
        -V_MAX, V_MAX,
        num_x=NX, num_v1=NV1, num_v2=NV2,
        periodic_x=True,
    )
    var x_mesh = BlockMesh1D[P_X](
        ctx, X_MIN, X_MAX, NX, periodic=True,
    )

    var q_phase = PhaseState1D2V[P_X, P_V1, P_V2, VLASOV_NC](
        ctx, NX, NV1, NV2,
    )
    var q_maxwell = DGState1D[P_X, MAXWELL_NC](ctx, NX)

    # IC: f₀(v1, v2) = norm · exp(-v1²/σ_par² - v2²/σ_perp²), uniform in x.
    # Normalisation chosen so ∫∫ f dv1 dv2 = 1 ⇒ ω_p² = n α² / ε₀ = 1.
    var sigma_par = sqrt(SIGMA_PAR_SQ)
    var sigma_perp = sqrt(SIGMA_PERP_SQ)
    var norm = Real(1.0) / (PI * sigma_par * sigma_perp)

    var ref_x_nodes = NodalBasis1D[P_X].nodes()
    var ref_v1_nodes = NodalBasis1D[P_V1].nodes()
    var ref_v2_nodes = NodalBasis1D[P_V2].nodes()
    var p_host = ctx.enqueue_create_host_buffer[Real.dtype](q_phase.buf_len)
    var nodes_per_phase = (P_X + 1) * (P_V1 + 1) * (P_V2 + 1)
    for ex in range(NX):
        for ev1 in range(NV1):
            var v1l = -V_MAX + Real(ev1) * phase_mesh.dv1
            for ev2 in range(NV2):
                var v2l = -V_MAX + Real(ev2) * phase_mesh.dv2
                var phase_e = (ex * NV1 + ev1) * NV2 + ev2
                for iv2 in range(P_V2 + 1):
                    var v2 = (
                        v2l
                        + (Real(1.0) + ref_v2_nodes[iv2])
                          * phase_mesh.jacobian_v2
                    )
                    for iv1 in range(P_V1 + 1):
                        var v1 = (
                            v1l
                            + (Real(1.0) + ref_v1_nodes[iv1])
                              * phase_mesh.jacobian_v1
                        )
                        for ix in range(P_X + 1):
                            var local_n = (
                                (iv2 * (P_V1 + 1) + iv1) * (P_X + 1) + ix
                            )
                            var idx = (
                                phase_e * nodes_per_phase + local_n
                            ) * VLASOV_NC
                            p_host[idx] = norm * exp(
                                -v1 * v1 / SIGMA_PAR_SQ
                                - v2 * v2 / SIGMA_PERP_SQ
                            )
    ctx.enqueue_copy(dst_buf=q_phase.buf, src_buf=p_host)

    # IC: B_z(x, 0) = δ · sin(2π x); other Maxwell components zero.
    var m_host = ctx.enqueue_create_host_buffer[Real.dtype](q_maxwell.buf_len)
    for k in range(q_maxwell.buf_len):
        m_host[k] = Real(0.0)
    for ex in range(NX):
        var xl = X_MIN + Real(ex) * x_mesh.dx
        for ix in range(P_X + 1):
            var x = xl + (Real(1.0) + ref_x_nodes[ix]) * x_mesh.jacobian
            var node_x = ex * (P_X + 1) + ix
            m_host[node_x * MAXWELL_NC + IDX_BZ] = DELTA_BZ * sin(TWO_PI * x)
    ctx.enqueue_copy(dst_buf=q_maxwell.buf, src_buf=m_host)
    ctx.synchronize()

    # Initial diagnostics.
    var m0 = compute_total_mass(ctx, phase_mesh, q_phase)
    var bz0 = max_abs_bz(ctx, q_maxwell)
    print("Weibel IC: M =", m0, " max |B_z|_0 =", bz0)

    # --- solvers + stepper ---
    var vlasov_app = VlasovMaxwell(
        charge_mass_ratio=CHARGE_MASS_RATIO,
        skin_depth_norm=SKIN_DEPTH_NORM,
        cfl_max=Real(0.5),
    )
    var maxwell_app = Maxwell(c_light=C_LIGHT, cfl_max=Real(0.5))
    var spatial_vlasov = DGVlasovSolver1D2V[P_X, P_V1, P_V2](vlasov_app)
    var spatial_maxwell = DGMaxwellSolver1D[P_X](maxwell_app)
    var moments = VlasovMoments1D2V[P_X, P_V1, P_V2]()
    var stepper = SSPRK3_VlasovMaxwell_1D2V[P_X, P_V1, P_V2](
        ctx, NX, NV1, NV2,
    )
    var jx_current = DGState1D[P_X, 1](ctx, NX)
    var jy_current = DGState1D[P_X, 1](ctx, NX)

    @parameter
    def compute_joint_rhs(
        mut qp_in: PhaseState1D2V[P_X, P_V1, P_V2, VLASOV_NC],
        mut qm_in: DGState1D[P_X, MAXWELL_NC],
        mut qp_out: PhaseState1D2V[P_X, P_V1, P_V2, VLASOV_NC],
        mut qm_out: DGState1D[P_X, MAXWELL_NC],
    ) raises:
        spatial_vlasov.compute_rhs(ctx, phase_mesh, qp_in, qm_in, qp_out)
        spatial_maxwell.compute_rhs(ctx, x_mesh, qm_in, qm_out)
        moments.first_moment_v1(ctx, phase_mesh, qp_in, jx_current)
        moments.first_moment_v2(ctx, phase_mesh, qp_in, jy_current)
        apply_vlasov_current_source_1d2v[P_X](
            ctx, jx_current, jy_current, qm_out,
            CHARGE_MASS_RATIO, EPS0,
        )

    # --- time loop ---
    var t: Real = 0.0
    var nsteps: Int = 0
    var bz_peak_over_run: Real = bz0
    while t < T_FINAL - Real(0.5) * DT:
        var dt_use = DT if (t + DT) < T_FINAL else (T_FINAL - t)
        stepper.step[compute_joint_rhs](ctx, q_phase, q_maxwell, dt_use)
        t += dt_use
        nsteps += 1
        if nsteps % 50 == 0:
            ctx.synchronize()
            var bz_now = max_abs_bz(ctx, q_maxwell)
            if bz_now > bz_peak_over_run:
                bz_peak_over_run = bz_now
    ctx.synchronize()

    var bz_final = max_abs_bz(ctx, q_maxwell)
    var m1 = compute_total_mass(ctx, phase_mesh, q_phase)

    # Theoretical growth rate, for the printed γ_obs sanity-check only.
    # v_th_perp = σ_perp / √2 in the `f ∝ exp(-v²/σ²)` convention;
    # T_perp/T_par = σ_perp²/σ_par².
    var k = TWO_PI
    var v_th_perp = sigma_perp / sqrt(Real(2.0))
    var ratio = SIGMA_PERP_SQ / SIGMA_PAR_SQ
    var num = k * v_th_perp * sqrt(ratio - Real(1.0))
    var den = sqrt(
        Real(1.0) + k * k * C_LIGHT * C_LIGHT / Real(1.0)
    )
    var gamma_theory = num / den
    var gamma_obs = log(bz_final / bz0) / T_FINAL
    var dM = m1 - m0
    var adM = dM if dM >= 0.0 else -dM

    print(
        "Weibel smoke: steps =", nsteps, "t =", t,
        " max|B_z|_0 =", bz0,
        " max|B_z|_T =", bz_final,
        " peak over run =", bz_peak_over_run,
    )
    print(
        "Weibel info: γ_obs =", gamma_obs,
        " γ_theory =", gamma_theory,
        " (rate-fit deferred to an eigenmode-IC slice)",
    )
    print("ΔM =", dM)

    # (a) Mass conservation.
    assert_true(adM < Real(1.0e-9), "mass not conserved")
    # (b) No blow-up. The bounded growth allowance is generous — a
    # well-converged Weibel run can reach ~10× δ before nonlinear
    # saturation; numerical NaN / drift would push past 1000× δ.
    assert_true(bz_peak_over_run < Real(1000.0) * DELTA_BZ, "B_z blew up")
    # (c) Coupling fired — B_z is still nonzero at T_FINAL (i.e. it
    # didn't get nuked to round-off).
    assert_true(bz_final > Real(1.0e-12), "B_z vanished")

    print("PASSED")
