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
# The physical mode is a Bohm-Gross / Langmuir wave with thermal
# correction `ω² = ω_p² + 3 k² v_th²`. With our parameters
# (ω_p = 1, k = 2π, v_th² = σ² = 0.5), `ω² ≈ 60.2`, period ≈ 0.81.
# Landau damping is non-trivial at this k·v_th (ω / (k·v_th) ≈ 1.75),
# so over the test horizon T = 1 we expect substantial damping of
# the amplitude.
#
# This slice's check is not a precise dispersion fit — that would
# need a longer run + FFT. Instead:
#
#   (a) Total energy `W = ½ ∫ E_x² dx + ½ ∫∫ v² f dv dx`
#       is conserved by the continuous Vlasov-Maxwell-current-source
#       system when `λ_skin = ε₀`. Numerically it should drift by
#       only DG / SSPRK3 truncation. Tolerance: 1e-2 relative.
#
#   (b) Total particle mass `M = ∫∫ f dv dx` is conserved to
#       round-off (the Vlasov solver is a conservation law and
#       periodic-x has no boundary fluxes; the v-boundary copyout
#       is benign because `f ≈ 0` at v = ±v_max).
#
#   (c) No NaNs / unbounded blow-up — the field amplitude stays
#       bounded by a few × the initial amplitude.
#
# Build / run:
#     pixi run mojo run -I src tests/test_vlasov_maxwell_langmuir_1d1v.mojo

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
comptime DT: Real = 5.0e-3
comptime EPS: Real = 0.01  # amplitude of the E_x perturbation


def compute_total_energy(
    var ctx: DeviceContext,
    mut phase_mesh: PhaseMesh1D1V[P_X, P_V],
    mut x_mesh: BlockMesh1D[P_X],
    mut q_phase: PhaseState1D1V[P_X, P_V, VLASOV_NC],
    mut q_maxwell: DGState1D[P_X, MAXWELL_NC],
) raises -> Real:
    """W = ½ ∫ E_x² dx + ½ ∫∫ v² f dv dx via GLL quadrature."""
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
    return w_E + w_K


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
    print("Vlasov-Maxwell Langmuir IC: W =", w0, " M =", m0)

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

    # --- time loop ---
    var t: Real = 0.0
    var nsteps: Int = 0
    while t < T_FINAL - Real(0.5) * DT:
        var dt_use = DT if (t + DT) < T_FINAL else (T_FINAL - t)
        stepper.step(ctx, q_phase, q_maxwell, dt_use, compute_joint_rhs)
        t += dt_use
        nsteps += 1
    ctx.synchronize()

    # Final diagnostics.
    var w1 = compute_total_energy(
        ctx, phase_mesh, x_mesh, q_phase, q_maxwell,
    )
    var m1 = compute_total_mass(ctx, phase_mesh, q_phase)

    # Peak |E_x| for the blow-up check.
    var max_abs_ex: Real = 0.0
    with q_maxwell.buf.map_to_host() as h:
        for ex in range(NX):
            for ix in range(P_X + 1):
                var node_x = ex * (P_X + 1) + ix
                var v = h[node_x * MAXWELL_NC + IDX_EX]
                var av = v if v >= 0.0 else -v
                if av > max_abs_ex:
                    max_abs_ex = av

    var dW_rel = (w1 - w0) / w0
    var adW_rel = dW_rel if dW_rel >= 0.0 else -dW_rel
    var dM = m1 - m0
    var adM = dM if dM >= 0.0 else -dM
    print(
        "Vlasov-Maxwell Langmuir T_FINAL: steps =", nsteps,
        "t =", t,
        "W =", w1, "(ΔW/W =", dW_rel, ")",
        "ΔM =", dM,
        "max |E_x| =", max_abs_ex,
    )

    # (a) Energy conservation. Continuous V-M-current-source conserves
    # energy when λ_skin = ε₀ (= 1 here). Numerical truncation drift
    # is bounded.
    assert_true(adW_rel < Real(1.0e-2), "energy not conserved")
    # (b) Mass conservation. Periodic-x has no boundary fluxes;
    # v-boundaries are physically inactive (f ≈ 0 at v = ±v_max).
    # Tolerance: copyout BC noise + GLL quadrature rounding ⇒ 1e-9.
    assert_true(adM < Real(1.0e-9), "mass not conserved")
    # (c) No blow-up.
    assert_true(max_abs_ex < Real(10.0) * EPS, "E_x blew up")

    print("PASSED")
