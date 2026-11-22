# Coupled Vlasov-Maxwell 1D1V — uniform plasma oscillator smoke test.
#
# A uniform-in-x Maxwellian distribution and a uniform-in-x initial
# `E_x` perturbation cannot develop any x-dependence — there are no
# spatial gradients to drive the spatial DG terms. The dynamics
# reduces to a closed ODE in the bulk fluid moments:
#
#     du/dt = α · E
#     dE/dt = -α · n · u / ε₀
#
# With `α = q/m = -1`, `n = ∫ f dv = 1` (we use a *normalised*
# Maxwellian so n=1), and `ε₀ = 1`, this is `u'' = -u`. Period `2π`.
#
# Exact solution for `E(0) = E0`, `u(0) = 0`:
#
#     E(t) = E0 · cos t,        J(t) = n·u(t) = -E0 · sin t.
#
# We integrate one full period with the coupled SSPRK3 stepper and
# compare `E_x` and the species current `J = ∫ v f dv` with that
# solution at every quarter period (t = π/2, π, 3π/2, 2π). Checking
# only the end of the period is not enough: with either half of the
# coupling removed `E_x` simply stays at `E0` and "returns" trivially.
# The quarter-period samples pin the frequency (E must vanish at π/2
# and 3π/2), the amplitude and sign of the particle response (J = ∓E0
# there), and the sign of the field response (E = -E0 at π).
#
# Also checked: `E_x` stays uniform in x to machine precision — no
# spurious spatial structure leaks in from the kernel.
#
# This is the smallest non-trivial test of the coupled stepper. A
# more ambitious Langmuir-oscillation test with a spatially varying
# perturbation (and Landau damping) is reserved for a later slice.
#
# Build / run:
#     pixi run mojo run -I src tests/test_vlasov_maxwell_uniform_oscillator_1d1v.mojo

from std.math import exp, sqrt, pi
from std.sys import has_accelerator
from max.gpu.host import DeviceContext
from std.testing import assert_true

from warpxm.apps.maxwell import Maxwell, MAXWELL_NC, IDX_EX
from warpxm.apps.vlasovmaxwell import (
    VlasovMaxwell, VLASOV_NC, VLASOV_AUX_NC,
)
from warpxm.basis.nodal_basis_1d import NodalBasis1D
from warpxm.core.bc import BC_COPYOUT
from warpxm.core.types import Real, PI
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
comptime NX: Int = 4
comptime NV: Int = 16
comptime SIGMA_SQ: Real = 0.5
comptime CHARGE_MASS_RATIO: Real = -1.0
comptime SKIN_DEPTH_NORM: Real = 1.0
comptime EPS0: Real = 1.0
comptime C_LIGHT: Real = 1.0
# n = ∫ exp(-v²/σ²)/(σ√π) dv ≈ 1 (with v_max=3, σ²=0.5 the truncation
# residue is ~ 1e-9). Plasma frequency in these normalised units is
# ω_p² = n · α² / ε₀ = 1, period = 2π.
comptime T_FINAL: Real = 2.0 * PI
# 80 steps per quarter period, so the samples land exactly on kπ/2.
comptime STEPS_PER_QUARTER: Int = 80
comptime DT: Real = T_FINAL / Real(4 * STEPS_PER_QUARTER)
comptime E0: Real = 0.01  # initial uniform field perturbation


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = DeviceContext()

    # --- meshes ---
    # The phase mesh's x-axis and the Maxwell mesh must align (same NX,
    # same domain, same periodicity) so the aux buffer indexes match.
    var phase_mesh = PhaseMesh1D1V[P_X, P_V](
        ctx,
        Real(0.0), Real(1.0), Real(-3.0), Real(3.0),
        num_x=NX, num_v=NV,
        periodic_x=True,
    )
    var x_mesh = BlockMesh1D[P_X](
        ctx, Real(0.0), Real(1.0), NX, periodic=True,
    )

    # --- state ---
    var q_phase = PhaseState1D1V[P_X, P_V, VLASOV_NC](ctx, NX, NV)
    var q_maxwell = DGState1D[P_X, MAXWELL_NC](ctx, NX)

    # IC: f₀(v) = exp(-v²/σ²) / (σ√π) — normalised so ∫ f dv ≈ 1.
    var sigma = sqrt(SIGMA_SQ)
    var norm = Real(1.0) / (sigma * sqrt(PI))
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

    # IC: uniform E_x = E0, all other Maxwell components zero.
    var m_host = ctx.enqueue_create_host_buffer[Real.dtype](q_maxwell.buf_len)
    for k in range(q_maxwell.buf_len):
        m_host[k] = Real(0.0)
    var node_count = NX * (P_X + 1)
    for n in range(node_count):
        m_host[n * MAXWELL_NC + IDX_EX] = E0
    ctx.enqueue_copy(dst_buf=q_maxwell.buf, src_buf=m_host)
    ctx.synchronize()

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

    # Scratch buffer for the species current at each RHS evaluation.
    var j_current = DGState1D[P_X, 1](ctx, NX)

    def compute_joint_rhs(
        mut qp_in: PhaseState1D1V[P_X, P_V, VLASOV_NC],
        mut qm_in: DGState1D[P_X, MAXWELL_NC],
        mut qp_out: PhaseState1D1V[P_X, P_V, VLASOV_NC],
        mut qm_out: DGState1D[P_X, MAXWELL_NC],
    ) raises {mut j_current, imm}:
        # Vlasov reads E_x from qm_in (the Maxwell aux at this stage).
        spatial_vlasov.compute_rhs(ctx, phase_mesh, qp_in, qm_in, qp_out)
        # Maxwell's spatial RHS (purely from its own state).
        spatial_maxwell.compute_rhs(ctx, x_mesh, qm_in, qm_out)
        # Species current density `J = ∫ v f dv`.
        moments.first_moment(ctx, phase_mesh, qp_in, j_current)
        # Add `-α/ε₀ · J` to `qm_out.E_x`. Closes the two-way coupling.
        apply_vlasov_current_source_1d[P_X](
            ctx, j_current, qm_out, CHARGE_MASS_RATIO, EPS0,
        )

    # --- time loop: one full plasma period, sampled every quarter ---
    # Exact (E/E0, J/E0) at t = π/2, π, 3π/2, 2π.
    var e_exact = [Real(0.0), Real(-1.0), Real(0.0), Real(1.0)]
    var j_exact = [Real(-1.0), Real(0.0), Real(1.0), Real(0.0)]
    var max_e_err: Real = 0.0
    var max_j_err: Real = 0.0
    var max_x_spread: Real = 0.0
    var t: Real = 0.0
    for quarter in range(4):
        for _ in range(STEPS_PER_QUARTER):
            stepper.step(ctx, q_phase, q_maxwell, DT, compute_joint_rhs)
            t += DT
        moments.first_moment(ctx, phase_mesh, q_phase, j_current)
        ctx.synchronize()
        var e_node0: Real
        with q_maxwell.buf.map_to_host() as h:
            e_node0 = h[IDX_EX]
            for n in range(node_count):
                var ex_val = h[n * MAXWELL_NC + IDX_EX]
                var e_err = abs(ex_val - E0 * e_exact[quarter])
                if e_err > max_e_err:
                    max_e_err = e_err
                var spread = abs(ex_val - e_node0)
                if spread > max_x_spread:
                    max_x_spread = spread
        var j_node0: Real
        with j_current.buf.map_to_host() as hj:
            j_node0 = hj[0]
            for n in range(node_count):
                var j_err = abs(hj[n] - E0 * j_exact[quarter])
                if j_err > max_j_err:
                    max_j_err = j_err
        print(
            "  t =", t, " E_x/E0 =", e_node0 / E0, " J/E0 =", j_node0 / E0,
        )

    print(
        "Vlasov-Maxwell oscillator 1D1V: max |E - E_exact| / E0 =",
        max_e_err / E0,
        " max |J - J_exact| / E0 =", max_j_err / E0,
        " x-spread =", max_x_spread,
    )

    # Measured: 2.0e-6 (E) and 1.5e-6 (J) relative to E0 — the SSPRK3
    # amplitude / phase error accumulated over one period.
    assert_true(
        max_e_err < Real(1.0e-5) * E0,
        "E_x does not follow E0·cos(ω_p t)",
    )
    assert_true(
        max_j_err < Real(1.0e-5) * E0,
        "species current does not follow -E0·sin(ω_p t)",
    )
    # E_x must stay uniform in x — the kernel correctly handles the
    # no-x-physics case. Machine precision.
    assert_true(
        max_x_spread < Real(1.0e-12),
        "E_x developed spurious x-structure",
    )

    print("PASSED")
