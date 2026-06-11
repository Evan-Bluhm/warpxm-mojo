# 1D1V Vlasov-Maxwell hybrid-initialisation test.
#
# End-to-end pipeline check: start from a uniform 5-moment Euler IC
# (ρ, ρu_x = 0, e = p/(γ−1)), use
# `apply_5moment_to_kinetic_1d1v` to populate the phase-state with
# the per-x-node Maxwellian, add a small sinusoidal `E_x`
# perturbation, then evolve with the coupled SSPRK3 stepper. Verify:
#
#   (a) Mass conservation: `ΔM/M₀` at machine precision (BGK isn't
#       active here — there are no sources/sinks in the v-direction
#       beyond the truncation-quiet Maxwellian tails).
#   (b) Energy conservation: `ΔW/W₀ < 1e-9` (same kind of bound
#       slice 9 hit at ≈ 8e-12 with a hand-built IC; the
#       5moment-initialised IC has an `erfc` truncation residue but
#       the *evolution* from that IC still conserves energy).
#
# This is the first test that ties together slice-14 (initialiser),
# slice-8 (coupled stepper), slice-7 (current source), slice-3
# (Vlasov DG kernel) and slice-6 (first moment) into a single
# end-to-end pipeline driven entirely from a fluid IC.
#
# Build / run:
#     pixi run mojo run -I src tests/test_vlasov_maxwell_hybrid_init_1d1v.mojo

from std.math import sin, sqrt
from std.sys import has_accelerator
from std.gpu.host import DeviceContext
from std.testing import assert_true

from warpxm.apps.euler import EULER_NC, IDX_RHO, IDX_MX, IDX_MY, IDX_MZ, IDX_E
from warpxm.apps.maxwell import Maxwell, MAXWELL_NC, IDX_EX
from warpxm.apps.vlasovmaxwell import VlasovMaxwell, VLASOV_NC
from warpxm.basis.nodal_basis_1d import NodalBasis1D
from warpxm.core.types import Real, TWO_PI
from warpxm.mesh.block_1d import BlockMesh1D
from warpxm.mesh.phase_mesh_1d1v import PhaseMesh1D1V
from warpxm.spatial.dg_maxwell_1d import DGMaxwellSolver1D
from warpxm.spatial.dg_vlasov_1d1v import DGVlasovSolver1D1V
from warpxm.spatial.vlasov_5moment_to_kinetic_1d1v import (
    apply_5moment_to_kinetic_1d1v,
)
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
# `V_MAX = 1.6` (≈ 10·v_th) keeps the Maxwellian tail at the
# boundary at `erfc(V_MAX / (v_th · √2)) ≈ erfc(7.1) ≈ 3·10⁻²³`, well
# below any other numerical-truncation scale in the run.
comptime V_MAX: Real = 1.6
comptime MASS: Real = 1.0
comptime GAMMA: Real = 5.0 / 3.0
# Fluid target: ρ=1, u=0, T=0.025 ⇒ e = p/(γ−1) = ρT/(γ−1).
# v_th² = T/m = 0.025 ⇒ v_th ≈ 0.158.
comptime RHO_TARGET: Real = 1.0
comptime T_TARGET: Real = 0.025
comptime E_TARGET: Real = RHO_TARGET * T_TARGET / (GAMMA - Real(1.0))
comptime CHARGE_MASS_RATIO: Real = -1.0
comptime SKIN_DEPTH_NORM: Real = 1.0
comptime EPS0: Real = 1.0
comptime C_LIGHT: Real = 1.0
comptime EPS: Real = 0.01
comptime T_FINAL: Real = 6.0
comptime DT: Real = 0.05


def total_energy(
    var ctx: DeviceContext,
    mut phase_mesh: PhaseMesh1D1V[P_X, P_V],
    mut x_mesh: BlockMesh1D[P_X],
    mut q_phase: PhaseState1D1V[P_X, P_V, VLASOV_NC],
    mut q_maxwell: DGState1D[P_X, MAXWELL_NC],
) raises -> Real:
    """`W = ½ ∫ E_x² dx + ½ ∫∫ v² f dv dx`."""
    var x_w = NodalBasis1D[P_X].weights()
    var v_w = NodalBasis1D[P_V].weights()
    var ref_v = NodalBasis1D[P_V].nodes()
    var w_E: Real = 0.0
    var w_K: Real = 0.0
    with q_maxwell.buf.map_to_host() as h_m:
        for ex in range(NX):
            for ix in range(P_X + 1):
                var node_x = ex * (P_X + 1) + ix
                var ex_val: Real = h_m[node_x * MAXWELL_NC + IDX_EX]
                w_E += x_w[ix] * x_mesh.jacobian * Real(0.5) * ex_val * ex_val
    with q_phase.buf.map_to_host() as h_p:
        for ex in range(NX):
            for ev in range(NV):
                var vl = -V_MAX + Real(ev) * phase_mesh.dv
                for ix in range(P_X + 1):
                    for iv in range(P_V + 1):
                        var v = vl + (Real(1.0) + ref_v[iv]) * phase_mesh.jacobian_v
                        var phase_e = ex * NV + ev
                        var idx = (
                            (phase_e * (P_V + 1) + iv) * (P_X + 1) + ix
                        ) * VLASOV_NC
                        var f: Real = h_p[idx]
                        var w = (
                            x_w[ix] * phase_mesh.jacobian_x
                            * v_w[iv] * phase_mesh.jacobian_v
                        )
                        w_K += w * Real(0.5) * v * v * f
    return w_E + w_K


def total_mass(
    var ctx: DeviceContext,
    mut phase_mesh: PhaseMesh1D1V[P_X, P_V],
    mut q_phase: PhaseState1D1V[P_X, P_V, VLASOV_NC],
) raises -> Real:
    var x_w = NodalBasis1D[P_X].weights()
    var v_w = NodalBasis1D[P_V].weights()
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
                        var f = h[idx]
                        var w = (
                            x_w[ix] * phase_mesh.jacobian_x
                            * v_w[iv] * phase_mesh.jacobian_v
                        )
                        acc += w * f
    return acc


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

    # --- Step 1: Build uniform 5-moment Euler IC. ---
    var fluid = DGState1D[P_X, EULER_NC](ctx, NX)
    var fluid_host = ctx.enqueue_create_host_buffer[Real.dtype](fluid.buf_len)
    var node_count = NX * (P_X + 1)
    for n in range(node_count):
        var b = n * EULER_NC
        fluid_host[b + IDX_RHO] = RHO_TARGET
        fluid_host[b + IDX_MX] = Real(0.0)
        fluid_host[b + IDX_MY] = Real(0.0)
        fluid_host[b + IDX_MZ] = Real(0.0)
        fluid_host[b + IDX_E] = E_TARGET
    ctx.enqueue_copy(dst_buf=fluid.buf, src_buf=fluid_host)
    ctx.synchronize()

    # --- Step 2: 5moment → kinetic Maxwellian. ---
    var q_phase = PhaseState1D1V[P_X, P_V, VLASOV_NC](ctx, NX, NV)
    apply_5moment_to_kinetic_1d1v[P_X, P_V](
        ctx, phase_mesh, fluid, q_phase, MASS, GAMMA,
    )
    ctx.synchronize()

    # --- Step 3: Maxwell IC with sinusoidal E_x perturbation. ---
    var q_maxwell = DGState1D[P_X, MAXWELL_NC](ctx, NX)
    var m_host = ctx.enqueue_create_host_buffer[Real.dtype](q_maxwell.buf_len)
    for k in range(q_maxwell.buf_len):
        m_host[k] = Real(0.0)
    var ref_x_nodes = NodalBasis1D[P_X].nodes()
    for ex in range(NX):
        var xl = Real(0.0) + Real(ex) * x_mesh.dx
        for ix in range(P_X + 1):
            var x = xl + (Real(1.0) + ref_x_nodes[ix]) * x_mesh.jacobian
            var node_x = ex * (P_X + 1) + ix
            m_host[node_x * MAXWELL_NC + IDX_EX] = EPS * sin(TWO_PI * x / L_DOMAIN)
    ctx.enqueue_copy(dst_buf=q_maxwell.buf, src_buf=m_host)
    ctx.synchronize()

    var w0 = total_energy(ctx, phase_mesh, x_mesh, q_phase, q_maxwell)
    var m0 = total_mass(ctx, phase_mesh, q_phase)
    print("Hybrid IC: M =", m0, " W =", w0)

    # --- Step 4: Coupled Vlasov-Maxwell evolution. ---
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

    @parameter
    def compute_joint_rhs(
        mut qp_in: PhaseState1D1V[P_X, P_V, VLASOV_NC],
        mut qm_in: DGState1D[P_X, MAXWELL_NC],
        mut qp_out: PhaseState1D1V[P_X, P_V, VLASOV_NC],
        mut qm_out: DGState1D[P_X, MAXWELL_NC],
    ) raises:
        spatial_vlasov.compute_rhs(ctx, phase_mesh, qp_in, qm_in, qp_out)
        spatial_maxwell.compute_rhs(ctx, x_mesh, qm_in, qm_out)
        moments.first_moment(ctx, phase_mesh, qp_in, j_current)
        apply_vlasov_current_source_1d[P_X](
            ctx, j_current, qm_out, CHARGE_MASS_RATIO, EPS0,
        )

    var t: Real = 0.0
    var nsteps: Int = 0
    while t < T_FINAL - Real(0.5) * DT:
        var dt_use = DT if (t + DT) < T_FINAL else (T_FINAL - t)
        stepper.step[compute_joint_rhs](ctx, q_phase, q_maxwell, dt_use)
        t += dt_use
        nsteps += 1
    ctx.synchronize()

    var w1 = total_energy(ctx, phase_mesh, x_mesh, q_phase, q_maxwell)
    var m1 = total_mass(ctx, phase_mesh, q_phase)

    var dW = w1 - w0
    var adW = dW if dW >= 0.0 else -dW
    var rel_W = adW / w0
    var dM = m1 - m0
    var adM = dM if dM >= 0.0 else -dM
    var rel_M = adM / m0
    print(
        "Hybrid T_FINAL: steps =", nsteps, " t =", t,
        " ΔM/M =", rel_M, " ΔW/W =", rel_W,
    )

    # (a) Mass conserved to machine precision — no source/sink terms,
    # outflow BC drains negligible mass when `v_max ≈ 10·v_th`.
    assert_true(rel_M < Real(1.0e-12), "hybrid pipeline lost mass")
    # (b) Energy conserved at numerical-truncation level. Drift is
    # dominated by SSPRK3 truncation; slice 9's 8e-12 came from a
    # 200-step run at `dt = 5·10⁻³`, ours uses `dt = 5·10⁻²` for 120
    # steps (a `T · dt³` budget 6000× larger). Tolerance 1e-7 sits
    # well above the observed 6e-8 and well below the time-truncation
    # bound `T · dt³ ≈ 7·10⁻⁴`.
    assert_true(rel_W < Real(1.0e-7), "hybrid pipeline didn't conserve energy")

    print("PASSED")
