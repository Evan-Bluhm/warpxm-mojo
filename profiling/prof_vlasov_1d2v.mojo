# Steady-state stepping driver for nsys / ncu profiling of the 1D2V
# Vlasov-Maxwell hot path (kinetic RHS, velocity moments, current source,
# Maxwell RHS, SSPRK3 combines). Same setup as
# `tests/test_vlasov_maxwell_weibel_growth_1d2v.mojo` on a mesh large
# enough to fill the GPU: 64 x 32 x 32 phase-space elements at P = 2
# (1.77M nodes). Prints steps/s over the measured window.
#
#     pixi run mojo run -I src profiling/prof_vlasov_1d2v.mojo

from std.time import perf_counter_ns
from std.collections import List
from std.math import exp, sin, sqrt, log
from std.sys import has_accelerator
from max.gpu.host import DeviceContext
from std.testing import assert_true

from warpxm.apps.maxwell import (
    Maxwell, MAXWELL_NC, IDX_BZ,
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
comptime NX: Int = 64
comptime NV1: Int = 32
comptime NV2: Int = 32
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
comptime N_WARMUP: Int = 5
comptime N_STEPS: Int = 30
comptime DT: Real = 5.0e-3
comptime DELTA: Real = 1.0e-6       # B_z seed amplitude
comptime SAMPLE_EVERY: Int = 4      # sample stride for the B_z amplitude
comptime FIT_T_MIN: Real = 10.0     # linear-fit window
comptime FIT_T_MAX: Real = 20.0


def compute_total_mass(
    var ctx: DeviceContext,
    mut phase_mesh: PhaseMesh1D2V[P_X, P_V1, P_V2],
    mut q_phase: PhaseState1D2V[P_X, P_V1, P_V2, VLASOV_NC],
) raises -> Real:
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
                                var w = (
                                    x_weights[ix] * phase_mesh.jacobian_x
                                    * v1_weights[iv1] * phase_mesh.jacobian_v1
                                    * v2_weights[iv2] * phase_mesh.jacobian_v2
                                )
                                acc += w * h[idx]
    return acc


def bz_sin_amplitude(
    var ctx: DeviceContext,
    mut x_mesh: BlockMesh1D[P_X],
    mut q_maxwell: DGState1D[P_X, MAXWELL_NC],
) raises -> Real:
    """Amplitude of the sin(kx) Fourier mode of B_z: (2/L) ∫ B_z sin(kx) dx."""
    var x_weights = NodalBasis1D[P_X].weights()
    var x_nodes = NodalBasis1D[P_X].nodes()
    var k_wave = TWO_PI / (X_MAX - X_MIN)
    var acc: Real = 0.0
    with q_maxwell.buf.map_to_host() as h:
        for ex in range(NX):
            var xl = X_MIN + Real(ex) * x_mesh.dx
            for ix in range(P_X + 1):
                var x = xl + (Real(1.0) + x_nodes[ix]) * x_mesh.jacobian
                var node_x = ex * (P_X + 1) + ix
                var bz: Real = h[node_x * MAXWELL_NC + IDX_BZ]
                acc += x_weights[ix] * x_mesh.jacobian * bz * sin(k_wave * x)
    return Real(2.0) * acc / (X_MAX - X_MIN)


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = DeviceContext()

    var phase_mesh = PhaseMesh1D2V[P_X, P_V1, P_V2](
        ctx,
        X_MIN, X_MAX,
        -V1_MAX, V1_MAX,
        -V2_MAX, V2_MAX,
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

    var k_wave = TWO_PI / (X_MAX - X_MIN)
    # IC: f = f₀ = norm · exp(-v1²/σ_par² - v2²/σ_perp²), n = 1 ⇒
    # norm = 1/(π σ_par σ_perp). No δf: the B_z seed alone projects onto
    # the growing mode.
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
            var v1l = -V1_MAX + Real(ev1) * phase_mesh.dv1
            for ev2 in range(NV2):
                var v2l = -V2_MAX + Real(ev2) * phase_mesh.dv2
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

    # IC: B_z(x, 0) = δ sin(kx), all other field components zero.
    var m_host = ctx.enqueue_create_host_buffer[Real.dtype](q_maxwell.buf_len)
    for kk in range(q_maxwell.buf_len):
        m_host[kk] = Real(0.0)
    for ex in range(NX):
        var xl = X_MIN + Real(ex) * x_mesh.dx
        for ix in range(P_X + 1):
            var x = xl + (Real(1.0) + ref_x_nodes[ix]) * x_mesh.jacobian
            var node_x = ex * (P_X + 1) + ix
            m_host[node_x * MAXWELL_NC + IDX_BZ] = DELTA * sin(k_wave * x)
    ctx.enqueue_copy(dst_buf=q_maxwell.buf, src_buf=m_host)
    ctx.synchronize()

    # Initial diagnostics.
    var m0 = compute_total_mass(ctx, phase_mesh, q_phase)
    var bz0 = bz_sin_amplitude(ctx, x_mesh, q_maxwell)
    print("Weibel IC: M =", m0, " B_z sin(kx) amplitude =", bz0)

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

    def compute_joint_rhs(
        mut qp_in: PhaseState1D2V[P_X, P_V1, P_V2, VLASOV_NC],
        mut qm_in: DGState1D[P_X, MAXWELL_NC],
        mut qp_out: PhaseState1D2V[P_X, P_V1, P_V2, VLASOV_NC],
        mut qm_out: DGState1D[P_X, MAXWELL_NC],
    ) raises {mut jx_current, mut jy_current, imm}:
        spatial_vlasov.compute_rhs(ctx, phase_mesh, qp_in, qm_in, qp_out)
        spatial_maxwell.compute_rhs(ctx, x_mesh, qm_in, qm_out)
        moments.first_moment_v1(ctx, phase_mesh, qp_in, jx_current)
        moments.first_moment_v2(ctx, phase_mesh, qp_in, jy_current)
        apply_vlasov_current_source_1d2v[P_X](
            ctx, jx_current, jy_current, qm_out,
            CHARGE_MASS_RATIO, EPS0,
        )

    for _ in range(N_WARMUP):
        stepper.step(ctx, q_phase, q_maxwell, DT, compute_joint_rhs)
    ctx.synchronize()
    var t0 = perf_counter_ns()
    for _ in range(N_STEPS):
        stepper.step(ctx, q_phase, q_maxwell, DT, compute_joint_rhs)
    ctx.synchronize()
    var elapsed_s = Real(perf_counter_ns() - t0) * Real(1.0e-9)
    var m1 = compute_total_mass(ctx, phase_mesh, q_phase)
    print(
        "BENCH config=vlasov_1d2v steps=" + String(N_STEPS)
        + " elapsed_s=" + String(elapsed_s)
        + " steps_per_s=" + String(Real(N_STEPS) / elapsed_s)
    )
    print("mass drift:", m1 - m0)
