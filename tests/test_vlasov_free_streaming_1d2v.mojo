# 1D2V Vlasov free-streaming smoke test.
#
# With zero electromagnetic field (`E = B = 0` ⇒ `a_v1 = a_v2 = 0`), the
# 1D2V Vlasov equation reduces to
#     ∂f/∂t + v1 · ∂f/∂x = 0,
# which advects every `(v1, v2)` slice independently along x at the slice's
# `v1` (the `v2` axis is a passive tag — nothing flows along it because
# `a_v2 = 0` is the only mechanism that does):
#     f(x, v1, v2, t) = f₀(x − v1 · t, v1, v2).
#
# This test:
#   * sets `f₀(x, v1, v2) = sin(2π x) · exp(-(v1² + v2²) / σ²)` on an
#     x-periodic 1D2V phase mesh,
#   * integrates with forward Euler steps (a slice-19-local choice;
#     `SSPRK3_1D2V` arrives in slice 20),
#   * compares the numerical solution to the analytic shifted initial
#     condition in L2 over the phase domain.
#
# This is a smoke test for the new spatial kernel only — the tolerance is
# loose so a single pass exercises every code path (volume sweeps along
# all three axes, all six surface branches via periodic x + outflow on
# v1/v2 where `f` is already exponentially small).
#
# Build / run:
#     pixi run mojo run -I src tests/test_vlasov_free_streaming_1d2v.mojo

from std.math import sin, exp, ceildiv
from std.sys import has_accelerator
from std.gpu.host import DeviceContext
from std.testing import assert_true
from layout import TileTensor, Idx, row_major

from warpxm.apps.vlasovmaxwell import (
    VlasovMaxwell, VLASOV_NC, VLASOV_AUX_NC,
)
from warpxm.basis.nodal_basis_1d import NodalBasis1D
from warpxm.core.types import Real, TWO_PI
from warpxm.mesh.phase_mesh_1d2v import PhaseMesh1D2V
from warpxm.spatial.dg_vlasov_1d2v import DGVlasovSolver1D2V
from warpxm.temporal.ssprk3 import _euler_step_kernel
from warpxm.variable.phase_state_1d2v import PhaseState1D2V
from warpxm.variable.state_1d import DGState1D


comptime P_X: Int = 2
comptime P_V1: Int = 2
comptime P_V2: Int = 2
comptime NX: Int = 16
comptime NV1: Int = 4
comptime NV2: Int = 4
comptime X_MIN: Real = 0.0
comptime X_MAX: Real = 1.0
comptime V1_MIN: Real = -3.0
comptime V1_MAX: Real = 3.0
comptime V2_MIN: Real = -3.0
comptime V2_MAX: Real = 3.0
comptime SIGMA_SQ: Real = 0.5
comptime T_FINAL: Real = 0.1
comptime DT: Real = 5.0e-4


@always_inline
def f_initial(x: Real, v1: Real, v2: Real) -> Real:
    return sin(TWO_PI * x) * exp(-(v1 * v1 + v2 * v2) / SIGMA_SQ)


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"

    var ctx = DeviceContext()
    var mesh = PhaseMesh1D2V[P_X, P_V1, P_V2](
        ctx,
        X_MIN, X_MAX,
        V1_MIN, V1_MAX,
        V2_MIN, V2_MAX,
        num_x=NX, num_v1=NV1, num_v2=NV2,
        periodic_x=True,
    )

    var q = PhaseState1D2V[P_X, P_V1, P_V2, VLASOV_NC](ctx, NX, NV1, NV2)
    var q_tmp = PhaseState1D2V[P_X, P_V1, P_V2, VLASOV_NC](ctx, NX, NV1, NV2)
    var rhs = PhaseState1D2V[P_X, P_V1, P_V2, VLASOV_NC](ctx, NX, NV1, NV2)

    # --- IC ---
    var ref_x_nodes = NodalBasis1D[P_X].nodes()
    var ref_v1_nodes = NodalBasis1D[P_V1].nodes()
    var ref_v2_nodes = NodalBasis1D[P_V2].nodes()
    var host = ctx.enqueue_create_host_buffer[Real.dtype](q.buf_len)
    var nodes_per_phase = (P_X + 1) * (P_V1 + 1) * (P_V2 + 1)
    for ex in range(NX):
        var xl = X_MIN + Real(ex) * mesh.dx
        for ev1 in range(NV1):
            var v1l = V1_MIN + Real(ev1) * mesh.dv1
            for ev2 in range(NV2):
                var v2l = V2_MIN + Real(ev2) * mesh.dv2
                var phase_e = (ex * NV1 + ev1) * NV2 + ev2
                for iv2 in range(P_V2 + 1):
                    var v2 = (
                        v2l + (Real(1.0) + ref_v2_nodes[iv2]) * mesh.jacobian_v2
                    )
                    for iv1 in range(P_V1 + 1):
                        var v1 = (
                            v1l
                            + (Real(1.0) + ref_v1_nodes[iv1]) * mesh.jacobian_v1
                        )
                        for ix in range(P_X + 1):
                            var x = (
                                xl
                                + (Real(1.0) + ref_x_nodes[ix]) * mesh.jacobian_x
                            )
                            var local_n = (
                                (iv2 * (P_V1 + 1) + iv1) * (P_X + 1) + ix
                            )
                            var idx = (
                                phase_e * nodes_per_phase + local_n
                            ) * VLASOV_NC
                            host[idx] = f_initial(x, v1, v2)
    ctx.enqueue_copy(dst_buf=q.buf, src_buf=host)
    ctx.synchronize()

    # --- solver setup ---
    var app = VlasovMaxwell(
        charge_mass_ratio=Real(-1.0),
        skin_depth_norm=Real(1.0),
        cfl_max=Real(0.5),
    )
    var spatial = DGVlasovSolver1D2V[P_X, P_V1, P_V2](app)

    # Zero Maxwell aux ⇒ av1 = av2 = 0 ⇒ pure free streaming along x.
    var aux = DGState1D[P_X, VLASOV_AUX_NC](ctx, NX)

    # --- time loop: forward Euler with the flat combine kernel ---
    var size = q.buf_len
    comptime BLOCK = 256
    var grid = ceildiv(size, BLOCK)
    var layout = row_major(Idx(size))

    var t: Real = 0.0
    var nsteps: Int = 0
    while t < T_FINAL - Real(0.5) * DT:
        var dt_use = DT if (t + DT) < T_FINAL else (T_FINAL - t)
        spatial.compute_rhs(ctx, mesh, q, aux, rhs)
        comptime k_euler = _euler_step_kernel[
            type_of(layout), type_of(layout), type_of(layout),
        ]
        ctx.enqueue_function[k_euler](
            TileTensor(q.buf, layout),
            TileTensor(rhs.buf, layout),
            TileTensor(q_tmp.buf, layout),
            dt_use, size, grid_dim=grid, block_dim=BLOCK,
        )
        ctx.enqueue_copy(dst_buf=q.buf, src_buf=q_tmp.buf)
        t += dt_use
        nsteps += 1
    ctx.synchronize()

    # --- error analysis: compare against f(x, v1, v2, T) = f₀(x − v1·T, v1, v2) ---
    var x_weights = NodalBasis1D[P_X].weights()
    var v1_weights = NodalBasis1D[P_V1].weights()
    var v2_weights = NodalBasis1D[P_V2].weights()
    var acc: Real = 0.0
    var max_abs_err: Real = 0.0
    with q.buf.map_to_host() as h:
        for ex in range(NX):
            var xl = X_MIN + Real(ex) * mesh.dx
            for ev1 in range(NV1):
                var v1l = V1_MIN + Real(ev1) * mesh.dv1
                for ev2 in range(NV2):
                    var v2l = V2_MIN + Real(ev2) * mesh.dv2
                    var phase_e = (ex * NV1 + ev1) * NV2 + ev2
                    for iv2 in range(P_V2 + 1):
                        var v2 = (
                            v2l
                            + (Real(1.0) + ref_v2_nodes[iv2]) * mesh.jacobian_v2
                        )
                        for iv1 in range(P_V1 + 1):
                            var v1 = (
                                v1l
                                + (Real(1.0) + ref_v1_nodes[iv1])
                                  * mesh.jacobian_v1
                            )
                            for ix in range(P_X + 1):
                                var x = (
                                    xl
                                    + (Real(1.0) + ref_x_nodes[ix])
                                      * mesh.jacobian_x
                                )
                                var local_n = (
                                    (iv2 * (P_V1 + 1) + iv1) * (P_X + 1) + ix
                                )
                                var idx = (
                                    phase_e * nodes_per_phase + local_n
                                ) * VLASOV_NC
                                # Wrap x − v1·T back into [X_MIN, X_MAX).
                                var period = X_MAX - X_MIN
                                var x_wrap = x - v1 * T_FINAL
                                while x_wrap < X_MIN:
                                    x_wrap += period
                                while x_wrap >= X_MAX:
                                    x_wrap -= period
                                var f_exact = f_initial(x_wrap, v1, v2)
                                var d = h[idx] - f_exact
                                var ad = d if d >= 0.0 else -d
                                if ad > max_abs_err:
                                    max_abs_err = ad
                                var w = (
                                    x_weights[ix] * mesh.jacobian_x
                                    * v1_weights[iv1] * mesh.jacobian_v1
                                    * v2_weights[iv2] * mesh.jacobian_v2
                                )
                                acc += w * d * d
    var l2 = acc ** Real(0.5)

    print(
        "free-streaming 1D2V: steps =", nsteps,
        "t_final =", t,
        "L2 err =", l2,
        "max |err| =", max_abs_err,
    )

    # Smoke-test tolerance. DG truncation error for P=2 with NX=16 is
    # h^(P+1) ≈ 2.5e-4 on the x-axis alone; forward Euler adds O(dt·T)
    # ≈ 1e-4 per step over T=0.1 / dt=5e-4 = 200 steps. The v-axis
    # contribution is bounded by the Gaussian's amplitude (≈ 1 near the
    # origin) and shouldn't change since `f` is exactly preserved
    # along v under free streaming.
    assert_true(l2 < Real(5.0e-2), "free-streaming 1D2V L2 error too high")
    print("PASSED")
