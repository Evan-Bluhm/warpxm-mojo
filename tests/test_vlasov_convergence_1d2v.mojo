# 1D2V Vlasov DG convergence regression — confirm rate ≈ (P + 1)
# on smooth free-streaming. Tensor-product extension of
# `test_vlasov_convergence_1d1v` to the slice-19 1D2V kernel
# (`spatial/dg_vlasov_1d2v.mojo`).
#
# Setup: zero Maxwell aux (`E = B = 0` ⇒ `a_v1 = a_v2 = 0`), so the
# 1D2V Vlasov equation reduces to `∂f/∂t + v1 ∂f/∂x = 0`. Each
# `(v1, v2)` slice advects along x at speed `v1`; `v2` is a passive
# tag. Analytic solution at `t = T`:
#     `f(x, v1, v2, T) = f₀(x − v1 T, v1, v2)`.
#
# IC: `f₀(x, v1, v2) = sin(2π x) · exp(-(v1² + v2²) / σ²)`. Smooth,
# bounded support in v to machine precision over the truncated v
# domain (`erfc(v_max / σ) ≈ 4·10⁻¹⁰` for `σ² = 0.5, v_max = 3`).
#
# Sweep: for `P_X ∈ {1, 2, 3}` and fixed `NV1 = NV2 = 4`, `P_V1 = P_V2
# = 2`, run `NX = 32` and `NX = 64` and fit `log2(err_lo / err_hi) ≈
# P_X + 1`. v-mesh is held constant so the L² error is dominated by
# the x-direction discretisation (`a_v1 = a_v2 = 0` ⇒ no v-direction
# advection error).
#
# Build / run:
#     pixi run mojo run -I src tests/test_vlasov_convergence_1d2v.mojo

from std.math import exp, log2, sin
from std.sys import has_accelerator
from std.gpu.host import DeviceContext
from std.testing import assert_true

from warpxm.apps.vlasovmaxwell import VlasovMaxwell, VLASOV_NC, VLASOV_AUX_NC
from warpxm.basis.nodal_basis_1d import NodalBasis1D
from warpxm.core.types import Real, TWO_PI
from warpxm.mesh.phase_mesh_1d2v import PhaseMesh1D2V
from warpxm.spatial.dg_vlasov_1d2v import DGVlasovSolver1D2V
from warpxm.temporal.ssprk3_1d2v import SSPRK3_1D2V, no_op_adjust_1d2v
from warpxm.variable.phase_state_1d2v import PhaseState1D2V
from warpxm.variable.state_1d import DGState1D


comptime NV1: Int = 4
comptime NV2: Int = 4
comptime P_V1: Int = 2
comptime P_V2: Int = 2
comptime X_MIN: Real = 0.0
comptime X_MAX: Real = 1.0
comptime V1_MIN: Real = -3.0
comptime V1_MAX: Real = 3.0
comptime V2_MIN: Real = -3.0
comptime V2_MAX: Real = 3.0
comptime SIGMA_SQ: Real = 0.5
comptime T_FINAL: Real = 0.1
comptime DT: Real = 2.0e-4


@always_inline
def f_initial(x: Real, v1: Real, v2: Real) -> Real:
    return sin(TWO_PI * x) * exp(-(v1 * v1 + v2 * v2) / SIGMA_SQ)


def _one_run[P_X: Int](
    var ctx: DeviceContext, num_x: Int,
) raises -> Real:
    var mesh = PhaseMesh1D2V[P_X, P_V1, P_V2](
        ctx,
        X_MIN, X_MAX,
        V1_MIN, V1_MAX,
        V2_MIN, V2_MAX,
        num_x=num_x, num_v1=NV1, num_v2=NV2,
        periodic_x=True,
    )
    var q = PhaseState1D2V[P_X, P_V1, P_V2, VLASOV_NC](
        ctx, num_x, NV1, NV2,
    )

    var ref_x_nodes = NodalBasis1D[P_X].nodes()
    var ref_v1_nodes = NodalBasis1D[P_V1].nodes()
    var ref_v2_nodes = NodalBasis1D[P_V2].nodes()
    var x_weights = NodalBasis1D[P_X].weights()
    var v1_weights = NodalBasis1D[P_V1].weights()
    var v2_weights = NodalBasis1D[P_V2].weights()
    var host = ctx.enqueue_create_host_buffer[Real.dtype](q.buf_len)
    var nodes_per_phase = (P_X + 1) * (P_V1 + 1) * (P_V2 + 1)
    for ex in range(num_x):
        var xl = X_MIN + Real(ex) * mesh.dx
        for ev1 in range(NV1):
            var v1l = V1_MIN + Real(ev1) * mesh.dv1
            for ev2 in range(NV2):
                var v2l = V2_MIN + Real(ev2) * mesh.dv2
                var phase_e = (ex * NV1 + ev1) * NV2 + ev2
                for iv2 in range(P_V2 + 1):
                    var v2 = (
                        v2l
                        + (Real(1.0) + ref_v2_nodes[iv2])
                          * mesh.jacobian_v2
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

    var app = VlasovMaxwell(
        charge_mass_ratio=Real(-1.0),
        skin_depth_norm=Real(1.0),
        cfl_max=Real(0.5),
    )
    var spatial = DGVlasovSolver1D2V[P_X, P_V1, P_V2](app)
    var stepper = SSPRK3_1D2V[P_X, P_V1, P_V2, VLASOV_NC](
        ctx, num_x, NV1, NV2,
    )
    # Zero Maxwell aux ⇒ a_v1 = a_v2 = 0 ⇒ pure free streaming along x.
    var aux = DGState1D[P_X, VLASOV_AUX_NC](ctx, num_x)

    @parameter
    def compute_rhs(
        mut q_in: PhaseState1D2V[P_X, P_V1, P_V2, VLASOV_NC],
        mut q_out: PhaseState1D2V[P_X, P_V1, P_V2, VLASOV_NC],
    ) raises:
        spatial.compute_rhs(ctx, mesh, q_in, aux, q_out)

    @parameter
    def no_adjust(
        mut q_: PhaseState1D2V[P_X, P_V1, P_V2, VLASOV_NC],
    ) raises:
        pass

    var t: Real = 0.0
    while t < T_FINAL - Real(0.5) * DT:
        var dt_use = DT if (t + DT) < T_FINAL else (T_FINAL - t)
        stepper.step[compute_rhs, no_adjust](ctx, q, dt_use)
        t += dt_use
    ctx.synchronize()

    var acc: Real = 0.0
    with q.buf.map_to_host() as h:
        for ex in range(num_x):
            var xl = X_MIN + Real(ex) * mesh.dx
            for ev1 in range(NV1):
                var v1l = V1_MIN + Real(ev1) * mesh.dv1
                for ev2 in range(NV2):
                    var v2l = V2_MIN + Real(ev2) * mesh.dv2
                    var phase_e = (ex * NV1 + ev1) * NV2 + ev2
                    for iv2 in range(P_V2 + 1):
                        var v2 = (
                            v2l
                            + (Real(1.0) + ref_v2_nodes[iv2])
                              * mesh.jacobian_v2
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
                                var w = (
                                    x_weights[ix] * mesh.jacobian_x
                                    * v1_weights[iv1] * mesh.jacobian_v1
                                    * v2_weights[iv2] * mesh.jacobian_v2
                                )
                                acc += w * d * d
    return acc ** Real(0.5)


def _check_rate[P_X: Int](var ctx: DeviceContext, expected: Real) raises:
    var err_lo = _one_run[P_X](ctx, 32)
    var err_hi = _one_run[P_X](ctx, 64)
    var rate = log2(err_lo / err_hi)
    print(
        "P_X =", P_X,
        " err_NX32 =", err_lo,
        " err_NX64 =", err_hi,
        " rate =", rate,
    )
    assert_true(
        rate > expected - Real(0.25),
        "1D2V Vlasov convergence rate below tolerance",
    )


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = DeviceContext()
    _check_rate[1](ctx, Real(2.0))
    _check_rate[2](ctx, Real(3.0))
    _check_rate[3](ctx, Real(4.0))
    print("PASSED")
