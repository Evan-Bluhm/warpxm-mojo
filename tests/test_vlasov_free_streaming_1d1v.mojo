# 1D1V Vlasov free-streaming smoke test.
#
# With no Lorentz acceleration (E = B = 0 ⇒ a_v = 0), the Vlasov
# equation reduces to
#     ∂f/∂t + v ∂f/∂x = 0,
# which advects every velocity slice independently along x at speed `v`:
#     f(x, v, t) = f₀(x − v · t, v).
#
# This test sets `f₀(x, v) = sin(2π x) · exp(−v² / σ²)` on an x-periodic
# 1D1V phase mesh, integrates with SSPRK3, and compares the numerical
# solution to the analytic shifted initial condition in L2.
#
# Build / run:
#     pixi run mojo run -I src tests/test_vlasov_free_streaming_1d1v.mojo

from std.math import sin, exp
from std.sys import has_accelerator
from std.gpu.host import DeviceContext
from std.testing import assert_true

from warpxm.apps.vlasovmaxwell import VlasovMaxwell, VLASOV_NC, VLASOV_AUX_NC
from warpxm.basis.nodal_basis_1d import NodalBasis1D
from warpxm.core.types import Real, TWO_PI
from warpxm.mesh.phase_mesh_1d1v import PhaseMesh1D1V
from warpxm.spatial.dg_vlasov_1d1v import DGVlasovSolver1D1V
from warpxm.temporal.ssprk3_1d1v import SSPRK3_1D1V
from warpxm.variable.phase_state_1d1v import PhaseState1D1V
from warpxm.variable.state_1d import DGState1D


comptime P_X: Int = 2
comptime P_V: Int = 2
comptime NX: Int = 16
comptime NV: Int = 8
comptime X_MIN: Real = 0.0
comptime X_MAX: Real = 1.0
comptime V_MIN: Real = -3.0
comptime V_MAX: Real = 3.0
comptime SIGMA_SQ: Real = 0.5
comptime T_FINAL: Real = 0.25
comptime DT: Real = 1.0e-3


@always_inline
def f_initial(x: Real, v: Real) -> Real:
    return sin(TWO_PI * x) * exp(-v * v / SIGMA_SQ)


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"

    var ctx = DeviceContext()
    var mesh = PhaseMesh1D1V[P_X, P_V](
        ctx,
        X_MIN, X_MAX, V_MIN, V_MAX,
        num_x=NX, num_v=NV,
        periodic_x=True,
    )

    var q = PhaseState1D1V[P_X, P_V, VLASOV_NC](ctx, NX, NV)

    # --- IC: f₀(x, v) = sin(2π x) · exp(-v² / σ²) ---
    var ref_x_nodes = NodalBasis1D[P_X].nodes()
    var ref_v_nodes = NodalBasis1D[P_V].nodes()
    var host = ctx.enqueue_create_host_buffer[Real.dtype](q.buf_len)
    for ex in range(NX):
        var xl = X_MIN + Real(ex) * mesh.dx
        for ev in range(NV):
            var vl = V_MIN + Real(ev) * mesh.dv
            for ix in range(P_X + 1):
                var x = xl + (Real(1.0) + ref_x_nodes[ix]) * mesh.jacobian_x
                for iv in range(P_V + 1):
                    var v = vl + (Real(1.0) + ref_v_nodes[iv]) * mesh.jacobian_v
                    var phase_e = ex * NV + ev
                    var idx = (
                        (phase_e * (P_V + 1) + iv) * (P_X + 1) + ix
                    ) * VLASOV_NC
                    host[idx] = f_initial(x, v)
    ctx.enqueue_copy(dst_buf=q.buf, src_buf=host)
    ctx.synchronize()

    # --- solver setup ---
    var app = VlasovMaxwell(
        charge_mass_ratio=Real(-1.0),
        skin_depth_norm=Real(1.0),
        cfl_max=Real(0.5),
    )
    var spatial = DGVlasovSolver1D1V[P_X, P_V](app)
    var stepper = SSPRK3_1D1V[P_X, P_V, VLASOV_NC](ctx, NX, NV)

    # Zero Maxwell aux ⇒ a_v = 0 ⇒ pure free streaming.
    var aux = DGState1D[P_X, VLASOV_AUX_NC](ctx, NX)

    @parameter
    def compute_rhs(
        mut q_in: PhaseState1D1V[P_X, P_V, VLASOV_NC],
        mut q_out: PhaseState1D1V[P_X, P_V, VLASOV_NC],
    ) raises:
        spatial.compute_rhs(ctx, mesh, q_in, aux, q_out)

    @parameter
    def no_adjust(mut q_: PhaseState1D1V[P_X, P_V, VLASOV_NC]) raises:
        pass

    # --- time loop ---
    var t: Real = 0.0
    var nsteps: Int = 0
    while t < T_FINAL - Real(0.5) * DT:
        var dt_use = DT if (t + DT) < T_FINAL else (T_FINAL - t)
        stepper.step[compute_rhs, no_adjust](
            ctx, q, dt_use,
        )
        t += dt_use
        nsteps += 1
    ctx.synchronize()

    # --- error analysis: compare against f(x, v, T) = f₀(x − v·T, v) ---
    var x_weights = NodalBasis1D[P_X].weights()
    var v_weights = NodalBasis1D[P_V].weights()
    var acc: Real = 0.0
    var max_abs_err: Real = 0.0
    with q.buf.map_to_host() as h:
        for ex in range(NX):
            var xl = X_MIN + Real(ex) * mesh.dx
            for ev in range(NV):
                var vl = V_MIN + Real(ev) * mesh.dv
                for ix in range(P_X + 1):
                    var x = xl + (Real(1.0) + ref_x_nodes[ix]) * mesh.jacobian_x
                    for iv in range(P_V + 1):
                        var v = vl + (Real(1.0) + ref_v_nodes[iv]) * mesh.jacobian_v
                        var phase_e = ex * NV + ev
                        var idx = (
                            (phase_e * (P_V + 1) + iv) * (P_X + 1) + ix
                        ) * VLASOV_NC
                        # Wrap the shifted x back into [X_MIN, X_MAX).
                        var period = X_MAX - X_MIN
                        var x_wrap = x - v * T_FINAL
                        while x_wrap < X_MIN:
                            x_wrap += period
                        while x_wrap >= X_MAX:
                            x_wrap -= period
                        var f_exact = f_initial(x_wrap, v)
                        var d = h[idx] - f_exact
                        var ad = d if d >= 0.0 else -d
                        if ad > max_abs_err:
                            max_abs_err = ad
                        # GLL quadrature weight contribution.
                        var w = (
                            x_weights[ix] * mesh.jacobian_x
                            * v_weights[iv] * mesh.jacobian_v
                        )
                        acc += w * d * d
    var l2 = acc ** Real(0.5)

    print(
        "free-streaming 1D1V: steps =", nsteps,
        "t_final =", t,
        "L2 err =", l2,
        "max |err| =", max_abs_err,
    )

    # The DG truncation error for P=2 with NX=16 / NV=8 is roughly
    # h^(P+1) ≈ 2.5e-4. SSPRK3 adds O(dt^3) ≈ 1e-9 at dt=1e-3. Use a
    # generous tolerance for this first end-to-end test — we're checking
    # that the kernel runs, the stepper composes cleanly, and the
    # answer is in the right ballpark.
    assert_true(l2 < Real(5.0e-2), "free-streaming L2 error too high")
    print("PASSED")
