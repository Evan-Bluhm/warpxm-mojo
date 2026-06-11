# 1D1V Vlasov constant-field acceleration test.
#
# With a constant external `E_x(x) = E₀` and an x-independent
# Maxwellian initial condition `f₀(v) = exp(-v²/σ²)`, the Vlasov
# equation reduces to pure v-advection at speed
#     a_v = (q/m) / λ_skin · E₀,
# so
#     f(x, v, t) = f₀(v - a_v t)     ∀ x.
#
# This test exercises the new Maxwell aux coupling on the kernel by
# checking that the numerical solution recovers a uniformly v-shifted
# Maxwellian. The x-direction term should contribute zero (f is
# x-independent and stays x-independent), which validates that the
# kernel handles the no-op x-volume / x-surface case correctly even
# while v-direction physics is active.
#
# Build / run:
#     pixi run mojo run -I src tests/test_vlasov_constant_field_1d1v.mojo

from std.math import exp
from std.sys import has_accelerator
from std.gpu.host import DeviceContext
from std.testing import assert_true

from warpxm.apps.vlasovmaxwell import (
    VlasovMaxwell, VLASOV_NC, VLASOV_AUX_NC, AUX_EX,
)
from warpxm.basis.nodal_basis_1d import NodalBasis1D
from warpxm.core.types import Real
from warpxm.mesh.phase_mesh_1d1v import PhaseMesh1D1V
from warpxm.spatial.dg_vlasov_1d1v import DGVlasovSolver1D1V
from warpxm.temporal.ssprk3_1d1v import SSPRK3_1D1V
from warpxm.variable.phase_state_1d1v import PhaseState1D1V
from warpxm.variable.state_1d import DGState1D


comptime P_X: Int = 2
comptime P_V: Int = 2
comptime NX: Int = 4
comptime NV: Int = 16
comptime X_MIN: Real = 0.0
comptime X_MAX: Real = 1.0
comptime V_MIN: Real = -3.0
comptime V_MAX: Real = 3.0
comptime SIGMA_SQ: Real = 0.5
comptime E0: Real = 0.1
comptime CHARGE_MASS_RATIO: Real = -1.0
comptime SKIN_DEPTH_NORM: Real = 1.0
# a_v = (-1) / 1 · 0.1 = -0.1, so f shifts by -0.1 · T in v.
comptime T_FINAL: Real = 2.0
comptime DT: Real = 5.0e-3


@always_inline
def f_maxwellian(v: Real) -> Real:
    """Un-normalised Maxwellian in v."""
    return exp(-v * v / SIGMA_SQ)


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
    var aux = DGState1D[P_X, VLASOV_AUX_NC](ctx, NX)

    # --- IC: f₀(x, v) = exp(-v²/σ²) — x-independent Maxwellian ---
    var ref_v_nodes = NodalBasis1D[P_V].nodes()
    var host = ctx.enqueue_create_host_buffer[Real.dtype](q.buf_len)
    for ex in range(NX):
        for ev in range(NV):
            var vl = V_MIN + Real(ev) * mesh.dv
            for ix in range(P_X + 1):
                for iv in range(P_V + 1):
                    var v = vl + (Real(1.0) + ref_v_nodes[iv]) * mesh.jacobian_v
                    var phase_e = ex * NV + ev
                    var idx = (
                        (phase_e * (P_V + 1) + iv) * (P_X + 1) + ix
                    ) * VLASOV_NC
                    host[idx] = f_maxwellian(v)
    ctx.enqueue_copy(dst_buf=q.buf, src_buf=host)

    # --- aux: E_x = E0 everywhere, other components zero ---
    var aux_host = ctx.enqueue_create_host_buffer[Real.dtype](aux.buf_len)
    for k in range(aux.buf_len):
        aux_host[k] = Real(0.0)
    for ex in range(NX):
        for ix in range(P_X + 1):
            var node_x = ex * (P_X + 1) + ix
            aux_host[node_x * VLASOV_AUX_NC + AUX_EX] = E0
    ctx.enqueue_copy(dst_buf=aux.buf, src_buf=aux_host)
    ctx.synchronize()

    # --- solver setup ---
    var app = VlasovMaxwell(
        charge_mass_ratio=CHARGE_MASS_RATIO,
        skin_depth_norm=SKIN_DEPTH_NORM,
        cfl_max=Real(0.5),
    )
    var spatial = DGVlasovSolver1D1V[P_X, P_V](app)
    var stepper = SSPRK3_1D1V[P_X, P_V, VLASOV_NC](ctx, NX, NV)

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
        stepper.step[compute_rhs, no_adjust](ctx, q, dt_use)
        t += dt_use
        nsteps += 1
    ctx.synchronize()

    # --- error analysis: f(x, v, T) = f₀(v - a_v · T) ---
    var a_v = CHARGE_MASS_RATIO / SKIN_DEPTH_NORM * E0
    var v_shift = a_v * T_FINAL
    var x_weights = NodalBasis1D[P_X].weights()
    var v_weights = NodalBasis1D[P_V].weights()
    var acc: Real = 0.0
    var max_abs_err: Real = 0.0
    # Also check x-independence: compute L_∞ of f(x, v, T) − f(x=x₀, v, T) over
    # x at every (ev, iv); should be ~ kernel-truncation noise.
    var max_x_variation: Real = 0.0
    with q.buf.map_to_host() as h:
        for ev in range(NV):
            var vl = V_MIN + Real(ev) * mesh.dv
            for iv in range(P_V + 1):
                var v = vl + (Real(1.0) + ref_v_nodes[iv]) * mesh.jacobian_v
                var f_exact = f_maxwellian(v - v_shift)
                # Sample at ex=0, ix=0 as the reference for x-comparison.
                var phase_e_ref = 0 * NV + ev
                var idx_ref = (
                    (phase_e_ref * (P_V + 1) + iv) * (P_X + 1) + 0
                ) * VLASOV_NC
                var f_ref = h[idx_ref]
                for ex in range(NX):
                    for ix in range(P_X + 1):
                        var phase_e = ex * NV + ev
                        var idx = (
                            (phase_e * (P_V + 1) + iv) * (P_X + 1) + ix
                        ) * VLASOV_NC
                        var f_num = h[idx]
                        var d = f_num - f_exact
                        var ad = d if d >= 0.0 else -d
                        if ad > max_abs_err:
                            max_abs_err = ad
                        var dvar = f_num - f_ref
                        var advar = dvar if dvar >= 0.0 else -dvar
                        if advar > max_x_variation:
                            max_x_variation = advar
                        var w = (
                            x_weights[ix] * mesh.jacobian_x
                            * v_weights[iv] * mesh.jacobian_v
                        )
                        acc += w * d * d
    var l2 = acc ** Real(0.5)

    print(
        "constant-field 1D1V: steps =", nsteps,
        "t_final =", t,
        "a_v =", a_v, "Δv =", v_shift,
        "L2 err =", l2,
        "max |err| =", max_abs_err,
        "max x-variation =", max_x_variation,
    )

    # Spatial DG truncation for P=2 with NV=16 ⇒ ≈ (6/16)^3 = 5e-2.
    # The Maxwellian is well-resolved (σ ≈ 0.71, dv ≈ 0.375 ⇒ ~2 cells per σ).
    # Use a generous tolerance for this first coupled test.
    assert_true(l2 < Real(2.0e-2), "v-shifted-Maxwellian L2 too high")
    # x-independence: should be ~ machine-precision (no x-physics happens).
    assert_true(
        max_x_variation < Real(1.0e-10),
        "f drifted in x despite x-independent IC + constant field",
    )
    print("PASSED")
