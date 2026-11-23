# 1D1V Vlasov constant-field acceleration test.
#
# With a constant external `E_x(x) = E₀` and an x-independent
# Maxwellian initial condition `f₀(v) = exp(-v²/σ²)`, the Vlasov
# equation reduces to pure v-advection at speed
#     a_v = (q/m) / λ_skin · E₀,
# so
#     f(x, v, t) = f₀(v - a_v t)     ∀ x.
#
# This test exercises the field coupling of `DGVlasovSolver[1, 1, …]` by
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
from max.gpu.host import DeviceContext
from std.testing import assert_true

from warpxm.prelude import *
from warpxm.physics.vlasov import *
from warpxm.apps.vlasovmaxwell import VLASOV_AUX_NC, AUX_EX


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


comptime State = PhaseState[1, 1, P_X, P_V, VLASOV_NC]
comptime Fields = DGState[1, P_X, VLASOV_AUX_NC]


@always_inline
def f_maxwellian(v: Real) -> Real:
    """Un-normalised Maxwellian in v."""
    return exp(-v * v / SIGMA_SQ)


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"

    var ctx = DeviceContext()
    var mesh = PhaseMesh[1, 1, P_X, P_V](
        ctx, lo=[X_MIN, V_MIN], hi=[X_MAX, V_MAX], n=[NX, NV],
        periodic=[True],
    )

    var q = PhaseState[1, 1, P_X, P_V, VLASOV_NC](
        ctx, mesh.num_phase_elements,
    )
    var aux = Fields(ctx, mesh.num_x_elements)

    # --- IC: f₀(x, v) = exp(-v²/σ²) — x-independent Maxwellian ---
    def initial(x: Real, v: Real) -> Real:
        return f_maxwellian(v)
    mesh.fill(q, initial)

    # --- aux: E_x = E0 everywhere, other components zero ---
    var aux_host = ctx.enqueue_create_host_buffer[Real.dtype](aux.buf_len)
    for k in range(aux.buf_len):
        aux_host[k] = Real(0.0)
    for ex in range(NX):
        for ix in range(P_X + 1):
            aux_host[Fields.index(ex, ix, AUX_EX)] = E0
    ctx.enqueue_copy(dst_buf=aux.buf, src_buf=aux_host)
    ctx.synchronize()

    # --- solver setup ---
    var app = VlasovMaxwell(
        charge_mass_ratio=CHARGE_MASS_RATIO,
        skin_depth_norm=SKIN_DEPTH_NORM,
        cfl_max=Real(0.5),
    )
    var spatial = DGVlasovSolver[1, 1, P_X, P_V](app)
    var stepper = SSPRK3Phase[1, 1, P_X, P_V, VLASOV_NC](
        ctx, mesh.num_phase_elements,
    )

    def compute_rhs(
        mut q_in: PhaseState[1, 1, P_X, P_V, VLASOV_NC],
        mut q_out: PhaseState[1, 1, P_X, P_V, VLASOV_NC],
    ) raises {mut aux, imm}:
        spatial.compute_rhs(ctx, mesh, q_in, aux, q_out)

    # --- time loop ---
    var t: Real = 0.0
    var nsteps: Int = 0
    while t < T_FINAL - Real(0.5) * DT:
        var dt_use = DT if (t + DT) < T_FINAL else (T_FINAL - t)
        stepper.step(
            ctx, q, dt_use, compute_rhs,
            no_op_adjust_phase[1, 1, P_X, P_V, VLASOV_NC],
        )
        t += dt_use
        nsteps += 1
    ctx.synchronize()

    # --- error analysis: f(x, v, T) = f₀(v - a_v · T) ---
    var a_v = CHARGE_MASS_RATIO / SKIN_DEPTH_NORM * E0
    var v_shift = a_v * T_FINAL
    var acc: Real = 0.0
    var max_abs_err: Real = 0.0
    def compare(x: Real, v: Real, w: Real, f_num: Real) {mut acc, mut max_abs_err, imm}:
        var d = f_num - f_maxwellian(v - v_shift)
        max_abs_err = fmax(max_abs_err, fabs(d))
        acc += w * d * d
    mesh.visit_nodes(q, compare)
    var l2 = sqrt(acc)

    # Also check x-independence: L_∞ of f(x, v, T) − f(x=x₀, v, T) over x at
    # every (ev, iv), with (ex, ix) = (0, 0) as the reference; should be
    # ~ kernel-truncation noise.
    var max_x_variation: Real = 0.0
    with q.buf.map_to_host() as h:
        for ev in range(NV):
            for iv in range(P_V + 1):
                var f_ref = h[State.index(ev, iv * (P_X + 1), 0)]
                for ex in range(NX):
                    for ix in range(P_X + 1):
                        var idx = State.index(
                            ex * NV + ev, iv * (P_X + 1) + ix, 0,
                        )
                        max_x_variation = fmax(
                            max_x_variation, fabs(h[idx] - f_ref),
                        )

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
