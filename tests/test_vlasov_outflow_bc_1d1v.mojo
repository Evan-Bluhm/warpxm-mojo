# 1D1V Vlasov v-axis outflow BC test.
#
# Stress-test the zero-ghost (`BC_VLASOV_OUTFLOW`) condition that
# `DGVlasovSolver` applies at every velocity boundary. Setup:
#
#   * IC: a Maxwellian centred at `v = 2.0`, so the right tail
#     `f(v_max = 3) ≈ exp(-2) ≈ 0.135` is non-negligible.
#   * Constant external `E_x = 0.5`, giving `a_v = α · E_x = -0.5`
#     (with α = q/m = -1). The whole distribution drifts toward
#     smaller v at speed 0.5; over T = 2 the centre moves from
#     v = 2 to v = 1, sweeping the original right tail through the
#     v_max boundary.
#
# Physical prediction: the Maxwellian centre drifts from v = 2 to
# v = 1 during T = 2, staying well inside the velocity domain
# `[-3, 3]`. **No mass should leave through either v-boundary**, so
# `ΔM/M₀` must be at machine precision.
#
# This is a test of the BC because at `v = v_max` the wave speed is
# `a_v = −0.5 < 0`, which makes that face an *inflow* boundary
# (characteristics enter the domain from `v > v_max`). The correct
# BC says "outside is empty, no inflow", giving `F* = a_v · 0 = 0`.
#
# With a copy-out ghost (ghost = interior) the upwind flux would
# be `F* = a_v · f_interior(v_max) ≈ −0.0675` at t = 0,
# **adding spurious mass to the interior** at rate
# `|F*| · L_x ≈ 0.0675` per unit time. Over T = 2 with a Maxwellian
# `f(v_max) ≈ 0.135` decaying as the centre drifts down, that
# accumulates to a few-percent spurious mass gain. The test
# tolerance below is much tighter than that, so this test would
# fail under copyout.
#
# Build / run:
#     pixi run mojo run -I src tests/test_vlasov_outflow_bc_1d1v.mojo

from std.math import exp
from std.sys import has_accelerator
from max.gpu.host import DeviceContext
from std.testing import assert_true

from warpxm.prelude import *
from warpxm.physics.vlasov import *
from warpxm.apps.vlasovmaxwell import VLASOV_AUX_NC, AUX_EX


comptime P_X: Int = 2
comptime P_V: Int = 3
comptime NX: Int = 4
comptime NV: Int = 16
comptime SIGMA_SQ: Real = 0.5
comptime U_CENTER: Real = 2.0
comptime E0: Real = 0.5            # external constant field
comptime CHARGE_MASS_RATIO: Real = -1.0  # ⇒ a_v = -0.5
comptime T_FINAL: Real = 2.0
comptime DT: Real = 5.0e-3


def compute_total_mass(
    mesh: PhaseMesh[1, 1, P_X, P_V],
    mut q: PhaseState[1, 1, P_X, P_V, VLASOV_NC],
) raises -> Real:
    var acc: Real = 0.0
    def add(x: Real, v: Real, w: Real, f: Real) {mut acc}:
        acc += w * f
    mesh.visit_nodes(q, add)
    return acc


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = DeviceContext()

    var mesh = PhaseMesh[1, 1, P_X, P_V](
        ctx, lo=[0.0, -3.0], hi=[1.0, 3.0], n=[NX, NV],
        periodic=[True],
    )

    var q = PhaseState[1, 1, P_X, P_V, VLASOV_NC](
        ctx, mesh.num_phase_elements,
    )
    var aux = DGState[1, P_X, VLASOV_AUX_NC](ctx, mesh.num_x_elements)

    # IC: f₀(v) = exp(-(v - U_CENTER)² / σ²) — Maxwellian near v_max,
    # x-independent.
    def initial(x: Real, v: Real) -> Real:
        var dv_shift = v - U_CENTER
        return exp(-dv_shift * dv_shift / SIGMA_SQ)
    mesh.fill(q, initial)

    # Aux: constant E_x = E0 at every x-node.
    var aux_host = ctx.enqueue_create_host_buffer[Real.dtype](aux.buf_len)
    for k in range(aux.buf_len):
        aux_host[k] = Real(0.0)
    for n in range(NX * (P_X + 1)):
        aux_host[n * VLASOV_AUX_NC + AUX_EX] = E0
    ctx.enqueue_copy(dst_buf=aux.buf, src_buf=aux_host)
    ctx.synchronize()

    # --- baseline mass + tail at v_max ---
    var m0 = compute_total_mass(mesh, q)
    var tail_at_vmax = exp(-(Real(3.0) - U_CENTER) * (Real(3.0) - U_CENTER) / SIGMA_SQ)
    print(
        "Vlasov outflow IC: M0 =", m0,
        " f(v_max) ≈", tail_at_vmax,
    )

    # --- solver + stepper ---
    var app = VlasovMaxwell(
        charge_mass_ratio=CHARGE_MASS_RATIO,
        skin_depth_norm=Real(1.0),
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

    var m1 = compute_total_mass(mesh, q)
    var dM = m1 - m0
    var adM = dM if dM >= 0.0 else -dM
    var rel_dM = adM / m0
    print(
        "Vlasov outflow T_FINAL: steps =", nsteps,
        "t =", t,
        " M1 =", m1, " ΔM =", dM,
        " |ΔM|/M0 =", rel_dM,
    )

    # Mass must be conserved to machine precision: with the zero ghost
    # state there is no spurious inflow at v_max even though
    # `f(v_max) ≈ 0.135` is substantial and the wave at that face is
    # inflow-directed. With a copy-out ghost this same simulation
    # would have grown the mass by a few percent.
    # Measured 2.0e-15.
    assert_true(rel_dM < Real(1.0e-13), "outflow BC didn't conserve mass")

    print("PASSED")
