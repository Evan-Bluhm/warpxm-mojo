# 2D2V Vlasov in a channel: specular walls in y, periodic in x.
#
# Companion to `test_vlasov_reflecting_wall_1d3v`. Here the walls are on
# the second configuration axis, so the mirrored velocity is `v_2`, the
# second velocity axis, while `v_1` and the periodic x direction must be
# left alone. With no fields and the image-consistent initial condition
#
#     f₀ = (1 + δ sin 2πx)(1 + ε cos πy)
#          · exp(−[(v_1 − U_1)² + (v_2 − U sin πy)²] / 2σ²)
#
# (even under `y → 2 − y`, `v_2 → −v_2`), the exact solution is
# `f₀(x − v_1 t, y − v_2 t, v)`. The uniform drift `U_1` makes `f`
# asymmetric in `v_1`, so mirroring the wrong velocity axis is an O(1)
# error. Checks order `P + 1` under refinement of x and y, and mass
# conservation to round-off.
#
# Build / run:
#     pixi run mojo run -I src tests/test_vlasov_reflecting_wall_2d2v.mojo

from std.math import cos, exp, sin
from std.sys import has_accelerator
from max.gpu.host import DeviceContext
from std.testing import assert_true

from warpxm.prelude import *
from warpxm.physics.vlasov import *
from warpxm.physics.maxwell import *
from warpxm.testing.convergence import check_convergence


comptime P: Int = 2
comptime NV1: Int = 6
comptime NV2: Int = 8
comptime V_MAX: Real = 4.0
comptime DELTA: Real = 0.2
comptime EPS: Real = 0.3
comptime DRIFT_X: Real = 0.5
comptime DRIFT_Y: Real = 0.8
comptime INV_2SIGMA_SQ: Real = 1.0 / (2.0 * 0.7 * 0.7)
comptime T_FINAL: Real = 0.6

comptime State = PhaseState[2, 2, P, P, VLASOV_NC]


def f_image(x: Real, y: Real, v1: Real, v2: Real) -> Real:
    var w1 = v1 - DRIFT_X
    var w2 = v2 - DRIFT_Y * sin(PI * y)
    return (
        (1.0 + DELTA * sin(TWO_PI * x)) * (1.0 + EPS * cos(PI * y))
        * exp(-INV_2SIGMA_SQ * (w1 * w1 + w2 * w2))
    )


def channel_run(var ctx: DeviceContext, nx: Int, ny: Int) raises -> Real:
    var mesh = PhaseMesh[2, 2, P, P](
        ctx,
        lo=[0.0, 0.0, -V_MAX, -V_MAX],
        hi=[1.0, 1.0, V_MAX, V_MAX],
        n=[nx, ny, NV1, NV2],
        periodic=[True, False],
        bc=[BC_COPYOUT, BC_COPYOUT, BC_VLASOV_REFLECT, BC_VLASOV_REFLECT],
    )
    var q = State(ctx, mesh.num_phase_elements)
    mesh.fill(q, f_image)
    var mass0 = mesh.integrate(q)

    var spatial = DGVlasovSolver[2, 2, P, P](VlasovMaxwell(cfl_max=Real(0.5)))
    var stepper = SSPRK3Phase[2, 2, P, P, VLASOV_NC](
        ctx, mesh.num_phase_elements,
    )
    var em = DGState[2, P, MAXWELL_NC](ctx, mesh.num_x_elements)   # zero

    def compute_rhs(mut q_in: State, mut q_out: State) raises {imm}:
        spatial.compute_rhs(ctx, mesh, q_in, em, q_out)
    def no_adjust(mut q_: State) raises:
        pass
    def suggest_dt(mut q_: State) raises {imm} -> Real:
        return spatial.suggest_dt(ctx, mesh, em)
    _ = stepper.run_until_cfl(
        ctx, q, Real(0.0), T_FINAL, compute_rhs, no_adjust, suggest_dt,
        dt_interval=1000000,
    )
    ctx.synchronize()

    var mass1 = mesh.integrate(q)
    var dm = (mass1 - mass0) / mass0
    print("  NX × NY =", nx, "×", ny, " relative mass change =", dm)
    # Measured −1.8e-14 and 4.1e-13: round-off over about 2000 steps.
    assert_true(abs(dm) < Real(1.0e-11), "specular walls did not conserve mass")

    var err_sq: Real = 0.0
    var norm_sq: Real = 0.0
    def add(
        x: Real, y: Real, v1: Real, v2: Real, w: Real, f: Real,
    ) {mut err_sq, mut norm_sq}:
        var exact = f_image(x - v1 * T_FINAL, y - v2 * T_FINAL, v1, v2)
        err_sq += w * (f - exact) * (f - exact)
        norm_sq += w * exact * exact
    mesh.visit_nodes(q, add)
    return sqrt(err_sq / norm_sq)


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = DeviceContext()
    # Measured 1.36e-3 → 1.68e-4 (rate 3.01); tolerances are 1.5x that.
    var lo = channel_run(ctx, 8, 12)
    var hi = channel_run(ctx, 16, 24)
    _ = check_convergence(
        "2D2V channel P = 2 (8 × 12 → 16 × 24)",
        lo, hi, Real(3.0), Real(2.0e-3), Real(2.6e-4),
    )
    print("PASSED")
