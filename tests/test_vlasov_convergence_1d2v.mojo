# 1D2V Vlasov DG convergence regression — confirm rate ≈ (P + 1)
# on smooth free-streaming. Tensor-product extension of
# `test_vlasov_convergence_1d1v` to 1D2V, on the generic kinetic stack
# (`DGVlasovSolver[1, 2, P_X, P_V]`, `spatial/kinetic/dg_vlasov.mojo`).
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
# Sweep: for `P_X ∈ {1, 2, 3}` and fixed `NV1 = NV2 = 4`, `P_V = 2`,
# run `NX = 32` and `NX = 64` and fit `log2(err_lo / err_hi) ≈
# P_X + 1`. v-mesh is held constant so the L² error is dominated by
# the x-direction discretisation (`a_v1 = a_v2 = 0` ⇒ no v-direction
# advection error).
#
# Build / run:
#     pixi run mojo run -I src tests/test_vlasov_convergence_1d2v.mojo

from std.math import exp, sin
from std.sys import has_accelerator
from max.gpu.host import DeviceContext
from std.testing import assert_true

from warpxm.prelude import *
from warpxm.physics.vlasov import *
from warpxm.apps.vlasovmaxwell import VLASOV_AUX_NC
from warpxm.testing.convergence import check_convergence


comptime NV1: Int = 4
comptime NV2: Int = 4
comptime P_V: Int = 2
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
    comptime Phase = PhaseState[1, 2, P_X, P_V, VLASOV_NC]
    var mesh = PhaseMesh[1, 2, P_X, P_V](
        ctx,
        lo=[X_MIN, V1_MIN, V2_MIN],
        hi=[X_MAX, V1_MAX, V2_MAX],
        n=[num_x, NV1, NV2],
        periodic=[True],
    )
    var q = Phase(ctx, mesh.num_phase_elements)

    mesh.fill(q, f_initial)

    var app = VlasovMaxwell(
        charge_mass_ratio=Real(-1.0),
        skin_depth_norm=Real(1.0),
        cfl_max=Real(0.5),
    )
    var spatial = DGVlasovSolver[1, 2, P_X, P_V](app)
    var stepper = SSPRK3Phase[1, 2, P_X, P_V, VLASOV_NC](
        ctx, mesh.num_phase_elements,
    )
    # Zero Maxwell aux ⇒ a_v1 = a_v2 = 0 ⇒ pure free streaming along x.
    var aux = DGState[1, P_X, VLASOV_AUX_NC](ctx, num_x)

    def compute_rhs(mut q_in: Phase, mut q_out: Phase) raises {imm}:
        spatial.compute_rhs(ctx, mesh, q_in, aux, q_out)

    _ = stepper.run_until(
        ctx, q, Real(0.0), T_FINAL, DT, compute_rhs,
        no_op_adjust_phase[1, 2, P_X, P_V, VLASOV_NC],
    )
    ctx.synchronize()

    var acc: Real = 0.0
    def add(x: Real, v1: Real, v2: Real, w: Real, f: Real) {mut acc}:
        # Wrap x − v1·T back into [X_MIN, X_MAX).
        var period = X_MAX - X_MIN
        var x_wrap = x - v1 * T_FINAL
        while x_wrap < X_MIN:
            x_wrap += period
        while x_wrap >= X_MAX:
            x_wrap -= period
        var d = f - f_initial(x_wrap, v1, v2)
        acc += w * d * d
    mesh.visit_nodes(q, add)
    return sqrt(acc)


def _check_rate[P_X: Int](
    var ctx: DeviceContext, expected: Real, tol_lo: Real, tol_hi: Real,
) raises:
    """Check the observed rate and bound both errors absolutely.

    `tol_lo` / `tol_hi` are about 3x the measured coarse / fine errors, so
    a defect that scales the error uniformly (and leaves the rate intact)
    still fails.
    """
    var err_lo = _one_run[P_X](ctx, 32)
    var err_hi = _one_run[P_X](ctx, 64)
    _ = check_convergence(
        "1D2V Vlasov P_X = " + String(P_X) + " (NX 32 → 64)",
        err_lo, err_hi, expected, tol_lo, tol_hi,
    )


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = DeviceContext()
    _check_rate[1](ctx, Real(2.0), Real(8.0e-3), Real(2.1e-3))
    _check_rate[2](ctx, Real(3.0), Real(1.2e-4), Real(1.5e-5))
    _check_rate[3](ctx, Real(4.0), Real(1.2e-6), Real(7.8e-8))
    print("PASSED")
