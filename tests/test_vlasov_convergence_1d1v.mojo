# 1D1V Vlasov DG convergence regression — confirm rate ≈ (P + 1)
# on smooth free-streaming. Direct analogue of `test_advection_convergence_2d`
# for the kinetic phase-space DG kernel (`spatial/dg_vlasov_1d1v.mojo`).
#
# Setup: zero Maxwell aux (`E = B = 0` ⇒ `a_v = 0`), so the Vlasov
# equation reduces to `∂f/∂t + v ∂f/∂x = 0`. Each velocity slice
# advects along x at speed `v`; analytic solution at `t = T` is
# `f(x, v, T) = f₀(x − v T, v)`.
#
# IC: `f₀(x, v) = sin(2π x) · exp(-v² / σ²)`. Smooth in both x and v
# with bounded support to machine precision over the truncated
# `v ∈ [-v_max, v_max]` domain (`erfc(v_max / σ) ≈ 4·10⁻¹⁰` for
# `σ² = 0.5, v_max = 3`).
#
# Sweep: for `P ∈ {1, 2, 3}` and fixed `NV = 8`, run `NX = 16` and
# `NX = 32` and fit `log2(err_lo / err_hi) ≈ P + 1`. NV is held
# constant across NX values so the L² error is dominated by the
# x-direction discretisation — the v-direction sees no advection
# (`a_v = 0`) and its only error source is GLL quadrature of the
# Gaussian IC, which is well below the x-truncation error for NV = 8.
#
# CFL: `dt < dx_min / max|v| / (2P + 1)`. Pinned to the finest-mesh
# CFL across the whole sweep (`dt = 2.0·10⁻⁴`) so SSPRK3's `O(dt³)`
# temporal truncation doesn't pollute the spatial-convergence rate.
#
# Build / run:
#     pixi run mojo run -I src tests/test_vlasov_convergence_1d1v.mojo

from std.math import exp, sin
from std.sys import has_accelerator
from max.gpu.host import DeviceContext
from std.testing import assert_true

from warpxm.prelude import *
from warpxm.physics.vlasov import *
from warpxm.apps.vlasovmaxwell import VLASOV_AUX_NC
from warpxm.testing.convergence import check_convergence


comptime NV: Int = 8
comptime P_V: Int = 3       # high enough v-order to keep v-direction error subdominant
comptime X_MIN: Real = 0.0
comptime X_MAX: Real = 1.0
comptime V_MIN: Real = -3.0
comptime V_MAX: Real = 3.0
comptime SIGMA_SQ: Real = 0.5
comptime T_FINAL: Real = 0.1
comptime DT: Real = 2.0e-4


@always_inline
def f_initial(x: Real, v: Real) -> Real:
    return sin(TWO_PI * x) * exp(-v * v / SIGMA_SQ)


def _one_run[P_X: Int](
    var ctx: DeviceContext, num_x: Int,
) raises -> Real:
    var mesh = PhaseMesh1D1V[P_X, P_V](
        ctx, X_MIN, X_MAX, V_MIN, V_MAX,
        num_x=num_x, num_v=NV, periodic_x=True,
    )
    var q = PhaseState1D1V[P_X, P_V, VLASOV_NC](ctx, num_x, NV)

    mesh.fill(q, f_initial)

    var app = VlasovMaxwell(
        charge_mass_ratio=Real(-1.0),
        skin_depth_norm=Real(1.0),
        cfl_max=Real(0.5),
    )
    var spatial = DGVlasovSolver1D1V[P_X, P_V](app)
    var stepper = SSPRK3_1D1V[P_X, P_V, VLASOV_NC](ctx, num_x, NV)

    # Zero Maxwell aux ⇒ a_v = 0 ⇒ pure free streaming along x.
    var aux = DGState[1, P_X, VLASOV_AUX_NC](ctx, num_x)

    def compute_rhs(
        mut q_in: PhaseState1D1V[P_X, P_V, VLASOV_NC],
        mut q_out: PhaseState1D1V[P_X, P_V, VLASOV_NC],
    ) raises {mut aux, imm}:
        spatial.compute_rhs(ctx, mesh, q_in, aux, q_out)

    def no_adjust(mut q_: PhaseState1D1V[P_X, P_V, VLASOV_NC]) raises {imm}:
        pass

    _ = stepper.run_until(
        ctx, q, Real(0.0), T_FINAL, DT, compute_rhs, no_adjust,
    )
    ctx.synchronize()

    var acc: Real = 0.0
    def add(x: Real, v: Real, w: Real, f: Real) {mut acc}:
        # Wrap shifted x back into [X_MIN, X_MAX).
        var period = X_MAX - X_MIN
        var x_wrap = x - v * T_FINAL
        while x_wrap < X_MIN:
            x_wrap += period
        while x_wrap >= X_MAX:
            x_wrap -= period
        var d = f - f_initial(x_wrap, v)
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
    # NX=32 vs 64 (same choice as `test_advection_convergence_1d`) — at NX=16
    # the P=1 case isn't yet in the asymptotic h^(P+1) regime.
    var err_lo = _one_run[P_X](ctx, 32)
    var err_hi = _one_run[P_X](ctx, 64)
    _ = check_convergence(
        "1D1V Vlasov P_X = " + String(P_X) + " (NX 32 → 64)",
        err_lo, err_hi, expected, tol_lo, tol_hi,
    )


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = DeviceContext()
    _check_rate[1](ctx, Real(2.0), Real(1.5e-2), Real(4.3e-3))
    _check_rate[2](ctx, Real(3.0), Real(2.9e-4), Real(3.3e-5))
    _check_rate[3](ctx, Real(4.0), Real(3.3e-6), Real(1.7e-7))
    print("PASSED")
