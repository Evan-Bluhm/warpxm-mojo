# 1D3V Vlasov right-hand side against the analytic operator.
#
# For a smooth `f(x, v1, v2, v3)` and smooth fields `E(x)`, `B(x)` the DG
# residual must converge to
#
#     −v1 ∂f/∂x − a · ∂f/∂v,      a = (q/m)/λ · (E + v × B)
#
# (`∂a_d/∂v_d = 0`, so the conservative and advective forms agree). All
# six field components are non-zero and vary in x, and `f` is not
# separable, so each of the nine Lorentz terms
#
#     a_1 = E_x + v_2 B_z − v_3 B_y
#     a_2 = E_y + v_3 B_x − v_1 B_z
#     a_3 = E_z + v_1 B_y − v_2 B_x
#
# contributes to the answer with its own sign; a wrong sign or a swapped
# component leaves an O(1) error that does not converge. The L² error of a
# degree-P nodal derivative converges at order P.
#
# Build / run:
#     pixi run mojo run -I src tests/test_vlasov_rhs_consistency_1d3v.mojo

from std.collections import Array
from std.math import cos, exp, sin
from std.sys import has_accelerator
from max.gpu.host import DeviceContext
from std.testing import assert_true

from warpxm.prelude import *
from warpxm.physics.vlasov import *
from warpxm.physics.maxwell import *
from warpxm.testing.convergence import check_convergence


comptime X_MIN: Real = 0.0
comptime X_MAX: Real = 1.0
# f at the velocity boundary is below exp(−4.2²) ≈ 2e-8, so the outflow
# ghost state (zero) is far below the discretisation error.
comptime V_MAX: Real = 4.5
comptime INV_SIGMA_SQ: Real = 2.0
comptime U1: Real = 0.3
comptime U2: Real = -0.2
comptime U3: Real = 0.1
comptime K1: Real = 0.3
comptime K2: Real = -0.4
comptime K3: Real = 0.5
comptime AMP: Real = 0.3
comptime CHARGE_MASS_RATIO: Real = 1.7
comptime SKIN_DEPTH_NORM: Real = 0.9


def fields(x: Real) -> Array[Real, MAXWELL_NC]:
    var v = Array[Real, MAXWELL_NC](fill=Real(0.0))
    v[IDX_EX] = 0.7 * sin(TWO_PI * x) + 0.2
    v[IDX_EY] = -0.4 * cos(TWO_PI * x) + 0.1
    v[IDX_EZ] = 0.3 * sin(TWO_PI * x) - 0.5
    v[IDX_BX] = 0.25 + 0.1 * cos(TWO_PI * x)
    v[IDX_BY] = 0.5 * cos(TWO_PI * x) - 0.15
    v[IDX_BZ] = 0.6 * sin(TWO_PI * x) - 0.3
    return v^


def f_exact(x: Real, v1: Real, v2: Real, v3: Real) -> Real:
    var phase = TWO_PI * x + K1 * v1 + K2 * v2 + K3 * v3
    var w1 = v1 - U1
    var w2 = v2 - U2
    var w3 = v3 - U3
    return (
        (1.0 + AMP * sin(phase))
        * exp(-0.5 * INV_SIGMA_SQ * (w1 * w1 + w2 * w2 + w3 * w3))
    )


def rhs_exact(x: Real, v1: Real, v2: Real, v3: Real) -> Real:
    var phase = TWO_PI * x + K1 * v1 + K2 * v2 + K3 * v3
    var w1 = v1 - U1
    var w2 = v2 - U2
    var w3 = v3 - U3
    var maxw = exp(-0.5 * INV_SIGMA_SQ * (w1 * w1 + w2 * w2 + w3 * w3))
    var g = 1.0 + AMP * sin(phase)
    var dg = AMP * cos(phase)
    var df_dx = TWO_PI * dg * maxw
    var df_dv1 = (K1 * dg - g * w1 * INV_SIGMA_SQ) * maxw
    var df_dv2 = (K2 * dg - g * w2 * INV_SIGMA_SQ) * maxw
    var df_dv3 = (K3 * dg - g * w3 * INV_SIGMA_SQ) * maxw
    var em = fields(x)
    var factor = CHARGE_MASS_RATIO / SKIN_DEPTH_NORM
    var a1 = factor * (em[IDX_EX] + v2 * em[IDX_BZ] - v3 * em[IDX_BY])
    var a2 = factor * (em[IDX_EY] + v3 * em[IDX_BX] - v1 * em[IDX_BZ])
    var a3 = factor * (em[IDX_EZ] + v1 * em[IDX_BY] - v2 * em[IDX_BX])
    return -v1 * df_dx - a1 * df_dv1 - a2 * df_dv2 - a3 * df_dv3


def rhs_error[P: Int](var ctx: DeviceContext, nx: Int, nv: Int) raises -> Real:
    """Relative L² error of the kernel's RHS."""
    comptime State = PhaseState[1, 3, P, P, VLASOV_NC]
    var mesh = PhaseMesh[1, 3, P, P](
        ctx,
        lo=[X_MIN, -V_MAX, -V_MAX, -V_MAX],
        hi=[X_MAX, V_MAX, V_MAX, V_MAX],
        n=[nx, nv, nv, nv],
        periodic=[True],
    )
    var x_mesh = BlockMesh[1, P](ctx, X_MIN, X_MAX, nx, periodic=True)
    var em = DGState[1, P, MAXWELL_NC](ctx, nx)
    x_mesh.fill(em, fields)

    var q = State(ctx, mesh.num_phase_elements)
    var rhs = State(ctx, mesh.num_phase_elements)
    mesh.fill(q, f_exact)

    var app = VlasovMaxwell(
        charge_mass_ratio=CHARGE_MASS_RATIO, skin_depth_norm=SKIN_DEPTH_NORM,
    )
    DGVlasovSolver[1, 3, P, P](app).compute_rhs(ctx, mesh, q, em, rhs)
    ctx.synchronize()

    var err_sq: Real = 0.0
    var norm_sq: Real = 0.0
    def add(
        x: Real, v1: Real, v2: Real, v3: Real, w: Real, r: Real,
    ) {mut err_sq, mut norm_sq}:
        var exact = rhs_exact(x, v1, v2, v3)
        err_sq += w * (r - exact) * (r - exact)
        norm_sq += w * exact * exact
    mesh.visit_nodes(rhs, add)
    return sqrt(err_sq / norm_sq)


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = DeviceContext()

    # Tolerances are about 1.5x the measured errors (0.116 → 0.0306 and
    # 0.0507 → 0.0082). A Lorentz term with the wrong sign gives an error
    # of order one at every resolution.
    var lo2 = rhs_error[2](ctx, 8, 12)
    var hi2 = rhs_error[2](ctx, 16, 24)
    _ = check_convergence(
        "1D3V RHS P = 2 (8 × 12³ → 16 × 24³)",
        lo2, hi2, Real(2.0), Real(0.17), Real(0.045),
    )
    # At P = 3 these meshes are still short of the asymptotic range
    # (measured rate 2.63), hence the wider slack on the rate.
    var lo3 = rhs_error[3](ctx, 4, 8)
    var hi3 = rhs_error[3](ctx, 8, 16)
    _ = check_convergence(
        "1D3V RHS P = 3 (4 × 8³ → 8 × 16³)",
        lo3, hi3, Real(3.0), Real(0.075), Real(0.012), rate_slack=Real(0.5),
    )
    print("PASSED")
