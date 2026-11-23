# 2D2V Vlasov: right-hand side against the analytic operator, and moments.
#
# The kinetic stack is generic over `[XDIM, VDIM]`. With two configuration
# axes it has to (i) advect along y at `v_2`, (ii) look the fields up on a
# `BlockMesh[2, P]`, whose elements are numbered x-innermost while the
# phase mesh numbers its axes last-innermost, and (iii) write the moments
# back in the `BlockMesh` numbering. None of that is exercised in 1D.
#
# (a) RHS. For smooth `f(x, y, v1, v2)` and fields varying in x and y the
#     residual must converge (order P) to
#
#         −v1 ∂f/∂x − v2 ∂f/∂y − a_1 ∂f/∂v1 − a_2 ∂f/∂v2,
#         a_1 = α (E_x + v_2 B_z),  a_2 = α (E_y − v_1 B_z).
#
#     x is periodic and y is a copy-out boundary; the element counts
#     differ on every axis, so a transposed index reads the wrong field.
#
# (b) Moments of a drifting Maxwellian with `n(x, y)`, `u(x, y)`.
#
# Build / run:
#     pixi run mojo run -I src tests/test_vlasov_rhs_consistency_2d2v.mojo

from std.collections import Array
from std.math import cos, exp, sin
from std.sys import has_accelerator
from max.gpu.host import DeviceContext
from std.testing import assert_true

from warpxm.prelude import *
from warpxm.physics.vlasov import *
from warpxm.physics.maxwell import *
from warpxm.testing.convergence import check_convergence


comptime P: Int = 2
comptime X_MIN: Real = 0.0
comptime X_MAX: Real = 1.0
comptime Y_MIN: Real = -0.5
comptime Y_MAX: Real = 1.0
comptime V_MAX: Real = 4.5
comptime INV_SIGMA_SQ: Real = 2.0
comptime U1: Real = 0.3
comptime U2: Real = -0.2
comptime KY: Real = 1.7
comptime K1: Real = 0.3
comptime K2: Real = -0.4
comptime AMP: Real = 0.3
comptime ALPHA: Real = -1.4

comptime State = PhaseState[2, 2, P, P, VLASOV_NC]


def fields(x: Real, y: Real) -> Array[Real, MAXWELL_NC]:
    var v = Array[Real, MAXWELL_NC](fill=Real(0.0))
    v[IDX_EX] = 0.7 * sin(TWO_PI * x) * cos(2.0 * y) + 0.2
    v[IDX_EY] = -0.4 * cos(TWO_PI * x) + 0.3 * y
    v[IDX_BZ] = 0.6 * sin(TWO_PI * x + y) - 0.3
    # Components a 2V species must not feel.
    v[IDX_EZ] = 5.0
    v[IDX_BX] = -7.0
    v[IDX_BY] = 3.0
    return v^


def f_exact(x: Real, y: Real, v1: Real, v2: Real) -> Real:
    var phase = TWO_PI * x + KY * y + K1 * v1 + K2 * v2
    var w1 = v1 - U1
    var w2 = v2 - U2
    return (1.0 + AMP * sin(phase)) * exp(-0.5 * INV_SIGMA_SQ * (w1 * w1 + w2 * w2))


def rhs_exact(x: Real, y: Real, v1: Real, v2: Real) -> Real:
    var phase = TWO_PI * x + KY * y + K1 * v1 + K2 * v2
    var w1 = v1 - U1
    var w2 = v2 - U2
    var maxw = exp(-0.5 * INV_SIGMA_SQ * (w1 * w1 + w2 * w2))
    var g = 1.0 + AMP * sin(phase)
    var dg = AMP * cos(phase)
    var em = fields(x, y)
    var a1 = ALPHA * (em[IDX_EX] + v2 * em[IDX_BZ])
    var a2 = ALPHA * (em[IDX_EY] - v1 * em[IDX_BZ])
    return (
        -v1 * TWO_PI * dg * maxw
        - v2 * KY * dg * maxw
        - a1 * (K1 * dg - g * w1 * INV_SIGMA_SQ) * maxw
        - a2 * (K2 * dg - g * w2 * INV_SIGMA_SQ) * maxw
    )


def make_mesh(
    var ctx: DeviceContext, nx: Int, ny: Int, nv1: Int, nv2: Int,
) raises -> PhaseMesh[2, 2, P, P]:
    return PhaseMesh[2, 2, P, P](
        ctx,
        lo=[X_MIN, Y_MIN, -V_MAX, -V_MAX],
        hi=[X_MAX, Y_MAX, V_MAX, V_MAX],
        n=[nx, ny, nv1, nv2],
        periodic=[True, False],
    )


def rhs_error(
    var ctx: DeviceContext, nx: Int, ny: Int, nv1: Int, nv2: Int,
) raises -> Real:
    var mesh = make_mesh(ctx, nx, ny, nv1, nv2)
    var x_mesh = BlockMesh[2, P](
        ctx, X_MIN, X_MAX, Y_MIN, Y_MAX, nx, ny,
        periodic_x=True, periodic_y=False,
    )
    var em = DGState[2, P, MAXWELL_NC](ctx, x_mesh.num_elements)
    x_mesh.fill(em, fields)
    var q = State(ctx, mesh.num_phase_elements)
    var rhs = State(ctx, mesh.num_phase_elements)
    mesh.fill(q, f_exact)

    var app = VlasovMaxwell(charge_mass_ratio=ALPHA, skin_depth_norm=Real(1.0))
    DGVlasovSolver[2, 2, P, P](app).compute_rhs(ctx, mesh, q, em, rhs)
    ctx.synchronize()

    var err_sq: Real = 0.0
    var norm_sq: Real = 0.0
    def add(
        x: Real, y: Real, v1: Real, v2: Real, w: Real, r: Real,
    ) {mut err_sq, mut norm_sq}:
        var exact = rhs_exact(x, y, v1, v2)
        err_sq += w * (r - exact) * (r - exact)
        norm_sq += w * exact * exact
    mesh.visit_nodes(rhs, add)
    return sqrt(err_sq / norm_sq)


# --- moments ---------------------------------------------------------------
comptime SIGMA1: Real = 0.8
comptime SIGMA2: Real = 0.9


def density(x: Real, y: Real) -> Real:
    return 1.0 + 0.4 * sin(TWO_PI * x) + 0.3 * y


def drift1(x: Real, y: Real) -> Real:
    return 0.5 * cos(TWO_PI * x) - 0.2 * y


def drift2(x: Real, y: Real) -> Real:
    return -0.3 + 0.2 * sin(TWO_PI * x) * y


def maxwellian(x: Real, y: Real, v1: Real, v2: Real) -> Real:
    var w1 = (v1 - drift1(x, y)) / SIGMA1
    var w2 = (v2 - drift2(x, y)) / SIGMA2
    return density(x, y) / (TWO_PI * SIGMA1 * SIGMA2) * exp(
        -0.5 * (w1 * w1 + w2 * w2)
    )


def check_moments(var ctx: DeviceContext) raises:
    comptime NX = 5
    comptime NY = 3
    var mesh = PhaseMesh[2, 2, P, P](
        ctx,
        lo=[X_MIN, Y_MIN, -6.0, -6.0],
        hi=[X_MAX, Y_MAX, 6.0, 6.0],
        n=[NX, NY, 16, 18],
        periodic=[True, False],
    )
    var x_mesh = BlockMesh[2, P](
        ctx, X_MIN, X_MAX, Y_MIN, Y_MAX, NX, NY,
        periodic_x=True, periodic_y=False,
    )
    var q = State(ctx, mesh.num_phase_elements)
    var mom = DGState[2, P, VLASOV_MOMENT_NC](ctx, x_mesh.num_elements)
    mesh.fill(q, maxwellian)
    VlasovMoments[2, 2, P, P](ctx, mesh).compute(ctx, mesh, q, mom)
    ctx.synchronize()

    var err = Array[Real, VLASOV_MOMENT_NC](fill=Real(0.0))
    def check(
        x: Real, y: Real, w: Real, m: Array[Real, VLASOV_MOMENT_NC],
    ) {mut err}:
        var n = density(x, y)
        var u1 = drift1(x, y)
        var u2 = drift2(x, y)
        var energy = 0.5 * n * (
            u1 * u1 + u2 * u2 + SIGMA1 * SIGMA1 + SIGMA2 * SIGMA2
        )
        err[MOM_DENSITY] = fmax(err[MOM_DENSITY], fabs(m[MOM_DENSITY] - n))
        err[MOM_FLUX_X] = fmax(err[MOM_FLUX_X], fabs(m[MOM_FLUX_X] - n * u1))
        err[MOM_FLUX_Y] = fmax(err[MOM_FLUX_Y], fabs(m[MOM_FLUX_Y] - n * u2))
        err[MOM_FLUX_Z] = fmax(err[MOM_FLUX_Z], fabs(m[MOM_FLUX_Z]))
        err[MOM_ENERGY] = fmax(err[MOM_ENERGY], fabs(m[MOM_ENERGY] - energy))
    x_mesh.visit_nodes(mom, check)
    print(
        "2D2V moments, max nodal error: n =", err[MOM_DENSITY],
        " n u_x =", err[MOM_FLUX_X], " n u_y =", err[MOM_FLUX_Y],
        " n u_z =", err[MOM_FLUX_Z], " energy =", err[MOM_ENERGY],
    )
    # Measured: 5.6e-10, 1.1e-9, 2.9e-9, 0, 1.0e-8.
    assert_true(err[MOM_DENSITY] < Real(2.0e-9), "density is off")
    assert_true(err[MOM_FLUX_X] < Real(4.0e-9), "x flux is off")
    assert_true(err[MOM_FLUX_Y] < Real(9.0e-9), "y flux is off")
    assert_true(err[MOM_FLUX_Z] == Real(0.0), "z flux of a 2V species is not 0")
    assert_true(err[MOM_ENERGY] < Real(3.0e-8), "energy is off")


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = DeviceContext()

    # Measured 0.0984 → 0.0261 (rate 1.92); tolerances are 1.5x that.
    var lo = rhs_error(ctx, 8, 6, 12, 14)
    var hi = rhs_error(ctx, 16, 12, 24, 28)
    _ = check_convergence(
        "2D2V RHS P = 2 (8 × 6 × 12 × 14, then doubled)",
        lo, hi, Real(2.0), Real(0.15), Real(0.04),
    )
    check_moments(ctx)
    print("PASSED")
