# 1D3V velocity moments against a drifting tri-Maxwellian.
#
# `f = n(x) / ((2π)^{3/2} σ_1 σ_2 σ_3) · exp(−Σ_d (v_d − u_d(x))² / 2σ_d²)`
# has the moments
#
#     ∫ f dv        = n
#     ∫ v_d f dv    = n u_d
#     ½ ∫ |v|² f dv = ½ n Σ_d (u_d² + σ_d²)
#
# with a different drift and temperature on each velocity axis and a
# density and drift that vary in x, so a moment written to the wrong
# component or the wrong x node, or a velocity axis read for another, is
# an O(1) error. The velocity mesh has a different element count on each
# axis for the same reason. All five moments come from one kernel pass
# (`VlasovMoments.compute`).
#
# Build / run:
#     pixi run mojo run -I src tests/test_vlasov_moments_1d3v.mojo

from std.collections import Array
from std.math import cos, exp, sin
from std.sys import has_accelerator
from max.gpu.host import DeviceContext
from std.testing import assert_true

from warpxm.prelude import *
from warpxm.physics.vlasov import *


comptime P_X: Int = 2
comptime P_V: Int = 2
comptime NX: Int = 5
comptime NV1: Int = 16
comptime NV2: Int = 18
comptime NV3: Int = 20
comptime X_MIN: Real = 0.0
comptime X_MAX: Real = 1.0
comptime V_MAX: Real = 6.0
comptime SIGMA1: Real = 0.8
comptime SIGMA2: Real = 0.9
comptime SIGMA3: Real = 0.7


def density(x: Real) -> Real:
    return 1.0 + 0.4 * sin(TWO_PI * x)


def drift1(x: Real) -> Real:
    return 0.5 * cos(TWO_PI * x)


def drift2(x: Real) -> Real:
    return -0.3 + 0.2 * sin(TWO_PI * x)


def drift3(x: Real) -> Real:
    return 0.4 - 0.1 * cos(TWO_PI * x)


def maxwellian(x: Real, v1: Real, v2: Real, v3: Real) -> Real:
    var w1 = (v1 - drift1(x)) / SIGMA1
    var w2 = (v2 - drift2(x)) / SIGMA2
    var w3 = (v3 - drift3(x)) / SIGMA3
    var norm = density(x) / (
        TWO_PI * sqrt(TWO_PI) * SIGMA1 * SIGMA2 * SIGMA3
    )
    return norm * exp(-0.5 * (w1 * w1 + w2 * w2 + w3 * w3))


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = DeviceContext()

    var mesh = PhaseMesh[1, 3, P_X, P_V](
        ctx,
        lo=[X_MIN, -V_MAX, -V_MAX, -V_MAX],
        hi=[X_MAX, V_MAX, V_MAX, V_MAX],
        n=[NX, NV1, NV2, NV3],
        periodic=[True],
    )
    var x_mesh = BlockMesh[1, P_X](ctx, X_MIN, X_MAX, NX, periodic=True)
    var q = PhaseState[1, 3, P_X, P_V, VLASOV_NC](ctx, mesh.num_phase_elements)
    var mom = DGState[1, P_X, VLASOV_MOMENT_NC](ctx, NX)
    mesh.fill(q, maxwellian)

    VlasovMoments[1, 3, P_X, P_V](ctx, mesh).compute(ctx, mesh, q, mom)
    ctx.synchronize()

    var err = Array[Real, VLASOV_MOMENT_NC](fill=Real(0.0))
    def check(x: Real, w: Real, m: Array[Real, VLASOV_MOMENT_NC]) {mut err}:
        var n = density(x)
        var u1 = drift1(x)
        var u2 = drift2(x)
        var u3 = drift3(x)
        var energy = 0.5 * n * (
            u1 * u1 + u2 * u2 + u3 * u3
            + SIGMA1 * SIGMA1 + SIGMA2 * SIGMA2 + SIGMA3 * SIGMA3
        )
        err[MOM_DENSITY] = fmax(err[MOM_DENSITY], fabs(m[MOM_DENSITY] - n))
        err[MOM_FLUX_X] = fmax(err[MOM_FLUX_X], fabs(m[MOM_FLUX_X] - n * u1))
        err[MOM_FLUX_Y] = fmax(err[MOM_FLUX_Y], fabs(m[MOM_FLUX_Y] - n * u2))
        err[MOM_FLUX_Z] = fmax(err[MOM_FLUX_Z], fabs(m[MOM_FLUX_Z] - n * u3))
        err[MOM_ENERGY] = fmax(err[MOM_ENERGY], fabs(m[MOM_ENERGY] - energy))
    x_mesh.visit_nodes(mom, check)

    print(
        "1D3V moments, max nodal error: n =", err[MOM_DENSITY],
        " n u_x =", err[MOM_FLUX_X], " n u_y =", err[MOM_FLUX_Y],
        " n u_z =", err[MOM_FLUX_Z], " energy =", err[MOM_ENERGY],
    )
    # Measured: 3.3e-10, 8.2e-10, 1.9e-9, 1.5e-10, 5.7e-9 (the GLL
    # quadrature error of a Gaussian about one element wide).
    assert_true(err[MOM_DENSITY] < Real(1.0e-9), "density is off")
    assert_true(err[MOM_FLUX_X] < Real(6.0e-9), "x flux is off")
    assert_true(err[MOM_FLUX_Y] < Real(6.0e-9), "y flux is off")
    assert_true(err[MOM_FLUX_Z] < Real(6.0e-9), "z flux is off")
    assert_true(err[MOM_ENERGY] < Real(2.0e-8), "energy is off")

    # The total mass seen by the moment kernel and by the host quadrature
    # are the same sum of 2.3 million terms in a different order (measured
    # difference 1.5e-12).
    var mass_moment = x_mesh.integrate(mom, MOM_DENSITY)
    var mass_phase = mesh.integrate(q)
    print("  mass: moments =", mass_moment, " phase quadrature =", mass_phase)
    assert_true(
        abs(mass_moment - mass_phase) < Real(1.0e-10), "mass mismatch",
    )
    print("PASSED")
