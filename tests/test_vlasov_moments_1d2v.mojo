# 1D2V Vlasov moment-kernel test.
#
# Sets `f(x, v1, v2) = exp(-(v1² + v2²) / σ²)` (x-independent 2D
# Maxwellian) and verifies the moments from `VlasovMoments[1, 2, P_X,
# P_V].compute` against analytic values:
#
#   ρ     (MOM_DENSITY) = (σ √π · erf(V/σ))²
#   J_x   (MOM_FLUX_X)  = 0     (odd integrand symmetry along v1)
#   J_y   (MOM_FLUX_Y)  = 0     (odd integrand symmetry along v2)
#   J_z   (MOM_FLUX_Z)  = 0     (no v3 axis in 1D2V; must be exactly zero)
#   E_kin (MOM_ENERGY)  = ((σ³√π/2)·erf(V/σ) − σ²·V·exp(-V²/σ²)) · (σ√π·erf(V/σ))
#
# Tolerances follow the slice-13/14/16 erfc budget: for σ² = 0.5,
# V = 3 (V/σ = 3√2 ≈ 4.243), `erfc(3√2) ≈ 1.5·10⁻⁹` and the second-
# moment truncation `σ²·V·exp(-V²/σ²) ≈ 2.3·10⁻⁸`. Use generous
# absolute tolerances that absorb both the tail residue and any
# remaining GLL quadrature error.
#
# Build / run:
#     pixi run mojo run -I src tests/test_vlasov_moments_1d2v.mojo

from std.math import exp, erf
from std.sys import has_accelerator
from max.gpu.host import DeviceContext
from std.testing import assert_true

from warpxm.prelude import *
from warpxm.physics.vlasov import *


comptime P_X: Int = 1
comptime P_V: Int = 3
comptime NX: Int = 4
comptime NV1: Int = 12
comptime NV2: Int = 12
comptime X_MIN: Real = 0.0
comptime X_MAX: Real = 1.0
comptime V1_MIN: Real = -3.0
comptime V1_MAX: Real = 3.0
comptime V2_MIN: Real = -3.0
comptime V2_MAX: Real = 3.0
comptime SIGMA_SQ: Real = 0.5
comptime MomentState = DGState[1, P_X, VLASOV_MOMENT_NC]


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"

    var ctx = DeviceContext()
    var mesh = PhaseMesh[1, 2, P_X, P_V](
        ctx,
        lo=[X_MIN, V1_MIN, V2_MIN],
        hi=[X_MAX, V1_MAX, V2_MAX],
        n=[NX, NV1, NV2],
        periodic=[True],
    )

    var q = PhaseState[1, 2, P_X, P_V, VLASOV_NC](
        ctx, mesh.num_phase_elements,
    )

    # IC: f(x, v1, v2) = exp(-(v1² + v2²) / σ²), x-independent.
    def maxwellian(x: Real, v1: Real, v2: Real) -> Real:
        return exp(-(v1 * v1 + v2 * v2) / SIGMA_SQ)
    mesh.fill(q, maxwellian)

    var mom = MomentState(ctx, NX)
    var moments = VlasovMoments[1, 2, P_X, P_V](ctx, mesh)
    moments.compute(ctx, mesh, q, mom)
    ctx.synchronize()

    # Analytic values for the truncated 2D Maxwellian.
    var sigma = sqrt(SIGMA_SQ)
    var V_over_sigma = V1_MAX / sigma  # = V2_MAX / sigma by construction
    var erf_term = erf(V_over_sigma)
    var rho_axis = sigma * sqrt(PI) * erf_term
    var rho_analytic = rho_axis * rho_axis
    var second_axis = (
        sigma * SIGMA_SQ * sqrt(PI) * Real(0.5) * erf_term
        - SIGMA_SQ * V1_MAX * exp(-V1_MAX * V1_MAX / SIGMA_SQ)
    )
    # E_kin = ∫ v1² f dv1 dv2 · 1 + symmetric in v2 ⇒ both axes equal,
    #         so the ½ in ½(v1² + v2²) folds back to 1·second_axis·rho_axis.
    var e_analytic = second_axis * rho_axis

    var max_rho_err: Real = 0.0
    var max_jx: Real = 0.0
    var max_jy: Real = 0.0
    var max_jz: Real = 0.0
    var max_e_err: Real = 0.0
    with mom.buf.map_to_host() as h:
        for e in range(NX):
            for n in range(P_X + 1):
                max_rho_err = fmax(
                    max_rho_err,
                    abs(h[MomentState.index(e, n, MOM_DENSITY)] - rho_analytic),
                )
                max_jx = fmax(
                    max_jx, abs(h[MomentState.index(e, n, MOM_FLUX_X)]),
                )
                max_jy = fmax(
                    max_jy, abs(h[MomentState.index(e, n, MOM_FLUX_Y)]),
                )
                max_jz = fmax(
                    max_jz, abs(h[MomentState.index(e, n, MOM_FLUX_Z)]),
                )
                max_e_err = fmax(
                    max_e_err,
                    abs(h[MomentState.index(e, n, MOM_ENERGY)] - e_analytic),
                )

    print(
        "1D2V moments: ρ_analytic =", rho_analytic,
        "max |err| =", max_rho_err,
    )
    print(
        "1D2V moments: E_analytic =", e_analytic,
        "max |err| =", max_e_err,
    )
    print(
        "1D2V moments: max |J_x| =", max_jx,
        " max |J_y| =", max_jy, " max |J_z| =", max_jz,
    )

    # Tolerances. ρ-axis truncation residue ≈ σ √π · erfc(V/σ) ≈ 1.4·10⁻⁹;
    # combined into ρ ≈ 2·rho_axis·that ≈ 3.9·10⁻⁹ (measured 4.2e-10). E_kin truncation σ²V·exp(-V²/σ²) ≈ 2.3·10⁻⁸ on
    # one axis multiplied by `rho_axis` ≈ 1.25 — call it 5e-8 (measured 2.6e-9).
    # J components should be at machine eps.
    assert_true(max_rho_err < Real(4.0e-9), "ρ moment off")
    assert_true(max_jx < Real(1.0e-15), "J_x not zero by symmetry")
    assert_true(max_jy < Real(1.0e-15), "J_y not zero by symmetry")
    assert_true(max_jz == Real(0.0), "J_z not zero without a v3 axis")
    assert_true(max_e_err < Real(2.5e-8), "E_kin moment off")
    print("PASSED")
