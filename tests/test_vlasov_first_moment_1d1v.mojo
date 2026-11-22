# 1D1V Vlasov first-moment kernel test.
#
# `J(x) = ∫ v · f(x, v) dv` is the species current density / momentum
# density — the quantity that couples Vlasov back into Maxwell's source
# in a future slice.
#
# Two checks, both with `f(x, v)` x-independent:
#
#   (a) symmetric Maxwellian f(v) = exp(-v²/σ²): J = 0 by oddness of the
#       integrand. Truncation symmetry makes the GLL result machine-zero.
#
#   (b) v-shifted Maxwellian f(v) = exp(-(v-u)²/σ²): J = u · σ√π up to
#       erfc tail truncation. The first-moment tail residue is
#       (σ²/2) · [exp(-(v_max-u)²/σ²) - exp(-(v_max+u)²/σ²)], so for
#       u = 0.5, σ² = 0.5, v_max = 3 the residue is ≈ 9.3e-7.
#
# Build / run:
#     pixi run mojo run -I src tests/test_vlasov_first_moment_1d1v.mojo

from std.math import exp, erf
from std.sys import has_accelerator
from max.gpu.host import DeviceContext
from std.testing import assert_true

from warpxm.prelude import *
from warpxm.physics.vlasov import *


comptime P_X: Int = 2
comptime P_V: Int = 3
comptime NX: Int = 4
comptime NV: Int = 16
comptime X_MIN: Real = 0.0
comptime X_MAX: Real = 1.0
comptime V_MIN: Real = -3.0
comptime V_MAX: Real = 3.0
comptime SIGMA_SQ: Real = 0.5


def fill_shifted_maxwellian(
    mut q: PhaseState1D1V[P_X, P_V, VLASOV_NC],
    mesh: PhaseMesh1D1V[P_X, P_V],
    u: Real,
) raises:
    """Fill `q` with `f(x, v) = exp(-(v - u)² / σ²)` (x-independent)."""
    def shifted(x: Real, v: Real) {imm} -> Real:
        var dv_shift = v - u
        return exp(-dv_shift * dv_shift / SIGMA_SQ)
    mesh.fill(q, shifted)


def first_moment_max_err(
    var ctx: DeviceContext,
    mut moments: VlasovMoments1D1V[P_X, P_V],
    mesh: PhaseMesh1D1V[P_X, P_V],
    mut q: PhaseState1D1V[P_X, P_V, VLASOV_NC],
    j_analytic: Real,
) raises -> Real:
    var j = DGState[1, P_X, 1](ctx, NX)
    moments.first_moment(ctx, mesh, q, j)
    ctx.synchronize()
    var worst: Real = 0.0
    with j.buf.map_to_host() as h:
        for k in range(j.buf_len):
            var d = h[k] - j_analytic
            var ad = d if d >= 0.0 else -d
            if ad > worst:
                worst = ad
    return worst


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
    var moments = VlasovMoments1D1V[P_X, P_V]()

    var sigma = sqrt(SIGMA_SQ)

    # --- (a) symmetric Maxwellian: J = 0 exactly ---
    fill_shifted_maxwellian(q, mesh, Real(0.0))
    var err_sym = first_moment_max_err(ctx, moments, mesh, q, Real(0.0))
    print("first-moment 1D1V (u=0): max |J - 0| =", err_sym)
    assert_true(err_sym < Real(1.0e-15), "symmetric J should be machine zero")

    # --- (b) shifted Maxwellian: J = u · σ√π (modulo tail residue) ---
    var u_shift: Real = 0.5
    fill_shifted_maxwellian(q, mesh, u_shift)
    # Account for the asymmetric-tail correction on the first moment.
    var dvmax_lo = V_MAX - u_shift
    var dvmax_hi = V_MAX + u_shift
    var tail_correction = (SIGMA_SQ * Real(0.5)) * (
        exp(-dvmax_lo * dvmax_lo / SIGMA_SQ)
        - exp(-dvmax_hi * dvmax_hi / SIGMA_SQ)
    )
    # And the truncated bulk gives σ√π/2 · (erf((vmax-u)/σ) + erf((vmax+u)/σ)).
    var bulk = sigma * sqrt(PI) * Real(0.5) * (
        erf(dvmax_lo / sigma) + erf(dvmax_hi / sigma)
    )
    var j_analytic = u_shift * bulk - tail_correction
    var err_shift = first_moment_max_err(ctx, moments, mesh, q, j_analytic)
    print(
        "first-moment 1D1V (u=", u_shift, "): J_analytic =", j_analytic,
        "max |err| =", err_shift,
    )
    assert_true(
        err_shift < Real(5.0e-9),  # measured 8.3e-10
        "shifted-Maxwellian J quadrature too inaccurate",
    )

    print("PASSED")
