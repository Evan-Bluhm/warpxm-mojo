# 1D1V Vlasov zeroth-moment kernel test.
#
# Sets `f(x, v) = exp(-v²/σ²)` (x-independent Maxwellian) and verifies
# that
#
#   ρ(x) = ∫_{-v_max}^{+v_max} exp(-v²/σ²) dv
#        = σ √π · erf(v_max / σ)
#
# matches at every x-node. Tail truncation outside [-v_max, +v_max]
# contributes a residue of order erfc(v_max/σ), which is tiny for
# v_max = 3, σ² = 0.5 (`erf(3 / sqrt(0.5)) ≈ 1 − 4·10⁻¹⁰`), so we can
# compare against `σ√π` to ≲ 1e-9.
#
# Build / run:
#     pixi run mojo run -I src tests/test_vlasov_zeroth_moment_1d1v.mojo

from std.math import exp, erf
from std.sys import has_accelerator
from max.gpu.host import DeviceContext
from std.testing import assert_true

from warpxm.prelude import *
from warpxm.physics.vlasov import *


comptime P_X: Int = 2
comptime P_V: Int = 3   # higher P in v for the Maxwellian quadrature accuracy
comptime NX: Int = 4
comptime NV: Int = 16
comptime X_MIN: Real = 0.0
comptime X_MAX: Real = 1.0
comptime V_MIN: Real = -3.0
comptime V_MAX: Real = 3.0
comptime SIGMA_SQ: Real = 0.5


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

    # IC: f(x, v) = exp(-v²/σ²), x-independent.
    def maxwellian(x: Real, v: Real) -> Real:
        return exp(-v * v / SIGMA_SQ)
    mesh.fill(q, maxwellian)

    var rho = DGState[1, P_X, 1](ctx, NX)
    var moments = VlasovMoments1D1V[P_X, P_V]()
    moments.zeroth_moment(ctx, mesh, q, rho)
    ctx.synchronize()

    # Analytic: ρ = σ √π · erf(v_max / σ).
    var sigma = sqrt(SIGMA_SQ)
    var rho_analytic = sigma * sqrt(PI) * erf(V_MAX / sigma)

    var max_abs_err: Real = 0.0
    with rho.buf.map_to_host() as h:
        for k in range(rho.buf_len):
            var d = h[k] - rho_analytic
            var ad = d if d >= 0.0 else -d
            if ad > max_abs_err:
                max_abs_err = ad

    print(
        "zeroth-moment 1D1V: ρ_analytic =", rho_analytic,
        "max |err| =", max_abs_err,
    )

    # GLL quadrature with P_v=3 over NV=16 cells is highly accurate for
    # a Gaussian; measured error 6.9e-12.
    assert_true(max_abs_err < Real(5.0e-11), "ρ quadrature too inaccurate")
    print("PASSED")
