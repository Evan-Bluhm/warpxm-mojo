# 1D1V Vlasov second-moment test: the `MOM_ENERGY` component of
# `VlasovMoments[1, 1, P_x, P_v]`.
#
# `E_kin(x) = ½ ∫ v² · f(x, v) dv` is the species energy density;
# combined with ρ = ∫ f dv and u = (∫ v f dv) / ρ it produces the
# bulk temperature `T = (2 · E_kin / ρ) - u²`.
#
# For `f(x, v) = exp(-v² / σ²)` (x-independent Maxwellian, NOT
# normalised — same IC the zeroth-moment test uses), the truncated
# analytic answer is
#
#     E_kin = ½ · σ³ · [(√π/2) · erf(v_max/σ) − (v_max/σ) · exp(−v_max²/σ²)]
#
# For σ² = 0.5, v_max = 3 the second term is ≈ 6.4e-8 (negligible) and
# the first is ≈ σ³ · √π / 2 to 1 part in 10⁹. Final value ≈ 0.157.
#
# Build / run:
#     pixi run mojo run -I src tests/test_vlasov_second_moment_1d1v.mojo

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
comptime SIGMA_SQ: Real = 0.5
comptime V_MAX: Real = 3.0


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"

    var ctx = DeviceContext()
    var mesh = PhaseMesh[1, 1, P_X, P_V](
        ctx, lo=[0.0, -V_MAX], hi=[1.0, V_MAX], n=[NX, NV],
        periodic=[True],
    )

    var q = PhaseState[1, 1, P_X, P_V, VLASOV_NC](
        ctx, mesh.num_phase_elements,
    )

    def maxwellian(x: Real, v: Real) -> Real:
        return exp(-v * v / SIGMA_SQ)
    mesh.fill(q, maxwellian)

    var mom = DGState[1, P_X, VLASOV_MOMENT_NC](ctx, mesh.num_x_elements)
    var moments = VlasovMoments[1, 1, P_X, P_V](ctx, mesh)
    moments.compute(ctx, mesh, q, mom)
    ctx.synchronize()

    var sigma = sqrt(SIGMA_SQ)
    var a = V_MAX / sigma
    var bulk_int = (sqrt(PI) * Real(0.5)) * erf(a) - a * exp(-a * a)
    # `bulk_int = ∫_{-a}^{a} u² exp(-u²) du`; rescale to v.
    var e_kin_analytic = Real(0.5) * sigma * sigma * sigma * bulk_int

    var max_abs_err: Real = 0.0
    with mom.buf.map_to_host() as h:
        for k in range(mom.total_nodes):
            var d = h[k * VLASOV_MOMENT_NC + MOM_ENERGY] - e_kin_analytic
            var ad = d if d >= 0.0 else -d
            if ad > max_abs_err:
                max_abs_err = ad

    print(
        "second-moment 1D1V: E_kin_analytic =", e_kin_analytic,
        "max |err| =", max_abs_err,
    )

    # GLL quadrature on a v²-weighted Gaussian is highly accurate; the
    # tolerance also covers the erfc(4.24) ≈ 1.5e-9 tail truncation
    # residue.
    assert_true(
        max_abs_err < Real(2.0e-10),  # measured 2.3e-11
        "second-moment quadrature too inaccurate",
    )
    print("PASSED")
