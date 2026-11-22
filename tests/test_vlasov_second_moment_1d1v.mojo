# 1D1V Vlasov second-moment kernel test.
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

from std.math import exp, sqrt, erf
from std.sys import has_accelerator
from max.gpu.host import DeviceContext
from std.testing import assert_true

from warpxm.apps.vlasovmaxwell import VLASOV_NC
from warpxm.basis.nodal_basis_1d import NodalBasis1D
from warpxm.core.types import Real, PI
from warpxm.mesh.phase_mesh_1d1v import PhaseMesh1D1V
from warpxm.spatial.vlasov_moments_1d1v import VlasovMoments1D1V
from warpxm.variable.phase_state_1d1v import PhaseState1D1V
from warpxm.variable.state_1d import DGState1D


comptime P_X: Int = 2
comptime P_V: Int = 3
comptime NX: Int = 4
comptime NV: Int = 16
comptime SIGMA_SQ: Real = 0.5
comptime V_MAX: Real = 3.0


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"

    var ctx = DeviceContext()
    var mesh = PhaseMesh1D1V[P_X, P_V](
        ctx,
        Real(0.0), Real(1.0), Real(-V_MAX), V_MAX,
        num_x=NX, num_v=NV,
        periodic_x=True,
    )

    var q = PhaseState1D1V[P_X, P_V, VLASOV_NC](ctx, NX, NV)

    var ref_v_nodes = NodalBasis1D[P_V].nodes()
    var host = ctx.enqueue_create_host_buffer[Real.dtype](q.buf_len)
    for ex in range(NX):
        for ev in range(NV):
            var vl = -V_MAX + Real(ev) * mesh.dv
            for ix in range(P_X + 1):
                for iv in range(P_V + 1):
                    var v = vl + (Real(1.0) + ref_v_nodes[iv]) * mesh.jacobian_v
                    var phase_e = ex * NV + ev
                    var idx = (
                        (phase_e * (P_V + 1) + iv) * (P_X + 1) + ix
                    ) * VLASOV_NC
                    host[idx] = exp(-v * v / SIGMA_SQ)
    ctx.enqueue_copy(dst_buf=q.buf, src_buf=host)
    ctx.synchronize()

    var e_dens = DGState1D[P_X, 1](ctx, NX)
    var moments = VlasovMoments1D1V[P_X, P_V]()
    moments.second_moment(ctx, mesh, q, e_dens)
    ctx.synchronize()

    var sigma = sqrt(SIGMA_SQ)
    var a = V_MAX / sigma
    var bulk_int = (sqrt(PI) * Real(0.5)) * erf(a) - a * exp(-a * a)
    # `bulk_int = ∫_{-a}^{a} u² exp(-u²) du`; rescale to v.
    var e_kin_analytic = Real(0.5) * sigma * sigma * sigma * bulk_int

    var max_abs_err: Real = 0.0
    with e_dens.buf.map_to_host() as h:
        for k in range(e_dens.buf_len):
            var d = h[k] - e_kin_analytic
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
