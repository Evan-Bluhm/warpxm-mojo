# 1D2V Vlasov moment-kernel test.
#
# Sets `f(x, v1, v2) = exp(-(v1² + v2²) / σ²)` (x-independent 2D
# Maxwellian) and verifies the four moment kernels against analytic
# values:
#
#   ρ           = (σ √π · erf(V/σ))²
#   J_x         = 0     (odd integrand symmetry along v1)
#   J_y         = 0     (odd integrand symmetry along v2)
#   E_kin       = ((σ³√π/2)·erf(V/σ) − σ²·V·exp(-V²/σ²)) · (σ√π·erf(V/σ))
#
# Tolerances follow the slice-13/14/16 erfc budget: for σ² = 0.5,
# V = 3 (V/σ = 3√2 ≈ 4.243), `erfc(3√2) ≈ 1.5·10⁻⁹` and the second-
# moment truncation `σ²·V·exp(-V²/σ²) ≈ 2.3·10⁻⁸`. Use generous
# absolute tolerances that absorb both the tail residue and any
# remaining GLL quadrature error.
#
# Build / run:
#     pixi run mojo run -I src tests/test_vlasov_moments_1d2v.mojo

from std.math import exp, sqrt, erf
from std.sys import has_accelerator
from max.gpu.host import DeviceContext
from std.testing import assert_true

from warpxm.apps.vlasovmaxwell import VLASOV_NC
from warpxm.basis.nodal_basis_1d import NodalBasis1D
from warpxm.core.types import Real, PI
from warpxm.mesh.phase_mesh_1d2v import PhaseMesh1D2V
from warpxm.spatial.vlasov_moments_1d2v import VlasovMoments1D2V
from warpxm.variable.phase_state_1d2v import PhaseState1D2V
from warpxm.variable.state_1d import DGState1D


comptime P_X: Int = 1
comptime P_V1: Int = 3
comptime P_V2: Int = 3
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


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"

    var ctx = DeviceContext()
    var mesh = PhaseMesh1D2V[P_X, P_V1, P_V2](
        ctx,
        X_MIN, X_MAX,
        V1_MIN, V1_MAX,
        V2_MIN, V2_MAX,
        num_x=NX, num_v1=NV1, num_v2=NV2,
        periodic_x=True,
    )

    var q = PhaseState1D2V[P_X, P_V1, P_V2, VLASOV_NC](ctx, NX, NV1, NV2)

    # IC: f(x, v1, v2) = exp(-(v1² + v2²) / σ²), x-independent.
    var ref_v1_nodes = NodalBasis1D[P_V1].nodes()
    var ref_v2_nodes = NodalBasis1D[P_V2].nodes()
    var host = ctx.enqueue_create_host_buffer[Real.dtype](q.buf_len)
    var nodes_per_phase = (P_X + 1) * (P_V1 + 1) * (P_V2 + 1)
    for ex in range(NX):
        for ev1 in range(NV1):
            var v1l = V1_MIN + Real(ev1) * mesh.dv1
            for ev2 in range(NV2):
                var v2l = V2_MIN + Real(ev2) * mesh.dv2
                var phase_e = (ex * NV1 + ev1) * NV2 + ev2
                for iv2 in range(P_V2 + 1):
                    var v2 = (
                        v2l
                        + (Real(1.0) + ref_v2_nodes[iv2]) * mesh.jacobian_v2
                    )
                    for iv1 in range(P_V1 + 1):
                        var v1 = (
                            v1l
                            + (Real(1.0) + ref_v1_nodes[iv1])
                              * mesh.jacobian_v1
                        )
                        for ix in range(P_X + 1):
                            var local_n = (
                                (iv2 * (P_V1 + 1) + iv1) * (P_X + 1) + ix
                            )
                            var idx = (
                                phase_e * nodes_per_phase + local_n
                            ) * VLASOV_NC
                            host[idx] = exp(-(v1 * v1 + v2 * v2) / SIGMA_SQ)
    ctx.enqueue_copy(dst_buf=q.buf, src_buf=host)
    ctx.synchronize()

    var rho = DGState1D[P_X, 1](ctx, NX)
    var jx = DGState1D[P_X, 1](ctx, NX)
    var jy = DGState1D[P_X, 1](ctx, NX)
    var e_dens = DGState1D[P_X, 1](ctx, NX)
    var moments = VlasovMoments1D2V[P_X, P_V1, P_V2]()
    moments.zeroth_moment(ctx, mesh, q, rho)
    moments.first_moment_v1(ctx, mesh, q, jx)
    moments.first_moment_v2(ctx, mesh, q, jy)
    moments.second_moment(ctx, mesh, q, e_dens)
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
    var max_e_err: Real = 0.0
    with rho.buf.map_to_host() as h_rho, \
         jx.buf.map_to_host() as h_jx, \
         jy.buf.map_to_host() as h_jy, \
         e_dens.buf.map_to_host() as h_e:
        for k in range(rho.buf_len):
            var dr = h_rho[k] - rho_analytic
            var adr = dr if dr >= 0.0 else -dr
            if adr > max_rho_err:
                max_rho_err = adr
            var ajx = h_jx[k] if h_jx[k] >= 0.0 else -h_jx[k]
            if ajx > max_jx:
                max_jx = ajx
            var ajy = h_jy[k] if h_jy[k] >= 0.0 else -h_jy[k]
            if ajy > max_jy:
                max_jy = ajy
            var de = h_e[k] - e_analytic
            var ade = de if de >= 0.0 else -de
            if ade > max_e_err:
                max_e_err = ade

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
        " max |J_y| =", max_jy,
    )

    # Tolerances. ρ-axis truncation residue ≈ σ √π · erfc(V/σ) ≈ 1.4·10⁻⁹;
    # combined into ρ ≈ 2·rho_axis·that ≈ 3.9·10⁻⁹ (measured 4.2e-10). E_kin truncation σ²V·exp(-V²/σ²) ≈ 2.3·10⁻⁸ on
    # one axis multiplied by `rho_axis` ≈ 1.25 — call it 5e-8 (measured 2.6e-9).
    # J components should be at machine eps.
    assert_true(max_rho_err < Real(4.0e-9), "ρ moment off")
    assert_true(max_jx < Real(1.0e-15), "J_x not zero by symmetry")
    assert_true(max_jy < Real(1.0e-15), "J_y not zero by symmetry")
    assert_true(max_e_err < Real(2.5e-8), "E_kin moment off")
    print("PASSED")
