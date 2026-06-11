# 5-moment Euler → 1D1V kinetic initialiser test.
#
# Build a uniform 5-moment Euler state with prescribed (ρ, u_x, T):
#   ρ = 1.0, u_x = 0.0, T = 0.5  ⇒  e = ½ρu² + p/(γ−1) = T/(γ−1)
#                                         = 0.5/(2/3) = 0.75   (γ = 5/3)
# Apply `apply_5moment_to_kinetic_1d1v` to populate the phase state
# with a Maxwellian, then verify via the three Vlasov moment kernels
# that the kinetic state recovers:
#   * n  = ρ/m       (zeroth moment matches ρ when m=1)
#   * J  = ρu_x/m    (first moment matches ρu_x when m=1)
#   * ½∫v² f dv = ½·V_th²·n = T/2  (kinetic energy in one v-DoF,
#                                     half of fluid's 3-DoF (3/2)·n·T)
#
# Build / run:
#     pixi run mojo run -I src tests/test_vlasov_5moment_to_kinetic_1d1v.mojo

from std.sys import has_accelerator
from std.gpu.host import DeviceContext
from std.testing import assert_true

from warpxm.apps.euler import EULER_NC, IDX_RHO, IDX_MX, IDX_MY, IDX_MZ, IDX_E
from warpxm.apps.vlasovmaxwell import VLASOV_NC
from warpxm.core.types import Real
from warpxm.mesh.phase_mesh_1d1v import PhaseMesh1D1V
from warpxm.spatial.vlasov_5moment_to_kinetic_1d1v import (
    apply_5moment_to_kinetic_1d1v,
)
from warpxm.spatial.vlasov_moments_1d1v import VlasovMoments1D1V
from warpxm.variable.phase_state_1d1v import PhaseState1D1V
from warpxm.variable.state_1d import DGState1D


comptime P_X: Int = 2
comptime P_V: Int = 3
comptime NX: Int = 4
comptime NV: Int = 32
comptime V_MIN: Real = -3.0
comptime V_MAX: Real = 3.0
comptime MASS: Real = 1.0
comptime GAMMA: Real = 5.0 / 3.0

# Target fluid moments.
comptime RHO_TARGET: Real = 1.0
comptime UX_TARGET: Real = 0.0
comptime T_TARGET: Real = 0.5
# e = ½ρu² + ρT/(γ−1) (ideal gas, no transverse momentum).
comptime E_TARGET: Real = (
    Real(0.5) * RHO_TARGET * UX_TARGET * UX_TARGET
    + RHO_TARGET * T_TARGET / (GAMMA - Real(1.0))
)


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = DeviceContext()

    var mesh = PhaseMesh1D1V[P_X, P_V](
        ctx,
        Real(0.0), Real(1.0), V_MIN, V_MAX,
        num_x=NX, num_v=NV,
        periodic_x=True,
    )

    # Build a uniform fluid IC.
    var fluid = DGState1D[P_X, EULER_NC](ctx, NX)
    var fluid_host = ctx.enqueue_create_host_buffer[Real.dtype](fluid.buf_len)
    var node_count = NX * (P_X + 1)
    for n in range(node_count):
        var base = n * EULER_NC
        fluid_host[base + IDX_RHO] = RHO_TARGET
        fluid_host[base + IDX_MX] = RHO_TARGET * UX_TARGET
        fluid_host[base + IDX_MY] = Real(0.0)
        fluid_host[base + IDX_MZ] = Real(0.0)
        fluid_host[base + IDX_E] = E_TARGET
    ctx.enqueue_copy(dst_buf=fluid.buf, src_buf=fluid_host)
    ctx.synchronize()

    # Apply the 5moment→kinetic initialiser.
    var q = PhaseState1D1V[P_X, P_V, VLASOV_NC](ctx, NX, NV)
    apply_5moment_to_kinetic_1d1v[P_X, P_V](
        ctx, mesh, fluid, q, MASS, GAMMA,
    )
    ctx.synchronize()

    # Verify via the three moment kernels — sample at every x-node.
    var moments = VlasovMoments1D1V[P_X, P_V]()
    var rho_k = DGState1D[P_X, 1](ctx, NX)
    var j_k = DGState1D[P_X, 1](ctx, NX)
    var e_k = DGState1D[P_X, 1](ctx, NX)
    moments.zeroth_moment(ctx, mesh, q, rho_k)
    moments.first_moment(ctx, mesh, q, j_k)
    moments.second_moment(ctx, mesh, q, e_k)
    ctx.synchronize()

    var rho_expected = RHO_TARGET / MASS
    var j_expected = RHO_TARGET * UX_TARGET / MASS
    # `e_k` is the kinetic-side `½ ∫ v² f dv`; for a centred Maxwellian
    # with `V_th² = T/m` and n = ρ/m: `½·n·V_th² = ρT/(2m²)`.
    var e_expected = (
        Real(0.5) * (RHO_TARGET / MASS) * (T_TARGET / MASS)
    )

    var max_drho: Real = 0.0
    var max_dj: Real = 0.0
    var max_de: Real = 0.0
    with rho_k.buf.map_to_host() as h:
        for k in range(rho_k.buf_len):
            var d = h[k] - rho_expected
            var ad = d if d >= 0.0 else -d
            if ad > max_drho:
                max_drho = ad
    with j_k.buf.map_to_host() as h:
        for k in range(j_k.buf_len):
            var d = h[k] - j_expected
            var ad = d if d >= 0.0 else -d
            if ad > max_dj:
                max_dj = ad
    with e_k.buf.map_to_host() as h:
        for k in range(e_k.buf_len):
            var d = h[k] - e_expected
            var ad = d if d >= 0.0 else -d
            if ad > max_de:
                max_de = ad

    print(
        "5moment→kinetic: rho_expected =", rho_expected,
        " max |Δρ| =", max_drho,
    )
    print(
        "5moment→kinetic: J_expected =", j_expected,
        " max |ΔJ| =", max_dj,
    )
    print(
        "5moment→kinetic: E_kin_expected =", e_expected,
        " max |ΔE_kin| =", max_de,
    )

    # Tolerances: the kinetic Maxwellian at v_max = 3 with V_th² = 0.5
    # has truncation residue `erfc(v_max / (V_th · √2)) = erfc(3) ≈
    # 2.2·10⁻⁵` of the mass — that's the dominant error for `Δρ`.
    # For `ΔE_kin` the residue is amplified by ~`v_max²` ≈ 9 because
    # the integrand carries `v²`, giving ~2·10⁻⁴. The first-moment is
    # protected by symmetry (odd integrand × symmetric mesh) and lands
    # at machine epsilon.
    assert_true(max_drho < Real(1.0e-4), "kinetic ρ doesn't match fluid")
    assert_true(max_dj < Real(1.0e-12), "kinetic J should be machine zero")
    assert_true(max_de < Real(1.0e-3), "kinetic E_kin doesn't match target")

    print("PASSED")
