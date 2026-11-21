# Vlasov BGK collision source — Maxwellian fixed-point test.
#
# A local Maxwellian is the BGK operator's fixed point: when `f` is
# already Maxwellian with consistent `(n, u, T)`, the reconstructed
# `f_M` from the three moments equals `f` and the source `-(f − f_M)/τ`
# vanishes per phase node. Verifying this:
#   (a) confirms the moment-reconstruction pipeline
#       (ρ → n, J → u, E_kin → T) computes the right (n, u, T);
#   (b) confirms the per-phase-node Maxwellian formula
#       `f_M = n/√(2π T) · exp(−(v−u)²/(2T))` matches the IC form
#       `exp(−v²/σ²)` (modulo the n-vs-σ relabelling: σ² = 2T means
#       `exp(−v²/σ²) = exp(−v²/(2T))`);
#   (c) confirms the kernel correctly accumulates the source onto an
#       existing RHS (this test pre-fills RHS with a non-zero pattern
#       and checks the BGK delta leaves it unchanged to ≲ ε).
#
# Build / run:
#     pixi run mojo run -I src tests/test_vlasov_bgk_maxwellian_1d1v.mojo

from std.math import exp
from std.sys import has_accelerator
from max.gpu.host import DeviceContext
from std.testing import assert_true

from warpxm.apps.vlasovmaxwell import VLASOV_NC
from warpxm.basis.nodal_basis_1d import NodalBasis1D
from warpxm.core.types import Real
from warpxm.mesh.phase_mesh_1d1v import PhaseMesh1D1V
from warpxm.spatial.vlasov_bgk_source_1d1v import VlasovBGKSource1D1V
from warpxm.variable.phase_state_1d1v import PhaseState1D1V


comptime P_X: Int = 2
comptime P_V: Int = 3
comptime NX: Int = 4
comptime NV: Int = 16
comptime SIGMA_SQ: Real = 0.5
comptime TAU: Real = 1.0


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = DeviceContext()

    var mesh = PhaseMesh1D1V[P_X, P_V](
        ctx,
        Real(0.0), Real(1.0), Real(-3.0), Real(3.0),
        num_x=NX, num_v=NV,
        periodic_x=True,
    )

    var q = PhaseState1D1V[P_X, P_V, VLASOV_NC](ctx, NX, NV)
    var rhs = PhaseState1D1V[P_X, P_V, VLASOV_NC](ctx, NX, NV)

    # IC: f(x, v) = exp(-v²/σ²). x-independent Maxwellian.
    var ref_v_nodes = NodalBasis1D[P_V].nodes()
    var host = ctx.enqueue_create_host_buffer[Real.dtype](q.buf_len)
    for ex in range(NX):
        for ev in range(NV):
            var vl = Real(-3.0) + Real(ev) * mesh.dv
            for ix in range(P_X + 1):
                for iv in range(P_V + 1):
                    var v = vl + (Real(1.0) + ref_v_nodes[iv]) * mesh.jacobian_v
                    var phase_e = ex * NV + ev
                    var idx = (
                        (phase_e * (P_V + 1) + iv) * (P_X + 1) + ix
                    ) * VLASOV_NC
                    host[idx] = exp(-v * v / SIGMA_SQ)
    ctx.enqueue_copy(dst_buf=q.buf, src_buf=host)

    # Pre-fill `rhs` with a non-zero pattern; the BGK source must
    # *add* its contribution (which should be zero here), not
    # overwrite.
    var rhs_host = ctx.enqueue_create_host_buffer[Real.dtype](rhs.buf_len)
    for k in range(rhs.buf_len):
        rhs_host[k] = Real(0.1) * Real(k % 7) + Real(0.01)
    ctx.enqueue_copy(dst_buf=rhs.buf, src_buf=rhs_host)
    ctx.synchronize()

    # Apply BGK source.
    var bgk = VlasovBGKSource1D1V[P_X, P_V](ctx, NX)
    bgk.apply(ctx, mesh, q, rhs, TAU)
    ctx.synchronize()

    # Verify rhs is unchanged (source contribution ≈ 0 everywhere).
    var max_abs_change: Real = 0.0
    with rhs.buf.map_to_host() as h:
        for k in range(rhs.buf_len):
            var d = h[k] - rhs_host[k]
            var ad = d if d >= 0.0 else -d
            if ad > max_abs_change:
                max_abs_change = ad

    print(
        "BGK Maxwellian fixed point: max |Δrhs| =", max_abs_change,
    )

    # Tolerance: the moment-reconstruction chain (zeroth → ρ, first →
    # J, second → E_kin) each carries ≈ 1e-9 quadrature error. Those
    # propagate into u = J/n and T = 2E/n − u², and the final
    # Maxwellian reconstruction amplifies them by a factor involving
    # `1/√T` and the Gaussian's sensitivity to width — empirically
    # the worst-case node lands at a few × 1e-8.
    assert_true(
        max_abs_change < Real(1.0e-7),
        "BGK source non-zero for a Maxwellian IC",
    )
    print("PASSED")
