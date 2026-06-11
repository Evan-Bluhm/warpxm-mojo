# Vlasov → Maxwell current-source kernel test.
#
# Builds a known J from a v-shifted Maxwellian, calls the current-source
# kernel on a non-zero Maxwell RHS, and verifies the new RHS equals
# `rhs_old + (-α / ε₀) · J` at every x-node. Touches only `IDX_EX`; the
# other Maxwell components must be unchanged.
#
# Build / run:
#     pixi run mojo run -I src tests/test_vlasov_current_source_1d.mojo

from std.math import exp, sqrt, erf
from std.sys import has_accelerator
from std.gpu.host import DeviceContext
from std.testing import assert_true

from warpxm.apps.maxwell import MAXWELL_NC, IDX_EX
from warpxm.apps.vlasovmaxwell import VLASOV_NC
from warpxm.basis.nodal_basis_1d import NodalBasis1D
from warpxm.core.types import Real, PI
from warpxm.mesh.phase_mesh_1d1v import PhaseMesh1D1V
from warpxm.spatial.vlasov_current_source_1d import (
    apply_vlasov_current_source_1d,
)
from warpxm.spatial.vlasov_moments_1d1v import VlasovMoments1D1V
from warpxm.variable.phase_state_1d1v import PhaseState1D1V
from warpxm.variable.state_1d import DGState1D


comptime P_X: Int = 2
comptime P_V: Int = 3
comptime NX: Int = 4
comptime NV: Int = 16
comptime SIGMA_SQ: Real = 0.5
comptime U_SHIFT: Real = 0.5
comptime ALPHA: Real = -1.0   # electron-like q/m
comptime EPS0: Real = 1.0


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

    # IC: v-shifted Maxwellian, x-independent ⇒ J is the same at every x-node.
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
                    var dv_shift = v - U_SHIFT
                    host[idx] = exp(-dv_shift * dv_shift / SIGMA_SQ)
    ctx.enqueue_copy(dst_buf=q.buf, src_buf=host)
    ctx.synchronize()

    # Build J via first-moment kernel.
    var moments = VlasovMoments1D1V[P_X, P_V]()
    var j = DGState1D[P_X, 1](ctx, NX)
    moments.first_moment(ctx, mesh, q, j)
    ctx.synchronize()

    # Snapshot J for the analytic prediction.
    var j_host = ctx.enqueue_create_host_buffer[Real.dtype](j.buf_len)
    ctx.enqueue_copy(dst_buf=j_host, src_buf=j.buf)
    ctx.synchronize()

    # Build a Maxwell RHS scratch buffer with a known starting pattern,
    # so we can verify the source is *added* (not overwritten).
    var rhs_maxwell = DGState1D[P_X, MAXWELL_NC](ctx, NX)
    var rhs_host = ctx.enqueue_create_host_buffer[Real.dtype](rhs_maxwell.buf_len)
    var node_count = NX * (P_X + 1)
    for n in range(node_count):
        for c in range(MAXWELL_NC):
            rhs_host[n * MAXWELL_NC + c] = Real(0.1) * Real(c + 1) + Real(0.01) * Real(n)
    ctx.enqueue_copy(dst_buf=rhs_maxwell.buf, src_buf=rhs_host)
    ctx.synchronize()

    # Apply the source.
    apply_vlasov_current_source_1d[P_X](
        ctx, j, rhs_maxwell, ALPHA, EPS0,
    )
    ctx.synchronize()

    # Verify per-node:
    #   rhs_new[IDX_EX] = rhs_old[IDX_EX] + (-α / ε₀) · J_node
    #   rhs_new[other]  = rhs_old[other]
    var max_err_ex: Real = 0.0
    var max_err_other: Real = 0.0
    var max_abs_j: Real = 0.0
    with rhs_maxwell.buf.map_to_host() as h:
        for n in range(node_count):
            var j_n = j_host[n]
            var aj = j_n if j_n >= 0.0 else -j_n
            if aj > max_abs_j:
                max_abs_j = aj
            for c in range(MAXWELL_NC):
                var pre = rhs_host[n * MAXWELL_NC + c]
                var got = h[n * MAXWELL_NC + c]
                var expect: Real
                if c == IDX_EX:
                    expect = pre + (-ALPHA / EPS0) * j_n
                else:
                    expect = pre
                var d = got - expect
                var ad = d if d >= 0.0 else -d
                if c == IDX_EX:
                    if ad > max_err_ex:
                        max_err_ex = ad
                else:
                    if ad > max_err_other:
                        max_err_other = ad

    print(
        "current-source 1D: max|J| =", max_abs_j,
        " max err E_x =", max_err_ex,
        " max err other =", max_err_other,
    )

    # E_x: small floating-point round-off when J is on the order of 0.6
    # and we add it to per-node RHS values up to ~0.7. 1e-12 is ample.
    assert_true(max_err_ex < Real(1.0e-12), "E_x update mismatch")
    # Other components must be untouched, bit-for-bit.
    assert_true(max_err_other == Real(0.0), "non-E_x component modified")
    # Sanity: J was actually nonzero (validates the test exercises the
    # source path, not just trivial zero arithmetic).
    var expected_j_floor = Real(0.5) * U_SHIFT * sqrt(SIGMA_SQ * PI)
    assert_true(
        max_abs_j > expected_j_floor,
        "J magnitude smaller than expected",
    )

    print("PASSED")
