# Smoke test for the 2D MHD positivity preserver.
#
# Builds two states and verifies:
#   (a) On a healthy uniform state with ρ, P_gas, |B| all positive and
#       smooth (constant), the preserver is a no-op — outputs == inputs
#       to machine precision.
#   (b) On a deliberately bad state where a single GLL node has ρ << 0,
#       the preserver clamps that node back up to ρ_eps while leaving
#       cell averages invariant.

from std.sys import has_accelerator
from std.gpu.host import DeviceContext
from std.testing import assert_true

from warpxm.adjusters.positivity_mhd_2d import PositivityMHD2D
from warpxm.apps.imhd_2d import MHD_NC, IDX_RHO, IDX_MX, IDX_MY, IDX_MZ, IDX_E, IDX_BX, IDX_BY, IDX_BZ
from warpxm.basis.nodal_basis_1d import NodalBasis1D
from warpxm.core.math_compat import fabs
from warpxm.core.types import Real
from warpxm.mesh.block_2d import BlockMesh2D
from warpxm.variable.state_2d import DGState2D


comptime P: Int = 2
comptime NX: Int = 4
comptime NY: Int = 4
comptime GAMMA: Real = 5.0 / 3.0
comptime RHO_EPS: Real = 1.0e-10
comptime P_EPS: Real = 1.0e-10


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"

    var ctx = DeviceContext()
    var mesh = BlockMesh2D[P](
        ctx, Real(0.0), Real(1.0), Real(0.0), Real(1.0),
        nx=NX, ny=NY, periodic_x=True, periodic_y=True,
    )
    var P1 = NodalBasis1D[P].num_nodes
    var nodes_per_elem = P1 * P1

    var positivity = PositivityMHD2D[P, MHD_NC](
        num_elements=mesh.num_elements, gamma=GAMMA,
        rho_eps=RHO_EPS, p_eps=P_EPS,
    )

    # ---- (a) Healthy state ------------------------------------------------
    var q1 = DGState2D[P, MHD_NC](ctx, mesh.num_elements)
    var host = ctx.enqueue_create_host_buffer[Real.dtype](q1.buf_len)
    for e in range(mesh.num_elements):
        for n in range(nodes_per_elem):
            var b = (e * nodes_per_elem + n) * MHD_NC
            host[b + IDX_RHO] = 1.0
            host[b + IDX_MX] = 0.3
            host[b + IDX_MY] = -0.2
            host[b + IDX_MZ] = 0.0
            host[b + IDX_BX] = 0.5
            host[b + IDX_BY] = -0.1
            host[b + IDX_BZ] = 0.0
            # P = 1.0
            var ke = Real(0.5) * (0.3 * 0.3 + 0.2 * 0.2)
            var mag = Real(0.5) * (0.5 * 0.5 + 0.1 * 0.1)
            host[b + IDX_E] = 1.0 / (GAMMA - Real(1.0)) + ke + mag
    ctx.enqueue_copy(dst_buf=q1.buf, src_buf=host)
    positivity.apply(ctx, mesh, q1)
    ctx.synchronize()

    var max_delta: Real = 0.0
    with q1.buf.map_to_host() as h:
        for i in range(q1.buf_len):
            var d = fabs(h[i] - host[i])
            if d > max_delta: max_delta = d
    print("(a) max change on healthy state:", max_delta)
    assert_true(max_delta < Real(1.0e-14), "healthy state was modified")

    # ---- (b) Single bad node (negative ρ) --------------------------------
    var q2 = DGState2D[P, MHD_NC](ctx, mesh.num_elements)
    # Reset to healthy then break node 0 of element 0.
    var host2 = ctx.enqueue_create_host_buffer[Real.dtype](q2.buf_len)
    for i in range(q1.buf_len):
        host2[i] = host[i]
    var b_bad = (0 * nodes_per_elem + 0) * MHD_NC
    var b_neighbor = (0 * nodes_per_elem + 1) * MHD_NC
    # Push ρ at node 0 to -0.5, compensate elsewhere by symmetry not required
    # for this test (we just want to see preserver clamp ρ).
    host2[b_bad + IDX_RHO] = -0.5
    # Bump another node to keep cell mean ≈ 1.
    host2[b_neighbor + IDX_RHO] = 2.5
    ctx.enqueue_copy(dst_buf=q2.buf, src_buf=host2)
    positivity.apply(ctx, mesh, q2)
    ctx.synchronize()

    var min_rho: Real = 1.0e30
    var cell0_mean_rho_before: Real = 0.0
    var cell0_mean_rho_after: Real = 0.0
    var weights = NodalBasis1D[P].weights()
    with q2.buf.map_to_host() as h:
        for n in range(nodes_per_elem):
            var b = (0 * nodes_per_elem + n) * MHD_NC
            var jn = n // P1
            var in_ = n % P1
            var w = weights[in_] * weights[jn] * Real(0.25)
            cell0_mean_rho_before += w * host2[b + IDX_RHO]
            cell0_mean_rho_after += w * h[b + IDX_RHO]
            if h[b + IDX_RHO] < min_rho: min_rho = h[b + IDX_RHO]

    print("(b) min ρ after preserver:           ", min_rho)
    print("    cell 0 mean ρ before / after:     ", cell0_mean_rho_before, " / ", cell0_mean_rho_after)
    assert_true(min_rho >= RHO_EPS, "ρ remained below eps after preserver")
    assert_true(
        fabs(cell0_mean_rho_after - cell0_mean_rho_before) < Real(1.0e-12),
        "preserver changed the cell mean",
    )
    print("PASSED")
