# Unit test for the Zhang-Shu Euler positivity preserver in 1D, 2D and 3D.
#
# One kernel (`_positivity_kernel_euler[P, NDIM]`) serves all three
# dimensions; this test drives each instantiation on a hand-built state
# and checks the properties the scheme relies on, element by element:
#
#   element 0  admissible, non-uniform      → returned bit-for-bit
#   element 1  one node with ρ < 0, at rest → min ρ lands on ρ_eps
#   element 2  one node with P < 0          → min P lands on P_eps
#   element 3  low-pressure element with a  → min P lands on P_eps
#              slightly negative node         (small-θ-change regime)
#
# For every element: all five cell means are unchanged, every node ends
# up admissible, and the result is `mean + θ (node − mean)` with a single
# θ per element. "Lands on the bound" (rather than merely "is above it")
# is what shows θ is the largest admissible value, i.e. that the limiter
# does not flatten more than it must.
#
# Build / run:
#     pixi run mojo run -I src tests/test_euler_positivity.mojo

from std.math import sin
from std.sys import has_accelerator
from max.gpu.host import DeviceContext
from std.testing import assert_true

from warpxm.adjusters.positivity_euler import (
    PositivityEuler1D, PositivityEuler2D, PositivityEuler3D,
)
from warpxm.apps.euler import EULER_NC, IDX_RHO, IDX_MX, IDX_MY, IDX_MZ, IDX_E
from warpxm.basis.nodal_basis_1d import NodalBasis1D
from warpxm.core.math_compat import fabs
from warpxm.core.types import Real
from warpxm.mesh.block_1d import BlockMesh1D
from warpxm.mesh.block_2d import BlockMesh2D
from warpxm.mesh.block_3d import BlockMesh3D
from warpxm.variable.state_1d import DGState1D
from warpxm.variable.state_2d import DGState2D
from warpxm.variable.state_3d import DGState3D


comptime P: Int = 2
comptime P1: Int = P + 1
comptime GAMMA: Real = 1.4
# Large floors so "the worst node sits on the floor" is resolvable.
comptime RHO_EPS: Real = 1.0e-3
comptime P_EPS: Real = 1.0e-3
comptime NUM_ELEMENTS: Int = 8


def _pressure(q: List[Real], b: Int) -> Real:
    var ke = Real(0.5) * (
        q[b + IDX_MX] * q[b + IDX_MX] + q[b + IDX_MY] * q[b + IDX_MY]
        + q[b + IDX_MZ] * q[b + IDX_MZ]
    ) / q[b + IDX_RHO]
    return (GAMMA - Real(1.0)) * (q[b + IDX_E] - ke)


def _set_node(
    mut q: List[Real], b: Int, rho: Real, mx: Real, my: Real, mz: Real, pp: Real,
):
    """Write a node from (ρ, m, P). ρ may be negative; then E is set from
    |ρ| so the node is 'bad' through its density only."""
    var rho_abs = rho if rho > Real(0.0) else -rho
    q[b + IDX_RHO] = rho
    q[b + IDX_MX] = mx
    q[b + IDX_MY] = my
    q[b + IDX_MZ] = mz
    q[b + IDX_E] = (
        pp / (GAMMA - Real(1.0))
        + Real(0.5) * (mx * mx + my * my + mz * mz) / rho_abs
    )


def _build(npe: Int) -> List[Real]:
    """Initial state on NUM_ELEMENTS elements of `npe` nodes each."""
    var q = List[Real](length=NUM_ELEMENTS * npe * EULER_NC, fill=Real(0.0))
    for e in range(NUM_ELEMENTS):
        # Element 3 (and 7) sit at low pressure; the rest at P ≈ 1.
        var p_base: Real = 1.0
        if e % 4 == 3: p_base = 1.0e-2
        # The density case is at rest: a node pushed to ρ_eps while carrying
        # momentum has an enormous kinetic energy, so in a moving element the
        # pressure step always takes over and the density step is never the
        # binding one.
        var moving: Real = 1.0
        if e % 4 == 1: moving = 0.0
        for n in range(npe):
            var s = sin(Real(1.7) * Real(n) + Real(0.9) * Real(e))
            _set_node(
                q, (e * npe + n) * EULER_NC,
                Real(1.0) + Real(0.1) * s,
                moving * Real(0.3) * (Real(1.0) + Real(0.05) * s),
                moving * Real(-0.2) * (Real(1.0) - Real(0.03) * s),
                moving * Real(0.1) * (Real(1.0) + Real(0.02) * s),
                p_base * (Real(1.0) + Real(0.2) * s),
            )
        var b0 = e * npe * EULER_NC
        if e % 4 == 1:
            # Negative density at node 0.
            _set_node(q, b0, Real(-0.5), Real(0.0), Real(0.0), Real(0.0), Real(1.0))
        elif e % 4 == 2:
            # Negative pressure at node 1, healthy density.
            _set_node(
                q, b0 + EULER_NC,
                Real(1.0), Real(0.3), Real(-0.2), Real(0.1), Real(-0.3),
            )
        elif e % 4 == 3:
            # Slightly negative pressure at node 2 of a low-pressure element.
            _set_node(
                q, b0 + 2 * EULER_NC,
                Real(1.0), Real(0.3), Real(-0.2), Real(0.1), Real(-1.0e-3),
            )
    return q^


def _verify(
    label: String, q0: List[Real], q1: List[Real], w: List[Real], npe: Int,
) raises:
    """Check `q1 = preserver(q0)` element by element. `w` are the node
    weights of the cell mean (they sum to 1)."""
    var worst_mean: Real = 0.0
    var worst_theta: Real = 0.0
    var worst_rho_gap: Real = 0.0
    var worst_p_gap: Real = 0.0
    for e in range(NUM_ELEMENTS):
        var kind = e % 4
        var b0 = e * npe * EULER_NC

        # Cell means before and after.
        var mean0 = List[Real](length=EULER_NC, fill=Real(0.0))
        var mean1 = List[Real](length=EULER_NC, fill=Real(0.0))
        for n in range(npe):
            for c in range(EULER_NC):
                mean0[c] += w[n] * q0[b0 + n * EULER_NC + c]
                mean1[c] += w[n] * q1[b0 + n * EULER_NC + c]
        for c in range(EULER_NC):
            var d = fabs(mean1[c] - mean0[c])
            if d > worst_mean: worst_mean = d

        # Single θ: recover it from the largest deviation, check the rest.
        var theta: Real = 1.0
        var big: Real = 0.0
        for n in range(npe):
            for c in range(EULER_NC):
                var dev = q0[b0 + n * EULER_NC + c] - mean0[c]
                if fabs(dev) > big:
                    big = fabs(dev)
                    theta = (q1[b0 + n * EULER_NC + c] - mean0[c]) / dev
        for n in range(npe):
            for c in range(EULER_NC):
                var i = b0 + n * EULER_NC + c
                var d = fabs(q1[i] - (mean0[c] + theta * (q0[i] - mean0[c])))
                if d > worst_theta: worst_theta = d

        var min_rho: Real = 1.0e30
        var min_p: Real = 1.0e30
        for n in range(npe):
            var b = b0 + n * EULER_NC
            if q1[b + IDX_RHO] < min_rho: min_rho = q1[b + IDX_RHO]
            var pp = _pressure(q1, b)
            if pp < min_p: min_p = pp

        if kind == 0:
            for i in range(b0, b0 + npe * EULER_NC):
                assert_true(q1[i] == q0[i], label + ": admissible element was modified")
        else:
            assert_true(theta > Real(0.0) and theta < Real(1.0), label + ": θ outside (0, 1)")
        if kind == 1:
            var gap = fabs(min_rho - RHO_EPS)
            if gap > worst_rho_gap: worst_rho_gap = gap
            assert_true(min_p > P_EPS, label + ": density case left P below the floor")
        if kind == 2 or kind == 3:
            var gap = fabs(min_p - P_EPS)
            if gap > worst_p_gap: worst_p_gap = gap
            assert_true(min_rho > RHO_EPS, label + ": pressure case left ρ below the floor")

    print(label)
    print("  max cell-mean change:        ", worst_mean)
    print("  max departure from single θ: ", worst_theta)
    print("  |min ρ − ρ_eps| (ρ case):     ", worst_rho_gap)
    print("  |min P − P_eps| (P cases):    ", worst_p_gap)
    assert_true(worst_mean < Real(1.0e-14), label + ": cell mean changed")
    assert_true(worst_theta < Real(1.0e-14), label + ": not a single-θ scaling")
    assert_true(worst_rho_gap < Real(1.0e-14), label + ": min ρ is not on the floor")
    assert_true(worst_p_gap < Real(1.0e-12), label + ": min P is not on the floor")


def _weights(ndim: Int) -> List[Real]:
    var w1 = NodalBasis1D[P].weights()
    var npe = P1 ** ndim
    var w = List[Real](length=npe, fill=Real(1.0))
    for n in range(npe):
        var rem = n
        for _ in range(ndim):
            w[n] *= Real(0.5) * w1[rem % P1]
            rem //= P1
    return w^


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = DeviceContext()

    # ---- 1D ----
    var npe1 = P1
    var mesh1 = BlockMesh1D[P](ctx, Real(0.0), Real(1.0), NUM_ELEMENTS, periodic=True)
    var s1 = DGState1D[P, EULER_NC](ctx, mesh1.num_elements)
    var init1 = _build(npe1)
    var host1 = ctx.enqueue_create_host_buffer[Real.dtype](s1.buf_len)
    for i in range(s1.buf_len): host1[i] = init1[i]
    ctx.enqueue_copy(dst_buf=s1.buf, src_buf=host1)
    var pos1 = PositivityEuler1D[P](
        num_elements=mesh1.num_elements, gamma=GAMMA, rho_eps=RHO_EPS, p_eps=P_EPS,
    )
    pos1.apply(ctx, mesh1, s1)
    ctx.synchronize()
    var out1 = List[Real](length=s1.buf_len, fill=Real(0.0))
    with s1.buf.map_to_host() as h:
        for i in range(s1.buf_len): out1[i] = h[i]
    _verify("1D", init1, out1, _weights(1), npe1)

    # ---- 2D ----
    var npe2 = P1 * P1
    var mesh2 = BlockMesh2D[P](
        ctx, Real(0.0), Real(1.0), Real(0.0), Real(1.0),
        nx=NUM_ELEMENTS // 2, ny=2, periodic_x=True, periodic_y=True,
    )
    var s2 = DGState2D[P, EULER_NC](ctx, mesh2.num_elements)
    var init2 = _build(npe2)
    var host2 = ctx.enqueue_create_host_buffer[Real.dtype](s2.buf_len)
    for i in range(s2.buf_len): host2[i] = init2[i]
    ctx.enqueue_copy(dst_buf=s2.buf, src_buf=host2)
    var pos2 = PositivityEuler2D[P](
        num_elements=mesh2.num_elements, gamma=GAMMA, rho_eps=RHO_EPS, p_eps=P_EPS,
    )
    pos2.apply(ctx, mesh2, s2)
    ctx.synchronize()
    var out2 = List[Real](length=s2.buf_len, fill=Real(0.0))
    with s2.buf.map_to_host() as h:
        for i in range(s2.buf_len): out2[i] = h[i]
    _verify("2D", init2, out2, _weights(2), npe2)

    # ---- 3D ----
    var npe3 = P1 * P1 * P1
    var mesh3 = BlockMesh3D[P](
        ctx, Real(0.0), Real(1.0), Real(0.0), Real(1.0), Real(0.0), Real(1.0),
        nx=NUM_ELEMENTS // 4, ny=2, nz=2,
        periodic_x=True, periodic_y=True, periodic_z=True,
    )
    var s3 = DGState3D[P, EULER_NC](ctx, mesh3.num_elements)
    var init3 = _build(npe3)
    var host3 = ctx.enqueue_create_host_buffer[Real.dtype](s3.buf_len)
    for i in range(s3.buf_len): host3[i] = init3[i]
    ctx.enqueue_copy(dst_buf=s3.buf, src_buf=host3)
    var pos3 = PositivityEuler3D[P](
        num_elements=mesh3.num_elements, gamma=GAMMA, rho_eps=RHO_EPS, p_eps=P_EPS,
    )
    pos3.apply(ctx, mesh3, s3)
    ctx.synchronize()
    var out3 = List[Real](length=s3.buf_len, fill=Real(0.0))
    with s3.buf.map_to_host() as h:
        for i in range(s3.buf_len): out3[i] = h[i]
    _verify("3D", init3, out3, _weights(3), npe3)

    print("PASSED")
