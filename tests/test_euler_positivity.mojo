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

from warpxm.prelude import *
from warpxm.physics.euler import *


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


def _build[NDIM: Int]() -> List[Real]:
    """Initial state on NUM_ELEMENTS elements, in `DGState` layout."""
    comptime State = DGState[NDIM, P, EULER_NC]
    comptime npe = State.num_nodes_per_elem
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
                q, State.index(e, n, 0),
                Real(1.0) + Real(0.1) * s,
                moving * Real(0.3) * (Real(1.0) + Real(0.05) * s),
                moving * Real(-0.2) * (Real(1.0) - Real(0.03) * s),
                moving * Real(0.1) * (Real(1.0) + Real(0.02) * s),
                p_base * (Real(1.0) + Real(0.2) * s),
            )
        if e % 4 == 1:
            # Negative density at node 0.
            _set_node(q, State.index(e, 0, 0), Real(-0.5), Real(0.0), Real(0.0), Real(0.0), Real(1.0))
        elif e % 4 == 2:
            # Negative pressure at node 1, healthy density.
            _set_node(
                q, State.index(e, 1, 0),
                Real(1.0), Real(0.3), Real(-0.2), Real(0.1), Real(-0.3),
            )
        elif e % 4 == 3:
            # Slightly negative pressure at node 2 of a low-pressure element.
            _set_node(
                q, State.index(e, 2, 0),
                Real(1.0), Real(0.3), Real(-0.2), Real(0.1), Real(-1.0e-3),
            )
    return q^


def _verify[NDIM: Int](label: String, q0: List[Real], q1: List[Real]) raises:
    """Check `q1 = preserver(q0)` element by element."""
    comptime State = DGState[NDIM, P, EULER_NC]
    comptime npe = State.num_nodes_per_elem
    var w = _weights[NDIM]()
    var worst_mean: Real = 0.0
    var worst_theta: Real = 0.0
    var worst_rho_gap: Real = 0.0
    var worst_p_gap: Real = 0.0
    for e in range(NUM_ELEMENTS):
        var kind = e % 4

        # Cell means before and after.
        var mean0 = List[Real](length=EULER_NC, fill=Real(0.0))
        var mean1 = List[Real](length=EULER_NC, fill=Real(0.0))
        for n in range(npe):
            for c in range(EULER_NC):
                mean0[c] += w[n] * q0[State.index(e, n, c)]
                mean1[c] += w[n] * q1[State.index(e, n, c)]
        for c in range(EULER_NC):
            var d = fabs(mean1[c] - mean0[c])
            if d > worst_mean: worst_mean = d

        # Single θ: recover it from the largest deviation, check the rest.
        var theta: Real = 1.0
        var big: Real = 0.0
        for n in range(npe):
            for c in range(EULER_NC):
                var dev = q0[State.index(e, n, c)] - mean0[c]
                if fabs(dev) > big:
                    big = fabs(dev)
                    theta = (q1[State.index(e, n, c)] - mean0[c]) / dev
        for n in range(npe):
            for c in range(EULER_NC):
                var i = State.index(e, n, c)
                var d = fabs(q1[i] - (mean0[c] + theta * (q0[i] - mean0[c])))
                if d > worst_theta: worst_theta = d

        var min_rho: Real = 1.0e30
        var min_p: Real = 1.0e30
        for n in range(npe):
            var b = State.index(e, n, 0)
            if q1[b + IDX_RHO] < min_rho: min_rho = q1[b + IDX_RHO]
            var pp = _pressure(q1, b)
            if pp < min_p: min_p = pp

        if kind == 0:
            for i in range(State.index(e, 0, 0), State.index(e + 1, 0, 0)):
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


def _weights[NDIM: Int]() -> List[Real]:
    """Node weights of the cell mean (they sum to 1)."""
    var w1 = NodalBasis1D[P].weights()
    comptime npe = P1 ** NDIM
    var w = List[Real](length=npe, fill=Real(1.0))
    for n in range(npe):
        var rem = n
        for _ in range(NDIM):
            w[n] *= Real(0.5) * w1[rem % P1]
            rem //= P1
    return w^


def _upload[NDIM: Int](
    mut s: DGState[NDIM, P, EULER_NC], values: List[Real],
) raises:
    with s.buf.map_to_host() as h:
        for i in range(s.buf_len): h[i] = values[i]


def _download[NDIM: Int](mut s: DGState[NDIM, P, EULER_NC]) raises -> List[Real]:
    var values = List[Real](length=s.buf_len, fill=Real(0.0))
    with s.buf.map_to_host() as h:
        for i in range(s.buf_len): values[i] = h[i]
    return values^


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = DeviceContext()

    # ---- 1D ----
    var mesh1 = BlockMesh[1, P](ctx, Real(0.0), Real(1.0), NUM_ELEMENTS, periodic=True)
    var s1 = DGState[1, P, EULER_NC](ctx, mesh1.num_elements)
    var init1 = _build[1]()
    _upload(s1, init1)
    var pos1 = PositivityEuler[1, P](
        num_elements=mesh1.num_elements, gamma=GAMMA, rho_eps=RHO_EPS, p_eps=P_EPS,
    )
    pos1.apply(ctx, mesh1, s1)
    ctx.synchronize()
    _verify[1]("1D", init1, _download(s1))

    # ---- 2D ----
    var mesh2 = BlockMesh[2, P](
        ctx, Real(0.0), Real(1.0), Real(0.0), Real(1.0),
        nx=NUM_ELEMENTS // 2, ny=2, periodic_x=True, periodic_y=True,
    )
    var s2 = DGState[2, P, EULER_NC](ctx, mesh2.num_elements)
    var init2 = _build[2]()
    _upload(s2, init2)
    var pos2 = PositivityEuler[2, P](
        num_elements=mesh2.num_elements, gamma=GAMMA, rho_eps=RHO_EPS, p_eps=P_EPS,
    )
    pos2.apply(ctx, mesh2, s2)
    ctx.synchronize()
    _verify[2]("2D", init2, _download(s2))

    # ---- 3D ----
    var mesh3 = BlockMesh[3, P](
        ctx, Real(0.0), Real(1.0), Real(0.0), Real(1.0), Real(0.0), Real(1.0),
        nx=NUM_ELEMENTS // 4, ny=2, nz=2,
        periodic_x=True, periodic_y=True, periodic_z=True,
    )
    var s3 = DGState[3, P, EULER_NC](ctx, mesh3.num_elements)
    var init3 = _build[3]()
    _upload(s3, init3)
    var pos3 = PositivityEuler[3, P](
        num_elements=mesh3.num_elements, gamma=GAMMA, rho_eps=RHO_EPS, p_eps=P_EPS,
    )
    pos3.apply(ctx, mesh3, s3)
    ctx.synchronize()
    _verify[3]("3D", init3, _download(s3))

    print("PASSED")
