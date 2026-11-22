# Smoke test for the 2D MHD positivity preserver.
#
# Builds two states and verifies:
#   (a) On a healthy uniform state with ρ, P_gas, |B| all positive and
#       smooth (constant), the preserver is a no-op — outputs == inputs
#       to machine precision.
#   (b) On a deliberately bad state where a single GLL node has ρ << 0,
#       the preserver clamps that node back up to ρ_eps while leaving
#       cell averages invariant.

from std.collections import Array
from std.sys import has_accelerator
from max.gpu.host import DeviceContext
from std.testing import assert_true

from warpxm.prelude import *
from warpxm.physics.imhd import *


comptime P: Int = 2
comptime NX: Int = 4
comptime NY: Int = 4
comptime GAMMA: Real = 5.0 / 3.0
comptime RHO_EPS: Real = 1.0e-10
comptime P_EPS: Real = 1.0e-10
comptime State = DGState[2, P, MHD_NC]


def _healthy() -> Array[Real, MHD_NC]:
    """Uniform state with P_gas = 1."""
    var ke = Real(0.5) * (0.3 * 0.3 + 0.2 * 0.2)
    var mag = Real(0.5) * (0.5 * 0.5 + 0.1 * 0.1)
    var v = Array[Real, MHD_NC](fill=Real(0.0))
    v[IDX_RHO] = 1.0
    v[IDX_MX] = 0.3
    v[IDX_MY] = -0.2
    v[IDX_BX] = 0.5
    v[IDX_BY] = -0.1
    v[IDX_E] = 1.0 / (GAMMA - Real(1.0)) + ke + mag
    return v^


def _cell0_mean_rho(mut q: State) raises -> Real:
    """GLL-weighted mean of ρ over element 0."""
    comptime P1 = P + 1
    var weights = NodalBasis1D[P].weights()
    var mean: Real = 0.0
    with q.buf.map_to_host() as h:
        for n in range(State.num_nodes_per_elem):
            var w = weights[n % P1] * weights[n // P1] * Real(0.25)
            mean += w * h[State.index(0, n, IDX_RHO)]
    return mean


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"

    var ctx = DeviceContext()
    var mesh = BlockMesh[2, P](
        ctx, Real(0.0), Real(1.0), Real(0.0), Real(1.0),
        nx=NX, ny=NY, periodic_x=True, periodic_y=True,
    )

    var positivity = PositivityMHD[2, P, MHD_NC](
        num_elements=mesh.num_elements, gamma=GAMMA,
        rho_eps=RHO_EPS, p_eps=P_EPS,
    )

    def healthy(x: Real, y: Real) -> Array[Real, MHD_NC]:
        return _healthy()

    # ---- (a) Healthy state ------------------------------------------------
    var q1 = State(ctx, mesh.num_elements)
    mesh.fill(q1, healthy)
    positivity.apply(ctx, mesh, q1)
    ctx.synchronize()

    var max_delta: Real = 0.0
    def delta(x: Real, y: Real, w: Real, v: Array[Real, MHD_NC]) {mut max_delta}:
        var want = _healthy()
        for c in range(MHD_NC):
            var d = fabs(v[c] - want[c])
            if d > max_delta: max_delta = d
    mesh.visit_nodes(q1, delta)
    print("(a) max change on healthy state:", max_delta)
    assert_true(max_delta < Real(1.0e-14), "healthy state was modified")

    # ---- (b) Single bad node (negative ρ) --------------------------------
    # Healthy everywhere, then break node 0 of element 0: push ρ to -0.5
    # and bump node 1 to keep the cell mean ≈ 1.
    var q2 = State(ctx, mesh.num_elements)
    mesh.fill(q2, healthy)
    with q2.buf.map_to_host() as h:
        h[State.index(0, 0, IDX_RHO)] = -0.5
        h[State.index(0, 1, IDX_RHO)] = 2.5
    var cell0_mean_rho_before = _cell0_mean_rho(q2)
    positivity.apply(ctx, mesh, q2)
    ctx.synchronize()

    var cell0_mean_rho_after = _cell0_mean_rho(q2)
    var min_rho: Real = 1.0e30
    with q2.buf.map_to_host() as h:
        for n in range(State.num_nodes_per_elem):
            var rho = h[State.index(0, n, IDX_RHO)]
            if rho < min_rho: min_rho = rho

    print("(b) min ρ after preserver:           ", min_rho)
    print("    cell 0 mean ρ before / after:     ", cell0_mean_rho_before, " / ", cell0_mean_rho_after)
    assert_true(min_rho >= RHO_EPS, "ρ remained below eps after preserver")
    assert_true(
        fabs(cell0_mean_rho_after - cell0_mean_rho_before) < Real(1.0e-12),
        "preserver changed the cell mean",
    )
    print("PASSED")
