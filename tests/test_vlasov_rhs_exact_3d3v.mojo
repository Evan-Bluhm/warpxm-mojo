# 2D3V and 3D3V Vlasov right-hand side, exact to round-off.
#
# When `f` is a polynomial of degree at most P in each phase coordinate
# and is continuous, the nodal DG residual is exact: the collocation
# derivative along each line of nodes is exact, the speed multiplies it
# node by node, and every interior jump vanishes. So with
#
#     f = Π_a (α_a + β_a s_a + ε_a s_a²) + Π_a (γ_a + δ_a s_a)
#
# over the phase coordinates `s = (x…, v1, v2, v3)` (a sum of two
# separable terms, so not separable; `ε = 0` at P = 1) and fields that
# are linear in x, the kernel must return
#
#     −Σ_a v_a ∂f/∂x_a − a · ∂f/∂v,      a = (q/m)/λ · (E + v × B)
#
# at every node to round-off. This covers what the 1D3V tests cannot:
# streaming along the second and third configuration axes, the neighbour
# index arithmetic in five and six dimensions (a wrong neighbour gives a
# non-zero jump), and the lookup of the fields on a `BlockMesh` that
# numbers its elements differently from the phase mesh. Every axis has a
# different element count and width so that a swapped axis cannot cancel.
#
# Configuration boundaries are copy-out (zero jump). Velocity boundaries
# take a zero ghost state, which is not this `f`, so nodes on the outer
# velocity faces are left out of the comparison; with GLL collocation the
# boundary flux touches only those nodes.
#
# Build / run:
#     pixi run mojo run -I src tests/test_vlasov_rhs_exact_3d3v.mojo

from std.collections import Array
from std.sys import has_accelerator
from max.gpu.host import DeviceContext
from std.testing import assert_true

from warpxm.prelude import *
from warpxm.physics.vlasov import *
from warpxm.physics.maxwell import *


comptime CHARGE_MASS_RATIO: Real = -1.3
comptime SKIN_DEPTH_NORM: Real = 0.8
comptime Coords = Array[Real, 6]
"""`(x, y, z, v1, v2, v3)`; `z` is unused in 2D3V."""

comptime LO: Coords = [-0.4, 0.1, -1.0, -2.0, -1.5, -2.5]
comptime HI: Coords = [0.8, 1.9, 0.5, 2.0, 1.5, 2.5]
comptime ALPHA: Coords = [1.0, 0.8, 1.2, 0.9, 1.1, 0.7]
comptime BETA: Coords = [0.3, -0.2, 0.25, 0.15, -0.3, 0.2]
comptime EPSILON: Coords = [0.1, 0.15, -0.1, -0.05, 0.08, 0.06]
comptime GAMMA: Coords = [0.5, 1.3, 0.7, 1.0, 0.6, 1.4]
comptime DELTA: Coords = [-0.4, 0.3, 0.2, -0.25, 0.35, -0.15]


def fields_at(x: Real, y: Real, z: Real) -> Array[Real, MAXWELL_NC]:
    var v = Array[Real, MAXWELL_NC](fill=Real(0.0))
    v[IDX_EX] = 0.7 + 0.3 * x - 0.2 * y + 0.4 * z
    v[IDX_EY] = -0.4 + 0.1 * x + 0.5 * y - 0.3 * z
    v[IDX_EZ] = 0.2 - 0.6 * x + 0.2 * y + 0.1 * z
    v[IDX_BX] = 0.25 + 0.2 * x + 0.3 * y - 0.1 * z
    v[IDX_BY] = -0.15 - 0.3 * x + 0.1 * y + 0.2 * z
    v[IDX_BZ] = 0.6 + 0.1 * x - 0.4 * y - 0.5 * z
    return v^


def f_poly[XDIM: Int, P: Int](s: Coords) -> Real:
    """The test polynomial. Axis 2 (`z`) is skipped when XDIM = 2."""
    var eps = materialize[EPSILON]()
    var al = materialize[ALPHA]()
    var be = materialize[BETA]()
    var ga = materialize[GAMMA]()
    var de = materialize[DELTA]()
    var first = Real(1.0)
    var second = Real(1.0)
    for a in range(6):
        if a == 2 and XDIM == 2:
            continue
        var quad = eps[a] if P >= 2 else Real(0.0)
        first *= al[a] + be[a] * s[a] + quad * s[a] * s[a]
        second *= ga[a] + de[a] * s[a]
    return first + second


def rhs_poly[XDIM: Int, P: Int](s: Coords) -> Real:
    """`−v · ∇_x f − a · ∇_v f` for `f_poly`."""
    var eps = materialize[EPSILON]()
    var al = materialize[ALPHA]()
    var be = materialize[BETA]()
    var ga = materialize[GAMMA]()
    var de = materialize[DELTA]()
    var grad = Array[Real, 6](fill=Real(0.0))
    for b in range(6):
        if b == 2 and XDIM == 2:
            continue
        var quad_b = eps[b] if P >= 2 else Real(0.0)
        var first = be[b] + Real(2.0) * quad_b * s[b]
        var second = de[b]
        for a in range(6):
            if a == b or (a == 2 and XDIM == 2):
                continue
            var quad = eps[a] if P >= 2 else Real(0.0)
            first *= al[a] + be[a] * s[a] + quad * s[a] * s[a]
            second *= ga[a] + de[a] * s[a]
        grad[b] = first + second
    var em = fields_at(s[0], s[1], s[2] if XDIM == 3 else Real(0.0))
    var factor = CHARGE_MASS_RATIO / SKIN_DEPTH_NORM
    var a1 = factor * (em[IDX_EX] + s[4] * em[IDX_BZ] - s[5] * em[IDX_BY])
    var a2 = factor * (em[IDX_EY] + s[5] * em[IDX_BX] - s[3] * em[IDX_BZ])
    var a3 = factor * (em[IDX_EZ] + s[3] * em[IDX_BY] - s[4] * em[IDX_BX])
    var out = -a1 * grad[3] - a2 * grad[4] - a3 * grad[5]
    for a in range(XDIM):
        out -= s[3 + a] * grad[a]
    return out


struct Tally(Copyable, Movable):
    var max_err: Real
    var max_rhs: Real
    var compared: Int

    def __init__(out self):
        self.max_err = 0.0
        self.max_rhs = 0.0
        self.compared = 0

    def add[XDIM: Int, P: Int](mut self, s: Coords, r: Real):
        """Compare one node, unless it sits on an outer velocity face."""
        var lo = materialize[LO]()
        var hi = materialize[HI]()
        for d in range(3, 6):
            var tol = Real(1.0e-9) * (hi[d] - lo[d])
            if s[d] < lo[d] + tol or s[d] > hi[d] - tol:
                return
        var exact = rhs_poly[XDIM, P](s)
        self.max_err = max(self.max_err, abs(r - exact))
        self.max_rhs = max(self.max_rhs, abs(exact))
        self.compared += 1


def check_2d3v[P: Int](var ctx: DeviceContext) raises:
    var lo = materialize[LO]()
    var hi = materialize[HI]()
    comptime State = PhaseState[2, 3, P, P, VLASOV_NC]
    var mesh = PhaseMesh[2, 3, P, P](
        ctx,
        lo=[lo[0], lo[1], lo[3], lo[4], lo[5]],
        hi=[hi[0], hi[1], hi[3], hi[4], hi[5]],
        n=[3, 5, 4, 3, 6],
        periodic=[False, False],
    )
    var x_mesh = BlockMesh[2, P](
        ctx, lo[0], hi[0], lo[1], hi[1], 3, 5,
        periodic_x=False, periodic_y=False,
    )
    mesh.check_companion(x_mesh)
    var em = DGState[2, P, MAXWELL_NC](ctx, x_mesh.num_elements)
    def em_ic(x: Real, y: Real) -> Array[Real, MAXWELL_NC]:
        return fields_at(x, y, Real(0.0))
    x_mesh.fill(em, em_ic)

    var q = State(ctx, mesh.num_phase_elements)
    var rhs = State(ctx, mesh.num_phase_elements)
    def ic(x: Real, y: Real, v1: Real, v2: Real, v3: Real) -> Real:
        return f_poly[2, P]([x, y, Real(0.0), v1, v2, v3])
    mesh.fill(q, ic)

    var app = VlasovMaxwell(
        charge_mass_ratio=CHARGE_MASS_RATIO, skin_depth_norm=SKIN_DEPTH_NORM,
    )
    DGVlasovSolver[2, 3, P, P](app).compute_rhs(ctx, mesh, q, em, rhs)
    ctx.synchronize()

    var tally = Tally()
    def add(
        x: Real, y: Real, v1: Real, v2: Real, v3: Real, w: Real, r: Real,
    ) {mut tally}:
        tally.add[2, P]([x, y, Real(0.0), v1, v2, v3], r)
    mesh.visit_nodes(rhs, add)
    print(
        "2D3V P =", P, ": max |rhs − exact| =", tally.max_err,
        " max |rhs| =", tally.max_rhs, " nodes compared =", tally.compared,
    )
    assert_true(tally.compared > 0, "no nodes compared")
    assert_true(
        tally.max_err < Real(1.0e-11) * tally.max_rhs, "2D3V RHS is not exact",
    )


def check_3d3v[P: Int](var ctx: DeviceContext) raises:
    var lo = materialize[LO]()
    var hi = materialize[HI]()
    comptime State = PhaseState[3, 3, P, P, VLASOV_NC]
    var mesh = PhaseMesh[3, 3, P, P](
        ctx,
        lo=[lo[0], lo[1], lo[2], lo[3], lo[4], lo[5]],
        hi=[hi[0], hi[1], hi[2], hi[3], hi[4], hi[5]],
        n=[3, 5, 2, 4, 3, 6],
        periodic=[False, False, False],
    )
    var x_mesh = BlockMesh[3, P](
        ctx, lo[0], hi[0], lo[1], hi[1], lo[2], hi[2], 3, 5, 2,
        periodic_x=False, periodic_y=False, periodic_z=False,
    )
    mesh.check_companion(x_mesh)
    var em = DGState[3, P, MAXWELL_NC](ctx, x_mesh.num_elements)
    x_mesh.fill(em, fields_at)

    var q = State(ctx, mesh.num_phase_elements)
    var rhs = State(ctx, mesh.num_phase_elements)
    def ic(
        x: Real, y: Real, z: Real, v1: Real, v2: Real, v3: Real,
    ) -> Real:
        return f_poly[3, P]([x, y, z, v1, v2, v3])
    mesh.fill(q, ic)

    var app = VlasovMaxwell(
        charge_mass_ratio=CHARGE_MASS_RATIO, skin_depth_norm=SKIN_DEPTH_NORM,
    )
    DGVlasovSolver[3, 3, P, P](app).compute_rhs(ctx, mesh, q, em, rhs)
    ctx.synchronize()

    var tally = Tally()
    def add(
        x: Real, y: Real, z: Real, v1: Real, v2: Real, v3: Real,
        w: Real, r: Real,
    ) {mut tally}:
        tally.add[3, P]([x, y, z, v1, v2, v3], r)
    mesh.visit_nodes(rhs, add)
    print(
        "3D3V P =", P, ": max |rhs − exact| =", tally.max_err,
        " max |rhs| =", tally.max_rhs, " nodes compared =", tally.compared,
    )
    assert_true(tally.compared > 0, "no nodes compared")
    assert_true(
        tally.max_err < Real(1.0e-11) * tally.max_rhs, "3D3V RHS is not exact",
    )


def check_companion_rejects_mismatch(var ctx: DeviceContext) raises:
    """`check_companion` must refuse a BlockMesh with other element
    counts or periodicity."""
    var mesh = PhaseMesh[2, 3, 1, 1](
        ctx,
        lo=[0.0, 0.0, -1.0, -1.0, -1.0], hi=[1.0, 1.0, 1.0, 1.0, 1.0],
        n=[3, 5, 2, 2, 2], periodic=[False, True],
    )
    var swapped = BlockMesh[2, 1](
        ctx, 0.0, 1.0, 0.0, 1.0, 5, 3, periodic_x=False, periodic_y=True,
    )
    var periodic = BlockMesh[2, 1](
        ctx, 0.0, 1.0, 0.0, 1.0, 3, 5, periodic_x=True, periodic_y=True,
    )
    var raised = 0
    try:
        mesh.check_companion(swapped)
    except e:
        raised += 1
    try:
        mesh.check_companion(periodic)
    except e:
        raised += 1
    assert_true(raised == 2, "check_companion accepted a mismatched mesh")


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = DeviceContext()
    check_companion_rejects_mismatch(ctx)
    check_2d3v[1](ctx)
    check_2d3v[2](ctx)
    check_3d3v[1](ctx)
    check_3d3v[2](ctx)
    print("PASSED")
