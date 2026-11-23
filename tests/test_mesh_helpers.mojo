# Unit test for the host-side mesh helpers: `fill`, `visit_nodes`,
# `integrate`, `integrate_fn`, `l2_error`, `volume`, `inv_dx_sum` on
# `BlockMesh` in 1D, 2D and 3D, and `fill` / `visit_nodes` on the phase meshes.
#
# Every other test builds its initial condition and diagnostics on these,
# so they are checked here against things computed without them:
#
#   * `fill` is compared node by node with the mesh's own device
#     coordinate buffer (`node_coords_buf`), which the mesh
#     constructor fills independently, using the raw
#     `[element, node, component]` buffer layout. A wrong node order or
#     component stride shows up here. Meshes have unequal element counts
#     and extents so a swapped axis cannot cancel.
#   * `integrate` / `integrate_fn` / `l2_error` are compared with closed
#     forms for polynomials the GLL rule integrates exactly.
#   * `visit_nodes` must hand back the coordinates and state `fill` wrote,
#     with weights that sum to the domain measure.

from std.collections import Array
from std.sys import has_accelerator
from std.testing import assert_true
from max.gpu.host import DeviceContext

from warpxm.basis.nodal_basis_1d import NodalBasis1D
from warpxm.prelude import *
from warpxm.physics.vlasov import *


comptime TOL: Real = 1.0e-13


def _close(a: Real, b: Real, what: String) raises:
    if fabs(a - b) > TOL * fmax(Real(1.0), fabs(b)):
        print(what, ": got", a, "expected", b)
    assert_true(fabs(a - b) <= TOL * fmax(Real(1.0), fabs(b)), what)


def _test_1d[P: Int](var ctx: DeviceContext) raises:
    comptime NC = 3
    var mesh = BlockMesh[1, P](ctx, Real(-1.0), Real(2.0), 7, True)
    var q = DGState[1, P, NC](ctx, mesh.num_elements)
    def ic(x: Real) -> Array[Real, NC]:
        return [x, Real(2.0) - x * x, Real(5.0)]
    mesh.fill(q, ic)

    # Node by node against the mesh's coordinate buffer.
    var worst: Real = 0.0
    with mesh.node_coords_buf.map_to_host() as xs:
        with q.buf.map_to_host() as h:
            for k in range(q.total_nodes):
                var x = xs[k]
                worst = fmax(worst, fabs(h[k * NC + 0] - x))
                worst = fmax(worst, fabs(h[k * NC + 1] - (Real(2.0) - x * x)))
                worst = fmax(worst, fabs(h[k * NC + 2] - Real(5.0)))
    assert_true(worst < TOL, "1D fill does not match the mesh node coordinates")

    _close(mesh.volume(), Real(3.0), "1D volume")
    _close(mesh.inv_dx_sum(), Real(7.0) / Real(3.0), "1D inv_dx_sum")
    _close(mesh.integrate(q, 0), Real(1.5), "1D ∫x")
    _close(mesh.integrate(q, 1), Real(3.0), "1D ∫(2 − x²)")
    _close(mesh.integrate(q, 2), Real(15.0), "1D ∫5")
    def sq(v: Array[Real, NC]) -> Real:
        return v[0] * v[2]
    _close(mesh.integrate_fn(q, sq), Real(7.5), "1D ∫ f(q)")
    def shifted(x: Real) -> Real:
        return x - Real(0.5)
    # ‖0.5‖₂ over a domain of length 3.
    _close(mesh.l2_error(q, 0, shifted), Real(0.5) * sqrt(Real(3.0)), "1D l2_error")
    def same(x: Real) -> Real:
        return Real(2.0) - x * x
    assert_true(mesh.l2_error(q, 1, same) < TOL, "1D l2_error of the exact field")

    var wsum: Real = 0.0
    var bad: Real = 0.0
    var count = 0
    def visit(x: Real, w: Real, v: Array[Real, NC]) {mut wsum, mut bad, mut count}:
        wsum += w
        bad = fmax(bad, fabs(v[0] - x))
        count += 1
    mesh.visit_nodes(q, visit)
    assert_true(count == q.total_nodes, "1D visit_nodes missed nodes")
    _close(wsum, Real(3.0), "1D quadrature weights")
    assert_true(bad < TOL, "1D visit_nodes coordinates disagree with fill")


def _test_2d[P: Int](var ctx: DeviceContext) raises:
    comptime NC = 2
    var mesh = BlockMesh[2, P](
        ctx, Real(-0.5), Real(1.0), Real(0.25), Real(1.25),
        nx=3, ny=5, periodic_x=True, periodic_y=True,
    )
    var q = DGState[2, P, NC](ctx, mesh.num_elements)
    def ic(x: Real, y: Real) -> Array[Real, NC]:
        return [x + Real(10.0) * y, x * y]
    mesh.fill(q, ic)

    var worst: Real = 0.0
    with mesh.node_coords_buf.map_to_host() as xy:
        with q.buf.map_to_host() as h:
            for k in range(q.total_nodes):
                var x = xy[2 * k]
                var y = xy[2 * k + 1]
                worst = fmax(worst, fabs(h[k * NC] - (x + Real(10.0) * y)))
                worst = fmax(worst, fabs(h[k * NC + 1] - x * y))
            # `index` agrees with the raw layout.
            var e = mesh.num_elements - 1
            var n = DGState[2, P, NC].num_nodes_per_elem - 1
            assert_true(
                DGState[2, P, NC].index(e, n, 1) == q.buf_len - 1,
                "DGState.index does not address the last entry",
            )
    assert_true(worst < TOL, "2D fill does not match the mesh node coordinates")

    _close(mesh.volume(), Real(1.5), "2D volume")
    _close(mesh.inv_dx_sum(), Real(2.0) + Real(5.0), "2D inv_dx_sum")
    # ∫x = 0.375 · 1, ∫y = 1.5 · 0.75.
    _close(mesh.integrate(q, 0), Real(0.375) + Real(10.0) * Real(1.125), "2D ∫(x + 10y)")
    _close(mesh.integrate(q, 1), Real(0.375) * Real(0.75), "2D ∫xy")
    def exact(x: Real, y: Real) -> Real:
        return x * y + Real(2.0)
    _close(mesh.l2_error(q, 1, exact), Real(2.0) * sqrt(Real(1.5)), "2D l2_error")

    var wsum: Real = 0.0
    var bad: Real = 0.0
    def visit(x: Real, y: Real, w: Real, v: Array[Real, NC]) {mut wsum, mut bad}:
        wsum += w
        bad = fmax(bad, fabs(v[1] - x * y))
    mesh.visit_nodes(q, visit)
    _close(wsum, Real(1.5), "2D quadrature weights")
    assert_true(bad < TOL, "2D visit_nodes coordinates disagree with fill")


def _test_3d[P: Int](var ctx: DeviceContext) raises:
    comptime NC = 2
    var mesh = BlockMesh[3, P](
        ctx, Real(0.0), Real(1.0), Real(0.0), Real(2.0), Real(-1.0), Real(2.0),
        nx=2, ny=3, nz=4,
    )
    var q = DGState[3, P, NC](ctx, mesh.num_elements)
    def ic(x: Real, y: Real, z: Real) -> Array[Real, NC]:
        return [x + Real(10.0) * y + Real(100.0) * z, x * y * z]
    mesh.fill(q, ic)

    var worst: Real = 0.0
    with mesh.node_coords_buf.map_to_host() as xyz:
        with q.buf.map_to_host() as h:
            for k in range(q.total_nodes):
                var x = xyz[3 * k]
                var y = xyz[3 * k + 1]
                var z = xyz[3 * k + 2]
                worst = fmax(
                    worst,
                    fabs(h[k * NC] - (x + Real(10.0) * y + Real(100.0) * z)),
                )
                worst = fmax(worst, fabs(h[k * NC + 1] - x * y * z))
    assert_true(worst < Real(100.0) * TOL, "3D fill does not match the mesh node coordinates")

    _close(mesh.volume(), Real(6.0), "3D volume")
    _close(mesh.inv_dx_sum(), Real(2.0) + Real(1.5) + Real(4.0) / Real(3.0), "3D inv_dx_sum")
    # ∫x = 0.5·2·3, ∫y = 2·1·3, ∫z = 1.5·1·2.
    _close(mesh.integrate(q, 0), Real(3.0) + Real(60.0) + Real(300.0), "3D ∫ linear")
    _close(mesh.integrate(q, 1), Real(0.5) * Real(2.0) * Real(1.5), "3D ∫xyz")
    def exact(x: Real, y: Real, z: Real) -> Real:
        return x * y * z - Real(1.0)
    _close(mesh.l2_error(q, 1, exact), sqrt(Real(6.0)), "3D l2_error")

    var wsum: Real = 0.0
    var bad: Real = 0.0
    def visit(
        x: Real, y: Real, z: Real, w: Real, v: Array[Real, NC],
    ) {mut wsum, mut bad}:
        wsum += w
        bad = fmax(bad, fabs(v[1] - x * y * z))
    mesh.visit_nodes(q, visit)
    _close(wsum, Real(6.0), "3D quadrature weights")
    assert_true(bad < TOL, "3D visit_nodes coordinates disagree with fill")


def _test_phase_1d1v(var ctx: DeviceContext) raises:
    comptime PX = 2
    comptime PV = 3
    var mesh = PhaseMesh[1, 1, PX, PV](
        ctx, lo=[0.0, -3.0], hi=[2.0, 1.0], n=[3, 5], periodic=[True],
    )
    var f = PhaseState[1, 1, PX, PV, VLASOV_NC](ctx, mesh.num_phase_elements)
    def ic(x: Real, v: Real) -> Real:
        return x + Real(10.0) * v
    mesh.fill(f, ic)

    # Against node coordinates built here from the GLL nodes, and the
    # documented layout: element (ex·Nv + ev), node iv·(P_x+1) + ix.
    var ref_x = NodalBasis1D[PX].nodes()
    var ref_v = NodalBasis1D[PV].nodes()
    var worst: Real = 0.0
    with f.buf.map_to_host() as h:
        for ex in range(3):
            for ev in range(5):
                for iv in range(PV + 1):
                    for ix in range(PX + 1):
                        var x = mesh.lo[0] + mesh.dx[0] * (
                            Real(ex) + Real(0.5) * (ref_x[ix] + Real(1.0))
                        )
                        var v = mesh.lo[1] + mesh.dx[1] * (
                            Real(ev) + Real(0.5) * (ref_v[iv] + Real(1.0))
                        )
                        var idx = PhaseState[1, 1, PX, PV, VLASOV_NC].index(
                            ex * 5 + ev, iv * (PX + 1) + ix, 0,
                        )
                        worst = fmax(
                            worst, fabs(h[idx] - (x + Real(10.0) * v)),
                        )
    assert_true(worst < TOL, "1D1V fill does not match the node coordinates")

    var total: Real = 0.0
    var wsum: Real = 0.0
    var bad: Real = 0.0
    def visit(x: Real, v: Real, w: Real, fv: Real) {mut total, mut wsum, mut bad}:
        total += w * fv
        wsum += w
        bad = fmax(bad, fabs(fv - (x + Real(10.0) * v)))
    mesh.visit_nodes(f, visit)
    _close(wsum, Real(8.0), "1D1V quadrature weights")
    # ∫∫ x = 2·4, ∫∫ v = 2·(−4).
    _close(total, Real(8.0) - Real(80.0), "1D1V ∫∫ f")
    _close(mesh.integrate(f), Real(8.0) - Real(80.0), "1D1V integrate")
    assert_true(bad < TOL, "1D1V visit_nodes coordinates disagree with fill")


def _test_phase_1d2v(var ctx: DeviceContext) raises:
    comptime PX = 1
    comptime PV = 2
    comptime State = PhaseState[1, 2, PX, PV, VLASOV_NC]
    var mesh = PhaseMesh[1, 2, PX, PV](
        ctx, lo=[0.0, -1.0, 0.0], hi=[2.0, 1.0, 3.0], n=[3, 2, 4],
        periodic=[True],
    )
    var f = State(ctx, mesh.num_phase_elements)
    def ic(x: Real, v1: Real, v2: Real) -> Real:
        return x + Real(10.0) * v1 + Real(100.0) * v2
    mesh.fill(f, ic)

    var ref_x = NodalBasis1D[PX].nodes()
    var ref_v = NodalBasis1D[PV].nodes()
    var worst: Real = 0.0
    with f.buf.map_to_host() as h:
        for ex in range(3):
            for ev1 in range(2):
                for ev2 in range(4):
                    var pe = (ex * 2 + ev1) * 4 + ev2
                    for iv2 in range(PV + 1):
                        for iv1 in range(PV + 1):
                            for ix in range(PX + 1):
                                var node = (iv2 * (PV + 1) + iv1) * (PX + 1) + ix
                                var x = mesh.lo[0] + mesh.dx[0] * (
                                    Real(ex) + Real(0.5) * (ref_x[ix] + Real(1.0))
                                )
                                var v1 = mesh.lo[1] + mesh.dx[1] * (
                                    Real(ev1) + Real(0.5) * (ref_v[iv1] + Real(1.0))
                                )
                                var v2 = mesh.lo[2] + mesh.dx[2] * (
                                    Real(ev2) + Real(0.5) * (ref_v[iv2] + Real(1.0))
                                )
                                var want = x + Real(10.0) * v1 + Real(100.0) * v2
                                worst = fmax(
                                    worst,
                                    fabs(h[State.index(pe, node, 0)] - want),
                                )
    assert_true(
        worst < Real(100.0) * TOL, "1D2V fill does not match the node coordinates",
    )

    var total: Real = 0.0
    var wsum: Real = 0.0
    def visit(
        x: Real, v1: Real, v2: Real, w: Real, fv: Real,
    ) {mut total, mut wsum}:
        total += w * fv
        wsum += w
    mesh.visit_nodes(f, visit)
    _close(wsum, Real(12.0), "1D2V quadrature weights")
    # ∫x = 2·2·3, ∫v1 = 0, ∫v2 = 4.5·2·2.
    _close(total, Real(12.0) + Real(100.0) * Real(18.0), "1D2V ∫∫∫ f")


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = DeviceContext()
    _test_1d[2](ctx)
    _test_1d[4](ctx)
    _test_2d[2](ctx)
    _test_2d[4](ctx)
    _test_3d[1](ctx)
    _test_3d[2](ctx)
    _test_phase_1d1v(ctx)
    _test_phase_1d2v(ctx)
    print("PASSED")
