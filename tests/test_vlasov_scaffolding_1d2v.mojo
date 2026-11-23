# 1D2V Vlasov-Maxwell scaffolding smoke test.
#
# Mirrors `test_vlasov_scaffolding_1d1v.mojo` for the 1D2V phase space,
# on the generic `PhaseMesh[1, 2, P_X, P_V]` / `PhaseState[1, 2, P_X, P_V,
# NC]`. No DG kernel is launched. Validates that:
#   * the mesh reports the requested element counts, widths and
#     Jacobians, and its velocity-node buffer holds one run per velocity
#     axis whose endpoints touch the domain extents;
#   * `PhaseState` allocates a zeroed buffer of the right length and
#     `PhaseState.index` follows the documented layout (phase element
#     with the last axis innermost, local node with the first axis
#     innermost), round-tripping a known pattern through device memory;
#   * `visit_nodes` pairs every buffer entry with the coordinates and
#     quadrature weight of its node: the weights sum to the phase volume,
#     the coordinates span exactly the domain on every axis, and `fill` /
#     `integrate` agree with it;
#   * `x_block_element` maps a phase element to its configuration
#     element, and `check_companion` accepts the matching `BlockMesh` and
#     rejects mismatched ones;
#   * `kernel_counts()` / `kernel_bc()` hand the kernels the element
#     counts and `BC_INTERIOR` for a periodic x axis or the per-side BC
#     kinds for a bounded one (the mesh stores no face tables: neighbours
#     and boundary faces follow from the element index);
#   * the constructor rejects empty axes, unsupported configuration BCs
#     and a specular wall on a velocity mesh that is not symmetric;
#   * `VlasovMaxwell.lorentz_acceleration_static[2, axis]` is
#     `(q/m)/λ · (E + v × B)` restricted to two velocity components.
#
# Build / run:
#     pixi run mojo run -I src tests/test_vlasov_scaffolding_1d2v.mojo

from std.collections import Array
from std.sys import has_accelerator
from max.gpu.host import DeviceContext
from std.testing import assert_true, assert_equal

from warpxm.prelude import *
from warpxm.physics.vlasov import *
from warpxm.core.bc import BC_INTERIOR, BC_PEC_WALL


comptime P_X: Int = 1
comptime P_V: Int = 1
comptime NX: Int = 3
comptime NV1: Int = 4
comptime NV2: Int = 5
comptime X_MIN: Real = 0.0
comptime X_MAX: Real = 1.0
comptime V1_MIN: Real = -2.0
comptime V1_MAX: Real = 2.0
comptime V2_MIN: Real = -3.0
comptime V2_MAX: Real = 3.0

comptime Mesh = PhaseMesh[1, 2, P_X, P_V]
comptime Phase = PhaseState[1, 2, P_X, P_V, VLASOV_NC]


def near(a: Real, b: Real) -> Bool:
    return abs(a - b) < Real(1.0e-12)


def mesh_raises(
    var ctx: DeviceContext,
    lo: Array[Real, 3], hi: Array[Real, 3], n: Array[Int, 3],
    bc: Array[Int, 2],
) -> Bool:
    """Whether building a non-periodic mesh with these arguments raises."""
    try:
        var m = Mesh(ctx, lo=lo, hi=hi, n=n, periodic=[False], bc=bc)
        _ = m.num_phase_elements
    except e:
        return True
    return False


def companion_raises(mesh: Mesh, x_mesh: BlockMesh[1, P_X]) -> Bool:
    try:
        mesh.check_companion(x_mesh)
    except e:
        return True
    return False


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"

    var ctx = DeviceContext()

    # --- mesh allocation ---
    var mesh = Mesh(
        ctx,
        lo=[X_MIN, V1_MIN, V2_MIN],
        hi=[X_MAX, V1_MAX, V2_MAX],
        n=[NX, NV1, NV2],
        periodic=[True],
    )
    assert_equal(mesh.n[0], NX, "n[0]")
    assert_equal(mesh.n[1], NV1, "n[1]")
    assert_equal(mesh.n[2], NV2, "n[2]")
    assert_equal(mesh.num_x_elements, NX, "num_x_elements")
    assert_equal(mesh.num_v_elements, NV1 * NV2, "num_v_elements")
    assert_equal(
        mesh.num_phase_elements, NX * NV1 * NV2, "num_phase_elements",
    )

    var expected_dx = [
        (X_MAX - X_MIN) / Real(NX),
        (V1_MAX - V1_MIN) / Real(NV1),
        (V2_MAX - V2_MIN) / Real(NV2),
    ]
    for a in range(3):
        assert_true(near(mesh.dx[a], expected_dx[a]), "dx")
        assert_true(
            near(mesh.jacobian[a], Real(0.5) * expected_dx[a]), "jacobian",
        )
    assert_true(
        near(mesh.lo[0], X_MIN) and near(mesh.hi[0], X_MAX), "x extent",
    )
    assert_true(
        near(mesh.lo[1], V1_MIN) and near(mesh.hi[1], V1_MAX), "v1 extent",
    )
    assert_true(
        near(mesh.lo[2], V2_MIN) and near(mesh.hi[2], V2_MAX), "v2 extent",
    )
    assert_true(near(mesh.phase_volume(), Real(24.0)), "phase volume")
    assert_true(near(mesh.v_abs_max(0), V1_MAX), "v_abs_max(0)")
    assert_true(near(mesh.v_abs_max(1), V2_MAX), "v_abs_max(1)")

    # Velocity-node buffer: the v1 run, then the v2 run; the GLL endpoints
    # of each touch the domain extents.
    var v1_total = NV1 * (P_V + 1)
    var v2_total = NV2 * (P_V + 1)
    assert_equal(mesh.node_v_len, v1_total + v2_total, "node_v_len")
    with mesh.node_v_buf.map_to_host() as h_v:
        assert_true(near(h_v[0], V1_MIN), "v1_min")
        assert_true(near(h_v[v1_total - 1], V1_MAX), "v1_max")
        assert_true(near(h_v[v1_total], V2_MIN), "v2_min")
        assert_true(near(h_v[v1_total + v2_total - 1], V2_MAX), "v2_max")
    print(
        "PhaseMesh[1, 2]: elements (x, v1, v2) =",
        mesh.n[0], mesh.n[1], mesh.n[2],
        " velocity nodes (v1, v2) =", v1_total, v2_total,
    )

    # --- state allocation ---
    var q = Phase(ctx, mesh.num_phase_elements)
    var expected_phase_elems = NX * NV1 * NV2
    var expected_nodes_per_phase = (P_X + 1) * (P_V + 1) * (P_V + 1)
    var expected_total_nodes = expected_phase_elems * expected_nodes_per_phase
    var expected_buf_len = expected_total_nodes * VLASOV_NC
    assert_equal(
        Phase.num_nodes_per_elem, expected_nodes_per_phase,
        "num_nodes_per_elem",
    )
    assert_equal(
        Mesh.num_nodes_per_elem, expected_nodes_per_phase,
        "mesh num_nodes_per_elem",
    )
    assert_equal(
        q.num_phase_elements, expected_phase_elems, "num_phase_elements",
    )
    assert_equal(q.total_nodes, expected_total_nodes, "total_nodes")
    assert_equal(q.buf_len, expected_buf_len, "buf_len")
    print(
        "PhaseState[1, 2]: phase elems =", q.num_phase_elements,
        " buf len =", q.buf_len,
    )

    # Zero-init check.
    with q.buf.map_to_host() as h_q:
        for k in range(q.buf_len):
            assert_true(h_q[k] == Real(0.0), "buffer not zero-initialised")

    # Round-trip a known pattern. The documented layout is
    #   phase element  e = (ex · NV1 + ev1) · NV2 + ev2
    #   local node     n = ix + (P_x+1) · (iv1 + (P_v+1) · iv2)
    #   buffer index   (e · nodes_per_elem + n) · NC + c,
    # which `PhaseState.index` must reproduce.
    var host = ctx.enqueue_create_host_buffer[Real.dtype](q.buf_len)
    for ex in range(NX):
        for ev1 in range(NV1):
            for ev2 in range(NV2):
                for iv2 in range(P_V + 1):
                    for iv1 in range(P_V + 1):
                        for ix in range(P_X + 1):
                            var phase_e = (ex * NV1 + ev1) * NV2 + ev2
                            var local_n = (
                                (iv2 * (P_V + 1) + iv1) * (P_X + 1) + ix
                            )
                            var idx = (
                                phase_e * expected_nodes_per_phase + local_n
                            ) * VLASOV_NC
                            assert_equal(
                                Phase.index(phase_e, local_n, 0), idx,
                                "PhaseState.index",
                            )
                            host[idx] = Real(
                                ex * 100000
                                + ev1 * 10000
                                + ev2 * 1000
                                + iv2 * 100
                                + iv1 * 10
                                + ix
                            )
    ctx.enqueue_copy(dst_buf=q.buf, src_buf=host)
    ctx.synchronize()

    with q.buf.map_to_host() as h_q:
        for ex in range(NX):
            for ev1 in range(NV1):
                for ev2 in range(NV2):
                    for iv2 in range(P_V + 1):
                        for iv1 in range(P_V + 1):
                            for ix in range(P_X + 1):
                                var phase_e = (ex * NV1 + ev1) * NV2 + ev2
                                var local_n = (
                                    (iv2 * (P_V + 1) + iv1) * (P_X + 1) + ix
                                )
                                var expect = Real(
                                    ex * 100000
                                    + ev1 * 10000
                                    + ev2 * 1000
                                    + iv2 * 100
                                    + iv1 * 10
                                    + ix
                                )
                                assert_true(
                                    h_q[Phase.index(phase_e, local_n, 0)]
                                    == expect,
                                    "round-trip",
                                )

    # --- node coordinates and quadrature weights ---
    # The pattern still in `q` names the element and node of every entry,
    # so `visit_nodes` must hand each one the coordinates of that node.
    # At P = 1 the GLL nodes are the element ends: the coordinate along an
    # axis is `lo + (e + i) · dx` and every weight is the product of the
    # three Jacobians.
    var num_visited = 0
    var weight_sum = Real(0.0)
    var bad_coords = 0
    var bad_weights = 0
    var c_min = Array[Real, 3](fill=Real(1.0e30))
    var c_max = Array[Real, 3](fill=Real(-1.0e30))
    var w_expected = mesh.jacobian[0] * mesh.jacobian[1] * mesh.jacobian[2]
    def check_node(
        x: Real, v1: Real, v2: Real, w: Real, f: Real,
    ) {
        mut num_visited, mut weight_sum, mut bad_coords, mut bad_weights,
        mut c_min, mut c_max, imm,
    }:
        var code = Int(f + Real(0.5))
        var ix = code % 10
        var iv1 = (code // 10) % 10
        var iv2 = (code // 100) % 10
        var ev2 = (code // 1000) % 10
        var ev1 = (code // 10000) % 10
        var ex = code // 100000
        if not (
            near(x, X_MIN + Real(ex + ix) * mesh.dx[0])
            and near(v1, V1_MIN + Real(ev1 + iv1) * mesh.dx[1])
            and near(v2, V2_MIN + Real(ev2 + iv2) * mesh.dx[2])
        ):
            bad_coords += 1
        if not near(w, w_expected):
            bad_weights += 1
        c_min[0] = fmin(c_min[0], x)
        c_max[0] = fmax(c_max[0], x)
        c_min[1] = fmin(c_min[1], v1)
        c_max[1] = fmax(c_max[1], v1)
        c_min[2] = fmin(c_min[2], v2)
        c_max[2] = fmax(c_max[2], v2)
        weight_sum += w
        num_visited += 1
    mesh.visit_nodes(q, check_node)
    assert_equal(num_visited, expected_total_nodes, "visit_nodes count")
    assert_equal(bad_coords, 0, "node coordinates do not match the layout")
    assert_equal(bad_weights, 0, "node quadrature weights")
    assert_true(
        abs(weight_sum - mesh.phase_volume()) < Real(1.0e-12),
        "weights do not sum to the phase volume",
    )
    assert_true(near(c_min[0], X_MIN) and near(c_max[0], X_MAX), "x range")
    assert_true(near(c_min[1], V1_MIN) and near(c_max[1], V1_MAX), "v1 range")
    assert_true(near(c_min[2], V2_MIN) and near(c_max[2], V2_MAX), "v2 range")
    print(
        "visit_nodes: nodes =", num_visited, " Σw =", weight_sum,
        " phase volume =", mesh.phase_volume(),
    )

    # `fill` writes through the same node map, and `integrate` uses the
    # same weights: a linear function is integrated exactly, and the
    # velocity terms vanish on the symmetric velocity domain, leaving
    # ∫ (1 + x) dx dv = 1.5 · 24.
    def linear(x: Real, v1: Real, v2: Real) -> Real:
        return Real(1.0) + x + Real(10.0) * v1 + Real(100.0) * v2
    mesh.fill(q, linear)
    var bad_fill = 0
    def check_fill(
        x: Real, v1: Real, v2: Real, w: Real, f: Real,
    ) {mut bad_fill}:
        if not near(f, Real(1.0) + x + Real(10.0) * v1 + Real(100.0) * v2):
            bad_fill += 1
    mesh.visit_nodes(q, check_fill)
    assert_equal(bad_fill, 0, "fill and visit_nodes disagree")
    var integral = mesh.integrate(q)
    assert_true(
        abs(integral - Real(36.0)) < Real(1.0e-10),
        "integrate of a linear function",
    )
    # One specific entry: element (ex, ev1, ev2) = (2, 1, 3), node
    # (ix, iv1, iv2) = (1, 0, 1) sits at x = 1, v1 = −1, v2 = 0.6 + 1.2.
    with q.buf.map_to_host() as h_q:
        var e_s = (2 * NV1 + 1) * NV2 + 3
        var n_s = 1 + (P_X + 1) * (0 + (P_V + 1) * 1)
        var x_s = X_MIN + Real(3) * mesh.dx[0]
        var v1_s = V1_MIN + Real(1) * mesh.dx[1]
        var v2_s = V2_MIN + Real(4) * mesh.dx[2]
        assert_true(
            near(
                h_q[Phase.index(e_s, n_s, 0)],
                Real(1.0) + x_s + Real(10.0) * v1_s + Real(100.0) * v2_s,
            ),
            "fill at a sampled node",
        )
    print("fill / integrate: ∫ f =", integral)

    # --- configuration-space correspondence ---
    # Phase element (ex, ev1, ev2) sits over configuration element ex.
    for ex in range(NX):
        for ev1 in range(NV1):
            for ev2 in range(NV2):
                assert_equal(
                    mesh.x_block_element((ex * NV1 + ev1) * NV2 + ev2), ex,
                    "x_block_element",
                )
    var x_mesh = BlockMesh[1, P_X](ctx, X_MIN, X_MAX, NX, periodic=True)
    var x_mesh_count = BlockMesh[1, P_X](
        ctx, X_MIN, X_MAX, NX + 1, periodic=True,
    )
    var x_mesh_extent = BlockMesh[1, P_X](
        ctx, X_MIN, Real(2.0) * X_MAX, NX, periodic=True,
    )
    var x_mesh_bounded = BlockMesh[1, P_X](
        ctx, X_MIN, X_MAX, NX, periodic=False,
    )
    assert_true(
        not companion_raises(mesh, x_mesh),
        "check_companion rejected the matching mesh",
    )
    assert_true(
        companion_raises(mesh, x_mesh_count),
        "check_companion accepted a different element count",
    )
    assert_true(
        companion_raises(mesh, x_mesh_extent),
        "check_companion accepted a different extent",
    )
    assert_true(
        companion_raises(mesh, x_mesh_bounded),
        "check_companion accepted a different periodicity",
    )

    # --- what the kernels receive ---
    # Periodic x: both sides are BC_INTERIOR. Velocity boundaries are not
    # listed; the kernel always uses a zero ghost state there.
    var counts = mesh.kernel_counts()
    assert_equal(Int(counts[0]), NX, "kernel_counts x")
    assert_equal(Int(counts[1]), NV1, "kernel_counts v1")
    assert_equal(Int(counts[2]), NV2, "kernel_counts v2")
    var inv_jac = mesh.kernel_inv_jacobian()
    for a in range(3):
        assert_true(
            near(inv_jac[a], Real(2.0) / expected_dx[a]),
            "kernel_inv_jacobian",
        )
    var bc_periodic = mesh.kernel_bc()
    assert_equal(Int(bc_periodic[0]), BC_INTERIOR, "x-left BC (periodic)")
    assert_equal(Int(bc_periodic[1]), BC_INTERIOR, "x-right BC (periodic)")

    # Bounded x with the default BCs: BC_COPYOUT on both sides.
    var mesh_np = Mesh(
        ctx,
        lo=[X_MIN, V1_MIN, V2_MIN],
        hi=[X_MAX, V1_MAX, V2_MAX],
        n=[NX, NV1, NV2],
        periodic=[False],
    )
    assert_equal(
        mesh_np.num_phase_elements, NX * NV1 * NV2,
        "num_phase_elements (non-periodic)",
    )
    var bc_np = mesh_np.kernel_bc()
    assert_equal(Int(bc_np[0]), BC_COPYOUT, "x-left BC (default)")
    assert_equal(Int(bc_np[1]), BC_COPYOUT, "x-right BC (default)")

    # Bounded x with one kind per side. The specular wall needs the v1
    # mesh symmetric about zero; v2 may be anything.
    var mesh_wall = Mesh(
        ctx,
        lo=[X_MIN, V1_MIN, V2_MIN],
        hi=[X_MAX, V1_MAX, Real(2.0) * V2_MAX],
        n=[NX, NV1, NV2],
        periodic=[False],
        bc=[BC_VLASOV_OUTFLOW, BC_VLASOV_REFLECT],
    )
    var bc_wall = mesh_wall.kernel_bc()
    assert_equal(Int(bc_wall[0]), BC_VLASOV_OUTFLOW, "x-left BC (outflow)")
    assert_equal(Int(bc_wall[1]), BC_VLASOV_REFLECT, "x-right BC (reflect)")
    print(
        "kernel_bc: periodic =", Int(bc_periodic[0]), Int(bc_periodic[1]),
        " default =", Int(bc_np[0]), Int(bc_np[1]),
        " outflow / reflect =", Int(bc_wall[0]), Int(bc_wall[1]),
    )

    # --- constructor validation ---
    var lo_ok: Array[Real, 3] = [X_MIN, V1_MIN, V2_MIN]
    var hi_ok: Array[Real, 3] = [X_MAX, V1_MAX, V2_MAX]
    var n_ok: Array[Int, 3] = [NX, NV1, NV2]
    var bc_ok: Array[Int, 2] = [BC_COPYOUT, BC_COPYOUT]
    assert_true(
        not mesh_raises(ctx, lo_ok, hi_ok, n_ok, bc_ok),
        "valid mesh rejected",
    )
    assert_true(
        mesh_raises(ctx, lo_ok, hi_ok, [NX, 0, NV2], bc_ok),
        "axis with no elements accepted",
    )
    assert_true(
        mesh_raises(ctx, lo_ok, [X_MAX, V1_MIN, V2_MAX], n_ok, bc_ok),
        "axis of zero width accepted",
    )
    assert_true(
        mesh_raises(ctx, lo_ok, hi_ok, n_ok, [BC_COPYOUT, BC_PEC_WALL]),
        "field BC kind accepted on a phase mesh",
    )
    assert_true(
        mesh_raises(ctx, lo_ok, hi_ok, n_ok, [BC_INTERIOR, BC_COPYOUT]),
        "BC_INTERIOR accepted on a bounded axis",
    )
    assert_true(
        mesh_raises(
            ctx, lo_ok, [X_MAX, Real(2.0) * V1_MAX, V2_MAX], n_ok,
            [BC_VLASOV_REFLECT, BC_COPYOUT],
        ),
        "specular wall accepted on an asymmetric v1 mesh",
    )

    # --- Lorentz acceleration (two velocity components) ---
    # `lorentz_acceleration_static[2, axis](E, B, v, factor)` with
    # `factor = (q/m)/λ`:
    #   axis 0:  factor · (E_x + v2·B_z)
    #   axis 1:  factor · (E_y − v1·B_z)
    # The `v3`, `B_x` and `B_y` terms belong to a third velocity
    # component and must not contribute.
    var qmr = Real(-1.0)
    var lam = Real(2.0)
    var coeff = VlasovMaxwell.v_flux_factor_static(qmr, lam)
    assert_true(near(coeff, Real(-0.5)), "v_flux_factor_static")

    # E only: Ex=1, Bz=2, v2=0 → av1 = coeff·1 = -0.5.
    var av1_a = VlasovMaxwell.lorentz_acceleration_static[2, 0](
        Real(1.0), Real(0.0), Real(0.0), Real(0.0), Real(0.0), Real(2.0),
        Real(0.0), Real(0.0), Real(0.0), coeff,
    )
    assert_true(near(av1_a, coeff), "av1 (Ex only)")

    # `v × B` term: Ex=0, Bz=2, v2=3 → av1 = coeff·(0 + 3·2) = -3.
    var av1_b = VlasovMaxwell.lorentz_acceleration_static[2, 0](
        Real(0.0), Real(0.0), Real(0.0), Real(0.0), Real(0.0), Real(2.0),
        Real(0.0), Real(3.0), Real(0.0), coeff,
    )
    assert_true(near(av1_b, coeff * Real(6.0)), "av1 (v×B term)")

    # Sign of v×B in v2: Ey=0, Bz=2, v1=3 → av2 = coeff·(0 − 3·2) = +3.
    var av2_a = VlasovMaxwell.lorentz_acceleration_static[2, 1](
        Real(0.0), Real(0.0), Real(0.0), Real(0.0), Real(0.0), Real(2.0),
        Real(3.0), Real(0.0), Real(0.0), coeff,
    )
    assert_true(near(av2_a, coeff * Real(-6.0)), "av2 (v×B sign)")

    # B=0 collapse: both reduce to (q/m)/λ · E component.
    var av1_c = VlasovMaxwell.lorentz_acceleration_static[2, 0](
        Real(0.7), Real(0.3), Real(0.0), Real(0.0), Real(0.0), Real(0.0),
        Real(4.0), Real(5.0), Real(0.0), coeff,
    )
    assert_true(near(av1_c, coeff * Real(0.7)), "av1 (B=0 collapse)")
    var av2_c = VlasovMaxwell.lorentz_acceleration_static[2, 1](
        Real(0.7), Real(0.3), Real(0.0), Real(0.0), Real(0.0), Real(0.0),
        Real(4.0), Real(5.0), Real(0.0), coeff,
    )
    assert_true(near(av2_c, coeff * Real(0.3)), "av2 (B=0 collapse)")

    # Everything at once, with the 3V-only inputs (E_z, B_x, B_y, v3) set
    # to values that would show up if they leaked in:
    #   av1 = coeff · (0.7 + 5·2) = coeff · 10.7
    #   av2 = coeff · (0.3 − 4·2) = coeff · (−7.7)
    var av1_d = VlasovMaxwell.lorentz_acceleration_static[2, 0](
        Real(0.7), Real(0.3), Real(11.0), Real(13.0), Real(17.0), Real(2.0),
        Real(4.0), Real(5.0), Real(19.0), coeff,
    )
    assert_true(near(av1_d, coeff * Real(10.7)), "av1 (full, 2V only)")
    var av2_d = VlasovMaxwell.lorentz_acceleration_static[2, 1](
        Real(0.7), Real(0.3), Real(11.0), Real(13.0), Real(17.0), Real(2.0),
        Real(4.0), Real(5.0), Real(19.0), coeff,
    )
    assert_true(near(av2_d, coeff * Real(-7.7)), "av2 (full, 2V only)")

    print("PASSED")
