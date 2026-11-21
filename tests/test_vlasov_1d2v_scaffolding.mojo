# 1D2V Vlasov-Maxwell scaffolding smoke test (slice 17).
#
# Mirrors `test_vlasov_1d1v_scaffolding.mojo` for the higher-
# dimensional 1D2V phase space. Validates that:
#   * `PhaseMesh1D2V` allocates three node-coordinate buffers and
#     hits the GLL endpoints at the requested domain extents on
#     every axis.
#   * `PhaseState1D2V` allocates a zero-initialised flat buffer of
#     the right length and round-trips a known pattern through
#     device memory using the v2-outer / v1-middle / x-inner index
#     formula.
#
# No DG kernel, no face connectivity, no flux logic yet — those
# arrive in follow-up slices.
#
# Build / run:
#     pixi run mojo run -I src tests/test_vlasov_1d2v_scaffolding.mojo

from std.sys import has_accelerator
from max.gpu.host import DeviceContext
from std.testing import assert_true, assert_equal

from warpxm.apps.vlasovmaxwell import VLASOV_NC, VlasovMaxwell
from warpxm.core.bc import BC_INTERIOR, BC_COPYOUT, BC_VLASOV_OUTFLOW
from warpxm.core.types import Real, INDEX_DTYPE
from warpxm.mesh.block_3d import (
    GHOST_NEG_3D, GHOST_POS_3D,
    FACE_DIR_LEFT_3D, FACE_DIR_RIGHT_3D,
    FACE_DIR_BOTTOM_3D, FACE_DIR_TOP_3D,
    FACE_DIR_DOWN_3D, FACE_DIR_UP_3D,
)
from warpxm.mesh.phase_mesh_1d2v import PhaseMesh1D2V
from warpxm.variable.phase_state_1d2v import PhaseState1D2V


comptime P_X: Int = 1
comptime P_V1: Int = 1
comptime P_V2: Int = 1
comptime NX: Int = 3
comptime NV1: Int = 4
comptime NV2: Int = 5


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"

    var ctx = DeviceContext()

    # --- mesh allocation ---
    var mesh = PhaseMesh1D2V[P_X, P_V1, P_V2](
        ctx,
        Real(0.0), Real(1.0),
        Real(-2.0), Real(2.0),
        Real(-3.0), Real(3.0),
        num_x=NX, num_v1=NV1, num_v2=NV2,
        periodic_x=True,
    )
    assert_equal(mesh.num_x, NX, "num_x")
    assert_equal(mesh.num_v1, NV1, "num_v1")
    assert_equal(mesh.num_v2, NV2, "num_v2")
    assert_equal(
        mesh.num_phase_elements, NX * NV1 * NV2, "num_phase_elements",
    )

    var expected_dx = Real(1.0) / Real(NX)
    var expected_dv1 = Real(4.0) / Real(NV1)
    var expected_dv2 = Real(6.0) / Real(NV2)
    assert_true(
        (mesh.dx - expected_dx) ** Real(2.0) < Real(1.0e-28), "dx",
    )
    assert_true(
        (mesh.dv1 - expected_dv1) ** Real(2.0) < Real(1.0e-28), "dv1",
    )
    assert_true(
        (mesh.dv2 - expected_dv2) ** Real(2.0) < Real(1.0e-28), "dv2",
    )

    # GLL endpoints touch the domain extents on every axis.
    var x_total = NX * (P_X + 1)
    var v1_total = NV1 * (P_V1 + 1)
    var v2_total = NV2 * (P_V2 + 1)
    with mesh.node_x_buf.map_to_host() as h_x:
        assert_true(
            (h_x[0] - Real(0.0)) ** Real(2.0) < Real(1.0e-24), "x_min",
        )
        assert_true(
            (h_x[x_total - 1] - Real(1.0)) ** Real(2.0) < Real(1.0e-24),
            "x_max",
        )
    with mesh.node_v1_buf.map_to_host() as h_v1:
        assert_true(
            (h_v1[0] - Real(-2.0)) ** Real(2.0) < Real(1.0e-24), "v1_min",
        )
        assert_true(
            (h_v1[v1_total - 1] - Real(2.0)) ** Real(2.0) < Real(1.0e-24),
            "v1_max",
        )
    with mesh.node_v2_buf.map_to_host() as h_v2:
        assert_true(
            (h_v2[0] - Real(-3.0)) ** Real(2.0) < Real(1.0e-24), "v2_min",
        )
        assert_true(
            (h_v2[v2_total - 1] - Real(3.0)) ** Real(2.0) < Real(1.0e-24),
            "v2_max",
        )
    print(
        "PhaseMesh1D2V: nodes (x, v1, v2) =", x_total, v1_total, v2_total,
    )

    # --- state allocation ---
    var q = PhaseState1D2V[P_X, P_V1, P_V2, VLASOV_NC](ctx, NX, NV1, NV2)
    var expected_phase_elems = NX * NV1 * NV2
    var expected_nodes_per_phase = (P_X + 1) * (P_V1 + 1) * (P_V2 + 1)
    var expected_total_nodes = expected_phase_elems * expected_nodes_per_phase
    var expected_buf_len = expected_total_nodes * VLASOV_NC
    assert_equal(q.num_x, NX, "state num_x")
    assert_equal(q.num_v1, NV1, "state num_v1")
    assert_equal(q.num_v2, NV2, "state num_v2")
    assert_equal(
        q.num_phase_elements, expected_phase_elems, "num_phase_elements",
    )
    assert_equal(q.buf_len, expected_buf_len, "buf_len")
    print(
        "PhaseState1D2V: phase elems =", q.num_phase_elements,
        " buf len =", q.buf_len,
    )

    # Zero-init check.
    with q.buf.map_to_host() as h_q:
        for k in range(q.buf_len):
            assert_true(h_q[k] == Real(0.0), "buffer not zero-initialised")

    # Round-trip a known pattern. The index formula is
    # ((((phase_e · (P_v2+1) + iv2) · (P_v1+1) + iv1) · (P_x+1) + ix)
    #  · NC + c.
    var host = ctx.enqueue_create_host_buffer[Real.dtype](q.buf_len)
    for ex in range(NX):
        for ev1 in range(NV1):
            for ev2 in range(NV2):
                for iv2 in range(P_V2 + 1):
                    for iv1 in range(P_V1 + 1):
                        for ix in range(P_X + 1):
                            var phase_e = (ex * NV1 + ev1) * NV2 + ev2
                            var local_n = (
                                (iv2 * (P_V1 + 1) + iv1) * (P_X + 1) + ix
                            )
                            var idx = (
                                phase_e * expected_nodes_per_phase + local_n
                            ) * VLASOV_NC
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
                    for iv2 in range(P_V2 + 1):
                        for iv1 in range(P_V1 + 1):
                            for ix in range(P_X + 1):
                                var phase_e = (ex * NV1 + ev1) * NV2 + ev2
                                var local_n = (
                                    (iv2 * (P_V1 + 1) + iv1) * (P_X + 1) + ix
                                )
                                var idx = (
                                    phase_e * expected_nodes_per_phase + local_n
                                ) * VLASOV_NC
                                var expect = Real(
                                    ex * 100000
                                    + ev1 * 10000
                                    + ev2 * 1000
                                    + iv2 * 100
                                    + iv1 * 10
                                    + ix
                                )
                                assert_true(
                                    h_q[idx] == expect, "round-trip",
                                )

    # --- face connectivity (periodic-x case) ---
    # Counts: periodic x ⇒ num_x faces per (ev1, ev2) row; v1 / v2 always
    # carry the +1 outflow boundary face.
    var expected_x_faces = NX * NV1 * NV2
    var expected_v1_faces = NX * (NV1 + 1) * NV2
    var expected_v2_faces = NX * NV1 * (NV2 + 1)
    var expected_total = expected_x_faces + expected_v1_faces + expected_v2_faces
    assert_equal(mesh.num_x_faces, expected_x_faces, "num_x_faces (periodic)")
    assert_equal(mesh.num_v1_faces, expected_v1_faces, "num_v1_faces")
    assert_equal(mesh.num_v2_faces, expected_v2_faces, "num_v2_faces")
    assert_equal(mesh.num_faces, expected_total, "num_faces (periodic)")
    print(
        "PhaseMesh1D2V faces: x =", mesh.num_x_faces,
        " v1 =", mesh.num_v1_faces,
        " v2 =", mesh.num_v2_faces,
        " total =", mesh.num_faces,
    )

    # All v1- and v2-boundary faces should carry BC_VLASOV_OUTFLOW and have
    # ghost sentinels on their outward sides. Interior axis faces should be
    # BC_INTERIOR.
    with mesh.face_bc_buf.map_to_host() as h_bc, \
         mesh.face_neg_elem_buf.map_to_host() as h_fn, \
         mesh.face_pos_elem_buf.map_to_host() as h_fp:
        # x-axis faces (all interior under periodic_x=True).
        for f in range(mesh.num_x_faces):
            assert_equal(
                Int(h_bc[f]), BC_INTERIOR, "x-face BC under periodic_x",
            )
        # v1-axis boundary faces sit at j_face ∈ {0, NV1}; with the layout
        # f_idx = off_v1 + ex*(NV1_p1 * NV2) + j_face * NV2 + ev2, the
        # bottom-row indices for each (ex, ev2) are off + ex*… + 0*NV2 + ev2.
        var off_v1 = mesh.num_x_faces
        var nv1_p1 = NV1 + 1
        for ex in range(NX):
            for ev2 in range(NV2):
                var f_bot = off_v1 + ex * (nv1_p1 * NV2) + 0 * NV2 + ev2
                var f_top = off_v1 + ex * (nv1_p1 * NV2) + NV1 * NV2 + ev2
                assert_equal(
                    Int(h_bc[f_bot]),
                    BC_VLASOV_OUTFLOW,
                    "v1-bottom BC",
                )
                assert_equal(
                    Int(h_bc[f_top]),
                    BC_VLASOV_OUTFLOW,
                    "v1-top BC",
                )
                assert_equal(
                    Int(h_fn[f_bot]),
                    Int(GHOST_NEG_3D),
                    "v1-bottom ghost neg",
                )
                assert_equal(
                    Int(h_fp[f_top]),
                    Int(GHOST_POS_3D),
                    "v1-top ghost pos",
                )
        # v2-axis boundary faces sit at k_face ∈ {0, NV2}.
        var off_v2 = mesh.num_x_faces + mesh.num_v1_faces
        var nv2_p1 = NV2 + 1
        for ex in range(NX):
            for ev1 in range(NV1):
                var f_dn = off_v2 + ex * (NV1 * nv2_p1) + ev1 * nv2_p1 + 0
                var f_up = off_v2 + ex * (NV1 * nv2_p1) + ev1 * nv2_p1 + NV2
                assert_equal(
                    Int(h_bc[f_dn]),
                    BC_VLASOV_OUTFLOW,
                    "v2-down BC",
                )
                assert_equal(
                    Int(h_bc[f_up]),
                    BC_VLASOV_OUTFLOW,
                    "v2-up BC",
                )
                assert_equal(
                    Int(h_fn[f_dn]),
                    Int(GHOST_NEG_3D),
                    "v2-down ghost neg",
                )
                assert_equal(
                    Int(h_fp[f_up]),
                    Int(GHOST_POS_3D),
                    "v2-up ghost pos",
                )

    # `elem_face_dir` should point each element at six valid face indices.
    with mesh.elem_face_dir_buf.map_to_host() as h_efd:
        for phase_e in range(NX * NV1 * NV2):
            for dir in range(6):
                var f_idx = Int(h_efd[phase_e * 6 + dir])
                assert_true(
                    f_idx >= 0 and f_idx < mesh.num_faces,
                    "elem_face_dir out of range",
                )

    # --- non-periodic-x face counts + boundary BCs ---
    var mesh_np = PhaseMesh1D2V[P_X, P_V1, P_V2](
        ctx,
        Real(0.0), Real(1.0),
        Real(-2.0), Real(2.0),
        Real(-3.0), Real(3.0),
        num_x=NX, num_v1=NV1, num_v2=NV2,
        periodic_x=False,
        bc_left=BC_COPYOUT, bc_right=BC_COPYOUT,
    )
    var expected_x_faces_np = (NX + 1) * NV1 * NV2
    assert_equal(
        mesh_np.num_x_faces,
        expected_x_faces_np,
        "num_x_faces (non-periodic)",
    )
    with mesh_np.face_bc_buf.map_to_host() as h_bc_np:
        # Leftmost x-face (i_face=0) and rightmost (i_face=NX) in each
        # (ev1, ev2) row should be BC_COPYOUT.
        for ev1 in range(NV1):
            for ev2 in range(NV2):
                var f_left = 0 * (NV1 * NV2) + ev1 * NV2 + ev2
                var f_right = NX * (NV1 * NV2) + ev1 * NV2 + ev2
                assert_equal(
                    Int(h_bc_np[f_left]),
                    BC_COPYOUT,
                    "x-left BC under non-periodic_x",
                )
                assert_equal(
                    Int(h_bc_np[f_right]),
                    BC_COPYOUT,
                    "x-right BC under non-periodic_x",
                )

    # --- Lorentz helpers (1D2V) ---
    # `acceleration_v1_1d2v_static(Ex, Bz, v2, q/m, λ) = (q/m)/λ · (Ex + v2·Bz)`
    # `acceleration_v2_1d2v_static(Ey, Bz, v1, q/m, λ) = (q/m)/λ · (Ey − v1·Bz)`
    var qmr = Real(-1.0)
    var lam = Real(2.0)
    var coeff = qmr / lam  # = -0.5

    # E×B drift sanity: Ex=1, Bz=2, v2=0 → av1 = coeff·1 = -0.5.
    var av1_a = VlasovMaxwell.acceleration_v1_1d2v_static(
        Real(1.0), Real(2.0), Real(0.0), qmr, lam,
    )
    assert_true((av1_a - coeff) ** Real(2.0) < Real(1.0e-28), "av1 (Ex only)")

    # `v × B` term: Ex=0, Bz=2, v2=3 → av1 = coeff·(0 + 3·2) = -3.
    var av1_b = VlasovMaxwell.acceleration_v1_1d2v_static(
        Real(0.0), Real(2.0), Real(3.0), qmr, lam,
    )
    assert_true(
        (av1_b - coeff * Real(6.0)) ** Real(2.0) < Real(1.0e-28),
        "av1 (v×B term)",
    )

    # Sign of v×B in v2: Ey=0, Bz=2, v1=3 → av2 = coeff·(0 − 3·2) = +3.
    var av2_a = VlasovMaxwell.acceleration_v2_1d2v_static(
        Real(0.0), Real(2.0), Real(3.0), qmr, lam,
    )
    assert_true(
        (av2_a - coeff * Real(-6.0)) ** Real(2.0) < Real(1.0e-28),
        "av2 (v×B sign)",
    )

    # B=0 collapse: both helpers reduce to (q/m)/λ · E component.
    var av1_c = VlasovMaxwell.acceleration_v1_1d2v_static(
        Real(0.7), Real(0.0), Real(5.0), qmr, lam,
    )
    assert_true(
        (av1_c - coeff * Real(0.7)) ** Real(2.0) < Real(1.0e-28),
        "av1 (B=0 collapse)",
    )
    var av2_c = VlasovMaxwell.acceleration_v2_1d2v_static(
        Real(0.3), Real(0.0), Real(5.0), qmr, lam,
    )
    assert_true(
        (av2_c - coeff * Real(0.3)) ** Real(2.0) < Real(1.0e-28),
        "av2 (B=0 collapse)",
    )

    print("PASSED")
