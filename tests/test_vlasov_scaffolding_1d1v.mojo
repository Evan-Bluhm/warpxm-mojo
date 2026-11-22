# Smoke test for the 1D1V Vlasov-Maxwell port — first two slices.
#
# Verifies that the phase-space scaffolding (mesh + face connectivity,
# state buffer, app constants + flux helpers) allocates correctly and
# computes the expected values. No DG kernel yet: that arrives with the
# spatial-solver slice.
#
# Build / run:
#     pixi run mojo run -I src tests/test_vlasov_scaffolding_1d1v.mojo

from std.collections import Array
from std.sys import has_accelerator
from max.gpu.host import DeviceContext
from std.testing import assert_true, assert_equal

from warpxm.prelude import *
from warpxm.physics.vlasov import *
from warpxm.apps.vlasovmaxwell import VLASOV_AUX_NC, AUX_EX
from warpxm.mesh.block import GHOST_NEG, GHOST_POS


comptime P_X: Int = 2
comptime P_V: Int = 2
comptime NX: Int = 4
comptime NV: Int = 8


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"

    var ctx = DeviceContext()

    # --- mesh allocation --------------------------------------------------
    var mesh = PhaseMesh1D1V[P_X, P_V](
        ctx,
        Real(0.0), Real(1.0),
        Real(-3.0), Real(3.0),
        num_x=NX, num_v=NV,
        periodic_x=True,
    )
    assert_equal(mesh.num_x, NX, "num_x mismatch")
    assert_equal(mesh.num_v, NV, "num_v mismatch")
    assert_equal(mesh.num_phase_elements, NX * NV, "num_phase_elements")

    # Face counts: periodic-x ⇒ NX x-faces per v-row, plus (NV + 1) v-face rows.
    var expected_x_faces = NX * NV
    var expected_v_faces = NX * (NV + 1)
    assert_equal(mesh.num_x_faces, expected_x_faces, "num_x_faces")
    assert_equal(mesh.num_v_faces, expected_v_faces, "num_v_faces")
    assert_equal(
        mesh.num_faces, expected_x_faces + expected_v_faces, "num_faces",
    )

    var expected_dx = Real(1.0) / Real(NX)
    var expected_dv = Real(6.0) / Real(NV)
    assert_true(
        (mesh.dx - expected_dx) ** Real(2.0) < Real(1.0e-28), "dx",
    )
    assert_true(
        (mesh.dv - expected_dv) ** Real(2.0) < Real(1.0e-28), "dv",
    )

    # Spot-check node coordinates: GLL endpoints hit the domain extents.
    var x_nodes_total = NX * (P_X + 1)
    var v_nodes_total = NV * (P_V + 1)
    with mesh.node_x_buf.map_to_host() as h_x:
        assert_true(
            (h_x[0] - Real(0.0)) ** Real(2.0) < Real(1.0e-24),
            "leftmost x-node",
        )
        assert_true(
            (h_x[x_nodes_total - 1] - Real(1.0)) ** Real(2.0) < Real(1.0e-24),
            "rightmost x-node",
        )
    with mesh.node_v_buf.map_to_host() as h_v:
        assert_true(
            (h_v[0] - Real(-3.0)) ** Real(2.0) < Real(1.0e-24),
            "leftmost v-node",
        )
        assert_true(
            (h_v[v_nodes_total - 1] - Real(3.0)) ** Real(2.0) < Real(1.0e-24),
            "rightmost v-node",
        )

    # Spot-check face connectivity. With periodic_x, the first x-face in
    # velocity row ev=0 should wrap (ex=num_x-1, ev=0) ↔ (ex=0, ev=0).
    with mesh.face_neg_elem_buf.map_to_host() as h_fn:
        with mesh.face_pos_elem_buf.map_to_host() as h_fp:
            var f0 = 0
            assert_equal(
                Int(h_fn[f0]), (NX - 1) * NV + 0,
                "periodic x-face 0 negative",
            )
            assert_equal(
                Int(h_fp[f0]), 0 * NV + 0,
                "periodic x-face 0 positive",
            )

            # First v-axis face (bottom of column ex=0) should have a
            # GHOST_NEG sentinel on the negative side.
            var v_bottom_first = mesh.num_x_faces + 0  # ex=0, j_face=0
            assert_equal(
                Int(h_fn[v_bottom_first]), Int(GHOST_NEG),
                "v-axis bottom-most face neg side should be GHOST_NEG",
            )
            assert_equal(
                Int(h_fp[v_bottom_first]), 0 * NV + 0,
                "v-axis bottom-most face pos side",
            )

            # Top-most v-face for ex=0 has GHOST_POS on the positive side.
            var v_top_first = mesh.num_x_faces + NV * NX + 0
            assert_equal(
                Int(h_fn[v_top_first]), 0 * NV + (NV - 1),
                "v-axis top-most face neg side",
            )
            assert_equal(
                Int(h_fp[v_top_first]), Int(GHOST_POS),
                "v-axis top-most face pos side should be GHOST_POS",
            )

    print(
        "PhaseMesh1D1V: x-faces =", mesh.num_x_faces,
        "v-faces =", mesh.num_v_faces,
        "total =", mesh.num_faces,
    )

    # --- state allocation -------------------------------------------------
    var q = PhaseState1D1V[P_X, P_V, VLASOV_NC](ctx, NX, NV)
    var expected_nodes_per_phase = (P_X + 1) * (P_V + 1)
    var expected_buf_len = NX * NV * expected_nodes_per_phase * VLASOV_NC
    assert_equal(q.buf_len, expected_buf_len, "buf_len")

    # Zero-fill check.
    with q.buf.map_to_host() as h_q:
        for k in range(q.buf_len):
            assert_true(h_q[k] == Real(0.0), "buffer not zero-initialised")

    # Round-trip a known pattern through device memory.
    var host = ctx.enqueue_create_host_buffer[Real.dtype](q.buf_len)
    for ex in range(NX):
        for ev in range(NV):
            for ix in range(P_X + 1):
                for iv in range(P_V + 1):
                    var phase_e = ex * NV + ev
                    var idx = (
                        (phase_e * (P_V + 1) + iv) * (P_X + 1) + ix
                    ) * VLASOV_NC
                    host[idx] = Real(ex * 1000 + ev * 100 + ix * 10 + iv)
    ctx.enqueue_copy(dst_buf=q.buf, src_buf=host)
    ctx.synchronize()
    with q.buf.map_to_host() as h_q:
        for ex in range(NX):
            for ev in range(NV):
                for ix in range(P_X + 1):
                    for iv in range(P_V + 1):
                        var phase_e = ex * NV + ev
                        var idx = (
                            (phase_e * (P_V + 1) + iv) * (P_X + 1) + ix
                        ) * VLASOV_NC
                        var expect = Real(
                            ex * 1000 + ev * 100 + ix * 10 + iv,
                        )
                        assert_true(h_q[idx] == expect, "round-trip")

    # --- app constants & flux helpers ------------------------------------
    var app = VlasovMaxwell(
        charge_mass_ratio=Real(-1.0),
        skin_depth_norm=Real(2.0),
        cfl_max=Real(0.5),
    )
    assert_true(app.charge_mass_ratio == Real(-1.0), "charge_mass_ratio")
    assert_true(app.skin_depth_norm == Real(2.0), "skin_depth_norm")
    assert_equal(VlasovMaxwell.num_components(), VLASOV_NC, "num_components")
    assert_equal(
        VlasovMaxwell.num_aux_components(), VLASOV_AUX_NC, "num_aux_components",
    )

    # v_flux_factor = (q/m) / λ = -1 / 2 = -0.5.
    var vff = VlasovMaxwell.v_flux_factor_static(
        app.charge_mass_ratio, app.skin_depth_norm,
    )
    assert_true(vff == Real(-0.5), "v_flux_factor")

    # 1D1V acceleration with Ex = 0.4: a_v = (-0.5) * 0.4 = -0.2.
    var av = VlasovMaxwell.acceleration_v_1d1v_static(
        Real(0.4), app.charge_mass_ratio, app.skin_depth_norm,
    )
    assert_true(
        (av - Real(-0.2)) ** Real(2.0) < Real(1.0e-28),
        "acceleration_v_1d1v",
    )

    # internal_flux_x: vx * f. With vx = 2 and f = 0.5 → 1.0.
    var f_in = Array[Real, VLASOV_NC](fill=Real(0.5))
    var fx = VlasovMaxwell.internal_flux_x_static[VLASOV_NC](
        f_in, Real(2.0),
    )
    assert_true(fx[0] == Real(1.0), "internal_flux_x")

    # internal_flux_v: av * f. With av = -0.2 and f = 0.5 → -0.1.
    var fv = VlasovMaxwell.internal_flux_v_static[VLASOV_NC](
        f_in, av,
    )
    assert_true(
        (fv[0] - Real(-0.1)) ** Real(2.0) < Real(1.0e-28),
        "internal_flux_v",
    )

    # Upwind numerical flux: positive wave speed pulls f_l, negative pulls f_r.
    var f_l = Array[Real, VLASOV_NC](fill=Real(1.0))
    var f_r = Array[Real, VLASOV_NC](fill=Real(3.0))
    var nf_pos = VlasovMaxwell.numerical_flux_static[VLASOV_NC](
        f_l, f_r, Real(2.0),
    )
    # wave_speed > 0 → upwind ⇒ F* = vx * f_l = 2.0 * 1.0 = 2.0.
    assert_true(nf_pos[0] == Real(2.0), "upwind pos wave_speed")
    var nf_neg = VlasovMaxwell.numerical_flux_static[VLASOV_NC](
        f_l, f_r, Real(-2.0),
    )
    # wave_speed < 0 → upwind ⇒ F* = vx * f_r = -2.0 * 3.0 = -6.0.
    assert_true(nf_neg[0] == Real(-6.0), "upwind neg wave_speed")
    var nf_zero = VlasovMaxwell.numerical_flux_static[VLASOV_NC](
        f_l, f_r, Real(0.0),
    )
    assert_true(nf_zero[0] == Real(0.0), "zero wave_speed flux")

    # CFL: with vx=2, av=-0.2, dx=0.25, dv=0.75 → s = 8 + 0.267 = 8.267,
    # dt_max = 0.5 / 8.267 ≈ 0.06048.
    var dt = app.dt_max_1d1v(Real(2.0), Real(-0.2), Real(0.25), Real(0.75))
    var expected_dt = Real(0.5) / (Real(2.0) / Real(0.25) + fabs(Real(-0.2)) / Real(0.75))
    assert_true(
        (dt - expected_dt) ** Real(2.0) < Real(1.0e-24),
        "dt_max_1d1v",
    )

    # Silence the unused-import warning for AUX_EX in this slice — it'll
    # be the canonical index when the spatial solver reads E_x from the
    # Maxwell aux buffer next slice.
    assert_equal(AUX_EX, 0, "AUX_EX index")

    print("PASSED")
