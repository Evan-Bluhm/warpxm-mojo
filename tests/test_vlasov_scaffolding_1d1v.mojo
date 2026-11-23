# Smoke test for the 1D1V instantiation of the generic kinetic scaffolding.
#
# Verifies, without running the DG kernel, that the pieces the kinetic
# solvers are built from allocate correctly and return the expected
# values:
#
#   * `PhaseMesh[1, 1, P_x, P_v]`: element counts, widths and Jacobians;
#     node coordinates and quadrature weights (through `visit_nodes` and
#     `integrate`); `x_block_element`; the BC kinds handed to the kernels
#     (`kernel_bc`) for periodic and bounded meshes; `check_companion`;
#     the constructor's validation errors. The mesh stores no face
#     tables (neighbours are found by index arithmetic), so there is no
#     connectivity to check.
#   * `PhaseState[1, 1, P_x, P_v, NC]`: buffer length, zero fill, the
#     `index` layout and a round trip through device memory.
#   * `VlasovMaxwell`: constants, `lorentz_acceleration_static`, the
#     internal fluxes and the upwind numerical flux.
#   * `DGVlasovSolver.suggest_dt` against its documented formula.
#
# Build / run:
#     pixi run mojo run -I src tests/test_vlasov_scaffolding_1d1v.mojo

from std.collections import Array
from std.sys import has_accelerator
from max.gpu.host import DeviceContext
from std.testing import assert_true, assert_equal, assert_raises

from warpxm.prelude import *
from warpxm.physics.vlasov import *
from warpxm.apps.vlasovmaxwell import (
    VLASOV_AUX_NC, AUX_EX, AUX_EY, AUX_BY, AUX_BZ,
)
from warpxm.core.bc import BC_INTERIOR


comptime P_X: Int = 2
comptime P_V: Int = 2
comptime NX: Int = 4
comptime NV: Int = 8
comptime X_MIN: Real = 0.0
comptime X_MAX: Real = 1.0
comptime V_MIN: Real = -3.0
comptime V_MAX: Real = 3.0

comptime Mesh = PhaseMesh[1, 1, P_X, P_V]
comptime State = PhaseState[1, 1, P_X, P_V, VLASOV_NC]
comptime Fields = DGState[1, P_X, VLASOV_AUX_NC]


def check_mesh(var ctx: DeviceContext, mesh: Mesh) raises:
    """Counts, widths, node coordinates and quadrature weights."""
    assert_equal(mesh.n[0], NX, "n[0]")
    assert_equal(mesh.n[1], NV, "n[1]")
    assert_equal(mesh.num_x_elements, NX, "num_x_elements")
    assert_equal(mesh.num_v_elements, NV, "num_v_elements")
    assert_equal(mesh.num_phase_elements, NX * NV, "num_phase_elements")
    assert_equal(Mesh.num_x_nodes_per_elem, P_X + 1, "x nodes per element")
    assert_equal(Mesh.num_v_nodes_per_elem, P_V + 1, "v nodes per element")
    assert_equal(
        Mesh.num_nodes_per_elem, (P_X + 1) * (P_V + 1), "nodes per element",
    )

    var expected_dx = (X_MAX - X_MIN) / Real(NX)
    var expected_dv = (V_MAX - V_MIN) / Real(NV)
    assert_true(fabs(mesh.dx[0] - expected_dx) < Real(1.0e-14), "dx")
    assert_true(fabs(mesh.dx[1] - expected_dv) < Real(1.0e-14), "dv")
    assert_true(
        fabs(mesh.jacobian[0] - Real(0.5) * expected_dx) < Real(1.0e-14),
        "x Jacobian",
    )
    assert_true(
        fabs(mesh.jacobian[1] - Real(0.5) * expected_dv) < Real(1.0e-14),
        "v Jacobian",
    )
    assert_true(mesh.lo[0] == X_MIN and mesh.hi[0] == X_MAX, "x extent")
    assert_true(mesh.lo[1] == V_MIN and mesh.hi[1] == V_MAX, "v extent")

    # In 1D the companion BlockMesh element is the x index of the phase
    # element, `phase_e = ex · NV + ev`.
    for ex in range(NX):
        for ev in range(NV):
            assert_equal(
                mesh.x_block_element(ex * NV + ev), ex, "x_block_element",
            )

    # Node coordinates and weights. On a zeroed state `visit_nodes` must
    # reach every phase node once: the GLL endpoints hit the domain
    # extents, and the weights sum to the phase volume.
    var q = State(ctx, mesh.num_phase_elements)
    var count = 0
    var w_sum = Real(0.0)
    var w_min = Real(1.0)
    var x_lo = Real(1.0e30)
    var x_hi = Real(-1.0e30)
    var v_lo = Real(1.0e30)
    var v_hi = Real(-1.0e30)
    var f_abs_max = Real(0.0)
    def survey(x: Real, v: Real, w: Real, f: Real) {
        mut count, mut w_sum, mut w_min, mut x_lo, mut x_hi, mut v_lo,
        mut v_hi, mut f_abs_max,
    }:
        count += 1
        w_sum += w
        w_min = fmin(w_min, w)
        x_lo = fmin(x_lo, x)
        x_hi = fmax(x_hi, x)
        v_lo = fmin(v_lo, v)
        v_hi = fmax(v_hi, v)
        f_abs_max = fmax(f_abs_max, fabs(f))
    mesh.visit_nodes(q, survey)
    var volume = (X_MAX - X_MIN) * (V_MAX - V_MIN)
    assert_equal(count, NX * NV * (P_X + 1) * (P_V + 1), "visited node count")
    assert_true(fabs(w_sum - volume) < Real(1.0e-13), "weights sum to volume")
    assert_true(w_min > Real(0.0), "quadrature weights are positive")
    assert_true(fabs(x_lo - X_MIN) < Real(1.0e-12), "leftmost x-node")
    assert_true(fabs(x_hi - X_MAX) < Real(1.0e-12), "rightmost x-node")
    assert_true(fabs(v_lo - V_MIN) < Real(1.0e-12), "lowest v-node")
    assert_true(fabs(v_hi - V_MAX) < Real(1.0e-12), "highest v-node")
    assert_true(f_abs_max == Real(0.0), "state not zero-initialised")

    # `fill` and `visit_nodes` agree on the node ordering, and the GLL
    # rule integrates a polynomial of degree ≤ 2P − 1 = 3 exactly:
    # ∫∫ (x + 10 v + 3 v²) dx dv = 6 · ½ + 0 + 3 · 18 = 57.
    def poly(x: Real, v: Real) -> Real:
        return x + Real(10.0) * v + Real(3.0) * v * v
    mesh.fill(q, poly)
    var mismatch = Real(0.0)
    def compare(x: Real, v: Real, w: Real, f: Real) {mut mismatch}:
        mismatch = fmax(
            mismatch, fabs(f - (x + Real(10.0) * v + Real(3.0) * v * v)),
        )
    mesh.visit_nodes(q, compare)
    assert_true(mismatch == Real(0.0), "fill / visit_nodes disagree")
    var integral = mesh.integrate(q)
    assert_true(fabs(integral - Real(57.0)) < Real(1.0e-12), "integrate")

    print(
        "PhaseMesh[1, 1]: phase elements =", mesh.num_phase_elements,
        "nodes =", count, "Σw =", w_sum, "∫ poly =", integral,
    )


def check_boundaries(var ctx: DeviceContext, mesh: Mesh) raises:
    """BC kinds handed to the kernels, companion-mesh check, and the
    constructor's validation."""
    # Periodic x: both sides read as interior faces.
    var bc_periodic = mesh.kernel_bc()
    assert_equal(Int(bc_periodic[0]), BC_INTERIOR, "periodic left")
    assert_equal(Int(bc_periodic[1]), BC_INTERIOR, "periodic right")
    var counts = mesh.kernel_counts()
    assert_equal(Int(counts[0]), NX, "kernel_counts x")
    assert_equal(Int(counts[1]), NV, "kernel_counts v")

    # Bounded x without `bc`: copy-out on both sides.
    var bounded = Mesh(
        ctx, lo=[X_MIN, V_MIN], hi=[X_MAX, V_MAX], n=[NX, NV],
        periodic=[False],
    )
    var bc_default = bounded.kernel_bc()
    assert_equal(Int(bc_default[0]), BC_COPYOUT, "default left")
    assert_equal(Int(bc_default[1]), BC_COPYOUT, "default right")

    # Bounded x with one kind per side.
    var walled = Mesh(
        ctx, lo=[X_MIN, V_MIN], hi=[X_MAX, V_MAX], n=[NX, NV],
        periodic=[False], bc=[BC_VLASOV_OUTFLOW, BC_VLASOV_REFLECT],
    )
    var bc_walled = walled.kernel_bc()
    assert_equal(Int(bc_walled[0]), BC_VLASOV_OUTFLOW, "outflow left")
    assert_equal(Int(bc_walled[1]), BC_VLASOV_REFLECT, "reflecting right")

    # A `bc` on a periodic axis is ignored.
    var periodic_with_bc = Mesh(
        ctx, lo=[X_MIN, V_MIN], hi=[X_MAX, V_MAX], n=[NX, NV],
        periodic=[True], bc=[BC_VLASOV_OUTFLOW, BC_VLASOV_REFLECT],
    )
    assert_equal(
        Int(periodic_with_bc.kernel_bc()[0]), BC_INTERIOR,
        "bc on a periodic axis",
    )

    # Companion configuration mesh: same extent, count and periodicity.
    var x_mesh = BlockMesh[1, P_X](ctx, X_MIN, X_MAX, NX, periodic=True)
    mesh.check_companion(x_mesh)
    with assert_raises():
        var other_count = BlockMesh[1, P_X](
            ctx, X_MIN, X_MAX, NX + 1, periodic=True,
        )
        mesh.check_companion(other_count)
    with assert_raises():
        var other_extent = BlockMesh[1, P_X](
            ctx, X_MIN, Real(2.0) * X_MAX, NX, periodic=True,
        )
        mesh.check_companion(other_extent)
    with assert_raises():
        var other_periodicity = BlockMesh[1, P_X](
            ctx, X_MIN, X_MAX, NX, periodic=False,
        )
        mesh.check_companion(other_periodicity)

    # Constructor validation.
    with assert_raises():  # no elements along v
        _ = Mesh(
            ctx, lo=[X_MIN, V_MIN], hi=[X_MAX, V_MAX], n=[NX, 0],
            periodic=[True],
        )
    with assert_raises():  # empty x interval
        _ = Mesh(
            ctx, lo=[X_MAX, V_MIN], hi=[X_MAX, V_MAX], n=[NX, NV],
            periodic=[True],
        )
    with assert_raises():  # a fluid BC kind on a configuration boundary
        _ = Mesh(
            ctx, lo=[X_MIN, V_MIN], hi=[X_MAX, V_MAX], n=[NX, NV],
            periodic=[False], bc=[BC_PEC_WALL, BC_COPYOUT],
        )
    with assert_raises():  # specular wall needs a v mesh symmetric about 0
        _ = Mesh(
            ctx, lo=[X_MIN, Real(-2.0)], hi=[X_MAX, V_MAX], n=[NX, NV],
            periodic=[False], bc=[BC_VLASOV_REFLECT, BC_COPYOUT],
        )


def check_state(var ctx: DeviceContext, mesh: Mesh) raises:
    """Buffer length, zero fill, index layout and a device round trip."""
    var q = State(ctx, mesh.num_phase_elements)
    comptime NODES = (P_X + 1) * (P_V + 1)
    assert_equal(State.num_nodes_per_elem, NODES, "nodes per phase element")
    assert_equal(q.num_phase_elements, NX * NV, "num_phase_elements")
    assert_equal(q.total_nodes, NX * NV * NODES, "total_nodes")
    assert_equal(q.buf_len, NX * NV * NODES * VLASOV_NC, "buf_len")

    with q.buf.map_to_host() as h_q:
        for k in range(q.buf_len):
            assert_true(h_q[k] == Real(0.0), "buffer not zero-initialised")

    # Layout: phase element `ex · NV + ev` (v innermost), local node
    # `iv · (P_x + 1) + ix` (x innermost), then the component.
    assert_equal(State.index(0, 0, 0), 0, "index origin")
    assert_equal(State.index(0, 1, 0), VLASOV_NC, "index node stride")
    assert_equal(State.index(1, 0, 0), NODES * VLASOV_NC, "index elem stride")
    assert_equal(
        State.index(NX * NV - 1, NODES - 1, VLASOV_NC - 1), q.buf_len - 1,
        "index of the last entry",
    )

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
                    assert_equal(
                        State.index(phase_e, iv * (P_X + 1) + ix, 0), idx,
                        "index layout",
                    )
                    host[idx] = Real(ex * 1000 + ev * 100 + ix * 10 + iv)
    ctx.enqueue_copy(dst_buf=q.buf, src_buf=host)
    ctx.synchronize()
    with q.buf.map_to_host() as h_q:
        for ex in range(NX):
            for ev in range(NV):
                for ix in range(P_X + 1):
                    for iv in range(P_V + 1):
                        var idx = State.index(
                            ex * NV + ev, iv * (P_X + 1) + ix, 0,
                        )
                        var expect = Real(
                            ex * 1000 + ev * 100 + ix * 10 + iv,
                        )
                        assert_true(h_q[idx] == expect, "round-trip")


def check_app_and_dt(var ctx: DeviceContext, mesh: Mesh) raises:
    """App constants, flux helpers and the solver's step suggestion."""
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
    assert_equal(AUX_EX, 0, "AUX_EX index")

    # v_flux_factor = (q/m) / λ = -1 / 2 = -0.5.
    var vff = VlasovMaxwell.v_flux_factor_static(
        app.charge_mass_ratio, app.skin_depth_norm,
    )
    assert_true(vff == Real(-0.5), "v_flux_factor")

    # 1V Lorentz acceleration with Ex = 0.4: a_v = (-0.5) * 0.4 = -0.2.
    # With one velocity dimension v × B has no component along v_1, so
    # the other field components and the velocity must not enter.
    var av = VlasovMaxwell.lorentz_acceleration_static[1, 0](
        Real(0.4), Real(0.0), Real(0.0),
        Real(0.0), Real(0.0), Real(0.0),
        Real(0.0), Real(0.0), Real(0.0),
        vff,
    )
    assert_true(fabs(av - Real(-0.2)) < Real(1.0e-14), "lorentz acceleration")
    var av_with_b = VlasovMaxwell.lorentz_acceleration_static[1, 0](
        Real(0.4), Real(7.0), Real(-5.0),
        Real(3.0), Real(-2.0), Real(11.0),
        Real(1.5), Real(-2.5), Real(3.5),
        vff,
    )
    assert_true(av_with_b == av, "1V acceleration depends on E_x only")

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
    assert_true(fabs(fv[0] - Real(-0.1)) < Real(1.0e-14), "internal_flux_v")

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

    # CFL. `suggest_dt` returns
    #     cfl_max / ((2 P_x + 1) max|v| / Δx + (2 P_v + 1) max|a| / Δv)
    # with max|v| the edge of the velocity mesh and, in 1V,
    # max|a| = |q/m| / λ · max|E_x|.
    var spatial = DGVlasovSolver[1, 1, P_X, P_V](app)
    var fields = Fields(ctx, mesh.num_x_elements)
    var dx = (X_MAX - X_MIN) / Real(NX)
    var dv = (V_MAX - V_MIN) / Real(NV)

    # Zero fields: only the streaming term. 0.5 / (5 · 3 / 0.25) = 1/120.
    var dt_free = spatial.suggest_dt(ctx, mesh, fields)
    var expected_free = Real(0.5) / (Real(2 * P_X + 1) * V_MAX / dx)
    assert_true(
        fabs(dt_free - expected_free) < Real(1.0e-14), "suggest_dt, no field",
    )

    # E_x = 0.1 everywhere except one node at -0.4, so max|E_x| = 0.4 and
    # max|a| = 0.5 · 0.4 = 0.2: 0.5 / (60 + 5 · 0.2 / 0.75). The other
    # field components are large but cannot accelerate along v_1 in 1V.
    with fields.buf.map_to_host() as h:
        for e in range(NX):
            for n in range(P_X + 1):
                h[Fields.index(e, n, AUX_EX)] = Real(0.1)
                h[Fields.index(e, n, AUX_EY)] = Real(50.0)
                h[Fields.index(e, n, AUX_BY)] = Real(-70.0)
                h[Fields.index(e, n, AUX_BZ)] = Real(90.0)
        h[Fields.index(NX - 2, 1, AUX_EX)] = Real(-0.4)
    var dt_field = spatial.suggest_dt(ctx, mesh, fields)
    var expected_field = Real(0.5) / (
        Real(2 * P_X + 1) * V_MAX / dx
        + Real(2 * P_V + 1) * Real(0.2) / dv
    )
    assert_true(
        fabs(dt_field - expected_field) < Real(1.0e-14), "suggest_dt, E_x",
    )
    print(
        "suggest_dt: no field =", dt_free, " with E_x =", dt_field,
        " expected =", expected_field,
    )


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"

    var ctx = DeviceContext()
    var mesh = Mesh(
        ctx, lo=[X_MIN, V_MIN], hi=[X_MAX, V_MAX], n=[NX, NV],
        periodic=[True],
    )
    check_mesh(ctx, mesh)
    check_boundaries(ctx, mesh)
    check_state(ctx, mesh)
    check_app_and_dt(ctx, mesh)

    print("PASSED")
